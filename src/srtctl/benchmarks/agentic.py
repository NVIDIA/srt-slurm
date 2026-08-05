# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentX agentic-code benchmark runner."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, AIPerfBenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


INFERENCEX_AGENTX_COMMIT = "f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd"
AIPERF_AGENTX_REF = "b7b16cf851885567988a643282266bce74e34437"
AGENTX_METHODOLOGY_DEFAULTS = {
    "INFERENCEX_AGENTX_COMMIT": INFERENCEX_AGENTX_COMMIT,
    "AIPERF_AGENTX_REF": AIPERF_AGENTX_REF,
    "AIPERF_TRACE_IDLE_GAP_CAP_SECONDS": "300",
    "AIPERF_WARMUP_REQUESTS_PER_LANE": "10",
    "AGENTIC_WARMUP_GRACE_PERIOD": "1800",
}


def _env_or_attr(config: SrtConfig, attr: str, env_name: str, default: object | None = None) -> object | None:
    value = getattr(config.benchmark, attr, None)
    if value is not None:
        return value
    return config.benchmark.env.get(env_name, default)


def _derive_model_prefix(model_name: str) -> str:
    normalized = model_name.lower()
    if "minimax" in normalized and "m3" in normalized:
        return "minimaxm3"
    if "deepseek-v4" in normalized or "dsv4" in normalized:
        return "dsv4"
    prefix = re.sub(r"[^a-z0-9]+", "", normalized.rsplit("/", 1)[-1])
    return prefix or "model"


def _format_concurrencies(config: SrtConfig) -> str:
    b = config.benchmark
    if b.concurrency is not None:
        return str(b.concurrency)
    if isinstance(b.concurrencies, list):
        return " ".join(str(c) for c in b.concurrencies)
    if isinstance(b.concurrencies, str):
        return " ".join(part for part in re.split(r"[x,\s]+", b.concurrencies) if part)
    return ""


@register_benchmark("agentic")
@register_benchmark("agentx")
class AgenticRunner(AIPerfBenchmarkRunner):
    """Run the InferenceX AgentX replay against an srt-slurm-launched server."""

    @property
    def name(self) -> str:
        return "AgentX"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/agentic/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "agentic")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        if b.concurrency is None and b.concurrencies is None:
            errors.append("benchmark.concurrency or benchmark.concurrencies is required for agentic")

        duration = _env_or_attr(config, "duration", "DURATION", 3600)
        try:
            if int(str(duration)) <= 0:
                errors.append(f"benchmark.duration must be positive, got: {duration}")
        except (TypeError, ValueError):
            errors.append(f"benchmark.duration must be an integer, got: {duration}")

        kv_offloading = str(_env_or_attr(config, "kv_offloading", "KV_OFFLOADING", "none") or "none")
        kv_backend = str(_env_or_attr(config, "kv_offload_backend", "KV_OFFLOAD_BACKEND", "") or "")
        kv_backend_metadata_raw = b.env.get("KV_OFFLOAD_BACKEND_METADATA")
        total_cpu_dram = _env_or_attr(config, "total_cpu_dram_gb", "TOTAL_CPU_DRAM_GB", None)

        if kv_offloading not in {"none", "dram"}:
            errors.append(f"benchmark.kv_offloading must be one of none/dram, got: {kv_offloading}")
        elif kv_offloading == "none" and kv_backend not in {"", "none"}:
            errors.append("benchmark.kv_offload_backend must be empty when kv_offloading=none")
        elif kv_offloading == "dram":
            if kv_backend in {"", "none"}:
                errors.append("benchmark.kv_offload_backend is required when kv_offloading=dram")
            try:
                if int(str(total_cpu_dram or 0)) <= 0:
                    errors.append("benchmark.total_cpu_dram_gb must be positive when kv_offloading=dram")
            except (TypeError, ValueError):
                errors.append(f"benchmark.total_cpu_dram_gb must be an integer, got: {total_cpu_dram}")

        kv_backend_metadata = None
        if kv_backend_metadata_raw not in (None, "", "null"):
            try:
                kv_backend_metadata = json.loads(str(kv_backend_metadata_raw))
            except json.JSONDecodeError:
                errors.append("benchmark.env.KV_OFFLOAD_BACKEND_METADATA must contain valid JSON")
            else:
                valid_keys = isinstance(kv_backend_metadata, dict) and set(kv_backend_metadata) in (
                    {"name"},
                    {"name", "version"},
                )
                valid_values = valid_keys and all(
                    isinstance(value, str) and value for value in kv_backend_metadata.values()
                )
                if not valid_values:
                    errors.append(
                        "benchmark.env.KV_OFFLOAD_BACKEND_METADATA must contain a non-empty name and optional version"
                    )
                    kv_backend_metadata = None

        if kv_offloading == "none" and kv_backend_metadata is not None:
            errors.append("benchmark.env.KV_OFFLOAD_BACKEND_METADATA must be empty when kv_offloading=none")
        elif kv_offloading == "dram":
            if kv_backend_metadata is None:
                errors.append("benchmark.env.KV_OFFLOAD_BACKEND_METADATA is required when kv_offloading=dram")
            elif kv_backend_metadata["name"] != kv_backend:
                errors.append("benchmark.env.KV_OFFLOAD_BACKEND_METADATA.name must match kv_offload_backend")

        if any(key.replace("_", "-") == "num-dataset-entries" for key in b.aiperf_args):
            errors.append(
                "Do not set benchmark.aiperf_args.num-dataset-entries for AgentX; "
                "the InferenceX ToT runner fixes the full-corpus ceiling at 393"
            )

        retired_overrides = {
            "AIPERF_AGENTIC_CACHE_WARMUP_DURATION": "duration-based cache warmup",
            "AIPERF_AGENTIC_WARMUP_MAX_TOKENS": "warmup token-count patch",
            "AIPERF_LOCAL_WEKA_DATASET": "patched local-dataset loader",
            "AIPERF_TOKENIZER": "non-ToT tokenizer CLI override",
            "AIPERF_APPLY_CHAT_TEMPLATE": "non-ToT chat-template CLI override",
            "AIPERF_SYNTHESIS_MAX_OSL": "non-ToT synthesis output cap",
            "AIPERF_TRANSFORMERS_SPEC": "AIPerf dependency rewrite",
            "AIPERF_ALLOW_GITHUB_TRANSFORMERS": "AIPerf dependency rewrite bypass",
            "AIPERF_DIR": "caller-provided AIPerf source tree",
            "AGENTX_USE_EXISTING_INFMAX_WORKSPACE": "caller-provided InferenceX workspace",
        }
        for env_name, description in retired_overrides.items():
            if env_name in b.env:
                errors.append(f"Remove benchmark.env.{env_name}; InferenceX ToT no longer uses the {description}")

        for env_name, expected in AGENTX_METHODOLOGY_DEFAULTS.items():
            configured = b.env.get(env_name)
            if configured is not None and str(configured) != expected:
                errors.append(f"benchmark.env.{env_name} must be {expected} to match InferenceX ToT")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        endpoint = f"http://localhost:{runtime.frontend_port}"
        model_name = config.served_model_name or config.model.path
        model_prefix = str(
            _env_or_attr(config, "model_prefix", "MODEL_PREFIX", None) or _derive_model_prefix(model_name)
        )
        framework = config.benchmark.env.get("FRAMEWORK")
        if not framework:
            framework = f"dynamo-{config.backend_type}" if config.frontend.type == "dynamo" else config.backend_type
        precision = config.benchmark.env.get("PRECISION", config.model.precision)
        concurrencies = _format_concurrencies(config)
        duration = str(_env_or_attr(config, "duration", "DURATION", 3600))
        result_filename = str(_env_or_attr(config, "result_filename", "RESULT_FILENAME", config.name))
        kv_offloading = str(_env_or_attr(config, "kv_offloading", "KV_OFFLOADING", "none") or "none")
        kv_backend = str(_env_or_attr(config, "kv_offload_backend", "KV_OFFLOAD_BACKEND", "") or "")
        total_cpu_dram = str(_env_or_attr(config, "total_cpu_dram_gb", "TOTAL_CPU_DRAM_GB", "0") or "0")

        return [
            "bash",
            self.script_path,
            endpoint,
            model_name,
            model_prefix,
            framework,
            precision,
            concurrencies,
            duration,
            result_filename,
            kv_offloading,
            kv_backend,
            total_cpu_dram,
        ]

    def get_environment(self, config: SrtConfig, runtime: RuntimeContext) -> dict[str, str]:
        del runtime
        environment = dict(config.benchmark.env)
        for name, value in AGENTX_METHODOLOGY_DEFAULTS.items():
            environment.setdefault(name, value)
        return environment
