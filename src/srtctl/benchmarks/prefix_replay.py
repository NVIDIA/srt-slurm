# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-wave exact-prefix replay benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("prefix-replay")
class PrefixReplayRunner(BenchmarkRunner):
    """Seed prefixes once, then replay them to isolate decode work."""

    @property
    def name(self) -> str:
        return "Prefix-Replay-Bench"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/sa-bench/prefix_replay.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "sa-bench")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors: list[str] = []
        b = config.benchmark
        if getattr(config.backend, "type", None) != "vllm":
            errors.append("prefix-replay currently requires the vLLM backend")
        if config.resources.is_disaggregated:
            errors.append("prefix-replay requires an aggregated backend")
        if b.isl is None or b.isl <= 0:
            errors.append("benchmark.isl must be positive for prefix-replay")
        if b.osl is None or b.osl <= 0:
            errors.append("benchmark.osl must be positive for prefix-replay")
        concurrencies = b.get_concurrency_list()
        if len(concurrencies) != 1 or concurrencies[0] <= 0:
            errors.append("prefix-replay requires exactly one positive benchmark concurrency")
        if b.seed_osl is not None and b.seed_osl <= 0:
            errors.append("benchmark.seed_osl must be positive")
        if b.settle_seconds is not None and b.settle_seconds < 0:
            errors.append("benchmark.settle_seconds must be non-negative")
        return errors

    def build_command(self, config: SrtConfig, runtime: RuntimeContext) -> list[str]:
        b = config.benchmark
        r = config.resources
        concurrency = b.get_concurrency_list()[0]
        total_gpus = (r.agg_nodes or 1) * r.gpus_per_node
        aggregated = config.backend.vllm_config.aggregated if config.backend.vllm_config else None
        aggregated = aggregated or {}
        dp_size = int(aggregated.get("data-parallel-size") or aggregated.get("data_parallel_size") or 1)
        tokenizer_path = str(runtime.model_path) if runtime.is_hf_model else "/model"
        endpoint = f"http://localhost:{runtime.frontend_port}"

        return [
            "bash",
            self.script_path,
            endpoint,
            str(b.isl),
            str(b.osl),
            str(concurrency),
            str(b.seed_osl or 1),
            tokenizer_path,
            config.served_model_name,
            str(dp_size),
            str(total_gpus),
            str(b.settle_seconds if b.settle_seconds is not None else 5),
        ]
