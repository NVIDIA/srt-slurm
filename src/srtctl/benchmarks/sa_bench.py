# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SA-Bench throughput/latency benchmark runner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

CONTAINER_DATASET_DIR = Path("/benchmark-dataset")


@register_benchmark("sa-bench")
class SABenchRunner(BenchmarkRunner):
    """SA-Bench throughput and latency benchmark.

    Tests serving throughput at various concurrency levels.

    Required config fields:
        - benchmark.concurrencies: Concurrency levels (e.g., "4x8x16x32")
        - benchmark.isl / benchmark.osl: Required when dataset_name is "random" (default)

    Optional:
        - benchmark.req_rate: Request rate (default: "inf")
        - benchmark.dataset_name: "random" (default) or "custom"
        - benchmark.dataset_path: Host path to JSONL file (required when dataset_name="custom")
    """

    @property
    def name(self) -> str:
        return "SA-Bench"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/sa-bench/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "sa-bench")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        is_custom = b.dataset_name == "custom"

        if not is_custom:
            if b.isl is None:
                errors.append("benchmark.isl is required for sa-bench (when dataset_name is not 'custom')")
            if b.osl is None:
                errors.append("benchmark.osl is required for sa-bench (when dataset_name is not 'custom')")
        if b.concurrencies is None:
            errors.append("benchmark.concurrencies is required for sa-bench")
        if is_custom and not b.dataset_path:
            errors.append("benchmark.dataset_path is required when dataset_name='custom'")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        r = config.resources
        endpoint = f"http://localhost:{runtime.frontend_port}"

        # Format concurrencies as x-separated string if it's a list
        concurrencies = b.concurrencies
        if isinstance(concurrencies, list):
            concurrencies = "x".join(str(c) for c in concurrencies)

        # Compute GPU info for result filename
        is_disaggregated = r.is_disaggregated
        if is_disaggregated:
            prefill_gpus = r.prefill_gpus
            decode_gpus = r.decode_gpus
            total_gpus = prefill_gpus + decode_gpus
        else:
            total_gpus = (r.agg_nodes or 1) * r.gpus_per_node
            prefill_gpus = 0
            decode_gpus = 0

        # Tokenizer path: HF model ID or container mount path
        tokenizer_path = str(runtime.model_path) if runtime.is_hf_model else "/model"

        # Resolve dataset name and container path
        dataset_name = b.dataset_name or "random"
        container_dataset_path = ""
        if dataset_name == "custom" and b.dataset_path:
            container_dataset_path = str(CONTAINER_DATASET_DIR / Path(b.dataset_path).name)

        cmd = [
            "bash",
            self.script_path,
            endpoint,
            str(b.isl or 0),
            str(b.osl or 0),
            str(concurrencies) if concurrencies else "",
            str(b.req_rate) if b.req_rate else "inf",
            tokenizer_path,
            config.served_model_name,
            str(is_disaggregated).lower(),
            str(total_gpus),
            str(prefill_gpus),
            str(decode_gpus),
            str(b.random_range_ratio) if b.random_range_ratio is not None else "0.8",
            str(b.num_prompts_mult) if b.num_prompts_mult is not None else "10",
            str(b.num_warmup_mult) if b.num_warmup_mult is not None else "2",
            b.custom_tokenizer or "",
            str(b.use_chat_template).lower(),
            dataset_name,
            container_dataset_path,
        ]
        return cmd

    def get_container_mounts(self, config: SrtConfig, runtime: RuntimeContext) -> dict[Path, Path]:
        mounts = dict(runtime.container_mounts)
        b = config.benchmark
        if b.dataset_name == "custom" and b.dataset_path:
            host_path = Path(b.dataset_path).resolve().parent
            mounts[host_path] = CONTAINER_DATASET_DIR
        return mounts
