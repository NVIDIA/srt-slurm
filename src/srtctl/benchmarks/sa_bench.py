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

# Container path where the host dataset cache dir is mounted (see
# benchmark.dataset_cache_dir). Kept stable so cache files written by one job
# are found by the next.
DATASET_CACHE_MOUNT = "/sa-bench-dataset-cache"

# Prompts per concurrency level for the measured and warmup runs. bench.sh
# derives num_prompts from these, and srtctl cache-inputs needs the same
# numbers to know which datasets to pre-generate.
DEFAULT_NUM_PROMPTS_MULT = 10
DEFAULT_NUM_WARMUP_MULT = 2
DEFAULT_RANDOM_RANGE_RATIO = 0.8


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
        - benchmark.dataset_path: Container path to dataset file (required when dataset_name="custom")
        - benchmark.dataset_cache_dir: Host dir for caching generated "random" datasets
        - benchmark.reuse_http_connections: Reuse a benchmark-scoped HTTP connection pool
          for the Dynamo adapter (default: false)
        - benchmark.slow_down_sleep_time / benchmark.slow_down_wait_time: When both are set and
          frontend is sglang, SA-Bench POSTs /slow_down on each decode worker leader (framework-derived
          URLs). Omit either field to disable slow_down.
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

        dataset_name = b.dataset_name or "random"

        # A configured host cache dir is mounted at DATASET_CACHE_MOUNT (see
        # get_container_mounts); bench.sh only ever sees the container path.
        dataset_cache_dir = DATASET_CACHE_MOUNT if b.dataset_cache_dir else ""

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
            str(b.random_range_ratio if b.random_range_ratio is not None else DEFAULT_RANDOM_RANGE_RATIO),
            str(b.num_prompts_mult if b.num_prompts_mult is not None else DEFAULT_NUM_PROMPTS_MULT),
            str(b.num_warmup_mult if b.num_warmup_mult is not None else DEFAULT_NUM_WARMUP_MULT),
            b.custom_tokenizer or "",
            str(b.use_chat_template).lower(),
            dataset_name,
            b.dataset_path or "",
            str(b.reuse_http_connections).lower(),
            dataset_cache_dir,
        ]
        return cmd

    def get_container_mounts(self, config: SrtConfig, runtime: RuntimeContext) -> dict[Path, Path]:
        """Mount the host dataset cache dir into the benchmark container.

        Created on the host when missing so the first run can populate it.
        """
        mounts = dict(runtime.container_mounts)
        cache_dir = config.benchmark.dataset_cache_dir
        if cache_dir:
            host_path = Path(cache_dir).expanduser()
            host_path.mkdir(parents=True, exist_ok=True)
            mounts[host_path.resolve()] = Path(DATASET_CACHE_MOUNT)
        return mounts
