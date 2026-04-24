# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf synthetic ISL/OSL benchmark runner.

Uses aiperf's built-in synthetic dataset generation to benchmark LLM serving
at specific input/output sequence lengths, sweeping concurrency levels.

Unlike trace-replay (which replays a trace file) or mooncake-router (which uses
a fixed-schedule trace), this benchmark generates synthetic requests on-the-fly
using aiperf's --synthetic-input-tokens-mean and --output-tokens-mean flags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, AIPerfBenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("aiperf-bench")
class AIPerfBenchRunner(AIPerfBenchmarkRunner):
    """Synthetic ISL/OSL benchmark using aiperf's built-in dataset generation.

    Uses --synthetic-input-tokens-mean and --output-tokens-mean to generate
    requests at specific input/output lengths, sweeping concurrency levels.
    Comparable to sa-bench but uses aiperf as the benchmark engine, which
    provides goodput metrics (TTFT/ITL thresholds) alongside throughput.

    Required config fields:
        - benchmark.isl: Input sequence length (mean tokens per request)
        - benchmark.osl: Output sequence length (mean tokens per response)
        - benchmark.concurrencies: Concurrency levels to sweep (e.g., "4x8x16" or [4, 8, 16])

    Optional config fields:
        - benchmark.ttft_threshold_ms: Goodput TTFT threshold in ms (default: 2000)
        - benchmark.itl_threshold_ms: Goodput ITL threshold in ms (default: 25)
        - benchmark.random_range_ratio: Controls input length variance.
            stddev = isl * (1 - ratio) / 2. Default 1.0 = no variance (fixed length).
    """

    @property
    def name(self) -> str:
        return "AIPerf Bench"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/aiperf-bench/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "aiperf-bench")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        if b.isl is None:
            errors.append("benchmark.isl is required for aiperf-bench")
        if b.osl is None:
            errors.append("benchmark.osl is required for aiperf-bench")
        if b.concurrencies is None:
            errors.append("benchmark.concurrencies is required for aiperf-bench")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"
        model_name = config.served_model_name or config.model.path

        # Format concurrencies as comma-separated string (handles both list and "NxM" string formats)
        concurrencies = ",".join(str(c) for c in b.get_concurrency_list())

        ttft_threshold = b.ttft_threshold_ms or 2000
        itl_threshold = b.itl_threshold_ms or 25

        # Derive stddev from random_range_ratio: ratio=0.8 → stddev = isl*0.1
        # Default ratio=1.0 → stddev=0 (fixed length, no variance)
        isl_stddev = int(b.isl * (1.0 - (b.random_range_ratio or 1.0)) / 2) if b.isl else 0

        tokenizer_path = str(runtime.model_path) if runtime.is_hf_model else "/model"

        req_rate = str(b.req_rate) if b.req_rate is not None else "inf"

        cmd = [
            "bash",
            self.script_path,
            endpoint,
            model_name,
            str(b.isl),
            str(b.osl),
            str(concurrencies) if concurrencies else "",
            str(ttft_threshold),
            str(itl_threshold),
            str(isl_stddev),
            tokenizer_path,
            req_rate,
        ]

        self.append_aiperf_args(cmd, config)

        if config.benchmark.enable_dcgm:
            dcgm_urls = [f"http://{node}:9400/metrics" for node in runtime.nodes.worker]
            cmd.extend(["--gpu-telemetry"] + dcgm_urls)

        return cmd
