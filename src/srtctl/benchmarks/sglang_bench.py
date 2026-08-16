# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang benchmark runner using sglang.bench_serving."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("sglang-bench")
class SGLangBenchRunner(BenchmarkRunner):
    """SGLang benchmark runner.

    Uses sglang.bench_serving to generate traffic. Supports profiling when
    profiling.type is set to "torch" or "nsys".

    Required config fields (in benchmark section):
        - benchmark.isl: Input sequence length
        - benchmark.osl: Output sequence length
        - benchmark.concurrencies: Concurrency levels (e.g., "128x256")

    Optional:
        - benchmark.req_rate: Request rate (default: "inf")
    """

    @property
    def name(self) -> str:
        return "SGLang-Bench"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/sglang-bench/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "sglang-bench")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        if b.isl is None:
            errors.append("benchmark.isl is required for sglang-bench")
        elif b.isl <= 0:
            errors.append("benchmark.isl must be a positive integer for sglang-bench")
        if b.osl is None:
            errors.append("benchmark.osl is required for sglang-bench")
        elif b.osl <= 0:
            errors.append("benchmark.osl must be a positive integer for sglang-bench")
        if b.concurrencies is None:
            errors.append("benchmark.concurrencies is required for sglang-bench")
        else:
            try:
                concurrency_list = b.get_concurrency_list()
            except Exception:  # noqa: BLE001
                concurrency_list = []
                errors.append(
                    "benchmark.concurrencies must be a list of ints or an 'x'-separated string for sglang-bench"
                )

            if not concurrency_list:
                errors.append("benchmark.concurrencies must not be empty for sglang-bench")
            elif any(c <= 0 for c in concurrency_list):
                errors.append("benchmark.concurrencies values must be positive integers for sglang-bench")

        if isinstance(b.req_rate, int) and b.req_rate <= 0:
            errors.append("benchmark.req_rate must be a positive integer or 'inf' for sglang-bench")
        if isinstance(b.req_rate, str) and b.req_rate.strip() in ("0", "0.0"):
            errors.append("benchmark.req_rate must be a positive number or 'inf' for sglang-bench")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"

        # Format concurrencies as x-separated string if it's a list
        concurrencies = b.concurrencies
        if isinstance(concurrencies, list):
            concurrencies = "x".join(str(c) for c in concurrencies)

        return [
            "bash",
            self.script_path,
            endpoint,
            str(b.isl),
            str(b.osl),
            str(concurrencies),
            str(b.req_rate) if b.req_rate is not None else "inf",
        ]
