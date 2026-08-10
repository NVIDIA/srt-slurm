# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("router")
class RouterRunner(BenchmarkRunner):
    """Router performance benchmark.

    Tests sglang-router specifically with prefix caching.

    Optional config fields:
        - benchmark.isl: Input sequence length (default: 14000)
        - benchmark.osl: Output sequence length (default: 200)
        - benchmark.num_requests: Number of requests (default: 200)
        - benchmark.concurrency: Concurrency level (default: 20)
        - benchmark.prefix_ratios: Prefix ratios to test (default: "0.1 0.3 0.5 0.7 0.9")
    """

    @property
    def name(self) -> str:
        return "Router Benchmark"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/router/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "router")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []

        # Router benchmark exercises the SGLang router's prefix-aware policies.
        if config.frontend.type != "sglang":
            errors.append("router benchmark requires frontend.type: sglang")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"

        # Format prefix ratios
        prefix_ratios = b.prefix_ratios or "0.1 0.3 0.5 0.7 0.9"
        if isinstance(prefix_ratios, list):
            prefix_ratios = " ".join(str(r) for r in prefix_ratios)

        return [
            "bash",
            self.script_path,
            endpoint,
            str(b.isl or 14000),
            str(b.osl or 200),
            str(b.num_requests or 200),
            str(getattr(b, "concurrency", None) or 20),
            prefix_ratios,
        ]
