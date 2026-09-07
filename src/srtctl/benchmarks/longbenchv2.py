# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LongBench v2 benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("longbenchv2")
class LongBenchV2Runner(BenchmarkRunner):
    """LongBench v2 long-context evaluation benchmark.

    Tests model performance on long-context tasks.

    Optional config fields:
        - benchmark.max_context_length: Max context length (default: 128000)
        - benchmark.num_threads: Concurrent threads (default: 16)
        - benchmark.max_tokens: Max tokens (default: 16384)
        - benchmark.num_examples: Number of examples (default: all)
        - benchmark.categories: Task categories to run (default: all)
    """

    # BenchmarkConfig fields this runner reads (beyond the shared ones); see benchmark_config_fields().
    config_fields: ClassVar[frozenset[str]] = frozenset(
        {"num_examples", "max_tokens", "num_threads", "max_context_length", "categories"}
    )

    @property
    def name(self) -> str:
        return "LongBench-v2"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/longbenchv2/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "longbenchv2")

    def validate_config(self, config: SrtConfig) -> list[str]:
        # Has sensible defaults
        return []

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"

        # Build categories string if provided
        categories = ""
        if b.categories:
            categories = ",".join(b.categories)

        return [
            "bash",
            self.script_path,
            endpoint,
            str(b.max_context_length or 128000),
            str(b.num_threads or 16),
            str(b.max_tokens or 16384),
            str(b.num_examples or ""),
            categories,
        ]
