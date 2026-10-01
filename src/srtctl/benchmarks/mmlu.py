# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MMLU accuracy benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("mmlu")
class MMLURunner(BenchmarkRunner):
    """MMLU accuracy evaluation benchmark.

    Uses sglang.test.run_eval with mmlu task.

    Optional config fields:
        - benchmark.num_examples: Number of examples (default: 200)
        - benchmark.max_tokens: Max tokens per response (default: 2048)
        - benchmark.repeat: Number of repeats (default: 8)
        - benchmark.num_threads: Concurrent threads (default: 512)
    """

    # BenchmarkConfig fields this runner reads (beyond the shared ones); see benchmark_config_fields().
    config_fields: ClassVar[frozenset[str]] = frozenset({"num_examples", "max_tokens", "repeat", "num_threads"})

    @property
    def name(self) -> str:
        return "MMLU"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/mmlu/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "mmlu")

    def validate_config(self, config: SrtConfig) -> list[str]:
        # MMLU has sensible defaults for everything
        return []

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"

        return [
            "bash",
            self.script_path,
            endpoint,
            str(b.num_examples or 200),
            str(b.max_tokens or 2048),
            str(b.repeat or 8),
            str(b.num_threads or 512),
        ]
