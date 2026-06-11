# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPQA accuracy benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("gpqa")
class GPQARunner(BenchmarkRunner):
    """GPQA (Graduate-level science QA) accuracy evaluation.

    Uses sglang.test.run_eval with gpqa task.

    Optional config fields:
        - benchmark.num_examples: Number of examples (default: 198)
        - benchmark.max_tokens: Max tokens per response (default: 32768)
        - benchmark.repeat: Number of repeats (default: 8)
        - benchmark.num_threads: Concurrent threads (default: 128)
    """

    @property
    def name(self) -> str:
        return "GPQA"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/gpqa/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "gpqa")

    def validate_config(self, config: SrtConfig) -> list[str]:
        # GPQA has sensible defaults
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
            str(b.num_examples or 198),
            str(b.max_tokens or 32768),
            str(b.repeat or 8),
            str(b.num_threads or 128),
        ]
