# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIME25 accuracy benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("aime25")
class AIME25Runner(BenchmarkRunner):
    """AIME25 (American Invitational Mathematics Examination 2025) accuracy eval.

    Uses sgl-eval (`sgl-eval run aime25`) against the OpenAI-compatible endpoint.
    Mirrors the GPQA runner; bench.sh maps the args below to sgl-eval flags
    (repeat -> --n-repeats).

    Optional config fields:
        - benchmark.num_examples: Number of examples (default: 30, AIME25 total)
        - benchmark.max_tokens: Max tokens per response (default: 32768)
        - benchmark.repeat: Number of repeats -> sgl-eval --n-repeats (default: 8)
        - benchmark.num_threads: Concurrent threads (default: 128)
        - benchmark.temperature: sgl-eval --temperature (default: sgl-eval's, 0.0 greedy)
        - benchmark.top_p: sgl-eval --top-p (default: sgl-eval's, 1.0)
    """

    @property
    def name(self) -> str:
        return "AIME25"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/aime25/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "aime25")

    def validate_config(self, config: SrtConfig) -> list[str]:
        # AIME25 has sensible defaults
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
            str(b.num_examples or 30),
            str(b.max_tokens or 32768),
            str(b.repeat or 8),
            str(b.num_threads or 128),
            "" if b.temperature is None else str(b.temperature),
            "" if b.top_p is None else str(b.top_p),
        ]
