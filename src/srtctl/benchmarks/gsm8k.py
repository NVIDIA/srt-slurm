# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GSM8K accuracy benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("gsm8k")
class GSM8KRunner(BenchmarkRunner):
    """GSM8K (Grade School Math 8K) accuracy evaluation.

    Uses sglang.test.run_eval with gsm8k task.

    Optional config fields:
        - benchmark.num_examples: Number of examples (default: 1319)
        - benchmark.max_tokens: Max tokens per response (default: 16384)
        - benchmark.num_threads: Concurrent threads (default: 512)
        - benchmark.num_shots: Few-shot examples (default: 5)
        - benchmark.temperature: Sampling temperature (default: server default)
        - benchmark.top_p: Nucleus sampling threshold (default: server default)
        - benchmark.top_k: Top-k sampling (default: server default)
    """

    @property
    def name(self) -> str:
        return "GSM8K"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/gsm8k/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "gsm8k")

    def validate_config(self, config: SrtConfig) -> list[str]:
        b = config.benchmark
        errors: list[str] = []
        for field in ("num_examples", "max_tokens", "num_threads"):
            value = getattr(b, field, None)
            if value is not None and value <= 0:
                errors.append(f"benchmark.{field} must be > 0")
        if b.num_shots is not None and b.num_shots < 0:
            errors.append("benchmark.num_shots must be >= 0")
        return errors

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
            str(1319 if b.num_examples is None else b.num_examples),
            str(16384 if b.max_tokens is None else b.max_tokens),
            str(512 if b.num_threads is None else b.num_threads),
            str(5 if b.num_shots is None else b.num_shots),
            str(b.temperature) if b.temperature is not None else "",
            str(b.top_p) if b.top_p is not None else "",
            str(b.top_k) if b.top_k is not None else "",
        ]
