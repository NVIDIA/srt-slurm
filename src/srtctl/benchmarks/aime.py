# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIME accuracy benchmark runner backed by NVIDIA NeMo Skills."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


SUPPORTED_AIME_DATASETS = ("aime24", "aime25", "aime26")


@register_benchmark("aime")
class AIMERunner(BenchmarkRunner):
    """AIME math evaluation using NVIDIA NeMo Skills.

    Optional config fields:
        - benchmark.aime_dataset: NeMo Skills dataset name (default: aime25)
        - benchmark.num_examples: Number of examples (default: all)
        - benchmark.max_tokens: Max tokens per response (default: 24576)
        - benchmark.repeat: Number of sampled repeats (default: 1)
        - benchmark.num_threads: Concurrent requests (default: 30)
        - benchmark.temperature: Sampling temperature (default: 0.0 for repeat=1, 0.7 for repeat>1)
        - benchmark.top_p: Nucleus sampling threshold (default: NeMo Skills default)
        - benchmark.top_k: Top-k sampling (default: NeMo Skills default)
    """

    @property
    def name(self) -> str:
        return "AIME"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/aime/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "aime")

    def validate_config(self, config: SrtConfig) -> list[str]:
        b = config.benchmark
        errors: list[str] = []

        dataset = b.aime_dataset or "aime25"
        if dataset not in SUPPORTED_AIME_DATASETS:
            supported = ", ".join(SUPPORTED_AIME_DATASETS)
            errors.append(f"benchmark.aime_dataset must be one of: {supported}")

        for field in ("num_examples", "max_tokens", "num_threads", "repeat"):
            value = getattr(b, field, None)
            if value is not None and value <= 0:
                errors.append(f"benchmark.{field} must be > 0")

        if b.top_p is not None and not 0 <= b.top_p <= 1:
            errors.append("benchmark.top_p must be between 0 and 1")
        if b.top_k is not None and b.top_k < -1:
            errors.append("benchmark.top_k must be >= -1")
        if b.temperature is not None and b.temperature < 0:
            errors.append("benchmark.temperature must be >= 0")

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
            config.served_model_name,
            b.aime_dataset or "aime25",
            str(b.num_examples) if b.num_examples is not None else "",
            str(b.max_tokens or 24576),
            str(b.num_threads or 30),
            str(b.repeat or 1),
            str(b.temperature) if b.temperature is not None else "",
            str(b.top_p) if b.top_p is not None else "",
            str(b.top_k) if b.top_k is not None else "",
        ]
