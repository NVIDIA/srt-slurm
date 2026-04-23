# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""lm-eval benchmark runner.

Runs the EleutherAI lm-evaluation-harness against the deployed OpenAI-compatible
endpoint. An external harness that exposes ``benchmark_lib.sh`` is used instead
when available — either via the ``LM_EVAL_LIB`` env var, or a workspace mounted
at ``/lm-eval-workspace`` (set host-side with ``LM_EVAL_WORKSPACE``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("lm-eval")
class LMEvalRunner(BenchmarkRunner):
    """lm-eval accuracy evaluation runner."""

    @property
    def name(self) -> str:
        return "lm-eval"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/lm-eval/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "lm-eval")

    def validate_config(self, config: SrtConfig) -> list[str]:
        # lm-eval has sensible defaults.
        return []

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        endpoint = f"http://localhost:{runtime.frontend_port}"
        return [
            "bash",
            self.script_path,
            endpoint,
        ]
