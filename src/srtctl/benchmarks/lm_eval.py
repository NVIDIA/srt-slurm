# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""lm-eval benchmark runner for InferenceX evals."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("lm-eval")
class LMEvalRunner(BenchmarkRunner):
    """lm-eval accuracy evaluation using InferenceX benchmark_lib.

    Runs lm-eval via the InferenceX benchmark_lib.sh harness,
    which handles task selection, result collection, and summary generation.
    """

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
        # lm-eval has sensible defaults
        return []

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        endpoint = f"http://localhost:{runtime.frontend_port}"
        # Always use the container mount path, not the host path.
        # INFMAX_WORKSPACE env var contains the host path (used for mount setup
        # in runtime.py), but inside the container it's at /infmax-workspace.
        infmax_workspace = "/infmax-workspace"

        return [
            "bash",
            self.script_path,
            endpoint,
            infmax_workspace,
        ]
