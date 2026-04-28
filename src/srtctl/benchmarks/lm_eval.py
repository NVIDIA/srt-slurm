# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""lm-eval benchmark runner for self-contained accuracy evals."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("lm-eval")
class LMEvalRunner(BenchmarkRunner):
    """lm-eval accuracy evaluation for OpenAI-compatible chat APIs."""

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
        del config
        return []

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        del config
        endpoint = f"http://localhost:{runtime.frontend_port}"
        return [
            "bash",
            self.script_path,
            endpoint,
        ]

    def get_environment(self, config: SrtConfig, runtime: RuntimeContext) -> dict[str, str]:
        del runtime
        env = dict(config.benchmark.env)
        env.setdefault("MODEL_NAME", config.served_model_name)
        env.setdefault("MODEL_PATH", "/model")
        return env
