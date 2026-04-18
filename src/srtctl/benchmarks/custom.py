# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom benchmark runner."""

from __future__ import annotations

from pathlib import Path

from srtctl.benchmarks.base import BenchmarkRunner, register_benchmark
from srtctl.core.runtime import RuntimeContext
from srtctl.core.schema import SrtConfig


@register_benchmark("custom")
class CustomBenchmarkRunner(BenchmarkRunner):
    """Run an arbitrary benchmark command inside a container."""

    @property
    def name(self) -> str:
        return "Custom"

    @property
    def script_path(self) -> str:
        return "<custom command>"

    def validate_config(self, config: SrtConfig) -> list[str]:
        if config.benchmark.command:
            return []
        return ["benchmark.command is required for benchmark.type=custom"]

    def build_command(self, config: SrtConfig, runtime: RuntimeContext) -> list[str]:
        del runtime
        assert config.benchmark.command is not None
        return ["bash", "-lc", config.benchmark.command]

    def get_container_image(self, config: SrtConfig, runtime: RuntimeContext) -> str | Path:
        return config.benchmark.container_image or runtime.container_image

    def get_environment(self, config: SrtConfig, runtime: RuntimeContext) -> dict[str, str]:
        del runtime
        return dict(config.benchmark.env)
