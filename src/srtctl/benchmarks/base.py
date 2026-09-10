# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes and registry for benchmark runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


# Path to bundled benchmark scripts
SCRIPTS_DIR = Path(__file__).parent / "scripts"


@runtime_checkable
class PrewarmPlan(Protocol):
    """Work a benchmark can do before its job is submitted.

    ``srtctl cache-inputs`` displays and runs a plan without knowing which
    benchmark produced it, so everything it shows the user comes from here.
    """

    @property
    def title(self) -> str:
        """Heading for the plan summary."""
        ...

    @property
    def summary_rows(self) -> tuple[tuple[str, str], ...]:
        """Label/value pairs describing what the plan would build."""
        ...

    @property
    def done_message(self) -> str:
        """Reported once the plan has run successfully."""
        ...

    def srun_command(self) -> list[str]:
        """The command that does the work, ready to display or execute."""
        ...

    def run(self) -> int:
        """Do the work and return its exit code."""
        ...


class BenchmarkRunner(ABC):
    """Abstract base class that all benchmark runners must inherit."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...

    @property
    @abstractmethod
    def script_path(self) -> str:
        """Path to the benchmark script inside the container."""
        ...

    @abstractmethod
    def validate_config(self, config: SrtConfig) -> list[str]:
        """Validate that config has all required fields.

        Returns:
            List of error messages (empty if valid)
        """
        ...

    @abstractmethod
    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        """Build the command to run the benchmark.

        Args:
            config: Full job configuration
            runtime: Runtime context with resolved paths

        Returns:
            Command as list of strings
        """
        ...

    def get_container_image(self, config: SrtConfig, runtime: RuntimeContext) -> str | Path:
        """Get the container image used for the benchmark process."""
        return runtime.container_image

    def get_container_mounts(self, config: SrtConfig, runtime: RuntimeContext) -> dict[Path, Path]:
        """Get mounts used for the benchmark process."""
        return runtime.container_mounts

    def get_environment(self, config: SrtConfig, runtime: RuntimeContext) -> dict[str, str]:
        """Get benchmark-specific environment variables."""
        return {}

    def plan_prewarm(
        self,
        config: SrtConfig,
        *,
        account: str | None = None,
        partition: str | None = None,
        time_limit: str | None = None,
        num_workers: int | None = None,
    ) -> PrewarmPlan | None:
        """Plan work this benchmark can do before its job is submitted.

        None means the benchmark has nothing to build ahead of time. Raise
        ValueError when it could, but the recipe is missing a field to do it.
        """
        return None


class AIPerfBenchmarkRunner(BenchmarkRunner):
    """Base class for AIPerf-driven benchmarks.

    Provides shared aiperf_args handling for subclasses.
    """

    def append_aiperf_args(self, cmd: list[str], config: SrtConfig) -> list[str]:
        """Append aiperf_args from config as CLI flags."""
        for key, value in config.benchmark.aiperf_args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        return cmd


# Registry of benchmark runners
_BENCHMARK_RUNNERS: dict[str, type[BenchmarkRunner]] = {}


def register_benchmark(name: str):
    """Decorator to register a benchmark runner class.

    Usage:
        @register_benchmark("sa-bench")
        class SABenchRunner(BenchmarkRunner):
            ...
    """

    def decorator(cls: type[BenchmarkRunner]) -> type[BenchmarkRunner]:
        _BENCHMARK_RUNNERS[name] = cls
        return cls

    return decorator


def get_runner(benchmark_type: str) -> BenchmarkRunner:
    """Get a runner instance for the given benchmark type.

    Args:
        benchmark_type: Type of benchmark (e.g., "sa-bench", "mmlu")

    Returns:
        Instantiated runner

    Raises:
        ValueError: If benchmark type is not registered
    """
    if benchmark_type not in _BENCHMARK_RUNNERS:
        available = ", ".join(sorted(_BENCHMARK_RUNNERS.keys()))
        raise ValueError(f"Unknown benchmark type: {benchmark_type}. Available: {available}")
    return _BENCHMARK_RUNNERS[benchmark_type]()


def list_benchmarks() -> list[str]:
    """List all registered benchmark types."""
    return sorted(_BENCHMARK_RUNNERS.keys())
