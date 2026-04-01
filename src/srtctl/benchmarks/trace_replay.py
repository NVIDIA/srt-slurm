# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace replay benchmark runner using aiperf.

Replays request traces with fixed timestamps to benchmark real-world workload
patterns, including multi-turn conversations with prefix caching potential.

Uses aiperf with --custom-dataset-type mooncake_trace and --fixed-schedule
to replay requests at their original timestamps from the trace file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, AIPerfBenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("trace-replay")
class TraceReplayRunner(AIPerfBenchmarkRunner):
    """Trace replay benchmark for testing with real-world request patterns.

    Replays requests from a trace file at their original timestamps using aiperf's
    fixed-schedule mode. Useful for benchmarking prefix caching, KV-aware routing,
    and real-world workload performance.

    Trace format (JSONL, one object per line):
        {"timestamp": <ms>, "input_length": <int>, "output_length": <int>,
         "hash_ids": [...], "context_len": <int>, "unique_user_prompt_len": <int>}

    The trace file format is compatible with Mooncake traces from the FAST25 paper.

    Required config fields:
        - benchmark.trace_file: Path to trace file (JSONL format)

    Optional config fields (in benchmark section):
        - benchmark.ttft_threshold_ms: Goodput TTFT threshold in ms (default: 2000)
        - benchmark.itl_threshold_ms: Goodput ITL threshold in ms (default: 25)
        - benchmark.concurrencies: List of concurrency levels to sweep (default: [1, 5, 25, 50])
                                   Can be a list [1, 5, 25, 50] or string "1x5x25x50"

    Profiling support:
        - profiling.type: "nsys" (iteration-based) or "nsys-time" (time-based) with trtllm backend
        - For nsys: set profiling.prefill.start_step/stop_step and profiling.decode.start_step/stop_step
        - For nsys-time: set profiling.delay_secs, profiling.duration_secs, profiling.benchmark_duration_secs

    Example config:
        benchmark:
          type: "trace-replay"
          trace_file: "traces/conversation_trace.jsonl"  # relative to workspace
          ttft_threshold_ms: 2000
          itl_threshold_ms: 25
          concurrencies: [1, 5, 25, 50]

    The trace file directory is automatically mounted into the container.
    """

    # Container mount point for trace files
    TRACE_MOUNT_PATH = Path("/trace-data")

    @property
    def name(self) -> str:
        return "Trace-Replay-Bench"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/trace-replay/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "trace-replay")

    def _resolve_trace_path(self, trace_file: str) -> Path:
        """Resolve trace file path to absolute path.

        Handles both absolute paths and paths relative to the workspace.
        """
        trace_path = Path(trace_file)
        if trace_path.is_absolute():
            return trace_path
        # Resolve relative to current working directory (workspace root)
        return Path.cwd() / trace_path

    def get_extra_mounts(self, config: SrtConfig) -> dict[Path, Path]:
        """Mount the trace file's directory into the container."""
        if config.benchmark.trace_file is None:
            return {}

        trace_path = self._resolve_trace_path(config.benchmark.trace_file)
        trace_dir = trace_path.parent.resolve()

        return {trace_dir: self.TRACE_MOUNT_PATH}

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        # trace_file is required
        if b.trace_file is None:
            errors.append("benchmark.trace_file is required for trace-replay benchmark")
        else:
            # Check that trace file exists
            trace_path = self._resolve_trace_path(b.trace_file)
            if not trace_path.exists():
                errors.append(f"Trace file not found: {trace_path}")

        # Validate thresholds if specified
        if b.ttft_threshold_ms is not None and b.ttft_threshold_ms <= 0:
            errors.append(f"benchmark.ttft_threshold_ms must be positive, got: {b.ttft_threshold_ms}")

        if b.itl_threshold_ms is not None and b.itl_threshold_ms <= 0:
            errors.append(f"benchmark.itl_threshold_ms must be positive, got: {b.itl_threshold_ms}")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"

        # Get model name - try served_model_name first, then model path
        model_name = config.served_model_name or config.model.path

        # Get benchmark parameters with defaults
        ttft_threshold = b.ttft_threshold_ms or 2000
        itl_threshold = b.itl_threshold_ms or 25

        # Get concurrency list - use default if not specified
        concurrency_list = b.get_concurrency_list()
        if not concurrency_list:
            concurrency_list = [1, 5, 25, 50]  # Default values
        concurrencies_str = "x".join(str(c) for c in concurrency_list)

        # Build container path for trace file
        trace_path = self._resolve_trace_path(b.trace_file)
        container_trace_path = self.TRACE_MOUNT_PATH / trace_path.name

        return [
            "bash",
            self.script_path,
            endpoint,
            model_name,
            str(container_trace_path),
            str(ttft_threshold),
            str(itl_threshold),
            concurrencies_str,
        ]
