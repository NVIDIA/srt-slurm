# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace replay benchmark runner using aiperf.

Replays a user-provided JSONL trace dataset at configurable concurrency levels.
Uses aiperf with --custom-dataset-type mooncake_trace for trace files in the
mooncake format (session_id, input_length, output_length, hash_ids, delay).

Unlike mooncake-router (which downloads a fixed trace and uses --fixed-schedule),
trace-replay takes a user-provided file and sweeps concurrency levels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, AIPerfBenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("trace-replay")
class TraceReplayRunner(AIPerfBenchmarkRunner):
    """Trace replay benchmark using aiperf with a user-provided dataset.

    Replays a JSONL trace file at various concurrency levels to measure
    serving performance with realistic request patterns.

    Required config fields:
        - benchmark.trace_file: Path to trace JSONL file (container path)
        - benchmark.concurrencies: Concurrency levels to sweep

    Optional config fields:
        - benchmark.ttft_threshold_ms: Goodput TTFT threshold (default: 2000)
        - benchmark.itl_threshold_ms: Goodput ITL threshold (default: 25)
    """

    @property
    def name(self) -> str:
        return "Trace Replay"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/trace-replay/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "trace-replay")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        if not b.trace_file:
            errors.append("benchmark.trace_file is required for trace-replay")

        if b.concurrencies is None:
            errors.append("benchmark.concurrencies is required for trace-replay")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"

        model_name = config.served_model_name or config.model.path

        # Format concurrencies as comma-separated string
        concurrencies = b.concurrencies
        if isinstance(concurrencies, list):
            concurrencies = ",".join(str(c) for c in concurrencies)

        ttft_threshold = getattr(b, "ttft_threshold_ms", None) or 2000
        itl_threshold = getattr(b, "itl_threshold_ms", None) or 25

        tokenizer_path = str(runtime.model_path) if runtime.is_hf_model else "/model"

        cmd = [
            "bash",
            self.script_path,
            endpoint,
            model_name,
            b.trace_file or "",
            str(concurrencies) if concurrencies else "",
            str(ttft_threshold),
            str(itl_threshold),
            tokenizer_path,
        ]

        self.append_aiperf_args(cmd, config)

        return cmd
