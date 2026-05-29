# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mooncake Router benchmark runner using aiperf.

This benchmark tests KV-aware routing performance using the Mooncake conversation
trace dataset from the FAST25 paper. It compares aggregated (round-robin) vs
disaggregated (KV-aware) routing for LLM serving.

Uses aiperf with --custom-dataset-type mooncake_trace and --fixed-schedule
to replay requests at their original timestamps.

Based on the dynamo exemplar for Qwen3-32B:
https://github.com/ai-dynamo/dynamo/tree/main/recipes/qwen3-32b
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, AIPerfBenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("mooncake-router")
class MooncakeRouterRunner(AIPerfBenchmarkRunner):
    """Mooncake Router benchmark for testing KV-aware routing using aiperf.

    Uses the Mooncake conversation trace dataset to benchmark prefix caching
    and KV-aware routing performance. The trace contains real-world multi-turn
    conversation patterns with high prefix sharing potential.

    Dataset characteristics (conversation trace):
        - 12,031 requests over ~59 minutes (3.4 req/s)
        - Avg input: 12,035 tokens, Avg output: 343 tokens
        - 36.64% cache efficiency potential

    Required config fields:
        - model.path: Model to benchmark (default: Qwen/Qwen3-32B)

    Optional config fields (in benchmark section):
        - benchmark.mooncake_workload: Trace type (default: "conversation")
            Options: "mooncake", "conversation", "synthetic", "toolagent"
        - benchmark.ttft_threshold_ms: Goodput TTFT threshold (default: 2000)
        - benchmark.itl_threshold_ms: Goodput ITL threshold (default: 25)
    """

    @property
    def name(self) -> str:
        return "Mooncake Router Benchmark"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/mooncake-router/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "mooncake-router")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        # Validate mooncake_workload if specified
        valid_workloads = {"mooncake", "conversation", "synthetic", "toolagent"}
        workload = getattr(b, "mooncake_workload", None) or "conversation"
        if workload not in valid_workloads:
            errors.append(f"benchmark.mooncake_workload must be one of {valid_workloads}, got: {workload}")

        # Validate thresholds
        ttft_threshold = getattr(b, "ttft_threshold_ms", None) or 2000
        if ttft_threshold <= 0:
            errors.append(f"benchmark.ttft_threshold_ms must be positive, got: {ttft_threshold}")

        itl_threshold = getattr(b, "itl_threshold_ms", None) or 25
        if itl_threshold <= 0:
            errors.append(f"benchmark.itl_threshold_ms must be positive, got: {itl_threshold}")

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
        workload = getattr(b, "mooncake_workload", None) or "conversation"
        ttft_threshold = getattr(b, "ttft_threshold_ms", None) or 2000
        itl_threshold = getattr(b, "itl_threshold_ms", None) or 25

        # Tokenizer path: HF model ID or container mount path
        # For HF models, use the model ID directly so transformers downloads it
        tokenizer_path = str(runtime.model_path) if runtime.is_hf_model else "/model"

        cmd = [
            "bash",
            self.script_path,
            endpoint,
            model_name,
            workload,
            str(ttft_threshold),
            str(itl_threshold),
            tokenizer_path,
        ]

        self.append_aiperf_args(cmd, config)

        return cmd
