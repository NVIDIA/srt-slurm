# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentX MVP benchmark runner using AIPerf."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, AIPerfBenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("agentx")
class AgentXRunner(AIPerfBenchmarkRunner):
    """InferenceX AgentX MVP replay using AIPerf.

    This runner targets an already-started OpenAI-compatible frontend and lets
    AIPerf's ``--scenario inferencex-agentx-mvp`` own the AgentX-specific replay
    rules: trajectory warmup, idle-gap capping, cache-bust markers, and
    submission validity metadata.
    """

    @property
    def name(self) -> str:
        return "AgentX"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/agentx/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "agentx")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        if b.concurrency is None:
            errors.append("benchmark.concurrency is required for agentx")
        if b.concurrencies is not None:
            errors.append("agentx uses a single benchmark.concurrency; do not set benchmark.concurrencies")
        if b.benchmark_duration is not None and b.benchmark_duration <= 0:
            errors.append("benchmark.benchmark_duration must be positive for agentx")
        if b.num_dataset_entries is not None and b.num_dataset_entries <= 0:
            errors.append("benchmark.num_dataset_entries must be positive for agentx")
        if b.failed_request_threshold is not None and not (0 <= b.failed_request_threshold <= 1):
            errors.append("benchmark.failed_request_threshold must be between 0 and 1 for agentx")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"
        model_name = config.served_model_name or config.model.path
        tokenizer_path = str(runtime.model_path) if runtime.is_hf_model else "/model"

        cmd = [
            "bash",
            self.script_path,
            endpoint,
            model_name,
            str(b.concurrency or ""),
            str(b.benchmark_duration or 1800),
            str(b.max_context_length or 0),
            tokenizer_path,
            b.agentx_dataset or b.dataset_name or "semianalysis_cc_traces_weka_with_subagents",
            str(b.num_dataset_entries or 472),
            str(b.failed_request_threshold if b.failed_request_threshold is not None else 0.10),
        ]

        self.append_aiperf_args(cmd, config)

        return cmd

    def get_environment(self, config: SrtConfig, runtime: RuntimeContext) -> dict[str, str]:
        del runtime
        return dict(config.benchmark.env)
