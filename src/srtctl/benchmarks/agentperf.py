# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentPerf benchmark runner.

Drives the agentperf-client trajectory-replay load generator
(https://github.com/ArtificialAnalysis-External/agentperf-client) against an
srt-slurm-launched server. This is a different client from the InferenceX
AgentX harness: agentperf-client is a standalone uv-managed Python project
with its own Rust streaming core, invoked as ``agentperf/run.py``, and it
produces per-phase trajectory outputs (``*__traj*.jsonl/.txt/.json``) plus a
per-request ``requests.jsonl``.

The client checkout is NOT vendored: mount it into the container via
``extra_mount`` and point ``benchmark.agentperf_client_dir`` at the container
path. Pin the checkout for comparable runs. The workload definition
(trajectory dataset, user-assignments file, settling time, phase timeout,
stop criteria) lives in the client's own config YAML, referenced by
``benchmark.agentperf_config``; srtctl injects only the endpoint, model and
concurrency so one workload file serves every topology.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


def _format_concurrencies(config: SrtConfig) -> str:
    b = config.benchmark
    if b.concurrency is not None:
        return str(b.concurrency)
    return ",".join(str(c) for c in config.benchmark.get_concurrency_list())


@register_benchmark("agentperf")
class AgentPerfRunner(BenchmarkRunner):
    """Run the agentperf-client trajectory replay against the frontend.

    Required config fields:
        - benchmark.agentperf_client_dir: Container path to an agentperf-client
          checkout (mount via extra_mount; pin the commit for comparable runs)
        - benchmark.agentperf_config: Container path to the client's config
          YAML (trajectory_path, user_assignments_path, settling_time_seconds,
          phase_timeout_seconds, stop criteria). The client validates this
          YAML on its own BEFORE merging CLI overrides, so it must carry
          syntactically valid placeholder base_url / model / concurrencies;
          srtctl's injected --base-url / --model / --concurrencies then
          override them.
        - benchmark.concurrency or benchmark.concurrencies

    Optional config fields:
        - benchmark.env: extra environment for the client (e.g.
          AGENTPERF_EXTRA_ARGS="--seed 100 --no-eval" appended verbatim to the
          run.py invocation)
    """

    @property
    def name(self) -> str:
        return "AgentPerf"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/agentperf/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "agentperf")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        if not b.agentperf_client_dir:
            errors.append("benchmark.agentperf_client_dir is required for agentperf (mount the client via extra_mount)")
        if not b.agentperf_config:
            errors.append("benchmark.agentperf_config is required for agentperf (the client's workload YAML)")
        if b.concurrency is None and b.concurrencies is None:
            errors.append("benchmark.concurrency or benchmark.concurrencies is required for agentperf")
        else:
            try:
                levels = [b.concurrency] if b.concurrency is not None else b.get_concurrency_list()
                if not levels:
                    # An empty list would make the client silently fall back to
                    # whatever levels the workload YAML declares.
                    errors.append("agentperf requires at least one concurrency level")
                elif any(int(c) <= 0 for c in levels):
                    errors.append(f"agentperf concurrencies must be positive, got: {levels}")
            except (TypeError, ValueError):
                errors.append(f"agentperf concurrencies must be integers, got: {b.concurrencies}")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"
        model_name = config.served_model_name or config.model.path

        return [
            "bash",
            self.script_path,
            endpoint,
            model_name,
            b.agentperf_client_dir or "",
            b.agentperf_config or "",
            _format_concurrencies(config),
        ]

    def get_environment(self, config: SrtConfig, runtime: RuntimeContext) -> dict[str, str]:
        del runtime
        return dict(config.benchmark.env)
