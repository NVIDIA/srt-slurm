# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentX agentic-coding benchmark runner for the InferenceX harness."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


# Container path of the InferenceX checkout. RuntimeContext.from_config mounts it
# there when INFMAX_WORKSPACE is set in the submitting environment.
INFMAX_CONTAINER_WORKSPACE = "/infmax-workspace"

# Entry point inside that checkout. Named for the directory it has always lived
# in; it drives single-node deployments just as well.
HARNESS_RELATIVE_PATH = "benchmarks/multi_node/agentic_srt.sh"

# The harness refuses a point whose server-side counters it cannot read, and the
# counter names are backend-specific.
SERVER_METRIC_PREFIXES = {
    "vllm": "vllm:",
    "sglang": "sglang:",
    "trtllm": "tensorrt_llm:",
}

# Every published AgentX point replays for an hour.
DEFAULT_DURATION_SECONDS = 3600

DEFAULT_RESULT_DIR = "/logs/agentic"


@register_benchmark("agentx")
class AgentXRunner(BenchmarkRunner):
    """Agentic-coding trace replay driven by the InferenceX AgentX harness.

    srt-slurm owns the deployment; the client is InferenceX's ``agentic_srt.sh``,
    which replays the ``cc-traces`` corpus through AIPerf's
    ``inferencex-agentx-mvp`` scenario. That client stack is not vendored here, so
    this runner needs an InferenceX checkout mounted at ``/infmax-workspace``
    through ``INFMAX_WORKSPACE`` -- the same contract the ``lm-eval`` runner uses.

    Recipes previously expressed this as ``benchmark.type: custom`` with the
    harness's environment contract inlined. That left the seven variables it
    requires (``MODEL``, ``MODEL_PREFIX``, ``FRAMEWORK``, ``PRECISION``, ``CONC``,
    ``RESULT_FILENAME``, ``DURATION``) either restated in every recipe or supplied
    by whichever launcher happened to submit the job, so a recipe could be
    complete on its own terms and still abort inside the container on
    ``check_env_vars``. This runner derives them from the recipe and reports what
    is missing before submission.

    ``benchmark.env`` is applied last and overrides everything derived here.
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
        env = config.benchmark.env

        if not config.benchmark.model_prefix and "MODEL_PREFIX" not in env:
            errors.append(
                "benchmark.model_prefix is required for benchmark.type=agentx: the harness "
                "selects its trace corpus from it (e.g. 'dsv4', 'qwen3.5', 'kimik3')"
            )

        if not self._concurrencies(config) and "CONC" not in env:
            errors.append("benchmark.concurrency or benchmark.concurrencies is required for benchmark.type=agentx")

        if config.backend.type not in SERVER_METRIC_PREFIXES and "AIPERF_REQUIRED_SERVER_METRIC_PREFIX" not in env:
            errors.append(
                f"no known server metric prefix for backend.type={config.backend.type!r}; "
                "set AIPERF_REQUIRED_SERVER_METRIC_PREFIX in benchmark.env"
            )

        return errors

    def build_command(self, config: SrtConfig, runtime: RuntimeContext) -> list[str]:
        del config
        return [
            "bash",
            self.script_path,
            f"http://localhost:{runtime.frontend_port}",
            INFMAX_CONTAINER_WORKSPACE,
        ]

    def get_environment(self, config: SrtConfig, runtime: RuntimeContext) -> dict[str, str]:
        benchmark = config.benchmark
        env = {
            "INFMAX_CONTAINER_WORKSPACE": INFMAX_CONTAINER_WORKSPACE,
            "RESULT_DIR": DEFAULT_RESULT_DIR,
            "PORT": str(runtime.frontend_port),
            "MODEL": config.served_model_name,
            "FRAMEWORK": config.backend.type,
            "PRECISION": config.model.precision,
            "DURATION": str(benchmark.duration_seconds or DEFAULT_DURATION_SECONDS),
            # The harness records this on every result row, and a single-node
            # deployment reports differently from a slice of a rack.
            "IS_MULTINODE": "true" if config.total_nodes > 1 else "false",
            "RESULT_FILENAME": benchmark.result_filename or config.name,
        }

        if benchmark.model_prefix:
            env["MODEL_PREFIX"] = benchmark.model_prefix

        concurrencies = self._concurrencies(config)
        if concurrencies:
            # CONC drives a single point. CONC_LIST replays several against one
            # engine configuration, which is only correct when no engine
            # parameter varies with concurrency.
            env["CONC"] = str(concurrencies[0])
            if len(concurrencies) > 1:
                env["CONC_LIST"] = " ".join(str(value) for value in concurrencies)

        prefix = SERVER_METRIC_PREFIXES.get(config.backend.type)
        if prefix:
            env["AIPERF_REQUIRED_SERVER_METRIC_PREFIX"] = prefix

        env.update(benchmark.env)
        return env

    @staticmethod
    def _concurrencies(config: SrtConfig) -> list[int]:
        if config.benchmark.concurrency is not None:
            return [config.benchmark.concurrency]
        return config.benchmark.get_concurrency_list()
