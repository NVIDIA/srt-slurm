# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLPerf Inference benchmark runner.

Drives NVIDIA's MLPerf submission harness (``nv_mlpinf``, from
``mlpinf/mlperf-inference`` under ``closed/NVIDIA``) against a cluster
srt-slurm has already deployed. LoadGen is what produces the submission-shaped
``mlperf_log_summary.txt`` / ``mlperf_log_detail.txt`` / ``mlperf_log_accuracy.json``
triple and the VALID/INVALID verdict, so this runner exists to get those
artifacts out of a topology srt-slurm already knows how to stand up.

Why nv_mlpinf rather than the mlcommons reference implementations: the
reference repo has one harness per benchmark, and they disagree on who owns the
server — llama2-70b builds the engine in-process, deepseek-r1 launches its own
server, and only gpt-oss-120b talks to an existing one (over SGLang's native
``/generate``, which neither trtllm-serve nor vLLM serves). nv_mlpinf instead
has a single ``llmlib`` for every benchmark with pluggable *cores*, and its
``dynamo_endpoint`` core is documented as a "minimal configuration for running
harness against pre-deployed Dynamo clusters" that "skips all build/runtime
flag loading from system configs". That is exactly srt-slurm's situation, and
it speaks OpenAI, so it works with any frontend that serves ``/v1``.

The harness checkout is NOT vendored: mount it via ``extra_mount`` and point
``benchmark.mlperf_harness_dir`` at ``closed/NVIDIA``. Pin the commit — LoadGen
decides VALID vs INVALID and the per-benchmark rules move between submission
rounds, so results are only comparable within one pin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

SCENARIOS = ("Offline", "Server", "Interactive")
TEST_MODES = ("PerformanceOnly", "AccuracyOnly")
CORE_TYPES = ("dynamo_endpoint", "trtllm_endpoint")

# Every core here issues OpenAI /v1/completions, so any frontend that serves
# that route works. This is the whole reason for preferring nv_mlpinf: the
# mlcommons gpt-oss-120b backend posts to SGLang's native /generate, which
# trtllm-serve and vllm serve answer with 404.
_OPENAI_FRONTENDS = frozenset({"dynamo", "sglang", "trtllm_serve", "vllm", "vllm-router", "static_router"})


@register_benchmark("mlperf")
class MLPerfRunner(BenchmarkRunner):
    """Run NVIDIA's MLPerf LoadGen harness against the frontend.

    Required config fields:
        - benchmark.mlperf_harness_dir: Container path to
          ``mlperf-inference/closed/NVIDIA`` (mount via extra_mount; pin the
          commit for comparable runs)
        - benchmark.mlperf_benchmark: nv_mlpinf benchmark name, e.g.
          ``deepseek-r1`` or ``gpt-oss-120b``

    Optional config fields:
        - benchmark.mlperf_scenario: ``Offline`` (default), ``Server``, or
          ``Interactive``
        - benchmark.mlperf_mode: ``PerformanceOnly`` (default) or
          ``AccuracyOnly``
        - benchmark.mlperf_core_type: ``dynamo_endpoint`` (default) or
          ``trtllm_endpoint``
        - benchmark.mlperf_system_name: passed as ``--system_name``
        - benchmark.mlperf_scratch_path: dataset/model root
          (``MLPERF_SCRATCH_PATH``); nv_mlpinf defaults it to
          ``/home/mlperf_inference_storage``, which is rarely mounted here
        - benchmark.env: extra environment, notably MLPERF_EXTRA_ARGS (appended
          verbatim to run_harness, e.g. ``--server_target_qps=40``) and
          MLPINF_USE_DYNAMO (see bench.sh for why it defaults to 0)
    """

    @property
    def name(self) -> str:
        return "MLPerf"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/mlperf/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "mlperf")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        if not b.mlperf_harness_dir:
            errors.append(
                "benchmark.mlperf_harness_dir is required for mlperf "
                "(mount mlperf-inference/closed/NVIDIA via extra_mount)"
            )
        if not b.mlperf_benchmark:
            errors.append("benchmark.mlperf_benchmark is required for mlperf (e.g. 'deepseek-r1')")

        if b.mlperf_scenario not in SCENARIOS:
            errors.append(f"benchmark.mlperf_scenario must be one of {list(SCENARIOS)}, got: {b.mlperf_scenario}")
        if b.mlperf_mode not in TEST_MODES:
            errors.append(f"benchmark.mlperf_mode must be one of {list(TEST_MODES)}, got: {b.mlperf_mode}")
        if b.mlperf_core_type not in CORE_TYPES:
            errors.append(f"benchmark.mlperf_core_type must be one of {list(CORE_TYPES)}, got: {b.mlperf_core_type}")

        # Both cores POST to /v1/completions. Catching a frontend that does not
        # serve it here beats discovering it as a 404 inside the harness's
        # warmup after the workers have already loaded weights.
        if config.frontend.type not in _OPENAI_FRONTENDS:
            errors.append(
                f"mlperf drives an OpenAI /v1/completions client, which requires "
                f"frontend.type in {sorted(_OPENAI_FRONTENDS)}, got: {config.frontend.type}"
            )

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"localhost:{runtime.frontend_port}"

        return [
            "bash",
            self.script_path,
            # host:port, not a URL — nv_mlpinf's --trtllm_server_urls wants the
            # bare authority and builds the base URL itself.
            endpoint,
            b.mlperf_harness_dir or "",
            b.mlperf_benchmark or "",
            b.mlperf_scenario,
            b.mlperf_mode,
            b.mlperf_core_type,
            b.mlperf_system_name or "",
            b.mlperf_scratch_path or "",
        ]

    def get_environment(self, config: SrtConfig, runtime: RuntimeContext) -> dict[str, str]:
        del runtime
        return dict(config.benchmark.env)
