# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLPerf Inference benchmark runner.

Drives the MLCommons reference LoadGen harness
(https://github.com/mlcommons/inference) against an srt-slurm-launched server.
LoadGen — not AIPerf, not sa-bench — is what produces a submission-shaped
``mlperf_log_summary.txt`` / ``mlperf_log_detail.txt`` / ``mlperf_log_accuracy.json``
triple, so this runner exists to get those artifacts out of a topology
srt-slurm already knows how to stand up.

Only the *server-url* shape of the reference harness is usable here. Most
MLPerf LLM reference implementations construct the engine in-process
(``SUT_VLLM``, the DeepSeek-R1 backends launch their own server), which
conflicts with srt-slurm owning the deployment. The ``gpt-oss-120b`` harness
instead takes ``--server-url`` and speaks HTTP to a server someone else
started — that is exactly the contract srt-slurm provides, so it is the
supported layout in this first iteration.

The harness checkout is NOT vendored: mount it into the container via
``extra_mount`` and point ``benchmark.mlperf_harness_dir`` at the container
path. Pin the checkout — the LoadGen version and the harness's dataset
handling both move between submission rounds, and results are only comparable
within one pin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

SCENARIOS = ("offline", "server")
MODES = ("performance", "accuracy", "both")
BACKENDS = ("sglang",)

# The reference backend posts token IDs to the server's native ``/generate``
# route, so the frontend in front of the workers has to speak that dialect.
_BACKEND_FRONTENDS = {"sglang": {"sglang"}}


@register_benchmark("mlperf")
class MLPerfRunner(BenchmarkRunner):
    """Run the MLCommons LoadGen harness against the frontend.

    Required config fields:
        - benchmark.mlperf_harness_dir: Container path to an mlcommons/inference
          checkout (mount via extra_mount; pin the commit for comparable runs)
        - benchmark.mlperf_benchmark: Benchmark directory under ``language/``
          (for example ``gpt-oss-120b``)
        - benchmark.mlperf_dataset: Container path to the tokenized dataset the
          harness loads with ``--input-file``

    Optional config fields:
        - benchmark.mlperf_scenario: ``offline`` (default) or ``server``
        - benchmark.mlperf_mode: ``performance`` (default), ``accuracy``, or
          ``both`` (a performance run followed by an accuracy run)
        - benchmark.mlperf_backend: harness backend, ``sglang`` (default)
        - benchmark.mlperf_user_conf: container path to a LoadGen ``user.conf``.
          This is where the scenario constraint lives (``target_qps``,
          ``min_duration``); without it the harness falls back to its own
          checked-in placeholder, which is not a submission-shaped run.
        - benchmark.concurrency: mapped to the harness ``--max-concurrency``
        - benchmark.env: extra environment for the harness, notably
          MLPERF_EXTRA_ARGS (appended verbatim to run_mlperf.py) and
          MLPERF_EVAL_ACCURACY=1 (run the accuracy scorer after an accuracy run)
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
                "benchmark.mlperf_harness_dir is required for mlperf (mount an mlcommons/inference checkout via extra_mount)"
            )
        if not b.mlperf_benchmark:
            errors.append("benchmark.mlperf_benchmark is required for mlperf (the directory under language/)")
        if not b.mlperf_dataset:
            errors.append("benchmark.mlperf_dataset is required for mlperf (the harness --input-file dataset)")

        if b.mlperf_scenario not in SCENARIOS:
            errors.append(f"benchmark.mlperf_scenario must be one of {list(SCENARIOS)}, got: {b.mlperf_scenario}")
        if b.mlperf_mode not in MODES:
            errors.append(f"benchmark.mlperf_mode must be one of {list(MODES)}, got: {b.mlperf_mode}")

        if b.mlperf_backend not in BACKENDS:
            errors.append(f"benchmark.mlperf_backend must be one of {list(BACKENDS)}, got: {b.mlperf_backend}")
        else:
            # The harness backend and the frontend have to agree on the wire
            # format. Catching this at config load beats discovering it as a
            # 404 inside LoadGen's connection preflight after the workers have
            # already loaded weights.
            expected = _BACKEND_FRONTENDS[b.mlperf_backend]
            if config.frontend.type not in expected:
                errors.append(
                    f"mlperf_backend '{b.mlperf_backend}' posts to the server's native /generate route, "
                    f"which requires frontend.type in {sorted(expected)}, got: {config.frontend.type}"
                )

        if b.concurrency is not None and b.concurrency <= 0:
            errors.append(f"mlperf concurrency must be positive, got: {b.concurrency}")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"

        return [
            "bash",
            self.script_path,
            endpoint,
            b.mlperf_harness_dir or "",
            b.mlperf_benchmark or "",
            b.mlperf_scenario,
            b.mlperf_mode,
            b.mlperf_backend,
            b.mlperf_dataset or "",
            b.mlperf_user_conf or "",
            str(b.concurrency) if b.concurrency is not None else "",
            # The accuracy scorer defaults to pulling a tokenizer from
            # HuggingFace. The model is already on shared storage for the
            # workers, so hand the harness that path instead of requiring
            # egress from the benchmark node.
            config.model.path,
        ]

    def get_environment(self, config: SrtConfig, runtime: RuntimeContext) -> dict[str, str]:
        del runtime
        return dict(config.benchmark.env)
