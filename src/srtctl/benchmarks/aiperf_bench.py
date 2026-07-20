# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf concurrency-sweep benchmark runner.

Same synthetic ISL/OSL concurrency sweep as ``sa-bench``, but driven by NVIDIA ``aiperf``
instead of the single-process Python ``benchmark_serving.py`` client. ``sa-bench`` under-reports
high-concurrency throughput for fast models because one asyncio process cannot drain hundreds of
concurrent SSE streams (streaming backpressure throttles the server). ``aiperf`` uses parallel
workers and reproduces InferenceX numbers. Targets the same frontend port as ``sa-bench``, so it
also isolates client cost from serving-stack cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, AIPerfBenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("aiperf")
class AIPerfSweepRunner(AIPerfBenchmarkRunner):
    """AIPerf synthetic concurrency-sweep benchmark.

    Required config fields:
        - benchmark.isl / benchmark.osl
        - benchmark.concurrencies (e.g. "1x8x64x512")

    Optional:
        - benchmark.num_prompts_mult   (request-count = concurrency * this; default 3)
        - benchmark.num_warmup_mult    (warmup-request-count = concurrency * this; default 1)
        - benchmark.aiperf_endpoint_type (aiperf --endpoint-type; default "chat")
        - benchmark.aiperf_args        (extra passthrough aiperf flags)

    Fixed-length enforcement is handled in bench.sh: input/output std=0 and the output is forced
    to exactly OSL (not merely capped). The forcing is backend-aware — vLLM gets an explicit
    min_tokens floor + legacy max_tokens field, others rely on --osl + ignore_eos — so the runner
    passes ``config.backend_type`` through to the script.
    """

    @property
    def name(self) -> str:
        return "AIPerf"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/aiperf/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "aiperf")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark
        if b.isl is None:
            errors.append("benchmark.isl is required for aiperf")
        if b.osl is None:
            errors.append("benchmark.osl is required for aiperf")
        if b.concurrencies is None:
            errors.append("benchmark.concurrencies is required for aiperf")
        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"

        concurrencies = b.concurrencies
        if isinstance(concurrencies, list):
            concurrencies = "x".join(str(c) for c in concurrencies)

        tokenizer_path = str(runtime.model_path) if runtime.is_hf_model else "/model"
        endpoint_type = getattr(b, "aiperf_endpoint_type", None) or "chat"

        cmd = [
            "bash",
            self.script_path,
            endpoint,
            config.served_model_name,
            tokenizer_path,
            str(b.isl or 0),
            str(b.osl or 0),
            str(concurrencies),
            str(b.num_prompts_mult) if b.num_prompts_mult is not None else "3",
            str(b.num_warmup_mult) if b.num_warmup_mult is not None else "1",
            endpoint_type,
            config.backend_type,
        ]
        self.append_aiperf_args(cmd, config)
        return cmd

    def get_environment(self, config: SrtConfig, runtime: RuntimeContext) -> dict[str, str]:
        env: dict[str, str] = {}
        pkg = getattr(config.benchmark, "aiperf_package", None)
        if pkg:
            env["AIPERF_PACKAGE"] = pkg
        return env
