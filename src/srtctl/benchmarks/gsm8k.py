# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GSM8K accuracy benchmark runner.

A single ``gsm8k`` type that adapts the eval harness to the backend:

- ``backend.type == "vllm"``: run a vendored copy of vLLM's ``gsm8k_eval.py``
  against the OpenAI ``/v1/completions`` endpoint. Only needs common deps
  (aiohttp/numpy/requests/tqdm) present in a vLLM container, and the served
  model name is sent automatically.
- otherwise (sglang, ...): run ``sglang.test.run_eval`` with the gsm8k task,
  which requires the ``sglang`` package in the container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

# sglang harness wrapper (needs the sglang package in the container).
_SGLANG_SCRIPT = "/srtctl-benchmarks/gsm8k/sglang-bench.sh"
# Vendored vLLM eval wrapper (pure OpenAI client, no sglang needed).
_VLLM_SCRIPT = "/srtctl-benchmarks/gsm8k/vllm-bench.sh"


@register_benchmark("gsm8k")
class GSM8KRunner(BenchmarkRunner):
    """GSM8K (Grade School Math 8K) accuracy evaluation.

    The harness is chosen automatically from the backend:

    - vLLM backend: vendored vLLM ``gsm8k_eval.py`` (5-shot, ``/v1/completions``,
      last-number scoring). The served model name is sent automatically, so no
      model needs to be set in the benchmark section.
    - sglang (and any other) backend: ``sglang.test.run_eval --eval-name gsm8k``.

    Optional config fields:
        - benchmark.num_examples: Number of questions (default: 1319)
        - benchmark.max_tokens: Max tokens per response
          (sglang default: 16384, vLLM default: 256)
        - benchmark.num_shots: Few-shot examples (default: 5)
        - benchmark.temperature: Sampling temperature
        - benchmark.top_p / benchmark.top_k: Sampling (sglang harness only)
        - benchmark.num_threads: Concurrent threads (sglang harness only, default: 512)
        - benchmark.repeat: Repeat the eval N times (vLLM path only, default: 1)
    """

    @property
    def name(self) -> str:
        return "GSM8K"

    @property
    def script_path(self) -> str:
        # Nominal path for logging; build_command selects the real script per backend.
        return _SGLANG_SCRIPT

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "gsm8k")

    def validate_config(self, config: SrtConfig) -> list[str]:
        b = config.benchmark
        errors: list[str] = []
        for field in ("num_examples", "max_tokens", "num_threads", "repeat"):
            value = getattr(b, field, None)
            if value is not None and value <= 0:
                errors.append(f"benchmark.{field} must be > 0")
        if b.num_shots is not None and b.num_shots < 0:
            errors.append("benchmark.num_shots must be >= 0")
        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        if config.backend_type == "vllm":
            return self._build_vllm_command(config, runtime)
        return self._build_sglang_command(config, runtime)

    def _build_vllm_command(self, config: SrtConfig, runtime: RuntimeContext) -> list[str]:
        b = config.benchmark
        return [
            "bash",
            _VLLM_SCRIPT,
            "http://localhost",
            str(runtime.frontend_port),
            config.served_model_name,
            str(b.num_examples or 1319),
            str(b.max_tokens or 256),
            str(b.num_shots if b.num_shots is not None else 5),
            str(b.temperature if b.temperature is not None else 0.0),
            str(b.repeat or 1),
        ]

    def _build_sglang_command(self, config: SrtConfig, runtime: RuntimeContext) -> list[str]:
        b = config.benchmark
        # TODO: support overriding endpoint via config to target external servers;
        # mmlu/gpqa/longbenchv2 share the same limitation today.
        endpoint = f"http://localhost:{runtime.frontend_port}"

        return [
            "bash",
            _SGLANG_SCRIPT,
            endpoint,
            str(b.num_examples or 1319),
            str(b.max_tokens or 16384),
            str(b.num_threads or 512),
            str(b.num_shots if b.num_shots is not None else 5),
            str(b.temperature) if b.temperature is not None else "",
            str(b.top_p) if b.top_p is not None else "",
            str(b.top_k) if b.top_k is not None else "",
        ]
