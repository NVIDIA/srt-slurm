#!/usr/bin/env bash
set -euo pipefail

# vLLM issue #49005: the MSA snapshot in nightly-dcfeb is incompatible with
# CUTLASS DSL 4.6. Keep the vLLM source fixed and restore its compatible DSL.
PYTHON="${PYTHON:-python3}"

uv pip install --system --force-reinstall --no-deps \
  "nvidia-cutlass-dsl==4.5.2" \
  "nvidia-cutlass-dsl-libs-base==4.5.2" \
  "quack-kernels==0.4.1"
# The 4.5.2 base and CUDA 13 wheels overlap. Reinstalling the CUDA wheel last
# matches vLLM's pre-4.6 packaging workaround and restores the CUDA payload.
uv pip install --system --force-reinstall --no-deps \
  "nvidia-cutlass-dsl-libs-cu13==4.5.2"

"${PYTHON}" <<'PY'
import importlib.metadata

import cutlass.cute as cute
import quack.activation

version = importlib.metadata.version("nvidia-cutlass-dsl")
if version != "4.5.2":
    raise SystemExit(f"Unexpected nvidia-cutlass-dsl version: {version}")
if not hasattr(cute.core, "ThrMma"):
    raise SystemExit("CUTLASS DSL 4.5.2 is missing cute.core.ThrMma")
quack_version = importlib.metadata.version("quack-kernels")
if quack_version != "0.4.1":
    raise SystemExit(f"Unexpected quack-kernels version: {quack_version}")

from vllm.third_party.fmha_sm100.api import sparse_topk_select  # noqa: F401

print(f"MiniMax MSA compatibility verified with nvidia-cutlass-dsl {version}")
print(f"Quack compatibility verified with quack-kernels {quack_version}")
print("MiniMax MSA sparse_topk_select import verified")
PY
