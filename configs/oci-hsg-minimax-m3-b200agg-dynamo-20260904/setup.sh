#!/usr/bin/env bash
set -euo pipefail

umask 002

exec 9>/logs/.setup-minimax-m3-b200agg-dynamo.lock
flock -x 9

if [[ ! -s /draft_model/config.json ]]; then
  echo "ERROR: shared EAGLE3-GQA draft artifact is not mounted at /draft_model" >&2
  exit 1
fi

python3 /configs/oci-hsg-minimax-m3-b200agg-dynamo-20260904/patch_vllm_simple_kv_offload.py

python3 - <<'PY'
import importlib.util
from importlib.metadata import version
from pathlib import Path

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("vLLM package is not installed")
root = Path(next(iter(spec.submodule_search_locations)))
worker = root / "v1/simple_kv_offload/worker.py"
source = worker.read_text()
required = (
    "split_storage_by_layer = any(",
    "region_offset = tensor.storage_offset() * tensor.element_size()",
)
if not all(text in source for text in required):
    raise SystemExit(f"SimpleCPUOffload heterogeneous-layer patch missing: {worker}")
print(f"Verified vLLM {version('vllm')} SimpleCPUOffload patch: {worker}")
print("Verified EAGLE3-GQA draft artifact: /draft_model")
print("Configured host KV ceiling: 200 GiB/node = 50 GiB/rank for TP4")
PY

flock -u 9
exec 9>&-
