#!/usr/bin/env bash
set -euo pipefail

# The f946 vLLM nightly already contains the MiniMax query-FP8 and indexer
# fixes used by the reference run. Keep this Lyris setup source-patch free and
# install only the matching Mooncake runtime.
MOONCAKE_VERSION=0.3.12.post1 \
  bash /configs/patches/install-mooncake-store-0312.sh

python3 - <<'PY'
from importlib.metadata import version

expected = "0.3.12.post1"
actual = version("mooncake-transfer-engine-cuda13")
if actual != expected:
    raise SystemExit(f"Mooncake version mismatch: expected {expected}, found {actual}")
print(f"Mooncake CUDA 13 runtime pinned to {actual}")
PY

# Per-GPU EngineCore ranks are mapped to physical GPU NUMA nodes by srt-slurm.
# Fail closed if the Slurm allocation does not expose both Grace sockets.
for numa_node in 0 1; do
  if ! numactl --cpunodebind="${numa_node}" --membind="${numa_node}" true; then
    echo "ERROR: NUMA node ${numa_node} is unavailable for EngineCore binding" >&2
    echo "Allowed CPUs: $(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)" >&2
    echo "Allowed memory nodes: $(awk '/^Mems_allowed_list:/ {print $2}' /proc/self/status)" >&2
    exit 1
  fi
done

echo "Verified: source-patch-free f946 runtime, Mooncake 0.3.12.post1, and both NUMA nodes"
