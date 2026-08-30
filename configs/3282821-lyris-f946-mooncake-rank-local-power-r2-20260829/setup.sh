#!/usr/bin/env bash
set -euo pipefail

# f946 already contains the MiniMax query-FP8 and indexer fixes used by the
# reference. Restore only the proven per-GPU Mooncake HCA mapping; keep NIXL,
# NUMA policy, and the power collection path unchanged for this canary.
python3 /configs/3282821-lyris-f946-mooncake-rank-local-power-r2-20260829/apply_f946_mooncake_rank_local_hca_patch.py

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

echo "Verified: f946 Mooncake-only rank-local HCA patch, Mooncake 0.3.12.post1, and both NUMA nodes"
