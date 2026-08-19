#!/usr/bin/env bash
set -euo pipefail

# Preserve only the established MiniMax indexer/EAGLE3 and query-FP8
# correctness setup. Recovery-v3 is intentionally not applied.
bash /configs/patches/vllm5e35-queryfp8-fix/minimax-m3-eagle3-grid-vllm5e35-query-fp8-fix.sh

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

# EngineCore applies per-rank numactl policies after this setup finishes.
# Validate both sockets now so Slurm/cgroup restrictions cannot silently turn
# the NUMA-1 ranks into unbound processes later during engine startup.
for numa_node in 0 1; do
  if ! numactl --cpunodebind="${numa_node}" --membind="${numa_node}" true; then
    echo "ERROR: NUMA node ${numa_node} is not available for EngineCore CPU/memory binding" >&2
    echo "Allowed CPUs: $(grep '^Cpus_allowed_list:' /proc/self/status | awk '{print $2}')" >&2
    echo "Allowed memory nodes: $(grep '^Mems_allowed_list:' /proc/self/status | awk '{print $2}')" >&2
    exit 1
  fi
done
echo "Verified: EngineCore can bind CPU and memory on NUMA nodes 0 and 1"

# Fail closed: neither the legacy interleave patch nor Recovery-v3 may be
# present. NUMA placement comes from the physical-GPU map rendered by
# srt-slurm for each per-GPU service.
python3 - <<'PY'
from pathlib import Path

numa_source = Path("/usr/local/lib/python3.12/dist-packages/vllm/utils/numa_utils.py").read_text()
if "srt-slurm-sa: interleave NUMA memory across nodes 0 and 1" in numa_source:
    raise SystemExit("Legacy --interleave=0,1 patch is still present")

nixl_root = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl"
)
recovery_markers = ("read_validation_timeout", "READ_ACK", "READ_NACK")
for path in nixl_root.rglob("*.py"):
    text = path.read_text(errors="replace")
    found = [marker for marker in recovery_markers if marker in text]
    if found:
        raise SystemExit(f"Recovery-v3 marker(s) {found} still present in {path}")

print("Verified: no legacy NUMA interleave and no Recovery-v3 markers")
PY

echo "Base MiniMax/query-FP8 setup verified; native Mooncake all-rail mode; Recovery-v3 absent"
