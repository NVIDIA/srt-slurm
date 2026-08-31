#!/usr/bin/env bash
set -euo pipefail

# Preserve physical CPU affinity while spreading host KV and model-runtime
# pages across both Grace NUMA nodes. This is the only memory-policy change
# relative to the rank-local-HCA-only f946 canary.
python3 /configs/patches/vllm_numa_interleave.py

# Keep the exact SHA-guarded f946 Mooncake HCA selection used by job 2844032.
python3 /configs/3282821-lyris-f946-mooncake-rank-local-power-r2-20260829/apply_f946_mooncake_rank_local_hca_patch.py

MOONCAKE_VERSION=0.3.12.post1 \
  bash /configs/patches/install-mooncake-store-0312.sh

python3 - <<'PY'
from importlib.metadata import version
from pathlib import Path

expected_mooncake = "0.3.12.post1"
actual_mooncake = version("mooncake-transfer-engine-cuda13")
if actual_mooncake != expected_mooncake:
    raise SystemExit(
        f"Mooncake version mismatch: expected {expected_mooncake}, found {actual_mooncake}"
    )

numa_utils = Path("/usr/local/lib/python3.12/dist-packages/vllm/utils/numa_utils.py")
content = numa_utils.read_text()
marker = "# srt-slurm-sa: interleave NUMA memory across nodes 0 and 1"
if content.count(marker) != 4:
    raise SystemExit(
        f"NUMA interleave patch verification failed in {numa_utils}: "
        f"marker count={content.count(marker)}, expected=4"
    )
print(f"Mooncake CUDA 13 runtime pinned to {actual_mooncake}")
print(f"Verified NUMA interleave patch in {numa_utils}")
PY

# Fail closed if the allocation cannot use both memory nodes under the exact
# policy that vLLM workers will receive.
numactl --interleave=0,1 true

echo "Verified: f946 rank-local Mooncake HCA, NUMA interleave 0,1, Mooncake 0.3.12.post1"
