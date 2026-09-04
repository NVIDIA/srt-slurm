#!/usr/bin/env bash
set -euo pipefail

umask 002

# Setup runs once per allocated node against a shared result directory. Keep
# cleanup and verification serialized while applying the narrowly scoped
# Mooncake scheduler hotfix from vLLM PR #55066.
exec 9>/logs/.setup-nightly44fe-mooncake013.lock
flock -x 9

AIPERF_RUN_TMP=/logs/agentx-tmp
mkdir -p "${AIPERF_RUN_TMP}"
find "${AIPERF_RUN_TMP}" -mindepth 1 -maxdepth 1 \
  -name 'aiperf_mmap_*' -exec rm -rf -- {} +
chmod g+rwx "${AIPERF_RUN_TMP}"
test -w "${AIPERF_RUN_TMP}"

# Setup executes before the per-role worker environments are injected, so the
# EAGLE3 helper cannot inherit these values from prefill_environment or
# decode_environment. Define the exact GQA model and shared Tyche cache here.
DRAFT_CACHE_ROOT=/lustre/fsw/coreai_comparch_inferencex/common/cache/draft-models
DRAFT_LOCAL_DIR_DEFAULT="${DRAFT_CACHE_ROOT}/Inferact--MiniMax-M3-EAGLE3-GQA"
export MINIMAX_M3_EAGLE3_DRAFT_MODEL="${MINIMAX_M3_EAGLE3_DRAFT_MODEL:-Inferact/MiniMax-M3-EAGLE3-GQA}"
export MINIMAX_M3_EAGLE3_DRAFT_LOCAL_DIR="${MINIMAX_M3_EAGLE3_DRAFT_LOCAL_DIR:-${DRAFT_LOCAL_DIR_DEFAULT}}"

# The main-model cache warmer does not stage speculative draft models. Reuse
# the established EAGLE3 helper so all ranks receive a complete local model
# directory rather than asking vLLM to interpret a nonexistent path as a Hub
# repository ID. The helper serializes the shared download with its own lock.
bash /configs/patches/minimax-m3-eagle3-draft.sh

if [[ -z "${MINIMAX_M3_EAGLE3_DRAFT_LOCAL_DIR:-}" ]] || \
   [[ ! -s "${MINIMAX_M3_EAGLE3_DRAFT_LOCAL_DIR}/config.json" ]]; then
  echo "ERROR: EAGLE3 draft model cache is missing config.json: ${MINIMAX_M3_EAGLE3_DRAFT_LOCAL_DIR:-unset}" >&2
  exit 1
fi
echo "Verified EAGLE3 draft model cache at ${MINIMAX_M3_EAGLE3_DRAFT_LOCAL_DIR}"

VLLM_PACKAGE_DIR=$(python3 - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise SystemExit("vLLM package is not discoverable")
print(Path(spec.origin).resolve().parent)
PY
)

verify_hash() {
  local expected=$1
  local path=$2
  local actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ERROR: source hash mismatch for ${path}: ${actual}" >&2
    exit 1
  fi
}

# These are the stock files in vllm/vllm-openai nightly 44fe2a392. Fail
# closed if the image changed or an old 5e35 mutation leaked into the run.
verify_hash e22fb8ac09874f1109752f211f43919e026958274b629412cf9da7bdde6bff54 \
  "${VLLM_PACKAGE_DIR}/v1/worker/gpu/dp_utils.py"
verify_hash 1506ec0e865e92f5d44a81bc6c26398be68fedc1e2302b5ddc1e1d4d5f0b8a55 \
  "${VLLM_PACKAGE_DIR}/v1/worker/gpu/spec_decode/autoregressive/speculator.py"

MOONCAKE_SCHEDULER="${VLLM_PACKAGE_DIR}/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py"
MOONCAKE_SCHEDULER_STOCK_SHA=82ce9a5d55f8b5e9d47953487a251f92bcef2eece9b8a30a0b79ffc0678bf6ef
MOONCAKE_SCHEDULER_PATCHED_SHA=a3c7daad29b8b250791186c90b225e24bc5b7d0613dece4698b3c831807ed03c
MOONCAKE_SCHEDULER_SHA=$(sha256sum "${MOONCAKE_SCHEDULER}" | awk '{print $1}')

if [[ "${MOONCAKE_SCHEDULER_SHA}" == "${MOONCAKE_SCHEDULER_STOCK_SHA}" ]]; then
  MOONCAKE_SCHEDULER="${MOONCAKE_SCHEDULER}" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["MOONCAKE_SCHEDULER"])
source = path.read_text()
old = '''            assert block_ids is not None, (
                f"Missing current block table for store request {req_meta.req_id}"
            )
            req_meta.block_ids = block_ids
'''
new = '''            if block_ids is None:
                # The request is outside this step's snapshot, e.g. it was
                # rescheduled after a KV load failure or is still waiting on
                # an async load while a pending spec produces a save. Drop
                # the save instead of crashing EngineCore, and roll the
                # tracker back so a later step re-attempts the chunk.
                logger.warning(
                    "Missing current block table for store request %s; "
                    "skipping its save this step",
                    req_meta.req_id,
                )
                tracker = self._request_trackers.get(req_meta.req_id)
                if tracker is not None:
                    tracker.num_saved_tokens = req_meta.token_ids_start
                meta.requests.remove(req_meta)
                continue
            req_meta.block_ids = block_ids
'''
if source.count(old) != 1:
    raise SystemExit(
        "Mooncake scheduler hotfix expected exactly one stock assertion block"
    )
path.write_text(source.replace(old, new))
PY
elif [[ "${MOONCAKE_SCHEDULER_SHA}" != "${MOONCAKE_SCHEDULER_PATCHED_SHA}" ]]; then
  echo "ERROR: unexpected Mooncake scheduler source hash: ${MOONCAKE_SCHEDULER_SHA}" >&2
  exit 1
fi

verify_hash "${MOONCAKE_SCHEDULER_PATCHED_SHA}" "${MOONCAKE_SCHEDULER}"

python3 -m py_compile \
  "${VLLM_PACKAGE_DIR}/v1/worker/gpu/dp_utils.py" \
  "${VLLM_PACKAGE_DIR}/v1/worker/gpu/spec_decode/autoregressive/speculator.py" \
  "${MOONCAKE_SCHEDULER}"

VLLM_PACKAGE_DIR="${VLLM_PACKAGE_DIR}" python3 - <<'PY'
import os
from importlib.metadata import version
from pathlib import Path

import vllm
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp

expected_vllm = "0.28.1rc1.dev130+g44fe2a392"
if vllm.__version__ != expected_vllm:
    raise SystemExit(
        f"vLLM version mismatch: expected {expected_vllm}, found {vllm.__version__}"
    )
if not callable(dispatch_cg_and_sync_dp):
    raise SystemExit("vLLM MRV2 dispatch_cg_and_sync_dp import probe failed")

expected_mooncake = "0.3.13"
actual_mooncake = version("mooncake-transfer-engine-cuda13")
if actual_mooncake != expected_mooncake:
    raise SystemExit(
        f"Mooncake version mismatch: expected {expected_mooncake}, found {actual_mooncake}"
    )

vllm_root = Path(os.environ["VLLM_PACKAGE_DIR"])
numa_source = (vllm_root / "utils/numa_utils.py").read_text()
if "srt-slurm-sa: interleave NUMA memory across nodes 0 and 1" in numa_source:
    raise SystemExit("Legacy --interleave=0,1 patch is still present")

nixl_root = vllm_root / "distributed/kv_transfer/kv_connector/v1/nixl"
recovery_markers = ("read_validation_timeout", "READ_ACK", "READ_NACK")
for path in nixl_root.rglob("*.py"):
    text = path.read_text(errors="replace")
    found = [marker for marker in recovery_markers if marker in text]
    if found:
        raise SystemExit(f"Recovery-v3 marker(s) {found} present in {path}")

mooncake_scheduler_source = (
    vllm_root
    / "distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py"
).read_text()
hotfix_marker = "Missing current block table for store request %s; "
if hotfix_marker not in mooncake_scheduler_source:
    raise SystemExit("vLLM PR #55066 Mooncake scheduler hotfix marker is absent")
if 'assert block_ids is not None, (' in mooncake_scheduler_source:
    raise SystemExit("Stock fatal Mooncake block-table assertion remains")

print(
    "Verified nightly-44fe runtime with vLLM PR #55066 scheduler hotfix: "
    f"vllm={vllm.__version__}, mooncake={actual_mooncake}, "
    "no legacy NUMA interleave, no Recovery-v3"
)
PY

# EngineCore applies the per-rank NUMA policies after setup. Confirm both
# sockets are visible inside Slurm's cgroup before worker startup.
for numa_node in 0 1; do
  if ! numactl --cpunodebind="${numa_node}" --membind="${numa_node}" true; then
    echo "ERROR: NUMA node ${numa_node} is unavailable for EngineCore binding" >&2
    echo "Allowed CPUs: $(grep '^Cpus_allowed_list:' /proc/self/status | awk '{print $2}')" >&2
    echo "Allowed memory nodes: $(grep '^Mems_allowed_list:' /proc/self/status | awk '{print $2}')" >&2
    exit 1
  fi
done

echo "Verified EngineCore NUMA access on nodes 0 and 1; native Tyche HCA mode"

flock -u 9
