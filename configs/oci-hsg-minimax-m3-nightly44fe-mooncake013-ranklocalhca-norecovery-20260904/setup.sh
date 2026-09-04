#!/usr/bin/env bash
set -euo pipefail

umask 002

# Setup runs once per allocated node against a shared result directory. Keep
# cleanup and verification serialized while applying the narrowly scoped
# Mooncake scheduler and per-GPU HCA fixes.
exec 9>/logs/.setup-nightly44fe-mooncake013-ranklocal.lock
flock -x 9

AIPERF_RUN_TMP=/logs/agentx-tmp
mkdir -p "${AIPERF_RUN_TMP}"
find "${AIPERF_RUN_TMP}" -mindepth 1 -maxdepth 1 \
  -name 'aiperf_mmap_*' -exec rm -rf -- {} +
chmod g+rwx "${AIPERF_RUN_TMP}"
test -w "${AIPERF_RUN_TMP}"

# Setup executes before the per-role worker environments are injected, so the
# EAGLE3 helper cannot inherit these values from prefill_environment or
# decode_environment. Define the exact GQA model and shared OCI-HSG cache here.
DRAFT_CACHE_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_inferencex/common/cache/draft-models
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

VLLM_NUMA_UTILS="${VLLM_PACKAGE_DIR}/utils/numa_utils.py"
VLLM_NUMA_UTILS_STOCK_SHA=e8d5d52e65bda7b3457f9f3863ecb5431a82bc2f7af97760425dd64626a75f68
VLLM_NUMA_UTILS_PATCHED_SHA=314a3600842650155f88a5131864bb53e146e1cc631b330d5cb0f403b9c3a060
VLLM_NUMA_UTILS_SHA=$(sha256sum "${VLLM_NUMA_UTILS}" | awk '{print $1}')

if [[ "${VLLM_NUMA_UTILS_SHA}" == "${VLLM_NUMA_UTILS_STOCK_SHA}" ]]; then
  VLLM_NUMA_UTILS="${VLLM_NUMA_UTILS}" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["VLLM_NUMA_UTILS"])
source = path.read_text()
old = '''    if cpu_binding is not None:
        logger.info(
            "Binding worker subprocess (local_rank=%s, gpu_index=%s) to CPUs %s and NUMA node %s",  # noqa: E501
            local_rank,
            gpu_index,
            cpu_binding,
            numa_node,
        )
        return f"--physcpubind={cpu_binding} --membind={numa_node}"

    logger.info(
        "Binding worker subprocess (local_rank=%s, gpu_index=%s) to NUMA node %s",
        local_rank,
        gpu_index,
        numa_node,
    )
    return f"--cpunodebind={numa_node} --membind={numa_node}"
'''
new = '''    # srt-slurm-sa: keep worker CPU affinity with preferred-node memory spill.
    # Mooncake and model warmup together can exceed one socket capacity even
    # when the Slurm job cgroup and host retain substantial free memory.
    if cpu_binding is not None:
        logger.info(
            "Binding worker subprocess (local_rank=%s, gpu_index=%s) to CPUs %s with preferred NUMA node %s",  # noqa: E501
            local_rank,
            gpu_index,
            cpu_binding,
            numa_node,
        )
        return f"--physcpubind={cpu_binding} --preferred={numa_node}"

    logger.info(
        "Binding worker subprocess (local_rank=%s, gpu_index=%s) to CPUs from preferred NUMA node %s",  # noqa: E501
        local_rank,
        gpu_index,
        numa_node,
    )
    return f"--cpunodebind={numa_node} --preferred={numa_node}"
'''
if source.count(old) != 1:
    raise SystemExit(
        "NUMA spill fix expected exactly one stock worker binding block"
    )
path.write_text(source.replace(old, new))
PY
elif [[ "${VLLM_NUMA_UTILS_SHA}" != "${VLLM_NUMA_UTILS_PATCHED_SHA}" ]]; then
  echo "ERROR: unexpected vLLM NUMA source hash: ${VLLM_NUMA_UTILS_SHA}" >&2
  exit 1
fi

verify_hash "${VLLM_NUMA_UTILS_PATCHED_SHA}" "${VLLM_NUMA_UTILS}"

MOONCAKE_WORKER="${VLLM_PACKAGE_DIR}/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"
MOONCAKE_WORKER_STOCK_SHA=303be42819e8308ff3e3e1b660748d6ba45eac1b2be9c77eb4269a13c0f6ea6e
MOONCAKE_WORKER_PATCHED_SHA=6b964a289777cb22d06c8be183d714079955098cbfbb955414e6171c1bba3c49
MOONCAKE_WORKER_SHA=$(sha256sum "${MOONCAKE_WORKER}" | awk '{print $1}')

if [[ "${MOONCAKE_WORKER_SHA}" == "${MOONCAKE_WORKER_STOCK_SHA}" ]]; then
  MOONCAKE_WORKER="${MOONCAKE_WORKER}" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["MOONCAKE_WORKER"])
source = path.read_text()
old = '''        # Initialize MooncakeDistributedStore with its own TransferEngine
        store_config = MooncakeStoreConfig.load_from_config()
        self.store = MooncakeDistributedStore()
'''
new = '''        # Initialize MooncakeDistributedStore with its own TransferEngine
        store_config = MooncakeStoreConfig.load_from_config()

        # srt-slurm-sa: select one GPU-local Mooncake HCA per per-GPU rank.
        # Giving every rank every HCA can make Mooncake 0.3.13 first-touch a
        # large segment on one NUMA node. Select one OCI-HSG rail per rank.
        # data_parallel_index is the explicit global DP rank supplied by the
        # per-GPU launcher; modulo the map length gives the node-local GPU.
        rank_local_hcas = [
            value.strip()
            for value in os.getenv(
                "VLLM_MOONCAKE_RANK_LOCAL_HCA_MAP", ""
            ).split(",")
            if value.strip()
        ]
        if rank_local_hcas:
            gpu_rank_for_hca = self.dp_rank * self.tp_size + self.tp_rank
            selected_hca = rank_local_hcas[
                gpu_rank_for_hca % len(rank_local_hcas)
            ]
            store_config.device_name = selected_hca.split(":", 1)[0]
            logger.info(
                "Per-GPU-rank Mooncake HCA selection: dp_rank=%d "
                "tp_rank=%d gpu_rank=%d device_name=%s",
                self.dp_rank,
                self.tp_rank,
                gpu_rank_for_hca,
                store_config.device_name,
            )

        self.store = MooncakeDistributedStore()
'''
if source.count(old) != 1:
    raise SystemExit(
        "Mooncake rank-local HCA fix expected exactly one stock setup block"
    )
path.write_text(source.replace(old, new))
PY
elif [[ "${MOONCAKE_WORKER_SHA}" != "${MOONCAKE_WORKER_PATCHED_SHA}" ]]; then
  echo "ERROR: unexpected Mooncake worker source hash: ${MOONCAKE_WORKER_SHA}" >&2
  exit 1
fi

verify_hash "${MOONCAKE_WORKER_PATCHED_SHA}" "${MOONCAKE_WORKER}"

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
  "${VLLM_NUMA_UTILS}" \
  "${VLLM_PACKAGE_DIR}/v1/worker/gpu/dp_utils.py" \
  "${VLLM_PACKAGE_DIR}/v1/worker/gpu/spec_decode/autoregressive/speculator.py" \
  "${MOONCAKE_WORKER}" \
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
numa_spill_marker = (
    "srt-slurm-sa: keep worker CPU affinity with preferred-node memory spill"
)
if numa_spill_marker not in numa_source:
    raise SystemExit("CPU-affinity-only NUMA worker patch marker is absent")
if 'return f"--physcpubind={cpu_binding} --membind={numa_node}"' in numa_source:
    raise SystemExit("Strict worker physcpubind/membind policy remains")
if 'return f"--cpunodebind={numa_node} --membind={numa_node}"' in numa_source:
    raise SystemExit("Strict worker cpunodebind/membind policy remains")

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

mooncake_worker_source = (
    vllm_root
    / "distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"
).read_text()
rank_local_hca_marker = (
    "srt-slurm-sa: select one GPU-local Mooncake HCA per per-GPU rank"
)
if rank_local_hca_marker not in mooncake_worker_source:
    raise SystemExit("Per-GPU-rank Mooncake HCA patch marker is absent")
if "gpu_rank_for_hca = self.dp_rank * self.tp_size + self.tp_rank" not in mooncake_worker_source:
    raise SystemExit("Mooncake HCA selection is not based on the explicit DP/TP rank")

print(
    "Verified nightly-44fe runtime with vLLM PR #55066 scheduler hotfix: "
    f"vllm={vllm.__version__}, mooncake={actual_mooncake}, "
    "worker CPU affinity with preferred-node memory spill, "
    "per-GPU-rank Mooncake HCA selection, no legacy NUMA interleave, "
    "no Recovery-v3"
)
PY

# This setup hook is shared by Dynamo's frontend and vLLM workers.  The
# frontend intentionally does not receive worker-only environment variables,
# but it still needs to install and validate the common runtime overlay.  Use
# the canonical OCI-HSG map for install-time hardware validation when the hook is
# running outside a worker; every vLLM worker continues to receive the explicit
# VLLM_MOONCAKE_RANK_LOCAL_HCA_MAP from the recipe environment.
CANONICAL_RANK_LOCAL_HCA_MAP=rdma_vf_rail0,rdma_vf_rail1,rdma_vf_rail2,rdma_vf_rail3
RANK_LOCAL_HCA_MAP=${VLLM_MOONCAKE_RANK_LOCAL_HCA_MAP:-${CANONICAL_RANK_LOCAL_HCA_MAP}}
if [[ -z "${VLLM_MOONCAKE_RANK_LOCAL_HCA_MAP:-}" ]]; then
  echo "VLLM_MOONCAKE_RANK_LOCAL_HCA_MAP is unset in this setup context; validating canonical OCI-HSG map ${CANONICAL_RANK_LOCAL_HCA_MAP}"
fi

IFS=',' read -r -a RANK_LOCAL_HCAS <<< "${RANK_LOCAL_HCA_MAP}"
if [[ ${#RANK_LOCAL_HCAS[@]} -ne 4 ]]; then
  echo "ERROR: expected four OCI-HSG rank-local HCAs, found ${#RANK_LOCAL_HCAS[@]}" >&2
  exit 1
fi

declare -A SEEN_NUMA_NODES=()
for index in "${!RANK_LOCAL_HCAS[@]}"; do
  hca=${RANK_LOCAL_HCAS[${index}]%%:*}
  numa_path="/sys/class/infiniband/${hca}/device/numa_node"
  if [[ ! -r "${numa_path}" ]]; then
    echo "ERROR: rank-local Mooncake HCA is unavailable: ${hca}" >&2
    exit 1
  fi
  actual_numa=$(<"${numa_path}")
  if [[ ! "${actual_numa}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${hca} has no usable NUMA mapping: ${actual_numa}" >&2
    exit 1
  fi
  SEEN_NUMA_NODES["${actual_numa}"]=1
  echo "Verified Mooncake rank-local HCA slot ${index}: ${hca} -> NUMA ${actual_numa}"
done

# EngineCore retains its native NUMA policy after setup while rank workers use
# a preferred-node spill policy. Confirm both policies work for every NUMA node
# discovered from the OCI-HSG rank-local rails; do not assume socket numbering.
for numa_node in "${!SEEN_NUMA_NODES[@]}"; do
  if ! numactl --cpunodebind="${numa_node}" --membind="${numa_node}" true; then
    echo "ERROR: NUMA node ${numa_node} is unavailable for EngineCore binding" >&2
    echo "Allowed CPUs: $(grep '^Cpus_allowed_list:' /proc/self/status | awk '{print $2}')" >&2
    echo "Allowed memory nodes: $(grep '^Mems_allowed_list:' /proc/self/status | awk '{print $2}')" >&2
    exit 1
  fi
  if ! numactl --cpunodebind="${numa_node}" --preferred="${numa_node}" true; then
    echo "ERROR: preferred-node spill policy is unavailable on NUMA node ${numa_node}" >&2
    exit 1
  fi
done

echo "Verified NUMA access on HCA-local nodes ${!SEEN_NUMA_NODES[*]}; preferred-node worker spill and rank-local OCI-HSG Mooncake HCAs"

flock -u 9
