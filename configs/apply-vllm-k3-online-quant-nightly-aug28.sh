#!/usr/bin/env bash
# Apply the Kimi-K3 DCP/PP/DSpark/Mooncake stack plus online FP8 composition
# to the Aug 28 nightly.

set -euo pipefail

readonly SITE_PACKAGES="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
readonly VLLM_ROOT="${SITE_PACKAGES}/vllm"
readonly VERSION_FILE="${VLLM_ROOT}/_version.py"
readonly PR53324_PATCH_FILE="${VLLM_PR53324_PATCH_FILE:-/configs/patches/vllm-pr53324-runtime-574d2e4-on-6f7df92a8.patch}"
readonly K3_AGENT_PATCH_FILE="${VLLM_K3_AGENT_PATCH_FILE:-/configs/patches/vllm-k3-agent-all-missing-on-6f7df92a8.patch}"
readonly K3_TAIL_GATE_PATCH_FILE="${VLLM_K3_TAIL_GATE_PATCH_FILE:-/configs/patches/vllm-k3-latent-tail-env-gate-on-6f7df92a8.patch}"
readonly K3_DEFERRED_FINALIZE_GATE_PATCH_FILE="${VLLM_K3_DEFERRED_FINALIZE_GATE_PATCH_FILE:-/configs/patches/vllm-k3-deferred-moe-finalize-env-gate-on-6f7df92a8.patch}"
readonly K3_CHECKPOINT_INDEX_INT64_PATCH_FILE="${VLLM_K3_CHECKPOINT_INDEX_INT64_PATCH_FILE:-/configs/patches/vllm-k3-prefill-checkpoint-index-int64-on-6f7df92a8.patch}"
readonly PR51392_PATCH_FILE="${VLLM_PR51392_PATCH_FILE:-/configs/patches/vllm-pr51392-online-quant-prequantized-on-6f7df92a8.patch}"
readonly K3_DCP_META_DEVICE_PATCH_FILE="${VLLM_K3_DCP_META_DEVICE_PATCH_FILE:-/configs/patches/vllm-k3-dcp-device-under-meta-on-6f7df92a8.patch}"
readonly PR53324_MARKER_FILE="${VLLM_ROOT}/.pr53324_574d2e4_on_6f7df92a8"
readonly K3_AGENT_MARKER_FILE="${VLLM_ROOT}/.k3_agent_all_728d3ad_on_6f7df92a8"
readonly K3_TAIL_GATE_MARKER_FILE="${VLLM_ROOT}/.k3_latent_tail_env_gate_on_6f7df92a8"
readonly K3_DEFERRED_FINALIZE_GATE_MARKER_FILE="${VLLM_ROOT}/.k3_deferred_moe_finalize_env_gate_on_6f7df92a8"
readonly K3_CHECKPOINT_INDEX_INT64_MARKER_FILE="${VLLM_ROOT}/.k3_prefill_checkpoint_index_int64_on_6f7df92a8"
readonly PR51392_MARKER_FILE="${VLLM_ROOT}/.pr51392_online_quant_prequantized_on_6f7df92a8"
readonly K3_DCP_META_DEVICE_MARKER_FILE="${VLLM_ROOT}/.k3_dcp_device_under_meta_on_6f7df92a8"

if [[ ! -r "${VERSION_FILE}" ]] || ! grep -q "6f7df92a8" "${VERSION_FILE}"; then
  echo "Refusing to patch: expected vLLM nightly commit 6f7df92a8." >&2
  echo "Version file: ${VERSION_FILE}" >&2
  exit 1
fi

apply_patch_once() {
  local label="$1"
  local patch_file="$2"
  local marker_file="$3"

  if [[ ! -r "${patch_file}" ]]; then
    echo "Missing ${label} runtime patch: ${patch_file}" >&2
    exit 1
  fi

  if [[ -f "${marker_file}" ]]; then
    echo "${label} runtime patch is already applied."
  elif patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 \
    < "${patch_file}" >/dev/null; then
    patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${patch_file}"
    touch "${marker_file}"
    echo "Applied ${label} to nightly commit 6f7df92a8."
  elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 \
    < "${patch_file}" >/dev/null; then
    touch "${marker_file}"
    echo "${label} runtime content is already present."
  else
    echo "${label} neither applies cleanly nor appears already applied." >&2
    exit 1
  fi
}

# PR #53324 supersedes the older Mooncake/DCP commits in
# xinli-sw/vllm:k3-agent-all. The supplemental patch carries only the runtime
# changes still missing from 6f7df92a8: PP speculative decoding and its DSpark
# loading fixes, DCP dummy-batch sequence lengths, hybrid load-failure recovery,
# and hybrid cache-accounting safeguards. Changes already merged upstream are
# deliberately not replayed.
apply_patch_once \
  "vLLM PR #53324 head 574d2e4" \
  "${PR53324_PATCH_FILE}" \
  "${PR53324_MARKER_FILE}"
apply_patch_once \
  "xinli-sw/vllm:k3-agent-all supplemental changes" \
  "${K3_AGENT_PATCH_FILE}" \
  "${K3_AGENT_MARKER_FILE}"
# PR #53682 (e6cc089) and PR #53773 (a447955) are ancestors of the Aug 28
# nightly and must not be replayed.
apply_patch_once \
  "Kimi-K3 latent-MoE tail environment gate" \
  "${K3_TAIL_GATE_PATCH_FILE}" \
  "${K3_TAIL_GATE_MARKER_FILE}"
apply_patch_once \
  "Kimi-K3 deferred MoE-finalize environment gate" \
  "${K3_DEFERRED_FINALIZE_GATE_PATCH_FILE}" \
  "${K3_DEFERRED_FINALIZE_GATE_MARKER_FILE}"
apply_patch_once \
  "Kimi-K3 prefill checkpoint 64-bit cache index" \
  "${K3_CHECKPOINT_INDEX_INT64_PATCH_FILE}" \
  "${K3_CHECKPOINT_INDEX_INT64_MARKER_FILE}"
apply_patch_once \
  "vLLM PR #51392 online quantization on prequantized models" \
  "${PR51392_PATCH_FILE}" \
  "${PR51392_MARKER_FILE}"
apply_patch_once \
  "Kimi-K3 DCP device selection under meta initialization" \
  "${K3_DCP_META_DEVICE_PATCH_FILE}" \
  "${K3_DCP_META_DEVICE_MARKER_FILE}"
python3 -m compileall -q \
  "${VLLM_ROOT}/config/quantization.py" \
  "${VLLM_ROOT}/config/speculative.py" \
  "${VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/mooncake/store" \
  "${VLLM_ROOT}/model_executor/layers/quantization" \
  "${VLLM_ROOT}/model_executor/model_loader" \
  "${VLLM_ROOT}/model_executor/models/interfaces.py" \
  "${VLLM_ROOT}/models/kimi_k3/nvidia" \
  "${VLLM_ROOT}/v1/core/kv_cache_coordinator.py" \
  "${VLLM_ROOT}/v1/core/kv_cache_utils.py" \
  "${VLLM_ROOT}/v1/core/sched/scheduler.py" \
  "${VLLM_ROOT}/v1/simple_kv_offload/manager.py" \
  "${VLLM_ROOT}/v1/worker/gpu/cudagraph_utils.py" \
  "${VLLM_ROOT}/v1/worker/gpu/model_runner.py" \
  "${VLLM_ROOT}/v1/worker/gpu/pp_utils.py" \
  "${VLLM_ROOT}/v1/worker/gpu/spec_decode"
