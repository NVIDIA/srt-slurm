#!/usr/bin/env bash
# Apply the residual Kimi-K3 DCP/PP/DSpark/Mooncake fixes to the Aug 31
# nightly. PRs #53324, #53682, #53773, and #54167, plus the production
# latent-tail/deferred-finalize paths, are already present in this image.

set -euo pipefail

readonly SITE_PACKAGES="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
readonly VLLM_ROOT="${SITE_PACKAGES}/vllm"
readonly VERSION_FILE="${VLLM_ROOT}/_version.py"
readonly K3_AGENT_PATCH_FILE="${VLLM_K3_AGENT_PATCH_FILE:-/configs/patches/vllm-k3-agent-all-missing-on-44fe2a392.patch}"
readonly K3_CHECKPOINT_INDEX_INT64_PATCH_FILE="${VLLM_K3_CHECKPOINT_INDEX_INT64_PATCH_FILE:-/configs/patches/vllm-k3-prefill-checkpoint-index-int64-on-44fe2a392.patch}"
readonly K3_MTP_BLOCK_INTERLEAVED_DCP_PATCH_FILE="${VLLM_K3_MTP_BLOCK_INTERLEAVED_DCP_PATCH_FILE:-/configs/patches/vllm-k3-mtp-block-interleaved-dcp-on-44fe2a392.patch}"
readonly K3_AGENT_MARKER_FILE="${VLLM_ROOT}/.k3_agent_all_residual_on_44fe2a392"
readonly K3_CHECKPOINT_INDEX_INT64_MARKER_FILE="${VLLM_ROOT}/.k3_prefill_checkpoint_index_int64_on_44fe2a392"
readonly K3_MTP_BLOCK_INTERLEAVED_DCP_MARKER_FILE="${VLLM_ROOT}/.k3_mtp_block_interleaved_dcp_on_44fe2a392"

if [[ ! -r "${VERSION_FILE}" ]] || ! grep -q "44fe2a392" "${VERSION_FILE}"; then
  echo "Refusing to patch: expected vLLM nightly commit 44fe2a392." >&2
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
    echo "Applied ${label} to nightly commit 44fe2a392."
  elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 \
    < "${patch_file}" >/dev/null; then
    touch "${marker_file}"
    echo "${label} runtime content is already present."
  else
    echo "${label} neither applies cleanly nor appears already applied." >&2
    exit 1
  fi
}

# This rebased supplemental patch contains the still-unmerged pieces needed by
# our Kimi configurations: PP speculative decoding/DSpark loading, hybrid KV
# load-failure recovery, native offload hybrid-DCP accounting, and associated
# KDA/DCP safeguards.
apply_patch_once \
  "Kimi-K3 agent runtime residual changes" \
  "${K3_AGENT_PATCH_FILE}" \
  "${K3_AGENT_MARKER_FILE}"
apply_patch_once \
  "Kimi-K3 prefill checkpoint 64-bit cache index" \
  "${K3_CHECKPOINT_INDEX_INT64_PATCH_FILE}" \
  "${K3_CHECKPOINT_INDEX_INT64_MARKER_FILE}"
apply_patch_once \
  "Kimi-K3 MTP with block-interleaved DCP" \
  "${K3_MTP_BLOCK_INTERLEAVED_DCP_PATCH_FILE}" \
  "${K3_MTP_BLOCK_INTERLEAVED_DCP_MARKER_FILE}"

python3 -m compileall -q \
  "${VLLM_ROOT}/config/speculative.py" \
  "${VLLM_ROOT}/v1/attention/backends/mla/flashinfer_mla.py" \
  "${VLLM_ROOT}/v1/attention/backends/mla/tokenspeed_mla.py" \
  "${VLLM_ROOT}/model_executor/models/interfaces.py" \
  "${VLLM_ROOT}/models/kimi_k3/nvidia" \
  "${VLLM_ROOT}/v1/core/sched/scheduler.py" \
  "${VLLM_ROOT}/v1/simple_kv_offload/manager.py" \
  "${VLLM_ROOT}/v1/worker/gpu/model_runner.py" \
  "${VLLM_ROOT}/v1/worker/gpu/pp_utils.py" \
  "${VLLM_ROOT}/v1/worker/gpu/spec_decode"
