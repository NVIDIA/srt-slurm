#!/usr/bin/env bash
# Apply the Aug 31 Kimi baseline plus online FP8 composition over the
# prequantized checkpoint and the DCP meta-device initialization fix.

set -euo pipefail

bash /configs/apply-vllm-k3-nightly-aug31-baseline.sh

readonly SITE_PACKAGES="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
readonly VLLM_ROOT="${SITE_PACKAGES}/vllm"
readonly PR51392_PATCH_FILE="${VLLM_PR51392_PATCH_FILE:-/configs/patches/vllm-pr51392-online-quant-prequantized-on-44fe2a392.patch}"
readonly K3_DCP_META_DEVICE_PATCH_FILE="${VLLM_K3_DCP_META_DEVICE_PATCH_FILE:-/configs/patches/vllm-k3-dcp-device-under-meta-on-44fe2a392.patch}"
readonly PR51392_MARKER_FILE="${VLLM_ROOT}/.pr51392_online_quant_prequantized_on_44fe2a392"
readonly K3_DCP_META_DEVICE_MARKER_FILE="${VLLM_ROOT}/.k3_dcp_device_under_meta_on_44fe2a392"

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
  "${VLLM_ROOT}/model_executor/layers/quantization" \
  "${VLLM_ROOT}/model_executor/model_loader" \
  "${VLLM_ROOT}/models/kimi_k3/nvidia/mla.py"
