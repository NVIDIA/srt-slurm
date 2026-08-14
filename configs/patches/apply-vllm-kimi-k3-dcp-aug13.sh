#!/usr/bin/env bash
# Port wzhao/kimi-k3-agentx-v2@face29e65 runtime changes onto the Aug 13
# nightly in place.

set -euo pipefail

readonly SITE_PACKAGES="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
readonly VLLM_ROOT="${SITE_PACKAGES}/vllm"
readonly VERSION_FILE="${VLLM_ROOT}/_version.py"
readonly PATCH_FILE="${VLLM_DCP_PATCH_FILE:-/configs/patches/vllm-wzhao-kimi-k3-agentx-v2-on-nightly-aug13.patch}"
readonly MARKER_FILE="${VLLM_ROOT}/.wzhao_kimi_k3_agentx_v2_face29e65_on_g3d204dfda"

if [[ -f "${MARKER_FILE}" ]]; then
  echo "Kimi-K3 DCP runtime patch is already applied."
  exit 0
fi

if [[ ! -r "${PATCH_FILE}" ]]; then
  echo "Missing vLLM patch: ${PATCH_FILE}" >&2
  exit 1
fi

if [[ ! -r "${VERSION_FILE}" ]] || ! grep -q "g3d204dfda" "${VERSION_FILE}"; then
  echo "Refusing to patch: expected nightly-aug13 vLLM commit g3d204dfda." >&2
  echo "Version file: ${VERSION_FILE}" >&2
  exit 1
fi

if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null; then
  echo "Kimi-K3 DCP runtime patch content is already present."
else
  echo "Patch neither applies cleanly nor appears already applied." >&2
  exit 1
fi

python3 -m compileall -q \
  "${VLLM_ROOT}/config/speculative.py" \
  "${VLLM_ROOT}/distributed/kv_transfer" \
  "${VLLM_ROOT}/models/kimi_k3" \
  "${VLLM_ROOT}/v1"

touch "${MARKER_FILE}"
echo "Applied Kimi-K3 DCP runtime patch to nightly-aug13."
