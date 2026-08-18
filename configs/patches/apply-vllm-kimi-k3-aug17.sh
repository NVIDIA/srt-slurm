#!/usr/bin/env bash
# Port the PR213 Kimi-K3 runtime changes (wzhao/kimi-k3-agentx-v2@face29e65)
# onto the Aug 17 nightly in place. The changes cover the Kimi-K3 DCP, DEP,
# hybrid-cache, speculative-decoding, and Mooncake paths used by this study.

set -euo pipefail

readonly SITE_PACKAGES="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
readonly VLLM_ROOT="${SITE_PACKAGES}/vllm"
readonly VERSION_FILE="${VLLM_ROOT}/_version.py"
readonly PATCH_FILE="${VLLM_KIMI_K3_PATCH_FILE:-/configs/patches/vllm-wzhao-kimi-k3-agentx-v2-on-nightly-aug17.patch}"
readonly MARKER_FILE="${VLLM_ROOT}/.wzhao_kimi_k3_agentx_v2_face29e65_on_g311b3513a"

if [[ -f "${MARKER_FILE}" ]]; then
  echo "Kimi-K3 PR213 runtime patch is already applied."
  exit 0
fi

if [[ ! -r "${PATCH_FILE}" ]]; then
  echo "Missing vLLM patch: ${PATCH_FILE}" >&2
  exit 1
fi

if [[ ! -r "${VERSION_FILE}" ]] || ! grep -q "g311b3513a" "${VERSION_FILE}"; then
  echo "Refusing to patch: expected nightly-aug17 vLLM commit g311b3513a." >&2
  echo "Version file: ${VERSION_FILE}" >&2
  exit 1
fi

if patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}" >/dev/null; then
  echo "Kimi-K3 PR213 runtime patch content is already present."
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
echo "Applied Kimi-K3 PR213 runtime patch to nightly-aug17."
