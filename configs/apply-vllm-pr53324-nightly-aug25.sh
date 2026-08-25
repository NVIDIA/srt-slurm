#!/usr/bin/env bash
# Apply the runtime changes from vLLM PR #53324 to the Aug 25 nightly.

set -euo pipefail

readonly SITE_PACKAGES="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
readonly VLLM_ROOT="${SITE_PACKAGES}/vllm"
readonly VERSION_FILE="${VLLM_ROOT}/_version.py"
readonly PATCH_FILE="${VLLM_PR53324_PATCH_FILE:-/configs/patches/vllm-pr53324-runtime-574d2e4-on-a9a17e7.patch}"
readonly MARKER_FILE="${VLLM_ROOT}/.pr53324_574d2e4_on_a9a17e709"

if [[ ! -r "${VERSION_FILE}" ]] || ! grep -q "a9a17e709" "${VERSION_FILE}"; then
  echo "Refusing to patch: expected vLLM nightly commit a9a17e709." >&2
  echo "Version file: ${VERSION_FILE}" >&2
  exit 1
fi

if [[ ! -r "${PATCH_FILE}" ]]; then
  echo "Missing PR #53324 runtime patch: ${PATCH_FILE}" >&2
  exit 1
fi

if [[ -f "${MARKER_FILE}" ]]; then
  echo "vLLM PR #53324 runtime patch is already applied."
elif patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 \
  < "${PATCH_FILE}" >/dev/null; then
  patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${PATCH_FILE}"
  python3 -m compileall -q \
    "${VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/mooncake/store" \
    "${VLLM_ROOT}/v1/core/kv_cache_coordinator.py" \
    "${VLLM_ROOT}/v1/core/kv_cache_utils.py"
  touch "${MARKER_FILE}"
  echo "Applied vLLM PR #53324 head 574d2e4 to nightly commit a9a17e709."
elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 \
  < "${PATCH_FILE}" >/dev/null; then
  echo "vLLM PR #53324 runtime content is already present."
else
  echo "PR #53324 neither applies cleanly nor appears already applied." >&2
  exit 1
fi
