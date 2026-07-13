#!/usr/bin/env bash

set -euo pipefail

PATCH_FILE="${NIXL_MIXED_KV_PATCH:-/configs/patches/nixl-mixed-kv.patch}"

if [[ ! -r "$PATCH_FILE" ]]; then
  echo "ERROR: NIXL mixed-KV patch is not readable: $PATCH_FILE" >&2
  exit 1
fi

if [[ -n "${VLLM_SITE_PACKAGES:-}" ]]; then
  SITE_PACKAGES="$VLLM_SITE_PACKAGES"
else
  VLLM_PACKAGE_DIR=$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.realpath(vllm.__file__)))')
  SITE_PACKAGES=$(dirname "$VLLM_PACKAGE_DIR")
fi

TARGET="$SITE_PACKAGES/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py"

if [[ ! -f "$TARGET" ]]; then
  echo "ERROR: vLLM NIXL worker source was not found: $TARGET" >&2
  exit 1
fi

echo "Applying NIXL mixed-KV patch to $TARGET"

if patch --dry-run --batch --forward --directory "$SITE_PACKAGES" -p1 < "$PATCH_FILE" >/dev/null; then
  patch --batch --forward --directory "$SITE_PACKAGES" -p1 < "$PATCH_FILE"
elif patch --dry-run --batch --reverse --directory "$SITE_PACKAGES" -p1 < "$PATCH_FILE" >/dev/null; then
  echo "NIXL mixed-KV patch is already applied"
else
  echo "ERROR: NIXL mixed-KV patch does not apply cleanly to $TARGET" >&2
  patch --dry-run --batch --forward --directory "$SITE_PACKAGES" -p1 < "$PATCH_FILE" || true
  exit 1
fi

if grep -Fq "All non-MLA kv cache tensors must have the same size" "$TARGET"; then
  echo "ERROR: obsolete uniform non-MLA KV-cache assertion is still present" >&2
  exit 1
fi

grep -Fq "Registered non-MLA KV cache regions with different block" "$TARGET"
grep -Fq "NIXL heterogeneous-TP transfer does not support non-MLA KV" "$TARGET"
python3 -m py_compile "$TARGET"

echo "NIXL mixed-KV patch verified"
