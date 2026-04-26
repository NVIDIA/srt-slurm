#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Install SGLang with the DeepSeek-V4 NIXL state-buffer transport fix.
# Remove this once https://github.com/sgl-project/sglang/pull/23773 is merged upstream.

set -euo pipefail

SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"
SGLANG_REMOTE="${SGLANG_REMOTE:-https://github.com/sgl-project/sglang.git}"
SGLANG_PR_NUMBER="${SGLANG_PR_NUMBER:-23773}"
SGLANG_PR_REF="refs/pull/${SGLANG_PR_NUMBER}/head"
SGLANG_LOCAL_BRANCH="${SGLANG_LOCAL_BRANCH:-nixl-dsv4-pr-${SGLANG_PR_NUMBER}}"

echo "=== Installing SGLang NIXL DSV4 fix from PR #${SGLANG_PR_NUMBER} ==="

if command -v flock >/dev/null 2>&1; then
    mkdir -p /tmp/srt-slurm-locks
    exec 9>/tmp/srt-slurm-locks/install-nixl-dsv4.lock
    flock 9
fi

mkdir -p "$(dirname "$SGLANG_DIR")"

if [ ! -d "$SGLANG_DIR/.git" ]; then
    echo "Recreating $SGLANG_DIR from $SGLANG_REMOTE"
    rm -rf "$SGLANG_DIR"
    git clone --depth 1 "$SGLANG_REMOTE" "$SGLANG_DIR"
fi

cd "$SGLANG_DIR"

git config --global --add safe.directory "$SGLANG_DIR" || true

if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$SGLANG_REMOTE"
else
    git remote add origin "$SGLANG_REMOTE"
fi

git fetch --depth 1 origin "$SGLANG_PR_REF"
git checkout -f -B "$SGLANG_LOCAL_BRANCH" FETCH_HEAD

INSTALLED_COMMIT="$(git rev-parse HEAD)"
echo "Checked out SGLang PR #${SGLANG_PR_NUMBER} at ${INSTALLED_COMMIT}"

NIXL_CONN="python/sglang/srt/disaggregation/nixl/conn.py"
if ! grep -q "send_state" "$NIXL_CONN" || ! grep -q "state_data_ptrs" "$NIXL_CONN"; then
    echo "ERROR: expected NIXL state-buffer transport changes were not found in $NIXL_CONN" >&2
    exit 1
fi

echo "=== SGLang NIXL DSV4 fix installed ==="
