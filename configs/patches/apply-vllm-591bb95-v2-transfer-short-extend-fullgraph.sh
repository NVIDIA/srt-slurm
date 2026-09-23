#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Apply vllm-591bb95-v2-transfer-short-extend-fullgraph.patch to the vLLM
# installed in the job container: promote the scheduler's one-real-token +
# K-placeholder transfer-resume batches onto the captured uniform-decode CUDA
# graph instead of falling back to eager.
#
# Serialized and made idempotent for the same reason install_dynamo_wheel.sh is:
# every srun task on a node runs the setup script concurrently against the same
# container root, and with dp_launch_mode: per_gpu that is one task per GPU.
# Concurrent `patch` runs against one file interleave and corrupt it.
#
# The patch is cut against the 591bb95 nightly. Applying it to a different vLLM
# build is expected to fail here rather than silently leave the engine unpatched,
# since an unpatched run looks healthy but measures a different code path.

set -euo pipefail

PATCH_FILE="/configs/patches/vllm-591bb95-v2-transfer-short-extend-fullgraph.patch"
# Present only after the patch has been applied; also the sentinel-free check
# that keeps a re-run (or a second task) from double-applying.
MARKER="_spec_short_extends_use_decode_kernels"

[ -f "${PATCH_FILE}" ] || { echo "missing ${PATCH_FILE}" >&2; exit 1; }

# -p1 strips the patch's a/ prefix, so the strip root is site-packages itself.
STRIP_ROOT="$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent.parent)')"
TARGET="${STRIP_ROOT}/vllm/v1/worker/gpu/model_runner.py"
[ -f "${TARGET}" ] || { echo "missing ${TARGET}" >&2; exit 1; }

command -v patch >/dev/null 2>&1 || apt-get install -y --no-install-recommends patch

LOCK="${STRIP_ROOT}/.srtctl_vllm_fullgraph_patch.lock"

(
    flock -x 200
    if grep -q "${MARKER}" "${TARGET}"; then
        echo "v2 transfer short-extend fullgraph patch already applied, skipping"
    else
        patch -p1 -d "${STRIP_ROOT}" --batch --forward <"${PATCH_FILE}"
        echo "v2 transfer short-extend fullgraph patch applied to ${TARGET}"
    fi
) 200>"${LOCK}"
