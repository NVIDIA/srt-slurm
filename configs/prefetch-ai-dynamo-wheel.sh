#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

DYNAMO_VERSION="${DYNAMO_VERSION:-1.2.0.dev20260426}"
DYNAMO_PACKAGE="${DYNAMO_PACKAGE:-ai-dynamo==${DYNAMO_VERSION}}"
DYNAMO_WHEEL_NAME="${DYNAMO_WHEEL_NAME:-ai_dynamo-${DYNAMO_VERSION}-py3-none-any.whl}"
DYNAMO_INDEX_URL="${DYNAMO_INDEX_URL:-https://pypi.org/simple}"
DYNAMO_EXTRA_INDEX_URL="${DYNAMO_EXTRA_INDEX_URL:-https://pypi.nvidia.com}"

source_dir="${SRTCTL_SOURCE_DIR:-$(pwd)}"
wheel_dir="${DYNAMO_WHEEL_HOST_DIR:-${source_dir}/configs/wheels}"
wheel_path="${wheel_dir}/${DYNAMO_WHEEL_NAME}"
lock_path="${wheel_dir}/.${DYNAMO_WHEEL_NAME}.lock"

mkdir -p "${wheel_dir}"

if [ -f "${wheel_path}" ]; then
    echo "ai-dynamo wheel already staged: ${wheel_path}"
    exit 0
fi

download_wheel() {
    python3 -m pip download \
        --no-deps \
        --pre \
        --only-binary=:all: \
        --index-url "${DYNAMO_INDEX_URL}" \
        --extra-index-url "${DYNAMO_EXTRA_INDEX_URL}" \
        --dest "${wheel_dir}" \
        "${DYNAMO_PACKAGE}"
}

if command -v flock >/dev/null 2>&1; then
    (
        flock -x 9
        if [ ! -f "${wheel_path}" ]; then
            echo "Staging ai-dynamo wheel: ${DYNAMO_PACKAGE} -> ${wheel_dir}"
            download_wheel
        fi
    ) 9>"${lock_path}"
else
    echo "Staging ai-dynamo wheel: ${DYNAMO_PACKAGE} -> ${wheel_dir}"
    download_wheel
fi

if [ ! -f "${wheel_path}" ]; then
    echo "ERROR: expected ${wheel_path} after download" >&2
    exit 1
fi
