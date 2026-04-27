#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

DYNAMO_VERSION="${DYNAMO_VERSION:-}"

if [ -z "${DYNAMO_VERSION}" ]; then
    echo "ERROR: DYNAMO_VERSION must be set for ai-dynamo wheel prefetch" >&2
    exit 1
fi

DYNAMO_PACKAGE="ai-dynamo==${DYNAMO_VERSION}"
DYNAMO_RUNTIME_PACKAGE="ai-dynamo-runtime==${DYNAMO_VERSION}"
DYNAMO_WHEEL_NAME="${DYNAMO_WHEEL_NAME:-ai_dynamo-${DYNAMO_VERSION}-py3-none-any.whl}"
DYNAMO_RUNTIME_WHEEL_PATTERN="${DYNAMO_RUNTIME_WHEEL_PATTERN:-ai_dynamo_runtime-${DYNAMO_VERSION}-*.whl}"
DYNAMO_INDEX_URL="${DYNAMO_INDEX_URL:-https://pypi.org/simple}"
DYNAMO_EXTRA_INDEX_URL="${DYNAMO_EXTRA_INDEX_URL:-https://pypi.nvidia.com}"

source_dir="${SRTCTL_SOURCE_DIR:-$(pwd)}"
wheel_dir="${DYNAMO_WHEEL_HOST_DIR:-${source_dir}/configs/wheels}"
wheel_path="${wheel_dir}/${DYNAMO_WHEEL_NAME}"
lock_path="${wheel_dir}/.${DYNAMO_WHEEL_NAME}.lock"

mkdir -p "${wheel_dir}"

runtime_wheel_path() {
    find "${wheel_dir}" -maxdepth 1 -type f -name "${DYNAMO_RUNTIME_WHEEL_PATTERN}" -print -quit
}

if [ -f "${wheel_path}" ] && [ -n "$(runtime_wheel_path)" ]; then
    echo "ai-dynamo wheels already staged: ${wheel_dir}"
    exit 0
fi

download_wheels() {
    python3 -m pip download \
        --no-deps \
        --pre \
        --only-binary=:all: \
        --index-url "${DYNAMO_INDEX_URL}" \
        --extra-index-url "${DYNAMO_EXTRA_INDEX_URL}" \
        --dest "${wheel_dir}" \
        "${DYNAMO_RUNTIME_PACKAGE}" \
        "${DYNAMO_PACKAGE}"
}

if command -v flock >/dev/null 2>&1; then
    (
        flock -x 9
        if [ ! -f "${wheel_path}" ] || [ -z "$(runtime_wheel_path)" ]; then
            echo "Staging ai-dynamo wheels: ${DYNAMO_RUNTIME_PACKAGE} ${DYNAMO_PACKAGE} -> ${wheel_dir}"
            download_wheels
        fi
    ) 9>"${lock_path}"
else
    echo "Staging ai-dynamo wheels: ${DYNAMO_RUNTIME_PACKAGE} ${DYNAMO_PACKAGE} -> ${wheel_dir}"
    download_wheels
fi

if [ ! -f "${wheel_path}" ]; then
    echo "ERROR: expected ${wheel_path} after download" >&2
    exit 1
fi

if [ -z "$(runtime_wheel_path)" ]; then
    echo "ERROR: expected ${DYNAMO_RUNTIME_WHEEL_PATTERN} in ${wheel_dir} after download" >&2
    exit 1
fi
