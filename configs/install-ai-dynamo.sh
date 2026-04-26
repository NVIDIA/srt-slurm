#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

DYNAMO_VERSION="${DYNAMO_VERSION:-1.2.0.dev20260426}"
DYNAMO_PACKAGE="${DYNAMO_PACKAGE:-ai-dynamo==${DYNAMO_VERSION}}"
DYNAMO_WHEEL_NAME="${DYNAMO_WHEEL_NAME:-ai_dynamo-${DYNAMO_VERSION}-py3-none-any.whl}"
DYNAMO_WHEEL_DIRS="${DYNAMO_WHEEL_DIRS:-/configs/wheels /configs}"
DYNAMO_INDEX_URL="${DYNAMO_INDEX_URL:-https://pypi.org/simple}"
DYNAMO_EXTRA_INDEX_URL="${DYNAMO_EXTRA_INDEX_URL:-https://pypi.nvidia.com}"

if python - <<PY
import importlib.metadata
import sys

wanted = "${DYNAMO_VERSION}"
try:
    installed = importlib.metadata.version("ai-dynamo")
except importlib.metadata.PackageNotFoundError:
    sys.exit(1)

sys.exit(0 if installed == wanted else 1)
PY
then
    echo "ai-dynamo ${DYNAMO_VERSION} already installed"
    exit 0
fi

find_wheel() {
    local pattern="$1"
    local wheel_dir
    for wheel_dir in ${DYNAMO_WHEEL_DIRS}; do
        [ -d "${wheel_dir}" ] || continue
        find "${wheel_dir}" -maxdepth 1 -type f -name "${pattern}" -print -quit
    done
}

dynamo_wheel="$(find_wheel "${DYNAMO_WHEEL_NAME}")"

if [ -n "${dynamo_wheel}" ]; then
    echo "Installing ai-dynamo ${DYNAMO_VERSION} from local wheel"
    python -m pip install \
        --pre \
        --no-deps \
        --no-index \
        "${dynamo_wheel}"
else
    echo "Installing ai-dynamo ${DYNAMO_VERSION} from package index"
    python -m pip install \
        --pre \
        --no-deps \
        --index-url "${DYNAMO_INDEX_URL}" \
        --extra-index-url "${DYNAMO_EXTRA_INDEX_URL}" \
        "${DYNAMO_PACKAGE}"
fi
