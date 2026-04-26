#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

DYNAMO_VERSION="${DYNAMO_VERSION:-1.2.0.dev20260426}"
DYNAMO_PACKAGE="${DYNAMO_PACKAGE:-ai-dynamo==${DYNAMO_VERSION}}"
DYNAMO_RUNTIME_PACKAGE="${DYNAMO_RUNTIME_PACKAGE:-ai-dynamo-runtime==${DYNAMO_VERSION}}"
DYNAMO_WHEEL_NAME="${DYNAMO_WHEEL_NAME:-ai_dynamo-${DYNAMO_VERSION}-py3-none-any.whl}"
DYNAMO_RUNTIME_WHEEL_PATTERN="${DYNAMO_RUNTIME_WHEEL_PATTERN:-ai_dynamo_runtime-${DYNAMO_VERSION}-*.whl}"
DYNAMO_WHEEL_DIRS="${DYNAMO_WHEEL_DIRS:-/configs/wheels /configs}"
DYNAMO_INDEX_URL="${DYNAMO_INDEX_URL:-https://pypi.org/simple}"
DYNAMO_EXTRA_INDEX_URL="${DYNAMO_EXTRA_INDEX_URL:-https://pypi.nvidia.com}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [ -z "${PYTHON_BIN}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "ERROR: neither python3 nor python found in PATH" >&2
        exit 127
    fi
fi

if "${PYTHON_BIN}" - <<PY
import importlib.metadata
import sys

wanted = "${DYNAMO_VERSION}"
packages = ("ai-dynamo", "ai-dynamo-runtime")
for package in packages:
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        sys.exit(1)
    if installed != wanted:
        sys.exit(1)

import dynamo.llm  # noqa: F401

sys.exit(0)
PY
then
    echo "ai-dynamo and ai-dynamo-runtime ${DYNAMO_VERSION} already installed"
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
runtime_wheel="$(find_wheel "${DYNAMO_RUNTIME_WHEEL_PATTERN}")"

find_links_args=()
for wheel_dir in ${DYNAMO_WHEEL_DIRS}; do
    [ -d "${wheel_dir}" ] || continue
    find_links_args+=(--find-links "${wheel_dir}")
done

if [ -n "${dynamo_wheel}" ] && [ -n "${runtime_wheel}" ]; then
    echo "Installing ai-dynamo-runtime and ai-dynamo ${DYNAMO_VERSION} from local wheels"
    "${PYTHON_BIN}" -m pip install \
        --pre \
        --no-deps \
        --no-index \
        "${find_links_args[@]}" \
        "${DYNAMO_RUNTIME_PACKAGE}" \
        "${DYNAMO_PACKAGE}"
else
    echo "Installing ai-dynamo-runtime and ai-dynamo ${DYNAMO_VERSION} from package index"
    "${PYTHON_BIN}" -m pip install \
        --pre \
        --no-deps \
        --index-url "${DYNAMO_INDEX_URL}" \
        --extra-index-url "${DYNAMO_EXTRA_INDEX_URL}" \
        "${DYNAMO_RUNTIME_PACKAGE}" \
        "${DYNAMO_PACKAGE}"
fi

"${PYTHON_BIN}" - <<'PY'
import dynamo.llm  # noqa: F401
PY
