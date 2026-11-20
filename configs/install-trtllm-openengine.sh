#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Install the OpenEngine bindings that TRT-LLM 1.3.x's `--grpc-protocol openengine` server imports
# from its optional `[openengine]` extra; 1.4.0rc0 and later bundle a private copy, and then this
# does nothing. Pins match Dynamo's lib/sidecar/trtllm/launch/common.sh. See docs/sidecars.md.

set -euo pipefail

has_bindings() {
    python3 -c '
import importlib.util, pathlib
spec = importlib.util.find_spec("tensorrt_llm")
bundled = spec and pathlib.Path(spec.submodule_search_locations[0], "grpc/openengine/_generated").is_dir()
if not bundled:
    import openengine.v1.openengine_pb2_grpc
' >/dev/null 2>&1
}

has_bindings && exit 0

# TRT-LLM runs several tasks per node in one container; serialize their pip installs.
lock_dir="$(python3 -c 'import sys; print(sys.prefix)')"
(
    flock -x 9
    has_bindings && exit 0
    echo "Installing the OpenEngine bindings from https://buf.build/gen/python"
    python3 -m pip install --break-system-packages --quiet --no-cache-dir --no-deps \
        --index-url https://buf.build/gen/python \
        "openengine-openengine-grpc-python==1.78.1.1.20260730172104+768a93c7b44e" \
        "openengine-openengine-protocolbuffers-python==33.5.0.1.20260730172104+768a93c7b44e" \
        "openengine-openengine-protocolbuffers-pyi==36.2.0.1.20260730172104+768a93c7b44e"
    has_bindings
) 9>"$lock_dir/.srtctl_openengine_install.lock"
