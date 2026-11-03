#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Install Infera (infera-router and the infera.engine.sglang worker launcher) from PyPI
# (amd-infera) into a stock SGLang image. INFERA_VERSION selects the release; its Linux
# x86_64 wheel bundles the infera-router binary. --no-deps keeps every package the image
# ships, and nats-py is the one dependency it lacks.

set -euo pipefail
: "${INFERA_VERSION:?set INFERA_VERSION to the Infera release}"

pip install --quiet --no-cache-dir --no-deps \
    "nats-py==2.16.0" \
    "amd-infera==${INFERA_VERSION}"
# Fail here, not at router start, when pip installed the pure-Python wheel (a release or a
# platform without the bundled router):
# this is the binary `python3 -m infera.server --router-backend rust` execs.
python3 -c "import infera.engine.sglang; from infera.server.launch_rust import _find_binary; _find_binary()"
