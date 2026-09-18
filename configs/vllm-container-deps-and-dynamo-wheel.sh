#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Combines vllm-container-deps.sh (numactl/msgpack/NUMA-bind patch) with
# install_dynamo_wheel.sh (installs the wheels mounted at /dynamo_wheels via
# extra_mount), since srtctl's setup_script only runs a single named script.

set -euo pipefail

bash /configs/patches/vllm-container-deps.sh
bash /configs/install_dynamo_wheel.sh
