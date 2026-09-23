#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# vllm-container-deps-and-dynamo-wheel.sh plus the 591bb95 v2 transfer
# short-extend fullgraph patch, since srtctl's setup_script only runs a single
# named script. The patch goes last so a pip install cannot overwrite the
# patched vLLM source afterwards.

set -euo pipefail

bash /configs/patches/vllm-container-deps.sh
bash /configs/install_dynamo_wheel.sh
bash /configs/patches/apply-vllm-591bb95-v2-transfer-short-extend-fullgraph.sh
