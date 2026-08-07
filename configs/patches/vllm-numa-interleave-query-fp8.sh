#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

bash /configs/patches/patch-minimax-m3-query-fp8-graph-stability.sh
bash /configs/patches/vllm-numa-interleave.sh
