#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

bash /configs/patches/vllm-container-deps.sh
python3 /configs/patches/vllm_revert_pr41015_fp4_cvt.py
