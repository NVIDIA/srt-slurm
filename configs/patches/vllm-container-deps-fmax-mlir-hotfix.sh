#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

bash /configs/patches/vllm-container-deps.sh
python3 /configs/patches/vllm_fmax_nvvm_signature_fix.py
python3 /configs/patches/cutlass_global_dtors_data_fix.py
