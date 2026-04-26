#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

pip install msgpack

# Upgrade flashinfer to v0.6.9. flashinfer-python / flashinfer-cubin only
# publish on PyPI; flashinfer-jit-cache is CUDA-specific and only on the
# cu130 index. --index-url replaces PyPI entirely, so split into two calls.
pip install --upgrade flashinfer-python==0.6.9 flashinfer-cubin==0.6.9
pip install --upgrade flashinfer-jit-cache==0.6.9 --index-url https://flashinfer.ai/whl/cu130

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

if [ -f /configs/patches/vllm_nvlink_one_sided_bf16_fix.py ]; then
    python3 /configs/patches/vllm_nvlink_one_sided_bf16_fix.py
fi