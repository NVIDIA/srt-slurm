#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

pip install msgpack

if [ -f /configs/install-ai-dynamo.sh ]; then
    bash /configs/install-ai-dynamo.sh
else
    python -m pip install --pre --no-deps --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com "ai-dynamo==1.2.0.dev20260426"
fi

# Upgrade FlashInfer for the NVLink one-sided all-to-all bf16 dispatch patch.
# flashinfer-python / flashinfer-cubin publish on PyPI; flashinfer-jit-cache is
# CUDA-specific and only on the cu130 index. --index-url replaces PyPI entirely,
# so split into two calls.
pip install --upgrade flashinfer-python==0.6.9 flashinfer-cubin==0.6.9
pip install --upgrade flashinfer-jit-cache==0.6.9 --index-url https://flashinfer.ai/whl/cu130

if [ -f /configs/patches/vllm_numa_bind_hash_fix.py ]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi

if [ -f /configs/patches/vllm_nvlink_one_sided_bf16_fix.py ]; then
    python3 /configs/patches/vllm_nvlink_one_sided_bf16_fix.py
fi
