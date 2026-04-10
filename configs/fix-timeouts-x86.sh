#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
BRANCH="fastdg"

# v0.5.8 + cherry-pick https://github.com/sgl-project/sglang/pull/18111
# Make sure to set SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
cd /sgl-workspace/sglang
git remote remove origin
git remote add origin https://github.com/trevor-m/sglang.git
git fetch origin
git checkout origin/${BRANCH}

# Increase device timeout from 100s -> 1000s
# This script adds in a target for 9.0 so we can also compile on Hopper
cd /sgl-workspace/DeepEP
sed -i 's/#define NUM_TIMEOUT_CYCLES 200000000000ull/#define NUM_TIMEOUT_CYCLES 2000000000000ull/' csrc/kernels/configs.cuh
TORCH_CUDA_ARCH_LIST="9.0,10.0;10.3" MAX_JOBS=$(nproc) pip install --break-system-packages --force-reinstall --no-build-isolation .
