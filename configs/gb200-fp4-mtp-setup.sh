#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
BRANCH="gb200-spec"

cd /sgl-workspace/sglang
git remote remove origin
git remote add origin https://github.com/sgl-project/sglang.git
git fetch origin
git checkout origin/${BRANCH}
pip install mooncake-transfer-engine==0.3.7.post2