#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

bash /configs/patches/vllm-container-deps.sh

python3 -m pip install nixl
python3 -m pip install --force-reinstall --no-deps nixl-cu13

python3 - <<'PY'
import importlib.metadata

for package in ("nixl", "nixl-cu13"):
    print(f"[nixl-cu13] {package}=={importlib.metadata.version(package)}")
PY
