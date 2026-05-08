#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ -x /opt/nsys/bin/nsys ]]; then
  ln -sf /opt/nsys/bin/nsys /usr/local/bin/nsys
  nsys --version
else
  echo "ERROR: /opt/nsys/bin/nsys is not executable; check the host nsys mount" >&2
  exit 1
fi
