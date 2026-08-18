#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENDPOINT="$1"
ISL="$2"
OSL="$3"
CONCURRENCY="$4"
SEED_OSL="$5"
TOKENIZER="$6"
MODEL="$7"
DP_SIZE="$8"
TOTAL_GPUS="$9"
SETTLE_SECONDS="${10}"

python "${SCRIPT_DIR}/prefix_replay.py" \
    --endpoint "${ENDPOINT}" \
    --isl "${ISL}" \
    --osl "${OSL}" \
    --concurrency "${CONCURRENCY}" \
    --seed-osl "${SEED_OSL}" \
    --tokenizer "${TOKENIZER}" \
    --model "${MODEL}" \
    --dp-size "${DP_SIZE}" \
    --total-gpus "${TOTAL_GPUS}" \
    --settle-seconds "${SETTLE_SECONDS}"
