#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GSM8K accuracy evaluation using vLLM's own eval script (bundled alongside).
# Expects: host port model [num_questions] [max_tokens] [num_shots] [temperature] [repeat]

set -euo pipefail

HOST=$1
PORT=$2
MODEL=$3
NUM_QUESTIONS=${4:-1319}
MAX_TOKENS=${5:-256}
NUM_SHOTS=${6:-5}
TEMPERATURE=${7:-0.0}
REPEAT=${8:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL="${SCRIPT_DIR}/gsm8k_eval.py"

echo "vLLM GSM8K Config: host=${HOST}; port=${PORT}; model=${MODEL:-<auto>}; num_questions=${NUM_QUESTIONS}; max_tokens=${MAX_TOKENS}; num_shots=${NUM_SHOTS}; temperature=${TEMPERATURE}; repeat=${REPEAT}"

# Create results directory
result_dir="/logs/accuracy"
mkdir -p "$result_dir"

# Only pass --model when a served model name is provided.
MODEL_ARGS=()
[ -n "${MODEL}" ] && MODEL_ARGS+=(--model "${MODEL}")

for i in $(seq 1 "${REPEAT}"); do
    echo "===== vLLM GSM8K run ${i}/${REPEAT} ($(date '+%Y-%m-%d %H:%M:%S')) ====="
    python3 "$EVAL" \
        --host "${HOST}" \
        --port "${PORT}" \
        "${MODEL_ARGS[@]}" \
        --num-questions "${NUM_QUESTIONS}" \
        --max-tokens "${MAX_TOKENS}" \
        --num-shots "${NUM_SHOTS}" \
        --temperature "${TEMPERATURE}" \
        --save-results "${result_dir}/gsm8k_run${i}.json"
    echo "===== vLLM GSM8K run ${i}/${REPEAT} done ====="
done

echo "vLLM GSM8K evaluation complete"
