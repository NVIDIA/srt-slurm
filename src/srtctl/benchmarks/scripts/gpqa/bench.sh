#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GPQA accuracy evaluation
# Expects: endpoint [num_examples] [max_tokens] [repeat] [num_threads]

set -e

ENDPOINT=$1
NUM_EXAMPLES=${2:-198}
MAX_TOKENS=${3:-32768}
REPEAT=${4:-8}
NUM_THREADS=${5:-128}

# Auto-detect model name from /v1/models endpoint; fall back to default
MODEL_NAME=$(curl -s "${ENDPOINT}/v1/models" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")
if [ -z "${MODEL_NAME}" ]; then
    MODEL_NAME="deepseek-ai/DeepSeek-R1"
    echo "Warning: Could not auto-detect model name, using default: ${MODEL_NAME}"
fi

echo "GPQA Config: endpoint=${ENDPOINT}; model=${MODEL_NAME}; num_examples=${NUM_EXAMPLES}; max_tokens=${MAX_TOKENS}; repeat=${REPEAT}; num_threads=${NUM_THREADS}"

# Create results directory
result_dir="/logs/accuracy"
mkdir -p "$result_dir"

# Set OPENAI_API_KEY if not set
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

echo "Running GPQA evaluation..."

python3 -m sglang.test.run_eval \
    --base-url "${ENDPOINT}" \
    --model "${MODEL_NAME}" \
    --eval-name gpqa \
    --num-examples "${NUM_EXAMPLES}" \
    --max-tokens "${MAX_TOKENS}" \
    --repeat "${REPEAT}" \
    --num-threads "${NUM_THREADS}"

# Copy result file
result_file=$(ls -t /tmp/gpqa_*.json 2>/dev/null | head -n1)
if [ -f "$result_file" ]; then
    cp "$result_file" "$result_dir/"
    echo "Results saved to: $result_dir/$(basename "$result_file")"
else
    echo "Warning: Could not find result file in /tmp"
fi

echo "GPQA evaluation complete"

