#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AIME25 accuracy evaluation via sgl-eval (https://github.com/sgl-project/sgl-eval).
# Mirrors gpqa/bench.sh; only the sgl-eval task name differs.
#
# Args (from srtctl AIME25Runner.build_command):
#   endpoint [num_examples] [max_tokens] [repeat] [num_threads] [temperature] [top_p]

set -e

ENDPOINT=$1                       # http://localhost:8000 (nginx) -- used as fallback
NUM_EXAMPLES=${2:-30}
MAX_TOKENS=${3:-32768}
REPEAT=${4:-8}
NUM_THREADS=${5:-128}
TEMPERATURE=${6:-}                # optional: forwarded to sgl-eval --temperature
TOP_P=${7:-}                      # optional: forwarded to sgl-eval --top-p

# Xet client hangs on this cluster; force plain HF download path.
export HF_HUB_DISABLE_XET=1

result_dir="/logs/accuracy"
mkdir -p "$result_dir"

echo "Installing sgl-eval..."
pip install --break-system-packages -q git+https://github.com/sgl-project/sgl-eval 2>&1 | tail -2

# Bypass nginx round-robin (some frontends hang on /v1/models, or list the model
# but 404 on chat completions). Read the frontend upstreams from the mounted
# nginx.conf and probe each with an actual chat completion (the real path the
# eval will use). /v1/models alone is NOT a reliable health check -- a frontend
# can return the model on /v1/models yet 404 chat. Pin to the first frontend
# that returns 200 on chat. Fall back to the nginx endpoint if none answer.
MODEL_NAME="deepseek-ai/DeepSeek-V4-Pro"
PROBE_BODY="{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}"
BASE_URL="${ENDPOINT}/v1"
NGINX_CONF="/logs/nginx.conf"
if [ -f "$NGINX_CONF" ]; then
    FRONTENDS=$(grep -oE 'server +[0-9.]+:[0-9]+' "$NGINX_CONF" | awk '{print $2}')
    echo "Selecting a frontend that actually serves chat (bypassing nginx round-robin)..."
    for fe in $FRONTENDS; do
        code=$(curl -s -m 25 -o /dev/null -w '%{http_code}' \
            "http://${fe}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d "$PROBE_BODY" 2>/dev/null || echo 000)
        echo "  probe http://${fe}/v1/chat/completions -> ${code}"
        if [ "$code" = "200" ]; then
            BASE_URL="http://${fe}/v1"
            break
        fi
    done
fi
echo "Using base-url: ${BASE_URL}"

echo "AIME25 Config (sgl-eval): base_url=${BASE_URL} num_examples=${NUM_EXAMPLES} max_tokens=${MAX_TOKENS} n_repeats=${REPEAT} num_threads=${NUM_THREADS}"
echo "Running AIME25 evaluation via sgl-eval..."

SAMPLING_ARGS=()
if [ -n "$TEMPERATURE" ]; then
    SAMPLING_ARGS+=(--temperature "$TEMPERATURE")
fi
if [ -n "$TOP_P" ]; then
    SAMPLING_ARGS+=(--top-p "$TOP_P")
fi

sgl-eval run aime25 \
    --base-url "${BASE_URL}" \
    --num-examples "${NUM_EXAMPLES}" \
    --n-repeats "${REPEAT}" \
    --max-tokens "${MAX_TOKENS}" \
    --num-threads "${NUM_THREADS}" \
    "${SAMPLING_ARGS[@]}" \
    --out-dir "$result_dir"

echo "Results saved under: $result_dir"
echo "AIME25 evaluation complete"
