#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# lm-eval accuracy evaluation using InferenceX benchmark_lib
# Expects: endpoint [infmax_workspace]

set -e

ENDPOINT=$1
INFMAX_WORKSPACE=${2:-/infmax-workspace}

# Extract HOST and PORT from endpoint (e.g., http://localhost:8000)
HOST=$(echo "$ENDPOINT" | sed -E 's|https?://||; s|:.*||')
PORT=$(echo "$ENDPOINT" | sed -E 's|.*:([0-9]+).*|\1|')

echo "lm-eval Config: endpoint=${ENDPOINT}; host=${HOST}; port=${PORT}; workspace=${INFMAX_WORKSPACE}"

# Auto-discover the served model name from /v1/models if MODEL_NAME is not set.
# This ensures we use the exact name the server recognizes, regardless of what
# $MODEL (the HuggingFace ID from the workflow) is set to.
if [[ -z "${MODEL_NAME:-}" ]]; then
    DISCOVERED_MODEL=$(curl -sf "${ENDPOINT}/v1/models" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || true)
    if [[ -n "$DISCOVERED_MODEL" ]]; then
        export MODEL_NAME="$DISCOVERED_MODEL"
        echo "Auto-discovered MODEL_NAME from /v1/models: ${MODEL_NAME}"
    else
        echo "WARNING: Could not discover model name from /v1/models, using MODEL_NAME=${MODEL_NAME:-$MODEL}"
    fi
else
    echo "Using MODEL_NAME from environment: ${MODEL_NAME}"
fi

# cd to workspace so that relative paths (e.g., utils/evals/*.yaml) resolve
cd "${INFMAX_WORKSPACE}"

# Source the InferenceX benchmark library
source "${INFMAX_WORKSPACE}/benchmarks/benchmark_lib.sh"

# Run lm-eval via benchmark_lib
# EVAL_CONC is set by the InferenceX workflow (median of conc list).
# benchmark_lib reads concurrency from EVAL_CONCURRENT_REQUESTS env var.
export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC:-256}"
echo "Running lm-eval with concurrent-requests=${EVAL_CONCURRENT_REQUESTS}..."
run_eval --framework lm-eval --port "$PORT"

# Set metadata env vars needed by append_lm_eval_summary
# These are passed through from the InferenceX environment
export TP="${TP:-${PREFILL_TP:-1}}"
export CONC="${CONC:-${EVAL_CONC}}"
export EP_SIZE="${EP_SIZE:-1}"
if [[ "${PREFILL_EP:-false}" == "true" ]]; then
    EP_SIZE="${PREFILL_TP:-1}"
fi
export EP_SIZE
export DP_ATTENTION="${DP_ATTENTION:-${PREFILL_DP_ATTN:-false}}"
export ISL="${ISL:-}"
export OSL="${OSL:-}"
export FRAMEWORK="${FRAMEWORK:-}"
export PRECISION="${PRECISION:-}"
export MODEL_PREFIX="${MODEL_PREFIX:-}"
export RUNNER_TYPE="${RUNNER_TYPE:-}"
export RESULT_FILENAME="${RESULT_FILENAME:-}"

# Generate the lm-eval summary
echo "Generating lm-eval summary..."
append_lm_eval_summary

# Copy eval artifacts to /logs/eval_results/
mkdir -p /logs/eval_results
echo "Copying eval artifacts to /logs/eval_results/..."
cp -v meta_env.json /logs/eval_results/ 2>/dev/null || true
cp -v results*.json /logs/eval_results/ 2>/dev/null || true
cp -v sample*.jsonl /logs/eval_results/ 2>/dev/null || true

echo "lm-eval evaluation complete"
