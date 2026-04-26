#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# lm-eval accuracy evaluation.
#
# By default runs the EleutherAI lm-evaluation-harness CLI directly against
# the OpenAI-compatible endpoint. If an external eval harness is mounted at
# /lm-eval-workspace (or pointed to via LM_EVAL_LIB) and exposes a compatible
# benchmark_lib.sh, that harness is sourced and used instead.
#
# Expects: endpoint

set -e

ENDPOINT=$1

# Extract HOST and PORT from endpoint (e.g., http://localhost:8000)
HOST=$(echo "$ENDPOINT" | sed -E 's|https?://||; s|:.*||')
PORT=$(echo "$ENDPOINT" | sed -E 's|.*:([0-9]+).*|\1|')

echo "lm-eval config: endpoint=${ENDPOINT}; host=${HOST}; port=${PORT}"

# Auto-discover the served model name from /v1/models if MODEL_NAME is not set.
# This ensures we use the exact name the server recognizes, regardless of what
# $MODEL (a HuggingFace ID from the launcher) is set to.
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

# Output directory for eval artifacts. Override with EVAL_OUTPUT_DIR.
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/logs/eval_results}"
mkdir -p "${EVAL_OUTPUT_DIR}"

# Concurrency. Prefer EVAL_CONC (set by orchestration) over EVAL_CONCURRENT_REQUESTS.
export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC:-${EVAL_CONCURRENT_REQUESTS:-256}}"
echo "Running lm-eval with concurrent-requests=${EVAL_CONCURRENT_REQUESTS}..."

eval_rc=0

# Let a host-provided harness take over if one is mounted. LM_EVAL_LIB points
# at a benchmark_lib.sh directly; otherwise we look for one at the standard
# /lm-eval-workspace mount point.
EXT_LIB="${LM_EVAL_LIB:-}"
if [[ -z "$EXT_LIB" && -f "/lm-eval-workspace/benchmarks/benchmark_lib.sh" ]]; then
    EXT_LIB="/lm-eval-workspace/benchmarks/benchmark_lib.sh"
fi

if [[ -n "$EXT_LIB" && -f "$EXT_LIB" ]]; then
    echo "Using external eval harness: $EXT_LIB"
    # cd to the workspace root so relative paths (utils/evals/*.yaml) resolve.
    cd "$(dirname "$(dirname "$EXT_LIB")")"
    # shellcheck disable=SC1090
    source "$EXT_LIB"
    run_eval --framework lm-eval --port "$PORT" || eval_rc=$?

    # Derive metadata env vars that append_lm_eval_summary expects but the
    # launcher passes under different names.
    export IS_MULTINODE="${IS_MULTINODE:-true}"
    export TP="${TP:-${PREFILL_TP:-1}}"
    export CONC="${CONC:-${EVAL_CONC:-${EVAL_CONCURRENT_REQUESTS:-1}}}"
    export EP_SIZE="${EP_SIZE:-${PREFILL_EP:-1}}"
    export DP_ATTENTION="${DP_ATTENTION:-${PREFILL_DP_ATTN:-false}}"
    export PREFILL_DP_ATTENTION="${PREFILL_DP_ATTENTION:-${PREFILL_DP_ATTN:-${DP_ATTENTION:-false}}}"
    export DECODE_DP_ATTENTION="${DECODE_DP_ATTENTION:-${DECODE_DP_ATTN:-${DP_ATTENTION:-false}}}"

    if declare -F append_lm_eval_summary >/dev/null 2>&1; then
        echo "Generating lm-eval summary..."
        append_lm_eval_summary || true
    fi

    echo "Copying eval artifacts to ${EVAL_OUTPUT_DIR}/..."
    cp -v meta_env.json "${EVAL_OUTPUT_DIR}/" 2>/dev/null || true
    cp -v results*.json "${EVAL_OUTPUT_DIR}/" 2>/dev/null || true
    cp -v sample*.jsonl "${EVAL_OUTPUT_DIR}/" 2>/dev/null || true
else
    # Default path: run the EleutherAI lm-evaluation-harness CLI directly.
    if ! command -v lm_eval >/dev/null 2>&1; then
        echo "lm_eval CLI not found; installing lm-evaluation-harness via pip..."
        python3 -m pip install --quiet --upgrade lm-eval || {
            echo "ERROR: lm_eval CLI is not available and pip install failed."
            exit 127
        }
    fi

    TASKS="${LM_EVAL_TASKS:-gsm8k}"
    MODEL_TYPE="${LM_EVAL_MODEL:-local-chat-completions}"
    BASE_URL="${ENDPOINT%/}/v1/chat/completions"

    echo "Running lm_eval on tasks=${TASKS} (model=${MODEL_NAME}, model_type=${MODEL_TYPE})"

    MODEL_ARGS="base_url=${BASE_URL},model=${MODEL_NAME},num_concurrent=${EVAL_CONCURRENT_REQUESTS},tokenized_requests=False"

    # shellcheck disable=SC2086
    lm_eval \
        --model "${MODEL_TYPE}" \
        --model_args "${MODEL_ARGS}" \
        --tasks "${TASKS}" \
        --apply_chat_template \
        --output_path "${EVAL_OUTPUT_DIR}" \
        ${LM_EVAL_EXTRA_ARGS:-} || eval_rc=$?
fi

if [[ "$eval_rc" -ne 0 ]]; then
    echo "lm-eval evaluation failed with exit code ${eval_rc}"
    exit "$eval_rc"
fi

echo "lm-eval evaluation complete"
