#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# lm-eval accuracy evaluation using InferenceX benchmark_lib.
# Expects: endpoint [infmax_workspace]

set -euo pipefail

ENDPOINT=$1
INFMAX_WORKSPACE=${2:-/infmax-workspace}

HOST=$(echo "$ENDPOINT" | sed -E 's|https?://||; s|:.*||')
PORT=$(echo "$ENDPOINT" | sed -E 's|.*:([0-9]+).*|\1|')

echo "lm-eval Config: endpoint=${ENDPOINT}; host=${HOST}; port=${PORT}; workspace=${INFMAX_WORKSPACE}"

if [[ ! -f "${INFMAX_WORKSPACE}/benchmarks/benchmark_lib.sh" ]]; then
    echo "ERROR: ${INFMAX_WORKSPACE}/benchmarks/benchmark_lib.sh not found." >&2
    echo "Set INFMAX_WORKSPACE on the host so srt-slurm can mount InferenceX at /infmax-workspace." >&2
    exit 1
fi

if [[ -z "${MODEL_NAME:-}" ]]; then
    DISCOVERED_MODEL=$(curl -sf "${ENDPOINT}/v1/models" 2>/dev/null \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
    if [[ -n "$DISCOVERED_MODEL" ]]; then
        export MODEL_NAME="$DISCOVERED_MODEL"
        echo "Auto-discovered MODEL_NAME from /v1/models: ${MODEL_NAME}"
    else
        echo "WARNING: Could not discover model name from /v1/models; using MODEL_NAME=${MODEL_NAME:-${MODEL:-unset}}"
    fi
else
    echo "Using MODEL_NAME from environment: ${MODEL_NAME}"
fi

cd "${INFMAX_WORKSPACE}"
source "${INFMAX_WORKSPACE}/benchmarks/benchmark_lib.sh"

export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC:-${EVAL_CONCURRENT_REQUESTS:-256}}"
echo "Running lm-eval with concurrent-requests=${EVAL_CONCURRENT_REQUESTS}..."

eval_rc=0
run_eval --framework lm-eval --port "$PORT" || eval_rc=$?

export IS_MULTINODE="${IS_MULTINODE:-true}"
export TP="${TP:-${PREFILL_TP:-1}}"
export CONC="${CONC:-${EVAL_CONC:-${EVAL_CONCURRENT_REQUESTS:-1}}}"
export EP_SIZE="${EP_SIZE:-${PREFILL_EP:-1}}"
export DP_ATTENTION="${DP_ATTENTION:-${PREFILL_DP_ATTN:-false}}"
export PREFILL_DP_ATTENTION="${PREFILL_DP_ATTENTION:-${PREFILL_DP_ATTN:-${DP_ATTENTION:-false}}}"
export DECODE_DP_ATTENTION="${DECODE_DP_ATTENTION:-${DECODE_DP_ATTN:-${DP_ATTENTION:-false}}}"

echo "Generating lm-eval summary..."
append_lm_eval_summary || true

mkdir -p /logs/eval_results
echo "Copying eval artifacts to /logs/eval_results/..."
cp -v meta_env.json /logs/eval_results/ 2>/dev/null || true
cp -v results*.json /logs/eval_results/ 2>/dev/null || true
cp -v sample*.jsonl /logs/eval_results/ 2>/dev/null || true

if [[ "$eval_rc" -ne 0 ]]; then
    echo "lm-eval evaluation failed with exit code ${eval_rc}"
    exit "$eval_rc"
fi

echo "lm-eval evaluation complete"
