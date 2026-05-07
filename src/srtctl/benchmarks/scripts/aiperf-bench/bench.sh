#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AIPerf Synthetic Benchmark
# Benchmarks LLM serving at specific ISL/OSL using aiperf's built-in synthetic
# dataset generation. Sweeps multiple concurrency levels.
#
# Usage: bench.sh ENDPOINT MODEL_NAME ISL OSL CONCURRENCIES [TTFT_THRESHOLD] [ITL_THRESHOLD] [ISL_STDDEV] [TOKENIZER_PATH] [REQ_RATE] [EXTRA_ARGS...]
#
# EXTRA_ARGS: Additional aiperf CLI flags (e.g., --benchmark-duration 600)

set -e

SCRIPT_DIR="$(dirname "$0")"

# Ensure Python output is unbuffered for real-time logging
export PYTHONUNBUFFERED=1

ENDPOINT=$1
MODEL_NAME=${2:-"test-model"}
ISL=${3:-1024}
OSL=${4:-128}
CONCURRENCIES=${5:-"1"}
TTFT_THRESHOLD=${6:-2000}
ITL_THRESHOLD=${7:-25}
ISL_STDDEV=${8:-0}
TOKENIZER_PATH=${9:-"/model"}
REQ_RATE=${10:-"inf"}
# Remaining args are extra aiperf flags
shift 10 2>/dev/null || true
EXTRA_ARGS=("$@")

# Optional: extra Prometheus endpoints for AIPerf server metrics
SERVER_METRICS_ARGS=()
if [ -n "${AIPERF_SERVER_METRICS_URLS:-}" ]; then
    IFS=',' read -r -a server_metrics_urls <<< "${AIPERF_SERVER_METRICS_URLS}"
    if [ ${#server_metrics_urls[@]} -gt 0 ]; then
        SERVER_METRICS_ARGS+=(--server-metrics "${server_metrics_urls[@]}")
    fi
else:
    SERVER_METRICS_ARGS+=("--no-server-metrics")
fi

# Setup directories (BASE_DIR defaults to /logs inside container, overridable for testing)
BASE_DIR="${BASE_DIR:-/logs}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_DIR}/artifacts}"
mkdir -p "${ARTIFACT_DIR}"

# Increase file descriptor limit for high concurrency
ulimit -n 600000 2>/dev/null || ulimit -n 65536 2>/dev/null || true

# Increase aiperf HTTP timeout
export AIPERF_HTTP_SO_RCVTIMEO=120

echo "=============================================="
echo "AIPerf Synthetic Benchmark"
echo "=============================================="
echo "Endpoint:      ${ENDPOINT}"
echo "Model:         ${MODEL_NAME}"
echo "ISL:           ${ISL}"
echo "OSL:           ${OSL}"
echo "ISL Stddev:    ${ISL_STDDEV}"
echo "Concurrencies: ${CONCURRENCIES}"
echo "TTFT Threshold: ${TTFT_THRESHOLD}ms"
echo "ITL Threshold:  ${ITL_THRESHOLD}ms"
echo "Request Rate:  ${REQ_RATE}"
echo "Tokenizer:     ${TOKENIZER_PATH}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra Args:    ${EXTRA_ARGS[*]}"
fi
echo "=============================================="

# Create isolated aiperf environment (avoids polluting container packages)
# AIPERF_PACKAGE env var controls the version (e.g., "aiperf>=0.7.0")
AIPERF_SPEC="${AIPERF_PACKAGE:-aiperf}"
AIPERF_VENV="/tmp/aiperf-${SLURM_JOB_ID:-$$}"

echo "Setting up aiperf environment: ${AIPERF_SPEC}"

# Install uv if not in container
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv "${AIPERF_VENV}"
uv pip install -p "${AIPERF_VENV}" "${AIPERF_SPEC}" tiktoken
export PATH="${AIPERF_VENV}/bin:${PATH}"
echo "aiperf $(aiperf --version 2>/dev/null || echo 'installed') in ${AIPERF_VENV}"

# Warmup run
echo ""
echo "Running warmup (concurrency=1, 5 requests)..."
WARMUP_DIR="${ARTIFACT_DIR}/warmup"
mkdir -p "${WARMUP_DIR}"
aiperf profile \
    -m "${MODEL_NAME}" \
    --endpoint-type chat \
    --streaming \
    --url "${ENDPOINT}" \
    --synthetic-input-tokens-mean "${ISL}" \
    --synthetic-input-tokens-stddev "${ISL_STDDEV}" \
    --output-tokens-mean "${OSL}" \
    --extra-inputs min_tokens:"${OSL}" \
    --extra-inputs ignore_eos:true \
    --concurrency 1 \
    --request-count 5 \
    --tokenizer "${TOKENIZER_PATH}" \
    --tokenizer-trust-remote-code \
    --random-seed 42 \
    --ui-type none \
    --artifact-dir "${WARMUP_DIR}"
echo "Warmup complete"

# Setup artifact directory prefix
MODEL_BASE_NAME="${MODEL_NAME##*/}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

if [ -n "$DEBUG_MODE" ]; then
    echo "WARNING: DEBUG_MODE ENABLED! MAKE SURE TO CANCEL YOUR JOB WHEN YOU ARE DONE!"
    sleep inf
fi

# Parse concurrencies (comma-separated)
IFS=',' read -r -a CONCURRENCY_LIST <<< "${CONCURRENCIES}"

for C in "${CONCURRENCY_LIST[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running concurrency=${C}"
    echo "=============================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting benchmark at concurrency ${C}"

    REQUEST_COUNT=$((C * 10))
    RUN_ARTIFACT_DIR="${ARTIFACT_DIR}/${MODEL_BASE_NAME}_isl${ISL}_osl${OSL}_c${C}_${TIMESTAMP}"
    mkdir -p "${RUN_ARTIFACT_DIR}"

    # Build optional request-rate flag (omit for "inf" / unlimited)
    REQ_RATE_ARGS=()
    if [ -n "${REQ_RATE}" ] && [ "${REQ_RATE}" != "inf" ]; then
        REQ_RATE_ARGS=(--request-rate "${REQ_RATE}")
    fi

    aiperf profile \
        -m "${MODEL_NAME}" \
        --endpoint-type completions \
        --streaming \
        --url "${ENDPOINT}" \
        --synthetic-input-tokens-mean "${ISL}" \
        --synthetic-input-tokens-stddev "${ISL_STDDEV}" \
        --output-tokens-stddev "${ISL_STDDEV}" \
        --output-tokens-mean "${OSL}" \
        --extra-inputs ignore_eos:true \
        --concurrency "${C}" \
        "${REQ_RATE_ARGS[@]}" \
        --request-count "${REQUEST_COUNT}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --tokenizer-trust-remote-code \
        --random-seed 42 \
        --ui-type none \
        --artifact-dir "${RUN_ARTIFACT_DIR}" \
        --extra-inputs "temperature:0.0" \
        --extra-inputs "best_of:1" \
        "${SERVER_METRICS_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"

    python3 "${SCRIPT_DIR}/format_results.py" "${RUN_ARTIFACT_DIR}" || true

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Concurrency ${C} complete"

    # List artifacts
    ls -la "${RUN_ARTIFACT_DIR}" 2>/dev/null || true
done

echo ""
echo "=============================================="
echo "AIPerf Synthetic Benchmark Complete"
echo "Results saved to: ${ARTIFACT_DIR}"
echo "=============================================="
