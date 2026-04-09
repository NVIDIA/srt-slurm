#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Trace Replay Benchmark using aiperf
# Replays a user-provided JSONL trace dataset at configurable concurrency levels.
# Uses aiperf with --custom-dataset-type mooncake_trace.
#
# Usage: bench.sh ENDPOINT MODEL_NAME TRACE_FILE CONCURRENCIES [TTFT_THRESHOLD] [ITL_THRESHOLD] [TOKENIZER_PATH]

set -e

ENDPOINT=$1
MODEL_NAME=${2:-"test-model"}
TRACE_FILE=$3
CONCURRENCIES=${4:-"1"}
TTFT_THRESHOLD=${5:-2000}
ITL_THRESHOLD=${6:-25}
TOKENIZER_PATH=${7:-"/model"}

# Optional: extra Prometheus endpoints for AIPerf server metrics
SERVER_METRICS_ARGS=()
if [ -n "${AIPERF_SERVER_METRICS_URLS:-}" ]; then
    IFS=',' read -r -a server_metrics_urls <<< "${AIPERF_SERVER_METRICS_URLS}"
    if [ ${#server_metrics_urls[@]} -gt 0 ]; then
        SERVER_METRICS_ARGS+=(--server-metrics "${server_metrics_urls[@]}")
    fi
fi

# Setup directories (BASE_DIR defaults to /logs inside container, overridable for testing)
BASE_DIR="${BASE_DIR:-/logs}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_DIR}/artifacts}"
mkdir -p "${ARTIFACT_DIR}"

# Increase aiperf HTTP timeout
export AIPERF_HTTP_SO_RCVTIMEO=120

echo "=============================================="
echo "Trace Replay Benchmark (aiperf)"
echo "=============================================="
echo "Endpoint: ${ENDPOINT}"
echo "Model: ${MODEL_NAME}"
echo "Trace File: ${TRACE_FILE}"
echo "Concurrencies: ${CONCURRENCIES}"
echo "TTFT Threshold: ${TTFT_THRESHOLD}ms"
echo "ITL Threshold: ${ITL_THRESHOLD}ms"
echo "Tokenizer Path: ${TOKENIZER_PATH}"
echo "=============================================="

# Validate trace file exists
if [ ! -f "${TRACE_FILE}" ]; then
    echo "ERROR: Trace file not found: ${TRACE_FILE}"
    exit 1
fi

# Install aiperf if not present
if ! command -v aiperf &> /dev/null; then
    echo "Installing aiperf..."
    pip install aiperf
fi

# Run small benchmark for warmup
echo "Running warmup..."
aiperf profile \
    -m "${MODEL_NAME}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --url "${ENDPOINT}" \
    --streaming \
    --ui simple \
    --extra-inputs ignore_eos:true \
    --concurrency 1 \
    --request-count 5
echo "Warmup complete"

# Setup artifact directory
MODEL_BASE_NAME="${MODEL_NAME##*/}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Parse concurrencies (comma-separated)
IFS=',' read -r -a CONCURRENCY_LIST <<< "${CONCURRENCIES}"

OVERALL_EXIT_CODE=0

for C in "${CONCURRENCY_LIST[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running concurrency=${C}"
    echo "=============================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting benchmark at concurrency ${C}"

    RUN_ARTIFACT_DIR="${ARTIFACT_DIR}/${MODEL_BASE_NAME}_trace_c${C}_${TIMESTAMP}"
    mkdir -p "${RUN_ARTIFACT_DIR}"

    # Disable set -e for aiperf so a failure at one concurrency level
    # doesn't abort the entire sweep — we want to try all levels.
    set +e
    aiperf profile \
        -m "${MODEL_NAME}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --input-file "${TRACE_FILE}" \
        --custom-dataset-type mooncake_trace \
        --url "${ENDPOINT}" \
        --streaming \
        --extra-inputs ignore_eos:true \
        --concurrency "${C}" \
        --random-seed 42 \
        --ui simple \
        --artifact-dir "${RUN_ARTIFACT_DIR}" \
        "${SERVER_METRICS_ARGS[@]}" \
        --goodput "time_to_first_token:${TTFT_THRESHOLD} inter_token_latency:${ITL_THRESHOLD}"
    BENCH_EXIT_CODE=$?
    set -e

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Concurrency ${C} complete (exit code: ${BENCH_EXIT_CODE})"

    if [ $BENCH_EXIT_CODE -ne 0 ]; then
        OVERALL_EXIT_CODE=$BENCH_EXIT_CODE
    fi

    # List artifacts
    ls -la "${RUN_ARTIFACT_DIR}" 2>/dev/null || true
done

echo ""
echo "=============================================="
echo "Trace Replay Benchmark Complete"
echo "Results saved to: ${ARTIFACT_DIR}"
echo "=============================================="

exit $OVERALL_EXIT_CODE
