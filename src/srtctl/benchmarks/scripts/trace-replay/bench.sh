#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Trace Replay Benchmark using aiperf
# Replays a user-provided JSONL trace dataset at configurable concurrency levels.
# Uses aiperf with --custom-dataset-type mooncake_trace.
#
# Usage: bench.sh ENDPOINT MODEL_NAME TRACE_FILE CONCURRENCIES [TTFT_THRESHOLD] [ITL_THRESHOLD] [TOKENIZER_PATH] [EXTRA_ARGS]
#
# EXTRA_ARGS: JSON-encoded string of additional aiperf flags (passed from Python)
#
# Profiling support (optional):
#   PROFILING_BACKEND: set to "trtllm" to use the no-op TRTLLM profiling lib
#                      (profiling is managed by worker env vars at launch time)
#   PROFILE_TYPE: "nsys" or "nsys-time" -- logged for diagnostics

set -e

SCRIPT_DIR="$(dirname "$0")"
LIB_DIR="${SCRIPT_DIR}/../lib"

# Source the appropriate profiling library
if [[ "${PROFILING_BACKEND:-}" == "trtllm" ]]; then
    # shellcheck source=../lib/profiling_trtllm.sh
    source "${LIB_DIR}/profiling_trtllm.sh"
else
    # shellcheck source=../lib/profiling.sh
    source "${LIB_DIR}/profiling.sh"
fi
profiling_init_from_env

cleanup() { stop_all_profiling; }
trap cleanup EXIT

# Ensure Python output is unbuffered for real-time logging
export PYTHONUNBUFFERED=1

ENDPOINT=$1
MODEL_NAME=${2:-"test-model"}
TRACE_FILE=$3
CONCURRENCIES=${4:-"1"}
TTFT_THRESHOLD=${5:-2000}
ITL_THRESHOLD=${6:-25}
TOKENIZER_PATH=${7:-"/model"}
# Remaining args are extra aiperf flags
shift 7 2>/dev/null || true
EXTRA_ARGS=("$@")

# Optional: extra Prometheus endpoints for AIPerf server metrics
SERVER_METRICS_ARGS=()
if [ -n "${AIPERF_SERVER_METRICS_URLS:-}" ]; then
    IFS=',' read -r -a server_metrics_urls <<< "${AIPERF_SERVER_METRICS_URLS}"
    if [ ${#server_metrics_urls[@]} -gt 0 ]; then
        SERVER_METRICS_ARGS+=(--server-metrics "${server_metrics_urls[@]}")
        SERVER_METRICS_ARGS+=(--server-metrics-formats json jsonl)
    fi
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
echo "Trace Replay Benchmark (aiperf)"
echo "=============================================="
echo "Endpoint: ${ENDPOINT}"
echo "Model: ${MODEL_NAME}"
echo "Trace File: ${TRACE_FILE}"
echo "Concurrencies: ${CONCURRENCIES}"
echo "TTFT Threshold: ${TTFT_THRESHOLD}ms"
echo "ITL Threshold: ${ITL_THRESHOLD}ms"
echo "Tokenizer Path: ${TOKENIZER_PATH}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra Args: ${EXTRA_ARGS[*]}"
fi
if [[ "${PROFILE_TYPE:-none}" != "none" ]]; then
    echo "Profiling: ${PROFILE_TYPE} (backend=${PROFILING_BACKEND:-sglang})"
fi
echo "=============================================="

# Validate trace file exists
if [ ! -f "${TRACE_FILE}" ]; then
    echo "ERROR: Trace file not found: ${TRACE_FILE}"
    exit 1
fi

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

# Run small benchmark for warmup
echo "Running warmup..."
WARMUP_DIR="${ARTIFACT_DIR}/warmup"
mkdir -p "${WARMUP_DIR}"
aiperf profile \
    -m "${MODEL_NAME}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --tokenizer-trust-remote-code \
    --url "${ENDPOINT}" \
    --streaming \
    --ui simple \
    --extra-inputs ignore_eos:true \
    --concurrency 1 \
    --request-count 5 \
    --artifact-dir "${WARMUP_DIR}"
echo "Warmup complete"

# Setup artifact directory
MODEL_BASE_NAME="${MODEL_NAME##*/}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Parse concurrencies (comma-separated)
IFS=',' read -r -a CONCURRENCY_LIST <<< "${CONCURRENCIES}"

# a no-op if profiling is not enabled
start_all_profiling

for C in "${CONCURRENCY_LIST[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running concurrency=${C}"
    echo "=============================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting benchmark at concurrency ${C}"

    RUN_ARTIFACT_DIR="${ARTIFACT_DIR}/${MODEL_BASE_NAME}_trace_c${C}_${TIMESTAMP}"
    mkdir -p "${RUN_ARTIFACT_DIR}"

    aiperf profile \
        -m "${MODEL_NAME}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --tokenizer-trust-remote-code \
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
        --goodput "time_to_first_token:${TTFT_THRESHOLD} inter_token_latency:${ITL_THRESHOLD}" \
        "${EXTRA_ARGS[@]}"

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Concurrency ${C} complete"

    # List artifacts
    ls -la "${RUN_ARTIFACT_DIR}" 2>/dev/null || true
done

# a no-op if profiling is not enabled
stop_all_profiling

echo ""
echo "=============================================="
echo "Trace Replay Benchmark Complete"
echo "Results saved to: ${ARTIFACT_DIR}"
echo "=============================================="
