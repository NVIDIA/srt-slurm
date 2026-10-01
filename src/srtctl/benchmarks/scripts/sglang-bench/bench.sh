#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SGLang benchmark using sglang.bench_serving
# Expects: endpoint isl osl concurrencies [req_rate]
#
# Environment variables for profiling:
#   PROFILE_TYPE: "nsys" or "torch" to enable profiling
#   PROFILE_PREFILL_IPS: Comma-separated list of prefill worker IPs
#   PROFILE_DECODE_IPS: Comma-separated list of decode worker IPs
#   PROFILE_PREFILL_START_STEP / PROFILE_PREFILL_STOP_STEP: Step range for prefill
#   PROFILE_DECODE_START_STEP / PROFILE_DECODE_STOP_STEP: Step range for decode

set -e

ENDPOINT=$1
ISL=$2
OSL=$3
CONCURRENCIES=$4
REQ_RATE=${5:-inf}

# Parse endpoint into host:port
HOST=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f1)
PORT=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f2 | cut -d/ -f1)

MODEL_NAME="${BENCH_MODEL_NAME:-deepseek-ai/DeepSeek-R1}"

echo "SGLang-Bench Config: endpoint=${ENDPOINT}; isl=${ISL}; osl=${OSL}; concurrencies=${CONCURRENCIES}; req_rate=${REQ_RATE}"

# Profiling shared helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/profiling.sh
source "${SCRIPT_DIR}/../lib/profiling.sh"
# shellcheck source=../lib/nsys.sh
source "${SCRIPT_DIR}/../lib/nsys.sh"
profiling_init_from_env

cleanup() { stop_all_profiling; }
trap cleanup EXIT

# Parse concurrency list
IFS='x' read -r -a CONCURRENCY_LIST <<< "$CONCURRENCIES"

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# Start profiling
start_all_profiling

# Run benchmark for each concurrency level
for concurrency in "${CONCURRENCY_LIST[@]}"; do
    echo "Running benchmark with concurrency: $concurrency"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"

    nsys_window_start
    set -x
    python3 -m sglang.bench_serving \
        --backend sglang-oai \
        --model "${MODEL_NAME}" \
        --host "${HOST}" --port "${PORT}" \
        --dataset-name random \
        --max-concurrency "${concurrency}" \
        --num-prompts 128 \
        --random-input-len "${ISL}" \
        --random-output-len "${OSL}" \
        --random-range-ratio 1 \
        --request-rate "${REQ_RATE}" \
        --warmup-requests 0
    set +x
    nsys_window_stop

    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Completed benchmark with concurrency: $concurrency"
    echo "-----------------------------------------"
done

stop_all_profiling

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "SGLang-Bench completed"
