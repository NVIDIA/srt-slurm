#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SA-Bench: Throughput/latency benchmark
# Expects: endpoint isl osl concurrencies req_rate model_name is_disaggregated total_gpus prefill_gpus decode_gpus

set -e

# Ensure benchmark dependencies are available.
# Creates an isolated venv with --system-site-packages so container packages are
# reused and only missing deps get installed — without touching system Python.
SA_BENCH_VENV="/tmp/sa-bench-venv"
SA_BENCH_DEPS=(aiohttp numpy pandas datasets Pillow tqdm transformers huggingface_hub)

ensure_sa_bench_deps() {
    # Quick check: if all deps import fine in current Python, skip venv entirely
    if python3 -c "import aiohttp, numpy, pandas, datasets, PIL, tqdm, transformers, huggingface_hub" 2>/dev/null; then
        echo "All sa-bench deps already available — skipping venv setup"
        return
    fi

    echo "Missing sa-bench deps — installing into venv at $SA_BENCH_VENV ..."
    if [ ! -d "$SA_BENCH_VENV" ]; then
        python3 -m venv --system-site-packages "$SA_BENCH_VENV"
    fi
    source "$SA_BENCH_VENV/bin/activate"
    pip install "${SA_BENCH_DEPS[@]}"
    echo "sa-bench deps ready"
}

ensure_sa_bench_deps

#
# Optional profiling (via worker profiling endpoints):
#   PROFILE_TYPE: "nsys" or "torch" to enable profiling (or "none" to disable)
#   PROFILE_OUTPUT_DIR: Directory inside the container to save profiler output (e.g., /logs/profiles)
#   WORKER_PORT: Default port to use when an endpoint is provided as IP only (defaults to 9090)
#
# Worker targets (prefer *_ENDPOINTS; *_IPS is supported for backward-compat):
#   PROFILE_PREFILL_ENDPOINTS: Comma-separated list of prefill worker endpoints (ip:port or ip)
#   PROFILE_DECODE_ENDPOINTS: Comma-separated list of decode worker endpoints (ip:port or ip)
#   PROFILE_AGG_ENDPOINTS: Comma-separated list of aggregated worker endpoints (ip:port or ip)
#   PROFILE_PREFILL_IPS / PROFILE_DECODE_IPS / PROFILE_AGG_IPS: Comma-separated IPs (uses WORKER_PORT)
#
# Step ranges (stop_step is exclusive; num_steps = stop_step - start_step):
#   PROFILE_PREFILL_START_STEP / PROFILE_PREFILL_STOP_STEP
#   PROFILE_DECODE_START_STEP / PROFILE_DECODE_STOP_STEP
#   PROFILE_AGG_START_STEP / PROFILE_AGG_STOP_STEP

ENDPOINT=$1
ISL=$2
OSL=$3
CONCURRENCIES=$4
REQ_RATE=${5:-inf}
MODEL_PATH=${6:-/model/}
MODEL_NAME=${7:-"model"}
IS_DISAGGREGATED=${8:-false}
TOTAL_GPUS=${9:-0}
PREFILL_GPUS=${10:-0}
DECODE_GPUS=${11:-0}
RANDOM_RANGE_RATIO=${12:-0.8}
NUM_PROMPTS_MULT=${13:-10}
NUM_WARMUP_MULT=${14:-2}
CUSTOM_TOKENIZER=${15:-}
USE_CHAT_TEMPLATE=${16:-true}
DATASET_NAME=${17:-random}
DATASET_PATH=${18:-}

# Build optional custom tokenizer args
CUSTOM_TOKENIZER_ARGS=()
if [ -n "$CUSTOM_TOKENIZER" ]; then
    CUSTOM_TOKENIZER_ARGS=(--custom-tokenizer "$CUSTOM_TOKENIZER")
fi

# Build optional chat template args
CHAT_TEMPLATE_ARGS=()
if [ "$USE_CHAT_TEMPLATE" = "true" ]; then
    CHAT_TEMPLATE_ARGS=(--use-chat-template)
    if [ -z "$CUSTOM_TOKENIZER" ]; then
        echo "[sa-bench] notice: use_chat_template=true but no custom_tokenizer set."
        echo "[sa-bench]   Models without a jinja chat_template (e.g. DeepSeek-V4)"
        echo "[sa-bench]   will fail fast in benchmark_serving.py with guidance."
        echo "[sa-bench]   For DSV4, set:"
        echo "[sa-bench]     benchmark.custom_tokenizer:"
        echo "[sa-bench]       sa_bench_tokenizers.sglang_deepseek_v4.SGLangDeepseekV4Tokenizer"
        echo "[sa-bench]   Or set benchmark.use_chat_template: false to skip it."
    fi
fi

# Build dataset args
DATASET_ARGS=(--dataset-name "$DATASET_NAME")
if [ -n "$DATASET_PATH" ]; then
    DATASET_ARGS+=(--dataset-path "$DATASET_PATH")
fi

# Random-length args only apply to random dataset
RANDOM_LEN_ARGS=()
if [ "$DATASET_NAME" = "random" ]; then
    RANDOM_LEN_ARGS=(
        --random-input-len "$ISL"
        --random-output-len "$OSL"
        --random-range-ratio "${RANDOM_RANGE_RATIO}"
    )
fi

# Optional SGLang /slow_down (set by srtctl for SA-Bench when YAML provides slow_down_* and frontend is sglang):
#   SA_BENCH_SLOW_DOWN_URLS: comma-separated http://host:port base URLs (decode workers)
#   SA_BENCH_SLOW_DOWN_SLEEP_TIME / SA_BENCH_SLOW_DOWN_WAIT_TIME
SLOW_DOWN_ARGS=()
if [ -n "${SA_BENCH_SLOW_DOWN_URLS:-}" ]; then
    IFS=',' read -r -a _sd_urls <<< "${SA_BENCH_SLOW_DOWN_URLS}"
    for u in "${_sd_urls[@]}"; do
        u="$(echo "$u" | xargs)"
        if [ -n "$u" ]; then
            SLOW_DOWN_ARGS+=(--slow-down-server "$u")
        fi
    done
fi
if [ ${#SLOW_DOWN_ARGS[@]} -gt 0 ]; then
    SLOW_DOWN_EXTRA=(
        --slow-down-sleep-time "${SA_BENCH_SLOW_DOWN_SLEEP_TIME:-1}"
        --slow-down-wait-time "${SA_BENCH_SLOW_DOWN_WAIT_TIME:-60}"
    )
else
    SLOW_DOWN_EXTRA=()
fi

# Parse endpoint into host:port
HOST=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f1)
PORT=$(echo "$ENDPOINT" | sed 's|http://||' | cut -d: -f2 | cut -d/ -f1)

WORK_DIR="$(dirname "$0")"

echo "SA-Bench Config: endpoint=${ENDPOINT}; isl=${ISL}; osl=${OSL}; concurrencies=${CONCURRENCIES}; req_rate=${REQ_RATE}; model=${MODEL_NAME}; dataset=${DATASET_NAME}; dataset_path=${DATASET_PATH}"

# Profiling shared helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/profiling.sh
source "${SCRIPT_DIR}/../lib/profiling.sh"
profiling_init_from_env

cleanup() { stop_all_profiling; }
trap cleanup EXIT

# Parse concurrency list
IFS='x' read -r -a CONCURRENCY_LIST <<< "$CONCURRENCIES"

# Quick curl to verify endpoint is working
echo "Verifying endpoint..."
curl -s "${ENDPOINT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_NAME}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
        \"stream\": false,
        \"max_tokens\": 10
    }" | head -c 200
echo ""

ulimit -n 65536 2>/dev/null || true  # May fail in containers without CAP_SYS_RESOURCE

# Benchmark
if [ "$DATASET_NAME" = "custom" ]; then
    dataset_label=$(basename "${DATASET_PATH%.*}")
    result_dir="/logs/sa-bench_custom_${dataset_label}"
else
    result_dir="/logs/sa-bench_isl_${ISL}_osl_${OSL}"
fi
mkdir -p "$result_dir"

# Start profiling before benchmark
start_all_profiling

for concurrency in "${CONCURRENCY_LIST[@]}"; do

    if [ "$NUM_WARMUP_MULT" -gt 0 ]; then
        num_warmup_prompts=$((concurrency * NUM_WARMUP_MULT))
        python3 -u "${WORK_DIR}/benchmark_serving.py" \
            --model "${MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
            --host "$HOST" --port "$PORT" \
            --backend "dynamo" --endpoint /v1/completions \
            --disable-tqdm \
            "${DATASET_ARGS[@]}" \
            --num-prompts "$num_warmup_prompts" \
            "${RANDOM_LEN_ARGS[@]}" \
            --ignore-eos \
            --request-rate 250 \
            --percentile-metrics ttft,tpot,itl,e2el \
            --max-concurrency "$concurrency" \
            --trust-remote-code \
            "${CHAT_TEMPLATE_ARGS[@]}" \
            "${CUSTOM_TOKENIZER_ARGS[@]}"
    fi

    num_prompts=$((concurrency * NUM_PROMPTS_MULT))

    # Generate result filename based on mode
    if [ "$IS_DISAGGREGATED" = "true" ]; then
        result_filename="results_concurrency_${concurrency}_gpus_${TOTAL_GPUS}_ctx_${PREFILL_GPUS}_gen_${DECODE_GPUS}.json"
    else
        result_filename="results_concurrency_${concurrency}_gpus_${TOTAL_GPUS}.json"
    fi

    echo "Running benchmark with concurrency: $concurrency"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"

    set -x
    python3 -u "${WORK_DIR}/benchmark_serving.py" \
        --model "${MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
        --host "$HOST" --port "$PORT" \
        --backend "dynamo" --endpoint /v1/completions \
        --disable-tqdm \
        "${DATASET_ARGS[@]}" \
        --num-prompts "$num_prompts" \
        "${RANDOM_LEN_ARGS[@]}" \
        --ignore-eos \
        --request-rate "${REQ_RATE}" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$concurrency" \
        --trust-remote-code \
        "${CHAT_TEMPLATE_ARGS[@]}" \
        "${CUSTOM_TOKENIZER_ARGS[@]}" \
        "${SLOW_DOWN_ARGS[@]}" \
        "${SLOW_DOWN_EXTRA[@]}" \
        --save-result --result-dir "$result_dir" --result-filename "$result_filename"
    set +x

    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Completed benchmark with concurrency: $concurrency"
    echo "-----------------------------------------"
done

stop_all_profiling

echo "SA-Bench complete. Results in $result_dir"

