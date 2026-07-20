#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AIPerf FIXED ISL/OSL concurrency-sweep benchmark.
#
# Drives the serving frontend with NVIDIA aiperf (high-performance client) instead of
# sa-bench's single-process Python client, which under-reports high-concurrency throughput
# due to SSE streaming backpressure.
#
# FIXED-LENGTH GUARANTEE (this is the reason the aiperf path exists — see AIPerfSweepRunner):
#   * Input/output lengths are pinned: --isl-stddev 0 / --osl-stddev 0 (std=0 on both).
#   * Output is forced to exactly OSL, not merely capped, so a server that stops early at EOS
#     cannot truncate it:
#       - all backends: --osl sets the ceiling (max_tokens/max_completion_tokens) and
#         --extra-inputs ignore_eos:true suppresses early EOS (the same mechanism sa-bench
#         uses across sglang/trtllm/dynamo).
#       - vLLM additionally gets a hard floor: --use-legacy-max-tokens (so --osl emits the
#         legacy `max_tokens` field vLLM honors) + --extra-inputs min_tokens:OSL. vLLM was
#         observed truncating/varying OSL without this explicit min/max forcing.
#   * --use-server-token-count measures the server's own completion-token count.
#
# Args (positional, from AIPerfSweepRunner.build_command):
#   1  ENDPOINT           http://localhost:<frontend_port>
#   2  MODEL_NAME         served model name
#   3  TOKENIZER_PATH     HF id or /model
#   4  ISL                input seq len
#   5  OSL                output seq len
#   6  CONCURRENCIES      x-separated list, e.g. "1x8x64x512"
#   7  NUM_PROMPTS_MULT   request-count = concurrency * this
#   8  NUM_WARMUP_MULT    warmup-request-count = concurrency * this
#   9  ENDPOINT_TYPE      aiperf endpoint type (default: chat)
#   10 BACKEND            backend type (vllm|sglang|trtllm|...); drives backend-aware OSL forcing
#   11+ passthrough       extra aiperf flags from benchmark.aiperf_args
set -uo pipefail

ENDPOINT="${1:?endpoint}"
MODEL_NAME="${2:?model}"
TOKENIZER_PATH="${3:?tokenizer}"
ISL="${4:?isl}"
OSL="${5:?osl}"
CONCURRENCIES="${6:?concurrencies}"
NUM_PROMPTS_MULT="${7:-3}"
NUM_WARMUP_MULT="${8:-1}"
ENDPOINT_TYPE="${9:-chat}"
BACKEND="${10:-}"
# Any args beyond the 10 positional ones are passthrough aiperf flags (benchmark.aiperf_args).
EXTRA_AIPERF_ARGS=("${@:11}")

ARTIFACT_BASE="/logs/artifacts"
mkdir -p "$ARTIFACT_BASE"

# Backend-aware output-length forcing. ignore_eos is universal; vLLM also gets an explicit
# min_tokens floor plus the legacy max_tokens field it honors.
OSL_FORCE_ARGS=(--extra-inputs ignore_eos:true)
MAX_TOKENS_ARGS=()
if [ "$BACKEND" = "vllm" ]; then
    OSL_FORCE_ARGS+=(--extra-inputs "min_tokens:${OSL}")
    MAX_TOKENS_ARGS=(--use-legacy-max-tokens)
fi

echo "=============================================="
echo "AIPerf fixed ISL/OSL concurrency-sweep benchmark"
echo "  Endpoint:      $ENDPOINT ($ENDPOINT_TYPE)"
echo "  Model:         $MODEL_NAME"
echo "  Backend:       ${BACKEND:-<unset>}"
echo "  ISL/OSL:       $ISL / $OSL (std=0, output forced to exactly OSL)"
echo "  Concurrencies: $CONCURRENCIES"
echo "=============================================="

# Install aiperf into an isolated venv that inherits system site-packages, so we do NOT
# uninstall distutils-installed packages (e.g. blinker 1.4) inside the serving container,
# which makes a bare `pip install aiperf` fail.
AIPERF_VENV="/tmp/aiperf_venv"
if [ ! -x "$AIPERF_VENV/bin/aiperf" ]; then
    echo "Installing aiperf into $AIPERF_VENV ..."
    python3 -m venv --system-site-packages "$AIPERF_VENV"
    "$AIPERF_VENV/bin/pip" install --no-cache-dir "${AIPERF_PACKAGE:-aiperf}" 2>&1 | tail -3
fi
AIPERF="$AIPERF_VENV/bin/aiperf"
"$AIPERF" --version || { echo "aiperf install failed"; exit 1; }

IFS='x' read -ra CONC_LIST <<< "$CONCURRENCIES"
FAIL=0
for conc in "${CONC_LIST[@]}"; do
    reqcount=$(( conc * NUM_PROMPTS_MULT ))
    warmup=$(( conc * NUM_WARMUP_MULT ))
    # aiperf rejects --warmup-request-count 0 (pydantic gt=0); omit the flag entirely for no warmup
    # (e.g. num_warmup_mult: 0). Passing 0 aborts the whole sweep with a validation error.
    warmup_args=()
    if [ "$warmup" -gt 0 ]; then
        warmup_args=(--warmup-request-count "$warmup")
    fi
    outdir="$ARTIFACT_BASE/conc_${conc}"
    mkdir -p "$outdir"
    echo ""
    echo "$(date '+%Y-%m-%d %H:%M:%S') - aiperf concurrency=$conc requests=$reqcount warmup=$warmup"
    set -x
    "$AIPERF" profile \
        -m "$MODEL_NAME" \
        --tokenizer "$TOKENIZER_PATH" \
        --url "$ENDPOINT" \
        --endpoint-type "$ENDPOINT_TYPE" \
        --ui-type none \
        --streaming \
        --concurrency "$conc" \
        --request-count "$reqcount" \
        "${warmup_args[@]}" \
        --isl "$ISL" --isl-stddev 0 \
        --osl "$OSL" --osl-stddev 0 \
        --use-server-token-count \
        "${MAX_TOKENS_ARGS[@]}" \
        "${OSL_FORCE_ARGS[@]}" \
        "${EXTRA_AIPERF_ARGS[@]}" \
        --artifact-dir "$outdir" || FAIL=1
    set +x
done

echo ""
echo "AIPerf sweep complete (fail=$FAIL). Artifacts under $ARTIFACT_BASE"
exit $FAIL
