#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run InferenceX AgentX against an srt-slurm deployment. This directory is
# mounted at /benchmarks; use benchmark.type: custom with command: bash /benchmarks/agentx.sh.
set -euo pipefail

required=(MODEL MODEL_PREFIX FRAMEWORK PRECISION CONC RESULT_FILENAME DURATION)
missing=()
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        missing+=("$name")
    fi
done
if (( ${#missing[@]} )); then
    echo "AgentX requires benchmark.env values for: ${missing[*]}" >&2
    exit 1
fi

# Keep caller-supplied benchmark.env overrides. These defaults match the
# InferenceX workflow settings, except power capture is opt-in here.
export RESULT_DIR="${RESULT_DIR:-/logs/agentic}"
export AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-$RESULT_DIR}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export IS_MULTINODE="${IS_MULTINODE:-false}"
export PORT="${PORT:-${SRT_FRONTEND_PORT:-8000}}"
export AIPERF_PYTHON_VERSION="${AIPERF_PYTHON_VERSION:-3.11}"
export AIPERF_FAILED_REQUEST_THRESHOLD="${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}"
export AIPERF_LIVE_FAILED_REQUEST_THRESHOLD="${AIPERF_LIVE_FAILED_REQUEST_THRESHOLD:-0.10}"
export AIPERF_TRACE_IDLE_GAP_CAP_SECONDS="${AIPERF_TRACE_IDLE_GAP_CAP_SECONDS:-300}"
export AIPERF_WARMUP_REQUESTS_PER_LANE="${AIPERF_WARMUP_REQUESTS_PER_LANE:-10}"
export AGENTIC_WARMUP_GRACE_PERIOD="${AGENTIC_WARMUP_GRACE_PERIOD:-1800}"
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES:-0}"
export AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS="${AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS:-3600}"
export AIPERF_EXPERIMENTAL_FAST="${AIPERF_EXPERIMENTAL_FAST:-0}"
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID="${AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID:-false}"
export AIPERF_UNSAFE_OVERRIDE="${AIPERF_UNSAFE_OVERRIDE:-false}"
export AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING="${AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING:-1}"
export ENABLE_AGENTX_POWER="${ENABLE_AGENTX_POWER:-0}"
export REQUIRE_POWER="${REQUIRE_POWER:-0}"
export AIPERF_DRAIN_TIMEOUT_SECONDS="${AIPERF_DRAIN_TIMEOUT_SECONDS:-1800}"
export AIPERF_DRAIN_POLL_SECONDS="${AIPERF_DRAIN_POLL_SECONDS:-10}"

mkdir -p "$RESULT_DIR" "$AGENTIC_OUTPUT_DIR"
checkout_root=$(mktemp -d "${TMPDIR:-/tmp}/srt-agentx-XXXXXX")
trap 'rm -rf "$checkout_root"' EXIT
export AIPERF_RUNTIME_DIR="${AIPERF_RUNTIME_DIR:-$checkout_root/runtime}"

repo_url="${AGENTX_INFERENCEX_REPO_URL:-https://github.com/SemiAnalysisAI/InferenceX.git}"
git clone --depth 1 --filter=blob:none "$repo_url" "$checkout_root/InferenceX"
if [[ -n "${AGENTX_INFERENCEX_REF:-}" ]]; then
    git -C "$checkout_root/InferenceX" fetch --depth 1 origin "$AGENTX_INFERENCEX_REF"
    git -C "$checkout_root/InferenceX" checkout --detach FETCH_HEAD
fi
git -C "$checkout_root/InferenceX" submodule update --init --depth 1 inferencex-e2e/utils/aiperf

export INFMAX_CONTAINER_WORKSPACE="$checkout_root/InferenceX/inferencex-e2e"
entrypoint="$INFMAX_CONTAINER_WORKSPACE/benchmarks/srt_agentic.sh"
if [[ ! -f "$entrypoint" || ! -f "$INFMAX_CONTAINER_WORKSPACE/utils/aiperf/pyproject.toml" ]]; then
    echo "AgentX entrypoint or AIPerf submodule missing in InferenceX checkout" >&2
    exit 1
fi
echo "Running InferenceX AgentX at $(git -C "$checkout_root/InferenceX" rev-parse HEAD)"
echo "Using AIPerf at $(git -C "$INFMAX_CONTAINER_WORKSPACE/utils/aiperf" rev-parse HEAD)"
bash "$entrypoint"
