#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AgentX MVP benchmark using AIPerf's inferencex-agentx-mvp scenario.
#
# Usage:
#   bench.sh ENDPOINT MODEL_NAME CONCURRENCY DURATION MAX_CONTEXT_LENGTH \
#            TOKENIZER_PATH PUBLIC_DATASET [NUM_DATASET_ENTRIES|auto] \
#            FAILED_REQUEST_THRESHOLD [EXTRA_AIPERF_ARGS...]

set -euo pipefail

ENDPOINT=$1
MODEL_NAME=$2
CONCURRENCY=$3
DURATION=${4:-1800}
MAX_CONTEXT_LENGTH=${5:-0}
TOKENIZER_PATH=${6:-/model}
PUBLIC_DATASET=${7:-semianalysis_cc_traces_weka_with_subagents}
NUM_DATASET_ENTRIES=${8:-auto}
FAILED_REQUEST_THRESHOLD=${9:-0.10}
shift 9 2>/dev/null || true
EXTRA_ARGS=("$@")

BASE_DIR="${BASE_DIR:-/logs}"
RESULT_DIR="${RESULT_DIR:-${BASE_DIR}/agentx}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${RESULT_DIR}/aiperf_artifacts}"
AIPERF_SPEC="${AIPERF_PACKAGE:-aiperf}"
AIPERF_VENV="${AIPERF_VENV:-/tmp/aiperf-agentx-${SLURM_JOB_ID:-$$}}"

mkdir -p "${RESULT_DIR}" "${ARTIFACT_DIR}"
export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES:-0}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AIPERF_DATASET_CONFIGURATION_TIMEOUT:-1800}"
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT:-1800}"

ulimit -n 600000 2>/dev/null || ulimit -n 65536 2>/dev/null || true

ensure_git() {
    if command -v git >/dev/null 2>&1; then
        return 0
    fi
    if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
        apt-get update && apt-get install -y git
        return 0
    fi
    echo "ERROR: git is required to install aiperf/transformers dependencies." >&2
    echo "Install git in the container image or run the benchmark step with container-remap-root." >&2
    exit 1
}

ensure_python_packaging() {
    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: python3 is required to run aiperf." >&2
        exit 1
    fi

    if ! python3 - <<'PY' >/dev/null 2>&1
import venv
PY
    then
        if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
            apt-get update && apt-get install -y python3-venv
        else
            echo "ERROR: python3 venv module is required to create ${AIPERF_VENV}." >&2
            exit 1
        fi
    fi

    if ! python3 -m pip --version >/dev/null 2>&1; then
        if python3 -m ensurepip --upgrade >/dev/null 2>&1; then
            return 0
        fi
        if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
            apt-get update && apt-get install -y python3-pip
        else
            echo "ERROR: python3 pip is required to install aiperf." >&2
            exit 1
        fi
    fi
}

ensure_aiperf() {
    ensure_git
    echo "Setting up AIPerf environment: ${AIPERF_SPEC}"
    if command -v uv >/dev/null 2>&1; then
        uv venv --system-site-packages "${AIPERF_VENV}"
        uv pip install -p "${AIPERF_VENV}" "${AIPERF_SPEC}"
        uv pip install -p "${AIPERF_VENV}" --upgrade "datasets>=4.7.0"
    else
        ensure_python_packaging
        python3 -m venv --system-site-packages "${AIPERF_VENV}"
        "${AIPERF_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
        "${AIPERF_VENV}/bin/python" -m pip install "${AIPERF_SPEC}"
        "${AIPERF_VENV}/bin/python" -m pip install --upgrade "datasets>=4.7.0"
    fi
    export PATH="${AIPERF_VENV}/bin:${PATH}"
    echo "aiperf $(aiperf --version 2>/dev/null || echo installed) in ${AIPERF_VENV}"
}

SERVER_METRICS_ARGS=()
if [ -n "${AIPERF_SERVER_METRICS_URLS:-}" ]; then
    IFS=',' read -r -a server_metrics_urls <<< "${AIPERF_SERVER_METRICS_URLS}"
    if [ ${#server_metrics_urls[@]} -gt 0 ]; then
        SERVER_METRICS_ARGS+=(--server-metrics "${server_metrics_urls[@]}")
        SERVER_METRICS_ARGS+=(--server-metrics-formats json jsonl)
    fi
fi

echo "=============================================="
echo "AgentX MVP Benchmark (AIPerf)"
echo "=============================================="
echo "Endpoint: ${ENDPOINT}"
echo "Model: ${MODEL_NAME}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Concurrency: ${CONCURRENCY}"
echo "Duration: ${DURATION}s"
echo "Public dataset: ${PUBLIC_DATASET}"
echo "Dataset entries: ${NUM_DATASET_ENTRIES}"
echo "Max context length: ${MAX_CONTEXT_LENGTH}"
echo "Failed request threshold: ${FAILED_REQUEST_THRESHOLD}"
echo "AIPerf package: ${AIPERF_SPEC}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra AIPerf args: ${EXTRA_ARGS[*]}"
fi
echo "=============================================="

ensure_aiperf

cmd=(
    aiperf profile
    --scenario inferencex-agentx-mvp
    --url "${ENDPOINT}"
    --endpoint /v1/chat/completions
    --endpoint-type chat
    --streaming
    --model "${MODEL_NAME}"
    --tokenizer "${TOKENIZER_PATH}"
    --tokenizer-trust-remote-code
    --concurrency "${CONCURRENCY}"
    --benchmark-duration "${DURATION}"
    --random-seed "${AIPERF_RANDOM_SEED:-42}"
    --failed-request-threshold "${FAILED_REQUEST_THRESHOLD}"
    --use-server-token-count
    --use-dynamo-conv-aware-routing
    --dynamo-session-timeout-seconds "${AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS:-3600}"
    --slice-duration 1.0
    --output-artifact-dir "${ARTIFACT_DIR}"
    --public-dataset "${PUBLIC_DATASET}"
    "${SERVER_METRICS_ARGS[@]}"
)

if [ -n "${NUM_DATASET_ENTRIES}" ] && [ "${NUM_DATASET_ENTRIES}" != "0" ] && [ "${NUM_DATASET_ENTRIES}" != "auto" ]; then
    cmd+=(--num-dataset-entries "${NUM_DATASET_ENTRIES}")
fi

if [ "${MAX_CONTEXT_LENGTH}" != "0" ] && [ -n "${MAX_CONTEXT_LENGTH}" ]; then
    cmd+=(--max-context-length "${MAX_CONTEXT_LENGTH}")
fi

cmd+=(--unsafe-override)

cmd+=("${EXTRA_ARGS[@]}")

printf "%q " "${cmd[@]}" > "${RESULT_DIR}/benchmark_command.txt"
printf "\n" >> "${RESULT_DIR}/benchmark_command.txt"

set +e
"${cmd[@]}" 2>&1 | tee "${RESULT_DIR}/benchmark.log"
replay_rc=${PIPESTATUS[0]}
set -e

set +e
python3 - "${ARTIFACT_DIR}" "${FAILED_REQUEST_THRESHOLD}" <<'PY'
import json
import math
import sys
from pathlib import Path


def resolve_aggregate_path(artifact_dir: Path) -> Path:
    direct = artifact_dir / "profile_export_aiperf.json"
    if direct.is_file():
        return direct
    aggregate = artifact_dir / "aggregate" / "profile_export_aiperf_aggregate.json"
    if aggregate.is_file():
        return aggregate
    for child in sorted(artifact_dir.iterdir()) if artifact_dir.is_dir() else []:
        candidate = child / "profile_export_aiperf.json"
        if child.is_dir() and candidate.is_file():
            return candidate
    return direct


def metric_avg(aggregate: dict, name: str) -> float | None:
    metric = aggregate.get(name)
    if not isinstance(metric, dict):
        return None
    value = metric.get("avg")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        return None
    return value


artifact_dir = Path(sys.argv[1])
threshold = float(sys.argv[2])
path = resolve_aggregate_path(artifact_dir)
if not path.is_file():
    print(f"ERROR: {path} not found", file=sys.stderr)
    sys.exit(1)

with path.open() as f:
    aggregate = json.load(f)

successes = metric_avg(aggregate, "request_count")
errors = metric_avg(aggregate, "error_request_count") or 0.0
completed = metric_avg(aggregate, "completed_request_count")
if successes is None:
    print("ERROR: request_count.avg is missing", file=sys.stderr)
    sys.exit(1)
if completed is None:
    completed = successes + errors
if completed <= 0:
    print("ERROR: aiperf completed zero requests", file=sys.stderr)
    sys.exit(1)

error_rate = errors / completed
print(
    "Validated aiperf request error rate: "
    f"{errors:g}/{completed:g} = {error_rate:.3%} <= {threshold:.3%}"
)
if error_rate > threshold:
    sys.exit(1)
PY
validation_rc=$?
set -e

if [ "${replay_rc}" -ne 0 ]; then
    echo "ERROR: AgentX replay exited with code ${replay_rc}" >&2
    exit "${replay_rc}"
fi

if [ "${validation_rc}" -ne 0 ]; then
    echo "ERROR: AgentX replay produced invalid results" >&2
    exit "${validation_rc}"
fi

echo "AgentX benchmark complete. Results saved to: ${RESULT_DIR}"
