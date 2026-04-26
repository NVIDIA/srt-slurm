#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AIME accuracy evaluation using NVIDIA NeMo Skills.
# Expects: endpoint model_name [aime_dataset] [num_examples] [max_tokens] [num_threads] [repeat] [temperature] [top_p] [top_k]

set -euo pipefail

ENDPOINT=${1:?endpoint is required}
MODEL_NAME=${2:-model}
AIME_DATASET=${3:-aime25}
NUM_EXAMPLES=${4:-}
MAX_TOKENS=${5:-24576}
NUM_THREADS=${6:-30}
REPEAT=${7:-1}
TEMPERATURE=${8:-}
TOP_P=${9:-}
TOP_K=${10:-}

case "$AIME_DATASET" in
    aime24|aime25|aime26) ;;
    *)
        echo "Unsupported AIME dataset: $AIME_DATASET. Expected one of: aime24, aime25, aime26" >&2
        exit 2
        ;;
esac

if ! [[ "$REPEAT" =~ ^[0-9]+$ ]] || [ "$REPEAT" -le 0 ]; then
    echo "repeat must be a positive integer, got: $REPEAT" >&2
    exit 2
fi

RESULT_ROOT="/logs/accuracy"
RESULT_DIR="${RESULT_ROOT}/${AIME_DATASET}"
METRICS_FILE="${RESULT_ROOT}/${AIME_DATASET}_metrics.json"
mkdir -p "$RESULT_DIR"
rm -f "$RESULT_DIR"/output.jsonl "$RESULT_DIR"/output-rs*.jsonl "$RESULT_DIR"/output*.done "$METRICS_FILE"

BASE_URL="${ENDPOINT%/}"
if [[ "$BASE_URL" != */v1 ]]; then
    BASE_URL="${BASE_URL}/v1"
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

echo "AIME Config: endpoint=${ENDPOINT}; base_url=${BASE_URL}; model=${MODEL_NAME}; dataset=${AIME_DATASET}; num_examples=${NUM_EXAMPLES:-all}; max_tokens=${MAX_TOKENS}; num_threads=${NUM_THREADS}; repeat=${REPEAT}; temperature=${TEMPERATURE:-auto}; top_p=${TOP_P:-default}; top_k=${TOP_K:-default}"

NEMO_SKILLS_VENV="${NEMO_SKILLS_VENV:-/tmp/nemo-skills-venv}"
NEMO_SKILLS_PACKAGE="${NEMO_SKILLS_PACKAGE:-git+https://github.com/NVIDIA-NeMo/Skills.git}"
PYTHON_BIN="python3"

has_nemo_skills() {
    "$1" - "$AIME_DATASET" <<'PY' >/dev/null 2>&1
import sys
from pathlib import Path

import hydra  # noqa: F401
import litellm  # noqa: F401
from nemo_skills.dataset.utils import get_dataset_path
from nemo_skills.evaluation.metrics import ComputeMetrics  # noqa: F401

dataset = sys.argv[1]
dataset_path = get_dataset_path(dataset)
assert (Path(dataset_path) / "prepare.py").exists()
PY
}

if [ "${NEMO_SKILLS_FORCE_INSTALL:-0}" = "1" ] || ! has_nemo_skills "$PYTHON_BIN"; then
    echo "Installing NeMo Skills into ${NEMO_SKILLS_VENV} ..."
    if [ ! -d "$NEMO_SKILLS_VENV" ]; then
        python3 -m venv --system-site-packages "$NEMO_SKILLS_VENV"
    fi
    PYTHON_BIN="${NEMO_SKILLS_VENV}/bin/python"
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install "$NEMO_SKILLS_PACKAGE"
fi

if ! has_nemo_skills "$PYTHON_BIN"; then
    echo "NeMo Skills is not available after installation" >&2
    exit 1
fi

echo "Preparing NeMo Skills dataset: ${AIME_DATASET}"
"$PYTHON_BIN" -m nemo_skills.dataset.prepare "$AIME_DATASET" --parallelism 1 --retries 0

DATASET_FILE=$("$PYTHON_BIN" - "$AIME_DATASET" <<'PY'
import sys
from pathlib import Path
from nemo_skills.dataset.utils import get_dataset_path

dataset = sys.argv[1]
print(Path(get_dataset_path(dataset)) / "test.jsonl")
PY
)

COMMON_ARGS=(
    "++input_file=${DATASET_FILE}"
    "++server.server_type=openai"
    "++server.base_url=${BASE_URL}"
    "++server.model=${MODEL_NAME}"
    "++prompt_config=generic/math"
    "++eval_type=math"
    "++eval_config.split=test"
    "++max_concurrent_requests=${NUM_THREADS}"
    "++inference.tokens_to_generate=${MAX_TOKENS}"
)

if [ -n "$NUM_EXAMPLES" ]; then
    COMMON_ARGS+=("++max_samples=${NUM_EXAMPLES}")
fi
if [ -n "$TOP_P" ]; then
    COMMON_ARGS+=("++inference.top_p=${TOP_P}")
fi
if [ -n "$TOP_K" ]; then
    COMMON_ARGS+=("++inference.top_k=${TOP_K}")
fi

run_generation() {
    local output_file=$1
    local seed=$2
    local temperature=$3

    local args=("${COMMON_ARGS[@]}" "++output_file=${output_file}" "++inference.temperature=${temperature}")
    if [ -n "$seed" ]; then
        args+=("++inference.random_seed=${seed}")
    fi

    "$PYTHON_BIN" -m nemo_skills.inference.generate "${args[@]}"
}

if [ "$REPEAT" -eq 1 ]; then
    run_generation "${RESULT_DIR}/output.jsonl" "" "${TEMPERATURE:-0.0}"
else
    sample_temperature="${TEMPERATURE:-0.7}"
    for ((seed = 0; seed < REPEAT; seed++)); do
        run_generation "${RESULT_DIR}/output-rs${seed}.jsonl" "$seed" "$sample_temperature"
    done
fi

echo "Computing NeMo Skills metrics..."
"$PYTHON_BIN" - "$AIME_DATASET" "$RESULT_DIR" "$METRICS_FILE" <<'PY'
import glob
import json
import sys
from pathlib import Path

from nemo_skills.evaluation.metrics import ComputeMetrics

benchmark = sys.argv[1]
result_dir = Path(sys.argv[2])
metrics_file = Path(sys.argv[3])

input_files = sorted(glob.glob(str(result_dir / "output-rs*.jsonl")))
if not input_files:
    input_files = [str(result_dir / "output.jsonl")]

metrics = ComputeMetrics(benchmark=benchmark).compute_metrics(input_files=input_files)
summary = {benchmark: metrics.get("_all_", metrics)}
metrics_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
(result_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
PY

echo "AIME evaluation complete. Results in ${RESULT_DIR}; metrics in ${METRICS_FILE}"
