#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENDPOINT="$1"
ISL="$2"
OSL="$3"
CONCURRENCY="$4"
SEED_OSL="$5"
TOKENIZER="$6"
MODEL="$7"
DP_SIZE="$8"
TOTAL_GPUS="$9"
SETTLE_SECONDS="${10}"

# Keep the client interpreter contract aligned with SA-Bench.  The serving
# containers are only required to provide python3; some intentionally omit the
# legacy `python` alias.
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Error: prefix-replay requires python3 (PYTHON_BIN=${PYTHON_BIN})" >&2
    exit 127
fi

# Import benchmark_serving.py's complete dependency set before spending time
# constructing the 256K-token prompts.  Match SA-Bench's fallback behavior if
# a container is missing any client-side package.
SA_BENCH_VENV="${SA_BENCH_VENV:-/tmp/sa-bench-venv}"
SA_BENCH_DEPS=(aiohttp numpy pandas datasets Pillow tqdm transformers huggingface_hub)
if ! "${PYTHON_BIN}" -c \
    "import aiohttp, numpy, pandas, datasets, PIL, tqdm, transformers, huggingface_hub" \
    2>/dev/null; then
    echo "Missing prefix-replay dependencies; installing into ${SA_BENCH_VENV} ..."
    if [[ ! -d "${SA_BENCH_VENV}" ]]; then
        "${PYTHON_BIN}" -m venv --system-site-packages "${SA_BENCH_VENV}"
    fi
    PYTHON_BIN="${SA_BENCH_VENV}/bin/python3"
    "${PYTHON_BIN}" -m pip install "${SA_BENCH_DEPS[@]}"
fi

exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/prefix_replay.py" \
    --endpoint "${ENDPOINT}" \
    --isl "${ISL}" \
    --osl "${OSL}" \
    --concurrency "${CONCURRENCY}" \
    --seed-osl "${SEED_OSL}" \
    --tokenizer "${TOKENIZER}" \
    --model "${MODEL}" \
    --dp-size "${DP_SIZE}" \
    --total-gpus "${TOTAL_GPUS}" \
    --settle-seconds "${SETTLE_SECONDS}"
