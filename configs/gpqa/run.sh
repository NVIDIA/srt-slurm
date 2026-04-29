#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GPQA reasoning eval — runs inside the NeMo Skills container.
#
# Phase 1: ns prepare_data gpqa  (downloads GPQA Diamond from HF; gated dataset
#          → set HF_TOKEN in benchmark.env)
# Phase 2: ns eval --benchmarks=gpqa:$REPEAT  (NeMo Skills' default multi-choice
#          extractor; pass@k via REPEAT)
#
# Server endpoint, model, and dataset can be overridden via env. Tuning knobs
# (max_tokens, repeat, etc.) match the upstream reasoning-eval reference
# (--benchmarks=gpqa:32, temperature=1.0, max_tokens=400000).
#
# Same nemo-run unquoting hazard as AIME applies — do not pass Hydra ++overrides
# with backslash-bearing values (e.g. custom extract_regex). Post-process the
# cached output-rs<seed>.jsonl files in Python if you need a broader extractor.

set -euo pipefail

ENDPOINT="${ENDPOINT:-http://localhost:8000/v1}"
MODEL="${MODEL:-dspro}"
DATASET="${DATASET:-gpqa}"
REPEAT="${REPEAT:-32}"
MAX_TOKENS="${MAX_TOKENS:-400000}"
NUM_THREADS="${NUM_THREADS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-/logs/accuracy/${DATASET}}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

echo "=== Config ==="
echo "  endpoint:    $ENDPOINT"
echo "  model:       $MODEL"
echo "  dataset:     $DATASET"
echo "  repeat:      $REPEAT"
echo "  max_tokens:  $MAX_TOKENS"
echo "  num_threads: $NUM_THREADS"
echo "  temperature: $TEMPERATURE"
echo "  top_p:       $TOP_P"
echo "  seed:        $SEED"
echo "  output_dir:  $OUTPUT_DIR"
echo

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set. GPQA Diamond is HF-gated; ns prepare_data"
  echo "         will fail unless the token is plumbed through benchmark.env."
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Phase 1: prepare_data ==="
ns prepare_data "$DATASET"

echo
echo "=== Phase 2: ns eval ==="
ns eval \
  --server_type=openai \
  --model="$MODEL" \
  --server_address="$ENDPOINT" \
  --benchmarks="${DATASET}:${REPEAT}" \
  --output_dir="$OUTPUT_DIR" \
  --starting_seed="$SEED" \
  "++inference.tokens_to_generate=${MAX_TOKENS}" \
  "++max_concurrent_requests=${NUM_THREADS}" \
  "++inference.temperature=${TEMPERATURE}" \
  "++inference.top_p=${TOP_P}" \
  "++inference.timeout=25000000"

echo
echo "=== Done ==="
echo "Metrics: ${OUTPUT_DIR}/eval-results/${DATASET}/metrics.json"
