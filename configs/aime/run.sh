#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AIME 2025 reasoning eval — runs inside the NeMo Skills container.
#
# Phase 1: ns prepare_data
# Phase 2: ns eval (default \boxed{} extraction; pass@k via REPEAT)
#
# Server endpoint, model, and dataset can be overridden via env. Tuning knobs
# (max_tokens, repeat, etc.) match the upstream reasoning-eval reference.
#
# === Custom answer-extraction regex (NOT applied here) ===
# The SGLang team's reasoning-eval reference suggests broadening the answer
# extractor for reasoning models with these two ns eval overrides:
#
#   ++eval_config.extract_from_boxed=False
#   ++eval_config.extract_regex=(?:\boxed\{|\*\*Answer\*\*[^0-9\-]{0,30}|(?i:final answer)[^0-9\-]{0,30}|(?i:answer)\s*(?:is|=|:)[^0-9\-]{0,30})(-?\d+)
#
# We intentionally don't pass them here. ns eval forwards Hydra ++overrides to
# parallel `python -m nemo_skills.inference.generate` subprocesses through
# nemo-run, which constructs the inner command line UNQUOTED — bash strips
# backslashes from the regex before Python re.compile sees it, the regex
# becomes invalid, and every generate subprocess crashes on import. Verified
# on cluster runs 4836 / 4838 (both produced empty output dirs and a false
# "Benchmark completed successfully").
#
# Default \boxed{} extraction gives usable numbers for reasoning models that
# follow the boxed-answer convention. If you need a broader extractor, do it
# post-hoc against the cached output-rs<seed>.jsonl files (Python re module,
# raw string, no shell layers).

set -euo pipefail

ENDPOINT="${ENDPOINT:-http://localhost:8000/v1}"
MODEL="${MODEL:-dspro}"
DATASET="${DATASET:-aime25}"
REPEAT="${REPEAT:-16}"
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
