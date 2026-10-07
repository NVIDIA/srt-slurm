#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# HLE (Humanity's Last Exam) eval — runs inside the NeMo Skills container.
#
# Phase 1: ns prepare_data hle (gated cais/hle dataset; needs HF_TOKEN)
# Phase 2: ns eval (HLE prompt; answers graded by NeMo Skills' HLE LLM judge)
#
# Many HLE answers are free-form, so there is no regex scorer: the judge compares
# each answer with the reference. NeMo Skills defaults the judge to
# o3-mini-2025-01-31 on api.openai.com (the official leaderboard's judge), which
# needs OPENAI_API_KEY. JUDGE_MODEL / JUDGE_SERVER_ADDRESS / JUDGE_SERVER_TYPE
# point it at another OpenAI-compatible server.
#
# Server endpoint, model, split and tuning knobs can be overridden via env.
# Sampling knobs match configs/aime/run.sh; REPEAT defaults to 1 because every
# repeat is another judge call per question.

set -euo pipefail

ENDPOINT="${ENDPOINT:-http://localhost:8000/v1}"
MODEL="${MODEL:?MODEL must be set to the served model name}"
SPLIT="${SPLIT:-text}"
NUM_EXAMPLES="${NUM_EXAMPLES:-}"
REPEAT="${REPEAT:-1}"
MAX_TOKENS="${MAX_TOKENS:-400000}"
NUM_THREADS="${NUM_THREADS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
# Keep 0: NeMo Skills' judge step reads output-rs0.jsonl, so a non-zero starting seed makes it
# fail with FileNotFoundError after generation has already finished (NeMo Skills 26.03).
SEED="${SEED:-0}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_SERVER_ADDRESS="${JUDGE_SERVER_ADDRESS:-}"
JUDGE_SERVER_TYPE="${JUDGE_SERVER_TYPE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/logs/accuracy/hle}"

# The default judge is OpenAI; fail before generation rather than after hours of it.
if [ -z "$JUDGE_SERVER_ADDRESS" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is required for the default HLE judge (or set JUDGE_SERVER_ADDRESS)" >&2
  exit 1
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

echo "=== Config ==="
echo "  endpoint:     $ENDPOINT"
echo "  model:        $MODEL"
echo "  split:        $SPLIT"
echo "  num_examples: ${NUM_EXAMPLES:-all}"
echo "  repeat:       $REPEAT"
echo "  max_tokens:   $MAX_TOKENS"
echo "  num_threads:  $NUM_THREADS"
echo "  temperature:  $TEMPERATURE"
echo "  top_p:        $TOP_P"
echo "  seed:         $SEED"
echo "  judge:        ${JUDGE_MODEL:-default} @ ${JUDGE_SERVER_ADDRESS:-default}"
echo "  output_dir:   $OUTPUT_DIR"
echo

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
[ -n "$NUM_EXAMPLES" ] && EXTRA_ARGS+=("++max_samples=${NUM_EXAMPLES}")
[ -n "$JUDGE_MODEL" ] && EXTRA_ARGS+=("--judge_model=${JUDGE_MODEL}")
[ -n "$JUDGE_SERVER_ADDRESS" ] && EXTRA_ARGS+=("--judge_server_address=${JUDGE_SERVER_ADDRESS}")
[ -n "$JUDGE_SERVER_TYPE" ] && EXTRA_ARGS+=("--judge_server_type=${JUDGE_SERVER_TYPE}")

echo "=== Phase 1: prepare_data ==="
ns prepare_data hle

echo
echo "=== Phase 2: ns eval ==="
ns eval \
  --server_type=openai \
  --model="$MODEL" \
  --server_address="$ENDPOINT" \
  --benchmarks="hle:${REPEAT}" \
  --split="$SPLIT" \
  --output_dir="$OUTPUT_DIR" \
  --starting_seed="$SEED" \
  "++inference.tokens_to_generate=${MAX_TOKENS}" \
  "++max_concurrent_requests=${NUM_THREADS}" \
  "++inference.temperature=${TEMPERATURE}" \
  "++inference.top_p=${TOP_P}" \
  "++inference.timeout=25000000" \
  "${EXTRA_ARGS[@]}"

METRICS="${OUTPUT_DIR}/eval-results/hle/metrics.json"

# `ns` can exit 0 after a failed prepare_data or judge step, so check for the result itself.
if [ ! -s "$METRICS" ]; then
  echo "ERROR: $METRICS was not written; see the Phase 1/Phase 2 output above" >&2
  echo "  (401/403 in prepare_data: HF_TOKEN missing or cais/hle terms not accepted;" >&2
  echo "   429/401 from the judge: OPENAI_API_KEY has no credit or is wrong)." >&2
  exit 1
fi

echo
echo "=== Done ==="
echo "Metrics: $METRICS"
