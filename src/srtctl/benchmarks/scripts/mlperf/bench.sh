#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MLPerf Inference (LoadGen) benchmark.
# srt-slurm owns server startup; this script prepares an isolated LoadGen
# runtime under /tmp and points the mounted mlcommons/inference harness at the
# ready frontend.
#
# Two things this preflight is deliberately careful about:
#   * The harness is installed into a job-scoped venv created with
#     --system-site-packages, never into the container's site-packages. The
#     reference requirements pin datasets/scikit-learn/numba versions that
#     would otherwise be dragged over the serving stack the workers are
#     running on, and the checkout can be mounted read-only.
#   * mlperf_loadgen is compiled from the pinned checkout. LoadGen is the
#     component that decides VALID vs INVALID, so it must come from the same
#     commit the results are reported against — never from a stray wheel.
#
# The dataset is staged from shared storage to node-local /tmp before
# measurement: LoadGen loads the whole QSL up front, and paying Lustre for it
# mid-run is a perf artifact of its own.

set -euo pipefail

ENDPOINT=$1
HARNESS_DIR=$2
BENCHMARK=$3
SCENARIO=$4
MODE=$5
BACKEND=$6
DATASET=$7
USER_CONF=${8:-}
MAX_CONCURRENCY=${9:-}
TOKENIZER=${10:-}

# When the client runs on a different node than the frontend, localhost is
# wrong; benchmark_stage injects the frontend's real host/port.
if [[ -n "${SRT_FRONTEND_HOST:-}" ]]; then
  PORT_FROM_ENDPOINT=$(echo "$ENDPOINT" | sed -E 's|.*:([0-9]+).*|\1|')
  ENDPOINT="http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT:-$PORT_FROM_ENDPOINT}"
fi

BENCH_DIR="$HARNESS_DIR/language/$BENCHMARK"
[[ -d "$HARNESS_DIR" ]] || { echo "ERROR: mlperf_harness_dir $HARNESS_DIR not found in container (mount via extra_mount)" >&2; exit 1; }
[[ -d "$BENCH_DIR" ]] || { echo "ERROR: mlperf_benchmark '$BENCHMARK' not found at $BENCH_DIR" >&2; exit 1; }
[[ -f "$BENCH_DIR/run_mlperf.py" ]] || { echo "ERROR: $BENCH_DIR has no run_mlperf.py — this benchmark does not use the server-url harness shape" >&2; exit 1; }
[[ -d "$HARNESS_DIR/loadgen" ]] || { echo "ERROR: $HARNESS_DIR/loadgen missing — mlperf_harness_dir must be an mlcommons/inference checkout" >&2; exit 1; }
[[ -f "$DATASET" ]] || { echo "ERROR: mlperf_dataset $DATASET not found in container" >&2; exit 1; }
if [[ -n "$USER_CONF" && ! -f "$USER_CONF" ]]; then
  echo "ERROR: mlperf_user_conf $USER_CONF not found in container" >&2
  exit 1
fi

RUNTIME="${MLPERF_RUNTIME:-/tmp/mlperf-${SLURM_JOB_ID:-$$}}"
export HOME="$RUNTIME/home"
export PIP_CACHE_DIR="$RUNTIME/pip-cache"
export VENV="$RUNTIME/venv"
export HF_HOME="$RUNTIME/hf"
mkdir -p "$HOME" "$PIP_CACHE_DIR" "$RUNTIME/data" "$HF_HOME"

# LoadGen's server scenario holds one in-flight request per scheduled query;
# lift the soft fd limit to the hard limit so high-QPS runs don't starve.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# ---- preflight: build the LoadGen runtime once per job ---------------------
# READY is self-validating: it records the arch, harness commit and benchmark
# it was built for, so a relocated MLPERF_RUNTIME reused across jobs (or a
# different pin) rebuilds instead of silently reporting numbers from the wrong
# LoadGen.
FINGERPRINT="$(uname -m):$(git -C "$HARNESS_DIR" rev-parse HEAD 2>/dev/null || echo unknown):$BENCHMARK"
ready() { [[ "$(cat "$RUNTIME/READY" 2>/dev/null)" == "$FINGERPRINT" ]]; }
HOLDING_LOCK=0
if ! ready; then
  # Atomic mkdir as a mutex: a concurrent job sharing this runtime waits for
  # the builder instead of double-building into the same venv.
  if mkdir "$RUNTIME/.build-lock" 2>/dev/null; then
    HOLDING_LOCK=1
  else
    echo "[mlperf] preflight in progress elsewhere; waiting for READY"
    for _ in $(seq 1 360); do
      ready && break
      sleep 5
    done
    ready || { echo "ERROR: timed out waiting for concurrent mlperf preflight in $RUNTIME" >&2; exit 1; }
  fi
fi
if ! ready; then
  echo "[mlperf] preflight: building LoadGen runtime in $RUNTIME (fingerprint $FINGERPRINT)"

  # --system-site-packages so the container's torch/transformers are reused;
  # a from-scratch install of those would dwarf the benchmark itself.
  [[ -x "$VENV/bin/python" ]] || python3 -m venv --system-site-packages "$VENV"
  "$VENV/bin/python" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel

  if [[ -f "$BENCH_DIR/requirements.txt" ]]; then
    "$VENV/bin/python" -m pip install --disable-pip-version-check -r "$BENCH_DIR/requirements.txt"
  fi
  # Built from the checkout, not PyPI: the wheel on PyPI tracks its own release
  # cadence and would decouple the pass/fail verdict from the pinned harness.
  "$VENV/bin/python" -m pip install --disable-pip-version-check "$HARNESS_DIR/loadgen"
  "$VENV/bin/python" -c 'import mlperf_loadgen; print("mlperf_loadgen", getattr(mlperf_loadgen, "__version__", "unknown"))'

  # Stage the dataset from shared storage to node-local /tmp.
  STAGED="$RUNTIME/data/$(basename "$DATASET")"
  cp "$DATASET" "$STAGED"
  [[ -s "$STAGED" ]] || { echo "ERROR: empty staged MLPerf dataset: $STAGED" >&2; exit 1; }

  echo "$FINGERPRINT" > "$RUNTIME/READY"
  echo "[mlperf] preflight complete"
fi
[[ "$HOLDING_LOCK" == 1 ]] && rmdir "$RUNTIME/.build-lock" 2>/dev/null || true

STAGED="$RUNTIME/data/$(basename "$DATASET")"
[[ -s "$STAGED" ]] || { echo "ERROR: staged dataset $STAGED missing after preflight" >&2; exit 1; }

# ---- run --------------------------------------------------------------------
RESULTS_DIR=/logs/mlperf
mkdir -p "$RESULTS_DIR"

cd "$BENCH_DIR"

# Fail with the reason rather than an argparse dump: only the server-url shape
# of the reference harness can measure a server srt-slurm already started.
if ! "$VENV/bin/python" run_mlperf.py --help 2>&1 | grep -q -- "--server-url"; then
  echo "ERROR: $BENCH_DIR/run_mlperf.py does not accept --server-url — this benchmark builds its own engine or launches its own server, which srt-slurm already owns" >&2
  exit 1
fi

COMMON_ARGS=(
  --scenario "$SCENARIO"
  --backend "$BACKEND"
  --server-url "$ENDPOINT"
  --input-file "$STAGED"
  --output-dir "$RESULTS_DIR"
)
# The harness defaults --mlperf-conf to a relative "inference/mlperf.conf",
# which never resolves from language/<benchmark>/. It then only *warns* and
# runs on without the official per-benchmark constraints, so point it at the
# checkout's real copy. MLPERF_EXTRA_ARGS is appended last and can override it.
[[ -f "$HARNESS_DIR/mlperf.conf" ]] && COMMON_ARGS+=(--mlperf-conf "$HARNESS_DIR/mlperf.conf")
[[ -n "$USER_CONF" ]] && COMMON_ARGS+=(--user-conf "$USER_CONF")
[[ -n "$MAX_CONCURRENCY" ]] && COMMON_ARGS+=(--max-concurrency "$MAX_CONCURRENCY")
# Simple space-separated tokens only — values are word-split, never shell-parsed.
read -r -a EXTRA_ARGS <<< "${MLPERF_EXTRA_ARGS:-}" || true

run_loadgen() {
  local mode=$1
  shift
  echo "[mlperf] $mode run: endpoint=$ENDPOINT benchmark=$BENCHMARK scenario=$SCENARIO backend=$BACKEND dataset=$STAGED"
  "$VENV/bin/python" run_mlperf.py "${COMMON_ARGS[@]}" "$@" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}

if [[ "$MODE" == "performance" || "$MODE" == "both" ]]; then
  run_loadgen performance
fi

if [[ "$MODE" == "accuracy" || "$MODE" == "both" ]]; then
  run_loadgen accuracy --accuracy

  # Scoring is opt-in: it detokenizes every response and, for gpt-oss, runs the
  # LiveCodeBench evaluators — minutes to hours of CPU on the benchmark node,
  # which is not something to spend by default while the cluster allocation is
  # still held.
  if [[ "${MLPERF_EVAL_ACCURACY:-0}" == "1" ]]; then
    ACC_DIR="$RESULTS_DIR/$SCENARIO/accuracy"
    ACC_LOG="$ACC_DIR/mlperf_log_accuracy.json"
    if [[ -f "$BENCH_DIR/eval_mlperf_accuracy.py" && -f "$ACC_LOG" ]]; then
      echo "[mlperf] scoring accuracy log $ACC_LOG"
      EVAL_ARGS=(--mlperf-log "$ACC_LOG" --reference-data "$STAGED" --output-file "$ACC_DIR/accuracy.json")
      [[ -n "$TOKENIZER" ]] && EVAL_ARGS+=(--tokenizer "${MLPERF_ACCURACY_TOKENIZER:-$TOKENIZER}")
      "$VENV/bin/python" eval_mlperf_accuracy.py "${EVAL_ARGS[@]}"
    else
      echo "[mlperf] MLPERF_EVAL_ACCURACY=1 but no eval_mlperf_accuracy.py or accuracy log; skipping scoring" >&2
    fi
  fi
fi
