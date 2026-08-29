#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# MLPerf Inference (LoadGen) benchmark, driven by NVIDIA's nv_mlpinf harness.
# srt-slurm owns the deployment; this script installs the harness into an
# isolated job-scoped runtime and points it at the ready frontend.
#
# The invocation mirrors the submission repo's own sflow task
# (closed/NVIDIA/scaleout/sflow/templates/dynamo_disagg_loadgen.yaml), so a run
# here and a run there differ in who deployed the cluster, not in how it is
# measured.

set -euo pipefail

ENDPOINT=$1          # host:port of the frontend (no scheme — nv_mlpinf adds it)
HARNESS_DIR=$2       # mlperf-inference/closed/NVIDIA
BENCHMARK=$3
SCENARIO=$4
TEST_MODE=$5
CORE_TYPE=$6
SYSTEM_NAME=${7:-${SYSTEM_NAME:-}}
SCRATCH_PATH=${8:-}

# When the client runs on a different node than the frontend, localhost is
# wrong; benchmark_stage injects the frontend's real host/port.
if [[ -n "${SRT_FRONTEND_HOST:-}" ]]; then
  PORT_FROM_ENDPOINT=${ENDPOINT##*:}
  ENDPOINT="${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT:-$PORT_FROM_ENDPOINT}"
fi

[[ -d "$HARNESS_DIR" ]] || { echo "ERROR: mlperf_harness_dir $HARNESS_DIR not found in container (mount via extra_mount)" >&2; exit 1; }
[[ -d "$HARNESS_DIR/src/nv_mlpinf" ]] || { echo "ERROR: $HARNESS_DIR has no src/nv_mlpinf — mlperf_harness_dir must point at mlperf-inference/closed/NVIDIA" >&2; exit 1; }
[[ -f "$HARNESS_DIR/pyproject.toml" ]] || { echo "ERROR: $HARNESS_DIR/pyproject.toml missing — cannot install the harness" >&2; exit 1; }

RUNTIME="${MLPERF_RUNTIME:-/tmp/mlperf-${SLURM_JOB_ID:-$$}}"
export HOME="$RUNTIME/home"
export PIP_CACHE_DIR="$RUNTIME/pip-cache"
export VENV="$RUNTIME/venv"
mkdir -p "$HOME" "$PIP_CACHE_DIR"

# LoadGen holds one in-flight request per scheduled query; lift the soft fd
# limit to the hard limit so high-QPS scenarios don't starve.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# ---- preflight: install the harness once per job -----------------------------
# READY is self-validating: it records the arch and harness commit it was built
# for, so a relocated MLPERF_RUNTIME reused across jobs (or a different pin)
# reinstalls instead of silently measuring with the wrong harness.
# safe.directory: the checkout is normally a bind-mount owned by whoever staged
# it, and git refuses to read a repo owned by another user ("dubious
# ownership"). Without this the commit silently degrades to "unknown", which
# makes the fingerprint constant and defeats the whole point of the marker —
# two different harness checkouts would share one installed runtime.
HARNESS_COMMIT="$(git -c safe.directory="$HARNESS_DIR" -C "$HARNESS_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ "$HARNESS_COMMIT" == "unknown" ]]; then
  # Not fatal — a tarball or export is a legitimate way to ship the harness —
  # but say so, because a reused MLPERF_RUNTIME can no longer detect a change.
  echo "[mlperf] WARNING: cannot read a commit from $HARNESS_DIR; the runtime fingerprint cannot" >&2
  echo "         detect a harness change. Use a job-scoped MLPERF_RUNTIME (the default) rather" >&2
  echo "         than a shared one." >&2
fi
FINGERPRINT="$(uname -m):$HARNESS_COMMIT"
ready() { [[ "$(cat "$RUNTIME/READY" 2>/dev/null)" == "$FINGERPRINT" ]]; }
HOLDING_LOCK=0
if ! ready; then
  # Atomic mkdir as a mutex: a concurrent job sharing this runtime waits for
  # the installer instead of double-installing into the same venv.
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
  echo "[mlperf] preflight: installing nv_mlpinf in $RUNTIME (fingerprint $FINGERPRINT)"

  # --system-site-packages so the container's torch/tensorrt_llm are reused; a
  # from-scratch install of those would dwarf the benchmark itself. Only pip is
  # upgraded: upgrading setuptools here shadows the container's copy for every
  # process using this venv, and TRT-LLM's torch build pins it (<82).
  [[ -x "$VENV/bin/python" ]] || python3 -m venv --system-site-packages "$VENV"
  "$VENV/bin/python" -m pip install --disable-pip-version-check --upgrade pip

  # --no-build-isolation matches the submission repo's own harness task: the
  # build deps are already in the image, and an isolated build would refetch
  # (and can fail without egress).
  #
  # Expect pip to report four unsatisfied constraints here, e.g.
  #   tensorrt-llm requires numpy>=2.0.0, but you have 1.26.4
  #   transformers requires huggingface-hub>=1.5.0, but you have 0.34.0
  #   torchtitan requires datasets>=3.6.0 / diffusers requires safetensors>=0.8.0
  # They come from layering this venv over the container: pip resolves the
  # harness's own pins and installs older copies that shadow the image's, while
  # the packages doing the complaining are the image's and are not what the
  # endpoint cores import. Measured harmless for the endpoint path — a
  # PerformanceOnly Interactive run completes and LoadGen writes a summary — but
  # not audited for the executor cores, which do import tensorrt_llm. The check
  # below asserts the shadowed copies are at least usable.
  (cd "$HARNESS_DIR" && "$VENV/bin/python" -m pip install -e ".[llm]" -q --no-build-isolation)
  "$VENV/bin/python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

import mlperf_loadgen  # noqa: F401  - fail here if LoadGen did not come along

# numpy is the one the install actively downgrades, and the harness reads its
# QSL tensors through it. A shadowed-but-broken copy would otherwise surface as
# an opaque failure during QSL construction, well after preflight "passed".
import numpy

print("[mlperf] numpy", numpy.__version__)
try:
    print("[mlperf] mlperf_loadgen", version("mlcommons_loadgen"))
except PackageNotFoundError:
    print("[mlperf] mlperf_loadgen (version unavailable)")
PY

  echo "$FINGERPRINT" > "$RUNTIME/READY"
  echo "[mlperf] preflight complete"
fi
[[ "$HOLDING_LOCK" == 1 ]] && rmdir "$RUNTIME/.build-lock" 2>/dev/null || true

# ---- run --------------------------------------------------------------------
RESULTS_DIR=/logs/mlperf
# nv_mlpinf appends <system>/<benchmark>/<scenario> to LOG_DIR, so keeping the
# mode in the path here is what separates a performance run from an accuracy
# one in the rollup.
LOG_DIR="$RESULTS_DIR/$TEST_MODE"
mkdir -p "$LOG_DIR"

# The checkout is authoritative over any copy baked into the image.
export PYTHONPATH="$HARNESS_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export LOG_DIR

# nv_mlpinf resolves every runtime path from project_base_dir, which defaults to
# the /work mount its own sflow tasks use. Without this it writes a paths.yml of
# non-existent defaults and then dies importing the submission checker:
#   nv_mlpinf/common/mlcommons/loadgen.py imports `constants` out of
#   mlcommons_inf_repo (= <project_base_dir>/3rdparty/mlc-inference, vendored in
#   the checkout), so a wrong base is a ModuleNotFoundError at CLI import time,
#   long before anything reaches the endpoint.
export PROJECT_BASE_DIR="${PROJECT_BASE_DIR:-$HARNESS_DIR}"
export MLCOMMONS_INF_REPO="${MLCOMMONS_INF_REPO:-$PROJECT_BASE_DIR/3rdparty/mlc-inference}"
[[ -d "$MLCOMMONS_INF_REPO" ]] || {
  echo "ERROR: MLCOMMONS_INF_REPO $MLCOMMONS_INF_REPO missing — the harness imports the submission checker from there. Mount the checkout's 3rdparty/ or set MLCOMMONS_INF_REPO in benchmark.env" >&2
  exit 1
}

export CONFIG_DIR="${CONFIG_DIR:-$PROJECT_BASE_DIR/configs}"

# nv_mlpinf resolves a per-(benchmark, system, scenario) config module before it
# ever looks at core_type, so an unregistered system is fatal even though
# dynamo_endpoint needs nothing out of that module: main.py builds
# <config_dir>/<benchmark>/<system_id>/<serving_framework>/<scenario>/harness.py
# and raises FileNotFoundError if it is missing. Without SYSTEM_NAME the id is
# auto-detected, and any machine outside the built-in list (every Vera Rubin
# bring-up node today) detects as UNREGISTERED_<arch>_<gpu>xN, which has no
# configs. Check it here so the failure names the fix.
BENCH_UNDERSCORE=${BENCHMARK//-/_}
CONFIG_BASE="$CONFIG_DIR/$BENCH_UNDERSCORE"
# The <serving_framework> segment is a per-benchmark constant with no CLI
# override, so glob it rather than hardcoding TRTLLM. It is harness.py
# specifically that must exist: a scenario directory can hold only server.py
# (the RunLLMServer action), which run_harness will not accept.
list_valid_points() {
  local found=0 path
  for path in "$CONFIG_BASE"/*/*/*/harness.py; do
    [[ -f "$path" ]] || continue
    found=1
    local scen system
    scen=$(basename "$(dirname "$path")")
    system=$(basename "$(dirname "$(dirname "$(dirname "$path")")")")
    echo "         $system  $scen" >&2
  done
  [[ "$found" == 1 ]] || echo "         (none under $CONFIG_BASE)" >&2
}
[[ -n "$SYSTEM_NAME" ]] && export SYSTEM_NAME
if [[ -d "$CONFIG_BASE" && -n "$SYSTEM_NAME" ]]; then
  if ! compgen -G "$CONFIG_BASE/$SYSTEM_NAME/*/$SCENARIO/harness.py" >/dev/null; then
    echo "ERROR: nv_mlpinf has no harness config for benchmark=$BENCHMARK system=$SYSTEM_NAME scenario=$SCENARIO." >&2
    echo "       run_harness resolves this module before it looks at core_type, so it is required" >&2
    echo "       even though $CORE_TYPE needs nothing from it. Valid system/scenario pairs here:" >&2
    list_valid_points
    echo "       Set benchmark.mlperf_system_name / mlperf_scenario to a pair above, or point" >&2
    echo "       benchmark.env CONFIG_DIR at a tree that provides your system." >&2
    exit 1
  fi
elif [[ -d "$CONFIG_BASE" ]]; then
  echo "[mlperf] WARNING: benchmark.mlperf_system_name is unset; nv_mlpinf will auto-detect, and an" >&2
  echo "         unregistered machine has no config. Valid system/scenario pairs here:" >&2
  list_valid_points
fi

# MLPINF_HTTP_USE_COMPLETIONS=1 selects /v1/completions over the chat route.
# MLPINF_USE_DYNAMO defaults to 0 even against Dynamo: at 1 it sets
# NOSKIP_FINAL_CHUNK, which keeps the client reading past finish_reason waiting
# for [DONE]; if the frontend half-closes, the read blocks forever and wedges
# LoadGen's no-timeout drain (submission-repo job 372979108 sat silent for 233
# minutes with 7,501 samples outstanding and produced no summary).
# MLPINF_FIRST_TOKEN_ALWAYS=1 because LoadGen requires a first-token latency
# before a sample latency in any non-Offline scenario, and a completion shorter
# than the engine's stream_interval arrives as one chunk that is both first and
# final — without this such a sample invalidates the run.
export MLPINF_HTTP_USE_COMPLETIONS="${MLPINF_HTTP_USE_COMPLETIONS:-1}"
export MLPINF_USE_DYNAMO="${MLPINF_USE_DYNAMO:-0}"
export MLPINF_FIRST_TOKEN_ALWAYS="${MLPINF_FIRST_TOKEN_ALWAYS:-1}"
# nv_mlpinf defaults this to /home/mlperf_inference_storage, which is an ASE
# cluster path and rarely mounted elsewhere.
[[ -n "$SCRATCH_PATH" ]] && export MLPERF_SCRATCH_PATH="$SCRATCH_PATH"

# The scratch tree feeds two different things, and only one of them fails early.
#   preprocessed_data/<benchmark>/  the QSL tensors — asserted during QSL
#                                   construction, so a miss is at least loud
#   models/<benchmark>/             the checkpoint the harness loads its
#                                   tokenizer from ("Loaded tokenizer from local
#                                   path: .../models/<benchmark>/...")
# harness_use_hf_tokenizer is False by default, so the tokenizer is expected on
# disk, not on the Hub. When the checkpoint is absent the harness can fall back
# to treating the configured name as a Hub repo id, which needs egress and — for
# a path-shaped name — raises HFValidationError. The submission repo hit exactly
# that: three 7-node runs finished their performance phase and then died in
# scoring. Check both here, while the message can still name the directory.
if [[ -n "${MLPERF_SCRATCH_PATH:-}" ]]; then
  for sub in "preprocessed_data/$BENCHMARK" "models/$BENCHMARK"; do
    [[ -d "$MLPERF_SCRATCH_PATH/$sub" ]] || {
      echo "[mlperf] WARNING: $MLPERF_SCRATCH_PATH/$sub is missing." >&2
      echo "         preprocessed_data/<benchmark> holds the QSL tensors; models/<benchmark> holds" >&2
      echo "         the checkpoint the harness loads its tokenizer from. Without the latter it may" >&2
      echo "         fall back to a Hub lookup, which needs egress and fails on a path-shaped name." >&2
    }
  done
fi

# Resolve the core the same way the harness will. CoreType.DYNAMO_ENDPOINT is a
# dangling registry entry in some checkouts — it names dynamo_endpoint_core,
# which does not exist there (newer trees alias it to TrtllmEndpointCore) — and
# the harness only discovers that deep inside wrap_lg_test, after the QSL is
# built and the dataset read. Fail here instead, while the message is still
# about the core.
if ! "$VENV/bin/python" -c "
import sys
from nv_mlpinf.llmlib.cores import BackendRegistry
try:
    BackendRegistry.get('$CORE_TYPE')
except Exception as exc:
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
    sys.exit(1)
" 2>/tmp/core-probe.err; then
  echo "ERROR: nv_mlpinf cannot load core '$CORE_TYPE' from this harness checkout:" >&2
  sed 's/^/         /' /tmp/core-probe.err >&2
  echo "       Set benchmark.mlperf_core_type to a core this checkout implements" >&2
  echo "       (trtllm_endpoint is what the submission repo's own Dynamo task uses)." >&2
  exit 1
fi

RUN_ARGS=(
  --benchmarks="$BENCHMARK"
  --scenarios="$SCENARIO"
  --core_type="$CORE_TYPE"
  --trtllm_server_urls="$ENDPOINT"
  --test_mode="$TEST_MODE"
)
[[ -n "$SYSTEM_NAME" ]] && RUN_ARGS+=(--system_name="$SYSTEM_NAME")
# Simple space-separated tokens only — values are word-split, never shell-parsed.
read -r -a EXTRA_ARGS <<< "${MLPERF_EXTRA_ARGS:-}" || true

echo "[mlperf] $TEST_MODE run: endpoint=$ENDPOINT benchmark=$BENCHMARK scenario=$SCENARIO core=$CORE_TYPE log_dir=$LOG_DIR"
set +e
# Run from the checkout: some harness defaults are paths relative to the working
# directory, not to project_base_dir — llm_gen_config_path resolves as
# src/nv_mlpinf/benchmarks/<benchmark>/generation_config.json, so anywhere else
# dies with "Generation config file not found" once the QSL is already built.
# The sflow tasks get this for free by running from WORK_DIR. LOG_DIR is
# absolute, so the results still land where the rollup expects.
cd "$HARNESS_DIR"
"$VENV/bin/nv-mlpinf" run_harness "${RUN_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
HARNESS_RC=$?
set -e
echo "[mlperf] run_harness exited with $HARNESS_RC"

# LoadGen's summary records the test it ran, not the knobs srtctl chose for it.
# rollup.py merges this sidecar in as srt_args.
SRT_RUN_DIR="$LOG_DIR" \
SRT_ENDPOINT="$ENDPOINT" SRT_BENCHMARK="$BENCHMARK" SRT_SCENARIO="$SCENARIO" \
SRT_MODE="$TEST_MODE" SRT_CORE_TYPE="$CORE_TYPE" SRT_SYSTEM_NAME="$SYSTEM_NAME" \
SRT_HARNESS_COMMIT="$HARNESS_COMMIT" SRT_EXTRA_ARGS="${MLPERF_EXTRA_ARGS:-}" \
"$VENV/bin/python" - <<'PY'
import json
import os
from pathlib import Path

FIELDS = {
    "endpoint": "SRT_ENDPOINT",
    "benchmark": "SRT_BENCHMARK",
    "scenario": "SRT_SCENARIO",
    "mode": "SRT_MODE",
    "core_type": "SRT_CORE_TYPE",
    "system_name": "SRT_SYSTEM_NAME",
    "harness_commit": "SRT_HARNESS_COMMIT",
    "extra_args": "SRT_EXTRA_ARGS",
}
# Unset optional knobs are empty strings; omit them rather than record "".
record = {name: os.environ[var] for name, var in FIELDS.items() if os.environ.get(var)}
out = Path(os.environ["SRT_RUN_DIR"]) / "srt_run.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, indent=1))
print(f"[mlperf] wrote {out}")
PY

exit $HARNESS_RC
