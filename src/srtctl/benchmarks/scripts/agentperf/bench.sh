#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AgentPerf trajectory-replay benchmark.
# srt-slurm owns server startup; this script prepares an isolated client
# runtime under /tmp and runs the mounted agentperf-client checkout against
# the ready frontend.
#
# The preflight is a faithful translation of the standalone disagg-harness
# invocation this benchmark was ported from (prepare_agentperf_client.sh), so
# results stay comparable:
#   * HOME/CARGO_HOME/RUSTUP_HOME/UV_PROJECT_ENVIRONMENT all live under one
#     job-scoped /tmp dir — the checkout can be mounted read-only and nothing
#     leaks between jobs;
#   * toolchain versions are pinned (override via AGENTPERF_RUST_VERSION /
#     AGENTPERF_UV_VERSION);
#   * the trajectory and user-assignments datasets are staged from shared
#     storage to node-local /tmp before measurement — reading the multi-GiB
#     trajectory file over Lustre mid-run is a perf artifact of its own.

set -euo pipefail

ENDPOINT=$1
MODEL_NAME=$2
CLIENT_DIR=$3
CONFIG_PATH=$4
CONCURRENCIES=$5

# When the client runs on a different node than the frontend, localhost is
# wrong; benchmark_stage injects the frontend's real host/port.
if [[ -n "${SRT_FRONTEND_HOST:-}" ]]; then
  PORT_FROM_ENDPOINT=$(echo "$ENDPOINT" | sed -E 's|.*:([0-9]+).*|\1|')
  ENDPOINT="http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT:-$PORT_FROM_ENDPOINT}"
fi

[[ -d "$CLIENT_DIR" ]] || { echo "ERROR: agentperf_client_dir $CLIENT_DIR not found in container (mount via extra_mount)" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "ERROR: agentperf_config $CONFIG_PATH not found in container" >&2; exit 1; }

RUNTIME="${AGENTPERF_RUNTIME:-/tmp/agentperf-${SLURM_JOB_ID:-$$}}"
export HOME="$RUNTIME/home"
export CARGO_HOME="$RUNTIME/cargo"
export RUSTUP_HOME="$RUNTIME/rustup"
export CARGO_TARGET_DIR="$RUNTIME/target"
export UV_PROJECT_ENVIRONMENT="$RUNTIME/venv"
export UV_CACHE_DIR="$RUNTIME/uv-cache"
export TIKTOKEN_CACHE_DIR="$RUNTIME/tiktoken-cache"
export PATH="$HOME/.local/bin:$CARGO_HOME/bin:/usr/local/bin:/usr/bin:/bin"
mkdir -p "$HOME" "$CARGO_HOME" "$RUSTUP_HOME" "$UV_CACHE_DIR" "$RUNTIME/data" "$TIKTOKEN_CACHE_DIR"

# The client holds one streaming connection per concurrent user; lift the
# soft fd limit to the hard limit so high-concurrency phases don't starve.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# ---- preflight: build the client runtime once per job ----------------------
# READY is self-validating: it records the arch + client commit it was built
# for, so a relocated AGENTPERF_RUNTIME reused across jobs (or clusters of a
# different arch) rebuilds instead of silently reusing a stale/wrong runtime.
FINGERPRINT="$(uname -m):$(git -C "$CLIENT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
ready() { [[ "$(cat "$RUNTIME/READY" 2>/dev/null)" == "$FINGERPRINT" ]]; }
HOLDING_LOCK=0
if ! ready; then
  # Atomic mkdir as a mutex: a concurrent job sharing this runtime waits for
  # the builder instead of double-building into the same venv.
  if mkdir "$RUNTIME/.build-lock" 2>/dev/null; then
    HOLDING_LOCK=1
  else
    echo "[agentperf] preflight in progress elsewhere; waiting for READY"
    for _ in $(seq 1 360); do
      ready && break
      sleep 5
    done
    ready || { echo "ERROR: timed out waiting for concurrent agentperf preflight in $RUNTIME" >&2; exit 1; }
  fi
fi
if ! ready; then
  echo "[agentperf] preflight: building client runtime in $RUNTIME (fingerprint $FINGERPRINT)"
  cd "$CLIENT_DIR"

  if [[ ! -x "$CARGO_HOME/bin/cargo" ]]; then
    RUST_VERSION="${AGENTPERF_RUST_VERSION:-1.93.1}"
    ARCH=$(uname -m)
    RUSTUP_INIT="$RUNTIME/rustup-init-$ARCH"
    wget --tries=3 --waitretry=5 -O "$RUSTUP_INIT" \
      "https://static.rust-lang.org/rustup/archive/1.28.1/${ARCH}-unknown-linux-gnu/rustup-init"
    chmod +x "$RUSTUP_INIT"
    "$RUSTUP_INIT" -y --profile minimal --default-toolchain "$RUST_VERSION" --no-modify-path
  fi
  cargo --version
  rustc --version

  if ! command -v uv >/dev/null 2>&1; then
    UV_VERSION="${AGENTPERF_UV_VERSION:-0.8.22}"
    python3 -m pip install --disable-pip-version-check --no-cache-dir --user "uv==$UV_VERSION"
  fi
  uv --version

  # --frozen fails fast on uv.lock drift instead of re-resolving (silently
  # different deps) or trying to rewrite uv.lock in a read-only checkout;
  # --no-sync afterwards because the env was just synced.
  uv sync --frozen
  # The default client_impl is the Rust streaming core; build it into the
  # redirected venv (maturin develop installs there, CARGO_TARGET_DIR keeps
  # build artifacts out of the possibly read-only checkout).
  uv run --no-sync maturin develop --release -m rustcore/Cargo.toml
  uv run --no-sync python -c 'import agentperf_rustcore; print("agentperf_rustcore", agentperf_rustcore.__version__)'
  # Warm the tokenizer cache so a mid-run egress hiccup can't kill startup.
  uv run --no-sync python -c "import tiktoken; tiktoken.get_encoding('o200k_base')" || true

  # Stage the trajectory and assignment data from shared storage to
  # node-local /tmp and write the config the benchmark actually uses with
  # those two paths replaced.
  uv run --no-sync python - "$CONFIG_PATH" "$RUNTIME/benchmark_config.yaml" "$RUNTIME/data" <<'PY'
from pathlib import Path
import shutil
import sys
import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
data_dir = Path(sys.argv[3])
config = yaml.safe_load(source.read_text())
for field in ("trajectory_path", "user_assignments_path"):
    value = config.get(field)
    if not value:
        continue
    source_data = Path(value)
    target_data = data_dir / source_data.name
    shutil.copy2(source_data, target_data)
    if not target_data.stat().st_size:
        raise RuntimeError(f"empty staged AgentPerf input: {target_data}")
    config[field] = str(target_data)
destination.write_text(yaml.safe_dump(config, sort_keys=False))
PY

  echo "$FINGERPRINT" > "$RUNTIME/READY"
  echo "[agentperf] preflight complete"
fi
[[ "$HOLDING_LOCK" == 1 ]] && rmdir "$RUNTIME/.build-lock" 2>/dev/null || true

# ---- run --------------------------------------------------------------------
RESULTS_DIR=/logs/agentperf
mkdir -p "$RESULTS_DIR"

cd "$CLIENT_DIR"
echo "[agentperf] endpoint=$ENDPOINT model=$MODEL_NAME concurrencies=$CONCURRENCIES config=$RUNTIME/benchmark_config.yaml"
# Simple space-separated tokens only — values are word-split, never shell-parsed.
read -r -a EXTRA_ARGS <<< "${AGENTPERF_EXTRA_ARGS:-}" || true
uv run --no-sync python agentperf/run.py \
  --config "$RUNTIME/benchmark_config.yaml" \
  --base-url "$ENDPOINT" \
  --model "$MODEL_NAME" \
  --concurrencies "$CONCURRENCIES" \
  --request-log-path "$RESULTS_DIR/requests.jsonl" \
  --results-dir "$RESULTS_DIR" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
