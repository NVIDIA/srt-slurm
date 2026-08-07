#!/usr/bin/env bash
# Run the local 2x L40S Dynamo/SGLang aggregate and disaggregate smoke
# benchmarks with tachometer-scraper telemetry.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DYNAMO_ROOT="${DYNAMO_ROOT:-/ephemeral/dynamo}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/outputs/local-l40s/$(date +%Y%m%d_%H%M%S)}"
TACHOMETER_BIN="${TACHOMETER_BIN:-/ephemeral/cargo-target/release/tachometer-scraper}"
AIPERF_BIN="${AIPERF_BIN:-}"

SERVER_PID=""
TACHOMETER_PID=""

log() {
  printf '[local-sglang] %s\n' "$*"
}

clean_stale() {
  pkill -9 -f 'python3? -m dynamo\.frontend' 2>/dev/null || true
  pkill -9 -f 'python3? -m dynamo\.sglang' 2>/dev/null || true
  pkill -9 -f 'tachometer-scraper' 2>/dev/null || true
  pkill -9 -f 'aiperf profile' 2>/dev/null || true
  sleep 3
}

stop_pid() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

stop_tachometer() {
  local pid="${TACHOMETER_PID:-}"
  [[ -n "$pid" ]] || return 0
  kill -INT "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  TACHOMETER_PID=""
}

cleanup() {
  local rc=$?
  stop_tachometer
  stop_pid "$SERVER_PID"
  clean_stale
  exit "$rc"
}
trap cleanup EXIT INT TERM

wait_http_ready() {
  local url="$1"
  local timeout="${2:-420}"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start >= timeout )); then
      log "Timed out waiting for $url"
      return 1
    fi
    sleep 5
  done
}

wait_chat_ready() {
  local timeout="${1:-600}"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS --max-time 30 \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1,\"stream\":false}" \
      http://localhost:8000/v1/chat/completions >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start >= timeout )); then
      log "Timed out waiting for chat completions to become ready"
      return 1
    fi
    sleep 5
  done
}

start_tachometer() {
  local phase="$1"
  local out_dir="$2"
  shift 2
  local storage_dir="$out_dir/tachometer/run"

  rm -rf "$storage_dir" "$out_dir/tachometer-local"
  mkdir -p "$out_dir/tachometer" "$out_dir/tachometer-local"
  if [[ ! -x "$TACHOMETER_BIN" ]]; then
    log "tachometer-scraper not executable at $TACHOMETER_BIN; skipping telemetry"
    return 0
  fi

  "$TACHOMETER_BIN" \
    "$@" \
    --storage "$storage_dir" \
    --local-dir "$out_dir/tachometer-local" \
    --freq "${TACHOMETER_FREQ:-1.0}" \
    --save-interval 2 \
    --sync-interval 0 \
    >"$out_dir/tachometer.log" 2>&1 &
  TACHOMETER_PID=$!
  log "Started tachometer for $phase as pid $TACHOMETER_PID"
}

run_aiperf() {
  local phase="$1"
  local out_dir="$2"

  "$AIPERF_BIN" profile "$MODEL" \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --concurrency "${AIPERF_CONCURRENCY:-2}" \
    --request-count "${AIPERF_REQUEST_COUNT:-8}" \
    --warmup-request-count "${AIPERF_WARMUP_REQUEST_COUNT:-1}" \
    --isl "${AIPERF_ISL:-128}" \
    --osl "${AIPERF_OSL:-32}" \
    --image-batch-size 0 \
    --audio-batch-size 0 \
    --video-batch-size 0 \
    --request-timeout-seconds 300 \
    --tokenizer-trust-remote-code \
    --output-artifact-dir "$out_dir/aiperf" \
    --profile-export-prefix "$phase-aiperf" \
    --ui none \
    --no-server-metrics \
    2>&1 | tee "$out_dir/aiperf.log"

  python3 - "$out_dir/aiperf/$phase-aiperf.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

errors = data.get("error_summary") or []
error_counts = data.get("error_request_count") or {}
error_total = sum(v.get("value", 0) for v in error_counts.values() if isinstance(v, dict))
if errors or error_total:
    print(f"aiperf reported errors in {path}: {errors or error_counts}", file=sys.stderr)
    raise SystemExit(1)
PY
}

run_phase() {
  local phase="$1"
  local launch_script="$2"
  shift 2
  local out_dir="$RUN_ROOT/$phase"

  log "Preparing $phase run in $out_dir"
  mkdir -p "$out_dir"
  clean_stale
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee "$out_dir/gpu-before.txt"

  (
    cd "$DYNAMO_ROOT"
    # shellcheck disable=SC1091
    source "$DYNAMO_ROOT/.venv/bin/activate"
    export HF_HOME="${HF_HOME:-/ephemeral/hf-cache}"
    export DYN_HTTP_PORT=8000
    export DYN_SYSTEM_PORT=8081
    export DYN_SYSTEM_PORT1=8081
    export DYN_SYSTEM_PORT2=8082
    export DYN_DISAGG_BOOTSTRAP_PORT=12345
    export _PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-4096}"
    exec setsid bash "$launch_script" "$@"
  ) >"$out_dir/server.log" 2>&1 &
  SERVER_PID=$!

  log "Started $phase server as pid $SERVER_PID"
  wait_http_ready "http://localhost:8000/v1/models" "${SERVER_READY_TIMEOUT:-420}"
  wait_http_ready "http://localhost:8081/metrics" 120
  if [[ "$phase" == "disagg" ]]; then
    wait_http_ready "http://localhost:8082/metrics" 120
  fi
  wait_chat_ready "${CHAT_READY_TIMEOUT:-600}"

  if [[ "$phase" == "disagg" ]]; then
    start_tachometer "$phase" "$out_dir" \
      --endpoint prefill=http://localhost:8081/metrics \
      --endpoint decode=http://localhost:8082/metrics
    run_aiperf "$phase" "$out_dir"
  else
    start_tachometer "$phase" "$out_dir" \
      --endpoint worker=http://localhost:8081/metrics
    run_aiperf "$phase" "$out_dir"
  fi

  stop_tachometer
  stop_pid "$SERVER_PID"
  SERVER_PID=""
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee "$out_dir/gpu-after.txt"
  log "Completed $phase run"
}

main() {
  if [[ -z "$AIPERF_BIN" ]]; then
    AIPERF_BIN="$(command -v aiperf || true)"
  fi
  if [[ -z "$AIPERF_BIN" || ! -x "$AIPERF_BIN" ]]; then
    log "aiperf not found; set AIPERF_BIN"
    exit 1
  fi
  mkdir -p "$RUN_ROOT"
  log "Writing artifacts to $RUN_ROOT"
  run_phase agg "$DYNAMO_ROOT/examples/backends/sglang/launch/agg.sh" --model-path "$MODEL" --tp 2
  run_phase disagg "$DYNAMO_ROOT/examples/backends/sglang/launch/disagg.sh"
  log "All runs completed: $RUN_ROOT"
}

main "$@"
