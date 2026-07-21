#!/usr/bin/env bash
set -euo pipefail
set -x

# Self-contained AIPerf driver for AgentX trace sweeps. The dataset is selected
# through the recipe environment, and the virtual environment must already be
# mounted.

AIPERF_DIR="${AIPERF_DIR:-/workspace/srt-slurm-sa/aiperf}"
AIPERF_VENV="${AIPERF_VENV:-$AIPERF_DIR/.venv}"
AIPERF_PYTHON="${AIPERF_VENV}/bin/python"
AIPERF_CLI="${AIPERF_VENV}/bin/aiperf"
AIPERF_FAILED_REQUEST_THRESHOLD="${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}"
AIPERF_MAX_OSL="${AIPERF_MAX_OSL:-8192}"
AIPERF_TRAJECTORY_START_MIN_RATIO="${AIPERF_TRAJECTORY_START_MIN_RATIO:-0.25}"
AIPERF_TRAJECTORY_START_MAX_RATIO="${AIPERF_TRAJECTORY_START_MAX_RATIO:-0.75}"
AIPERF_AGENTIC_CACHE_WARMUP_DURATION="${AIPERF_AGENTIC_CACHE_WARMUP_DURATION:-600}"
AIPERF_WARMUP_GRACE_PERIOD="${AIPERF_WARMUP_GRACE_PERIOD:-1800}"
AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING="${AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING:-0}"
AIPERF_USE_LEGACY_DYNAMO_SESSION_CONTROL="${AIPERF_USE_LEGACY_DYNAMO_SESSION_CONTROL:-0}"
AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS="${AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS:-}"

check_env_vars() {
    local missing=0
    for name in "$@"; do
        if [ -z "${!name:-}" ]; then
            echo "ERROR: required environment variable is missing: $name" >&2
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        exit 1
    fi
}

run_agentic_replay_and_write_outputs() {
    local result_dir="$1"
    mkdir -p "$result_dir"
    env | sort > "$result_dir/env.txt"
    printf '%s\n' "$REPLAY_CMD" > "$result_dir/aiperf_command.txt"
    set +e
    bash -lc "$REPLAY_CMD" 2>&1 | tee "$result_dir/aiperf.log"
    local replay_rc="${PIPESTATUS[0]}"
    set -e
    return "$replay_rc"
}

check_env_vars MODEL MODEL_PREFIX FRAMEWORK PRECISION CONC RESULT_FILENAME DURATION

if [ ! -x "$AIPERF_PYTHON" ]; then
    echo "ERROR: AIPerf Python is missing or not executable: $AIPERF_PYTHON" >&2
    exit 1
fi
if [ ! -x "$AIPERF_CLI" ]; then
    echo "ERROR: AIPerf CLI is missing or not executable: $AIPERF_CLI" >&2
    exit 1
fi

"$AIPERF_PYTHON" - <<'PY'
import aiperf
import huggingface_hub

print(f"Using aiperf {getattr(aiperf, '__version__', 'unknown')} from {aiperf.__file__}")
print(f"Using huggingface_hub from {huggingface_hub.__file__}")
PY

AIPERF_PROFILE_HELP="$(COLUMNS=240 "$AIPERF_CLI" profile --help 2>&1)"

RESULT_DIR="${RESULT_DIR:-/logs/agentic}"
AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-$RESULT_DIR}"
PORT="${PORT:-8000}"
AIPERF_BASE_URL="${AIPERF_BASE_URL:-http://localhost:$PORT}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-1000000}"
NUM_DATASET_ENTRIES="${NUM_DATASET_ENTRIES:-472}"
PUBLIC_DATASET="${PUBLIC_DATASET:-semianalysis_cc_traces_weka_with_subagents_060826}"
HF_WEKA_DATASET="${HF_WEKA_DATASET:-}"
SLICE_DURATION="${SLICE_DURATION:-1.0}"
VLLM_START_PROFILE_AFTER_SECONDS="${VLLM_START_PROFILE_AFTER_SECONDS:-900}"

mkdir -p "$RESULT_DIR"

export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES:-0}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AIPERF_DATASET_CONFIGURATION_TIMEOUT:-1800}"
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT:-1800}"

REPLAY_CMD="$AIPERF_CLI profile"
REPLAY_CMD+=" --scenario inferencex-agentx-mvp"
REPLAY_CMD+=" --url $AIPERF_BASE_URL"
REPLAY_CMD+=" --endpoint /v1/chat/completions"
REPLAY_CMD+=" --endpoint-type chat"
REPLAY_CMD+=" --streaming"
REPLAY_CMD+=" --model $MODEL"
REPLAY_CMD+=" --concurrency $CONC"
REPLAY_CMD+=" --benchmark-duration $DURATION"
REPLAY_CMD+=" --random-seed 42"
REPLAY_CMD+=" --failed-request-threshold $AIPERF_FAILED_REQUEST_THRESHOLD"
REPLAY_CMD+=" --trajectory-start-min-ratio $AIPERF_TRAJECTORY_START_MIN_RATIO"
REPLAY_CMD+=" --trajectory-start-max-ratio $AIPERF_TRAJECTORY_START_MAX_RATIO"
if [ -n "$AIPERF_AGENTIC_CACHE_WARMUP_DURATION" ] && [ "$AIPERF_AGENTIC_CACHE_WARMUP_DURATION" != "none" ]; then
    REPLAY_CMD+=" --agentic-cache-warmup-duration $AIPERF_AGENTIC_CACHE_WARMUP_DURATION"
fi
if [ -n "$AIPERF_WARMUP_GRACE_PERIOD" ] && [ "$AIPERF_WARMUP_GRACE_PERIOD" != "none" ]; then
    if grep -q -- "--warmup-grace-period" <<<"$AIPERF_PROFILE_HELP"; then
        REPLAY_CMD+=" --warmup-grace-period $AIPERF_WARMUP_GRACE_PERIOD"
    else
        echo "NOTE: current aiperf does not support --warmup-grace-period; skipping AIPERF_WARMUP_GRACE_PERIOD=$AIPERF_WARMUP_GRACE_PERIOD." \
            | tee "$RESULT_DIR/aiperf_warmup_grace_flag_note.txt"
    fi
fi
REPLAY_CMD+=" --use-server-token-count"
REPLAY_CMD+=" --no-gpu-telemetry"
REPLAY_CMD+=" --tokenizer-trust-remote-code"
if [ "$AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING" = "1" ]; then
    if grep -q -- "--use-dynamo-conv-aware-routing" <<<"$AIPERF_PROFILE_HELP"; then
        REPLAY_CMD+=" --use-dynamo-conv-aware-routing"
    else
        echo "ERROR: current aiperf does not support --use-dynamo-conv-aware-routing." >&2
        exit 1
    fi
    if [ -n "$AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS" ]; then
        if ! [[ "$AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS must be a positive integer." >&2
            exit 1
        fi
        if grep -q -- "--dynamo-session-timeout-seconds" <<<"$AIPERF_PROFILE_HELP"; then
            REPLAY_CMD+=" --dynamo-session-timeout-seconds $AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS"
        else
            echo "ERROR: current aiperf does not support --dynamo-session-timeout-seconds." >&2
            exit 1
        fi
    fi
    if [ "$AIPERF_USE_LEGACY_DYNAMO_SESSION_CONTROL" = "1" ]; then
        if grep -q -- "--use-legacy-dynamo-session-control" <<<"$AIPERF_PROFILE_HELP"; then
            REPLAY_CMD+=" --use-legacy-dynamo-session-control"
        else
            echo "ERROR: current aiperf does not support --use-legacy-dynamo-session-control." >&2
            exit 1
        fi
    fi
elif [ "$AIPERF_USE_LEGACY_DYNAMO_SESSION_CONTROL" = "1" ]; then
    echo "ERROR: legacy Dynamo session control requires AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=1." >&2
    exit 1
fi
if grep -q -- "--vllm-start-profile-after-seconds" <<<"$AIPERF_PROFILE_HELP"; then
    REPLAY_CMD+=" --vllm-start-profile-after-seconds $VLLM_START_PROFILE_AFTER_SECONDS"
else
    echo "NOTE: current aiperf does not support --vllm-start-profile-after-seconds; skipping it." \
        | tee "$RESULT_DIR/aiperf_profile_flag_note.txt"
fi
if [ -n "$AIPERF_MAX_OSL" ] && [ "$AIPERF_MAX_OSL" != "none" ]; then
    if grep -q -- "--trace-max-osl" <<<"$AIPERF_PROFILE_HELP"; then
        REPLAY_CMD+=" --trace-max-osl $AIPERF_MAX_OSL"
    elif grep -q -- "--synthesis-max-osl" <<<"$AIPERF_PROFILE_HELP"; then
        REPLAY_CMD+=" --synthesis-max-osl $AIPERF_MAX_OSL"
    else
        echo "NOTE: current aiperf does not support max OSL capping; skipping AIPERF_MAX_OSL=$AIPERF_MAX_OSL." \
            | tee "$RESULT_DIR/aiperf_max_osl_flag_note.txt"
    fi
fi
if grep -q -- "--max-context-length" <<<"$AIPERF_PROFILE_HELP"; then
    REPLAY_CMD+=" --max-context-length $MAX_CONTEXT_LENGTH"
else
    echo "NOTE: current aiperf does not support --max-context-length; skipping MAX_CONTEXT_LENGTH=$MAX_CONTEXT_LENGTH." \
        | tee "$RESULT_DIR/aiperf_max_context_flag_note.txt"
fi
REPLAY_CMD+=" --num-dataset-entries $NUM_DATASET_ENTRIES"
REPLAY_CMD+=" --slice-duration $SLICE_DURATION"
REPLAY_CMD+=" --output-artifact-dir $RESULT_DIR/aiperf_artifacts"
if [ "$DURATION" -lt 900 ] || [ "${AIPERF_UNSAFE_OVERRIDE:-false}" = "true" ]; then
    REPLAY_CMD+=" --unsafe-override"
fi
if [ -n "$HF_WEKA_DATASET" ]; then
    REPLAY_CMD+=" --hf-weka-dataset $HF_WEKA_DATASET"
else
    REPLAY_CMD+=" --public-dataset $PUBLIC_DATASET"
fi

run_agentic_replay_and_write_outputs "$RESULT_DIR"
