#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# lm-eval accuracy evaluation using InferenceX benchmark_lib
# Expects: endpoint [infmax_workspace]

set -e

ENDPOINT=$1
INFMAX_WORKSPACE=${2:-/infmax-workspace}

# Extract HOST and PORT from endpoint (e.g., http://localhost:8000)
HOST=$(echo "$ENDPOINT" | sed -E 's|https?://||; s|:.*||')
PORT=$(echo "$ENDPOINT" | sed -E 's|.*:([0-9]+).*|\1|')

echo "lm-eval Config: endpoint=${ENDPOINT}; host=${HOST}; port=${PORT}; workspace=${INFMAX_WORKSPACE}"

# Serving images commonly make their system Python environment read-only.  The
# InferenceX eval harness installs a pinned lm-eval runtime before executing, so
# give it a job-local writable environment while retaining the serving image's
# already-installed framework dependencies.  Prepending the venv to PATH keeps
# benchmark_lib.sh's existing `python3 -m ...` interface unchanged.
LM_EVAL_RUNTIME_DIR="${SRTCTL_LM_EVAL_RUNTIME_DIR:-${TMPDIR:-/tmp}/srtctl-lm-eval-${SLURM_JOB_ID:-$$}}"
LM_EVAL_VENV="${LM_EVAL_RUNTIME_DIR}/venv"
LM_EVAL_CACHE_DIR="${SRTCTL_LM_EVAL_CACHE_DIR:-${LM_EVAL_RUNTIME_DIR}/cache}"
LM_EVAL_RESULT_DIR="${SRTCTL_LM_EVAL_RESULT_DIR:-}"

# Serving containers often expose the host user's home and shared model cache
# read-only. lm-eval still needs writable Hugging Face and XDG caches for task
# datasets, even when model weights are already present. Keep all client-side
# downloads inside the disposable job-local runtime instead of mutating the
# serving cache or relying on $HOME/.cache.
export XDG_CACHE_HOME="${LM_EVAL_CACHE_DIR}/xdg"
export HF_HOME="${LM_EVAL_CACHE_DIR}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

# The Slurm step can receive a reduced PATH even when the serving image keeps
# its ROCm/PyTorch environment under /opt/venv.  lm-eval imports torch for its
# API client helpers, so preserve the framework environment's site-packages in
# the disposable eval venv instead of downloading a second (and potentially
# incompatible) torch build.  Prefer an explicit override, then the standard
# SGLang image environment, and finally the Python visible on PATH.
LM_EVAL_FRAMEWORK_PYTHON="${SRTCTL_LM_EVAL_FRAMEWORK_PYTHON:-}"
if [[ -z "${LM_EVAL_FRAMEWORK_PYTHON}" ]]; then
    for candidate in /opt/venv/bin/python3 "$(command -v python3)"; do
        if [[ -x "${candidate}" ]] && "${candidate}" -c 'import torch' >/dev/null 2>&1; then
            LM_EVAL_FRAMEWORK_PYTHON="${candidate}"
            break
        fi
    done
fi
if [[ -z "${LM_EVAL_FRAMEWORK_PYTHON}" ]]; then
    echo "ERROR: no serving-image Python with torch is available for lm-eval" >&2
    exit 1
fi
LM_EVAL_FRAMEWORK_SITE_PACKAGES="$("${LM_EVAL_FRAMEWORK_PYTHON}" - <<'PY'
import sys

print("\n".join(path for path in sys.path if "site-packages" in path))
PY
)"

if [[ ! -x "${LM_EVAL_VENV}/bin/python3" ]]; then
    rm -rf "${LM_EVAL_VENV}"
    mkdir -p "${LM_EVAL_RUNTIME_DIR}"
    python3 -m venv --system-site-packages "${LM_EVAL_VENV}"
fi
LM_EVAL_SITE_PACKAGES="$("${LM_EVAL_VENV}/bin/python3" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "${LM_EVAL_FRAMEWORK_SITE_PACKAGES}" > "${LM_EVAL_SITE_PACKAGES}/srtctl-framework.pth"
export PATH="${LM_EVAL_VENV}/bin:${PATH}"
hash -r

if [[ "$(python3 -c 'import sys; print(sys.prefix)')" != "${LM_EVAL_VENV}" ]]; then
    echo "ERROR: failed to activate writable lm-eval runtime at ${LM_EVAL_VENV}" >&2
    exit 1
fi
echo "lm-eval Runtime: python=$(command -v python3); prefix=${LM_EVAL_VENV}"
python3 -c 'import torch' || {
    echo "ERROR: job-local lm-eval runtime cannot import serving-image torch" >&2
    exit 1
}

# Some serving images seed virtual environments with a pip version old enough
# that it does not recognize --break-system-packages.  InferenceX's shared eval
# installer passes that option, so make the job-local pip understand it before
# handing control to benchmark_lib.sh.  This only mutates the disposable venv.
if ! python3 -m pip install --help 2>/dev/null | grep -q -- '--break-system-packages'; then
    echo "lm-eval Runtime: upgrading job-local pip for --break-system-packages support"
    python3 -m pip install --upgrade 'pip>=23.0'
fi

# Auto-discover the served model name from /v1/models if MODEL_NAME is not set.
# This ensures we use the exact name the server recognizes, regardless of what
# $MODEL (the HuggingFace ID from the workflow) is set to.
if [[ -z "${MODEL_NAME:-}" ]]; then
    DISCOVERED_MODEL=$(curl -sf "${ENDPOINT}/v1/models" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || true)
    if [[ -n "$DISCOVERED_MODEL" ]]; then
        export MODEL_NAME="$DISCOVERED_MODEL"
        echo "Auto-discovered MODEL_NAME from /v1/models: ${MODEL_NAME}"
    else
        echo "WARNING: Could not discover model name from /v1/models, using MODEL_NAME=${MODEL_NAME:-$MODEL}"
    fi
else
    echo "Using MODEL_NAME from environment: ${MODEL_NAME}"
fi

# cd to workspace so that relative paths (e.g., utils/evals/*.yaml) resolve
cd "${INFMAX_WORKSPACE}"

# Source the InferenceX benchmark library
source "${INFMAX_WORKSPACE}/benchmarks/benchmark_lib.sh"

# Run lm-eval via benchmark_lib
# EVAL_CONC is set by the InferenceX workflow (median of conc list).
# benchmark_lib reads concurrency from EVAL_CONCURRENT_REQUESTS env var.
export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC:-${EVAL_CONCURRENT_REQUESTS:-256}}"
echo "Running lm-eval with concurrent-requests=${EVAL_CONCURRENT_REQUESTS}..."
eval_rc=0
run_eval --framework lm-eval --port "$PORT" || eval_rc=$?

# Derive metadata env vars that append_lm_eval_summary needs but do_sweep.py
# does not pass directly (it passes PREFILL_TP/EP/etc, not TP/EP_SIZE/CONC).
export IS_MULTINODE="${IS_MULTINODE:-true}"
export TP="${TP:-${PREFILL_TP:-1}}"
export CONC="${CONC:-${EVAL_CONC:-${EVAL_CONCURRENT_REQUESTS:-1}}}"
export EP_SIZE="${EP_SIZE:-${PREFILL_EP:-1}}"
export DP_ATTENTION="${DP_ATTENTION:-${PREFILL_DP_ATTN:-false}}"
# Remap srt-slurm's DP_ATTN names to InferenceX's DP_ATTENTION names
export PREFILL_DP_ATTENTION="${PREFILL_DP_ATTENTION:-${PREFILL_DP_ATTN:-${DP_ATTENTION:-false}}}"
export DECODE_DP_ATTENTION="${DECODE_DP_ATTENTION:-${DECODE_DP_ATTN:-${DP_ATTENTION:-false}}}"

# Generate the lm-eval summary
echo "Generating lm-eval summary..."
append_lm_eval_summary || true

# Copy eval artifacts to /logs/eval_results/
mkdir -p /logs/eval_results
echo "Copying eval artifacts to /logs/eval_results/..."
cp -v meta_env.json /logs/eval_results/ 2>/dev/null || true
cp -v results*.json /logs/eval_results/ 2>/dev/null || true
cp -v sample*.jsonl /logs/eval_results/ 2>/dev/null || true

# Integrations that maintain a separate result tree can request an additional
# copy without coupling this benchmark to a particular host mount layout. The
# default remains /logs/eval_results for existing srt-slurm consumers.
if [[ -n "${LM_EVAL_RESULT_DIR}" ]]; then
    mkdir -p "${LM_EVAL_RESULT_DIR}"
    echo "Copying eval artifacts to ${LM_EVAL_RESULT_DIR}/..."
    cp -v meta_env.json "${LM_EVAL_RESULT_DIR}/" 2>/dev/null || true
    cp -v results*.json "${LM_EVAL_RESULT_DIR}/" 2>/dev/null || true
    cp -v sample*.jsonl "${LM_EVAL_RESULT_DIR}/" 2>/dev/null || true
fi

if [[ "$eval_rc" -ne 0 ]]; then
    echo "lm-eval evaluation failed with exit code ${eval_rc}"
    exit "$eval_rc"
fi

echo "lm-eval evaluation complete"
