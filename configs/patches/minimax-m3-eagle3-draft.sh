#!/usr/bin/env bash
set -euo pipefail

DRAFT_MODEL="${MINIMAX_M3_EAGLE3_DRAFT_MODEL:-Inferact/MiniMax-M3-EAGLE3}"
DRAFT_LOCAL_DIR="${MINIMAX_M3_EAGLE3_DRAFT_LOCAL_DIR:-}"

echo "=== MiniMax-M3 setup: warming EAGLE3 draft model ${DRAFT_MODEL} ==="
echo "HF_HOME=${HF_HOME:-unset}"
if [[ -n "${DRAFT_LOCAL_DIR}" ]]; then
    echo "MINIMAX_M3_EAGLE3_DRAFT_LOCAL_DIR=${DRAFT_LOCAL_DIR}"
fi

download_draft_model() {
    if command -v hf >/dev/null 2>&1; then
        hf download "$@"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$@"
    else
        echo "Neither hf nor huggingface-cli is available in the container" >&2
        exit 1
    fi
}

download_with_optional_local_dir() {
    if [[ -n "${DRAFT_LOCAL_DIR}" ]]; then
        mkdir -p "${DRAFT_LOCAL_DIR}"
        if [[ -s "${DRAFT_LOCAL_DIR}/config.json" ]]; then
            echo "Draft model already present at ${DRAFT_LOCAL_DIR}"
            return
        fi
        download_draft_model "${DRAFT_MODEL}" --local-dir "${DRAFT_LOCAL_DIR}"
        return
    fi

    download_draft_model "${DRAFT_MODEL}"
}

if [[ -n "${DRAFT_LOCAL_DIR}" ]] && command -v flock >/dev/null 2>&1; then
    mkdir -p "${DRAFT_LOCAL_DIR}"
    exec 9>"${DRAFT_LOCAL_DIR}.lock"
    flock -x 9
    download_with_optional_local_dir
else
    download_with_optional_local_dir
fi

echo "=== MiniMax-M3 EAGLE3 draft setup complete ==="
