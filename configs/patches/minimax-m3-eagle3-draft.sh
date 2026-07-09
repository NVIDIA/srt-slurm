#!/usr/bin/env bash
set -euo pipefail

DRAFT_MODEL="${MINIMAX_M3_EAGLE3_DRAFT_MODEL:-Inferact/MiniMax-M3-EAGLE3}"

echo "=== MiniMax-M3 setup: warming EAGLE3 draft model ${DRAFT_MODEL} ==="
echo "HF_HOME=${HF_HOME:-unset}"

if command -v hf >/dev/null 2>&1; then
    hf download "${DRAFT_MODEL}"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${DRAFT_MODEL}"
else
    echo "Neither hf nor huggingface-cli is available in the container" >&2
    exit 1
fi

echo "=== MiniMax-M3 EAGLE3 draft setup complete ==="
