#!/bin/bash
# SRT patch: connector prefetch pass (defer-ahead) for external KV loads.
# Base container deps + the prefetch-pass patch. Inert unless the job also
# sets VLLM_CONNECTOR_PREFETCH_DEPTH > 0 (recommended: 8 on prefill workers).
set -euo pipefail

bash /configs/patches/vllm-container-deps.sh

python3 /configs/patches/vllm_connector_prefetch_pass.py
