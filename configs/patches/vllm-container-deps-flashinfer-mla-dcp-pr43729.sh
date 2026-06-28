#!/bin/bash
# Apply vLLM PR #43729 to nightly-jun27 (68ee830) before any worker starts.

set -euo pipefail

bash /configs/patches/vllm-container-deps.sh
python3 /configs/patches/vllm_flashinfer_mla_dcp_pr43729.py
