#!/bin/bash
set -euo pipefail

bash /configs/patches/vllm-container-deps.sh
python3 /configs/patches/vllm_revert_pr41015_fp4_cvt.py
