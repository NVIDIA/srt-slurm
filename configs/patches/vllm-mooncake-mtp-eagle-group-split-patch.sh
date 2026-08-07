#!/bin/bash
# Keep same-spec eagle and non-eagle physical KV groups separate in Mooncake's
# coordinator (see vllm_mooncake_mtp_eagle_group_split.py).
# Transitional c188b96 workaround only. vllm-internal e5827fabc1 +
# 89510b05dc are the official fixes; do not carry this patch into newer images.
set -euo pipefail

python3 /configs/patches/vllm_mooncake_mtp_eagle_group_split.py
