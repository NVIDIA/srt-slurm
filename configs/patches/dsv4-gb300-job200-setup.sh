#!/usr/bin/env bash
set -euo pipefail

# Package the setup order used by GB300 validation job 200 without requiring
# post_install_script support from srtctl.
bash /configs/patches/vllm-container-deps.sh
python3 /configs/patches/vllm_connector_prefetch_pass.py

if [ ! -f /srtctl-runtime/dynamo_wheels.py ]; then
    echo "ERROR: /srtctl-runtime/dynamo_wheels.py not found" >&2
    exit 1
fi
python3 /srtctl-runtime/dynamo_wheels.py install

bash /configs/patches/dynamo-multiconnector-nopr123-expiry3600-compat-patch.sh
