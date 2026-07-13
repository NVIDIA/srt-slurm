#!/usr/bin/env bash
set -euo pipefail

# Package the Dynamo setup order without requiring post_install_script support
# from srtctl. Connector prefetch support is built into the selected image.
bash /configs/patches/vllm-container-deps.sh

if [ ! -f /srtctl-runtime/dynamo_wheels.py ]; then
    echo "ERROR: /srtctl-runtime/dynamo_wheels.py not found" >&2
    exit 1
fi
python3 /srtctl-runtime/dynamo_wheels.py install

bash /configs/patches/dynamo-multiconnector-nopr123-expiry3600-compat-patch.sh
