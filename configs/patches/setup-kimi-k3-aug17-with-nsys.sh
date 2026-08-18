#!/usr/bin/env bash
# Apply the Aug17 Kimi-K3 runtime patch and install the latest Nsight Systems CLI.

set -euo pipefail

bash /configs/patches/apply-vllm-kimi-k3-aug17.sh
bash /configs/patches/install-nsys-cli.sh
