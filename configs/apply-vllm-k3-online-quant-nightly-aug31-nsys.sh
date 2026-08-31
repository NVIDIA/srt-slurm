#!/usr/bin/env bash
set -euo pipefail

bash /configs/apply-vllm-k3-online-quant-nightly-aug31.sh
bash /configs/patches/install-nsys-cli.sh
