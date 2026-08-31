#!/usr/bin/env bash
set -euo pipefail

bash /configs/apply-vllm-k3-nightly-aug31-baseline.sh
bash /configs/patches/install-nsys-cli.sh
