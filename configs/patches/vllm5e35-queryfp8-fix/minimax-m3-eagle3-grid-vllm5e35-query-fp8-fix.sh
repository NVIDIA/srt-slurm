#!/usr/bin/env bash
set -euo pipefail

# Preserve the established MiniMax-M3 EAGLE3/indexer setup first.
bash /configs/patches/minimax-m3-eagle3-draft-indexer-grid-8b00f41.sh

# Then apply the vLLM 5e35 CUDA-graph query_fp8 correctness fix.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "${script_dir}/apply-vllm5e35-minimax-m3-query-fp8-fix.sh"
