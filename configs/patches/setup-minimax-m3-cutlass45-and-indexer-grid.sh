#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/setup-minimax-m3-cutlass45.sh"
"${SCRIPT_DIR}/patch-minimax-m3-indexer-grid-8b00f41.sh"
