#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/patch-minimax-m3-indexer-grid-8b00f41.sh"

PYTHON="${PYTHON:-python3}"
LMCACHE_VERSION="${LMCACHE_VERSION:-0.5.1}"

if ! "${PYTHON}" -c 'import lmcache' >/dev/null 2>&1; then
  echo "LMCache is absent from the worker image; installing lmcache==${LMCACHE_VERSION} without dependency changes"
  "${PYTHON}" -m pip install --no-cache-dir --no-deps "lmcache==${LMCACHE_VERSION}"
fi

"${PYTHON}" - <<'PY'
from importlib.metadata import version

import lmcache

print(f"LMCache import verified: {lmcache.__file__}")
print(f"LMCache distribution version: {version('lmcache')}")
PY
