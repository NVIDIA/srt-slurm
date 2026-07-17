#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/patch-minimax-m3-indexer-grid-8b00f41.sh"

PYTHON="${PYTHON:-python3}"
MOONCAKE_VERSION="${MOONCAKE_VERSION:-0.3.11.post1}"

if ! "${PYTHON}" -c 'import mooncake' >/dev/null 2>&1; then
  echo "Mooncake is absent from the worker image; installing the CUDA 13 transfer engine without dependency changes"
  "${PYTHON}" -m pip install --no-cache-dir --no-deps \
    "mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}"
fi

"${PYTHON}" - <<'PY'
from importlib.metadata import version

import mooncake

print(f"Mooncake import verified: {mooncake.__file__}")
for distribution in (
    "mooncake-transfer-engine-cuda13",
    "mooncake-transfer-engine",
):
    try:
        print(f"Mooncake distribution version: {version(distribution)} ({distribution})")
        break
    except Exception:
        pass
else:
    raise SystemExit("Mooncake imported but no transfer-engine distribution was found")
PY
