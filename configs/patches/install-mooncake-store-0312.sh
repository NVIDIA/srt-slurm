#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
MOONCAKE_VERSION="${MOONCAKE_VERSION:-0.3.12.post1}"

CURRENT_MOONCAKE_VERSION="$("${PYTHON}" - <<'PY' 2>/dev/null || true
from importlib.metadata import PackageNotFoundError, version

for distribution in ("mooncake-transfer-engine-cuda13", "mooncake-transfer-engine"):
    try:
        print(version(distribution))
        break
    except PackageNotFoundError:
        pass
PY
)"

if [[ "${CURRENT_MOONCAKE_VERSION}" != "${MOONCAKE_VERSION}" ]]; then
  echo "Installing Mooncake CUDA 13 transfer engine ${MOONCAKE_VERSION}; found ${CURRENT_MOONCAKE_VERSION:-none}"
  "${PYTHON}" -m pip install --no-cache-dir --no-deps --force-reinstall \
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
