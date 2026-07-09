#!/usr/bin/env bash
set -euo pipefail

FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.14}"
FLASHINFER_PATCH="/configs/patches/flashinfer-cutedsl-splitk-gemm.patch"

echo "=== MiniMax-M3 setup: FlashInfer ${FLASHINFER_VERSION} + CuTeDSL split-K GEMM ==="

python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache || true

python3 -m pip install --no-deps "flashinfer-python[cu13]==${FLASHINFER_VERSION}" \
    || { echo "FlashInfer ${FLASHINFER_VERSION} install failed" >&2; exit 1; }
python3 -m pip install --no-deps "flashinfer-cubin==${FLASHINFER_VERSION}" \
    --index-url https://flashinfer.ai/whl \
    || { echo "FlashInfer cubin ${FLASHINFER_VERSION} install failed" >&2; exit 1; }
python3 -m pip install --no-deps "flashinfer-jit-cache==${FLASHINFER_VERSION}+cu130" \
    --index-url https://flashinfer.ai/whl/cu130 \
    || { echo "FlashInfer JIT cache ${FLASHINFER_VERSION}+cu130 install failed" >&2; exit 1; }

if [[ ! -f "${FLASHINFER_PATCH}" ]]; then
    echo "FlashInfer patch not found: ${FLASHINFER_PATCH}" >&2
    exit 1
fi

if ! command -v patch >/dev/null 2>&1; then
    apt-get update -y
    apt-get install -y --no-install-recommends patch \
        || { echo "Failed to install patch(1)" >&2; exit 1; }
fi

SITE_PACKAGES=$(
    python3 - <<'PY'
import importlib.util

spec = importlib.util.find_spec("flashinfer")
if spec is None or spec.submodule_search_locations is None:
    raise SystemExit("could not locate installed flashinfer package")
print(spec.submodule_search_locations[0].rsplit("/", 1)[0])
PY
)

patch -p1 -d "${SITE_PACKAGES}" < "${FLASHINFER_PATCH}" \
    || { echo "FlashInfer CuTeDSL split-K GEMM patch failed" >&2; exit 1; }

echo "=== MiniMax-M3 FlashInfer setup complete ==="
