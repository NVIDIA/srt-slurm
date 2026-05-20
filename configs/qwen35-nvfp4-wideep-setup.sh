#!/bin/bash
# Bump dynamo prefill warmup timeout 1800s -> 7200s.
# Multi-prefill (4P, 8P) cold-start exceeds the upstream 1800s default on first run
# (flashinfer autotune / DeepEP compile cache). 7200s is a safety margin; subsequent
# runs with primed caches complete in <600s.
# Idempotent: re-runs detect sentinel and skip.
set -euo pipefail
set -x

DYNAMO_INIT_LLM="/usr/local/lib/python3.12/dist-packages/dynamo/sglang/init_llm.py"

if [[ ! -f "${DYNAMO_INIT_LLM}" ]]; then
    echo "FATAL: ${DYNAMO_INIT_LLM} not found in container" >&2
    exit 1
fi

python3 - "${DYNAMO_INIT_LLM}" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
sentinel = "WARMUP_TIMEOUT_BUMP_APPLIED"
if sentinel in src:
    print(f"already patched: {p}")
    sys.exit(0)
needle = "await asyncio.wait_for(_do_warmup(), timeout=1800)"
replacement = "await asyncio.wait_for(_do_warmup(), timeout=7200)  # WARMUP_TIMEOUT_BUMP_APPLIED: 1800 -> 7200 for multi-prefill"
if needle not in src:
    print(f"FATAL: needle not found in {p}", file=sys.stderr)
    sys.exit(2)
new = src.replace(needle, replacement, 1)
p.write_text(new)
pycache_dir = p.parent / '__pycache__'
if pycache_dir.exists():
    for pyc in pycache_dir.glob('init_llm*.pyc'):
        pyc.unlink()
        print(f"removed stale {pyc}")
print(f"PATCHED {p}: warmup timeout bumped 1800s -> 7200s")
PY

echo "=== Warmup timeout bump complete ==="
