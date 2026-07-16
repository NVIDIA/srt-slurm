#!/usr/bin/env bash
set -euo pipefail

# Backport vLLM commit 8b00f4123776a47a6d8e315242ee5f0dd0b817cf.
# The experiment pins nightly-94c0, so accept only that exact source or the
# exact post-backport result. This prevents a future layout change from being
# silently treated as the same controlled experiment.
PYTHON="${PYTHON:-python3}"
SOURCE="${VLLM_INDEX_TOPK_FILE:-$("${PYTHON}" -c '
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm")
assert spec and spec.submodule_search_locations
print(Path(next(iter(spec.submodule_search_locations))) / "models/minimax_m3/common/ops/index_topk.py")
')}"

"${PYTHON}" - "${SOURCE}" <<'PY'
from __future__ import annotations

import hashlib
import os
import stat
import sys
import tempfile
from pathlib import Path


source = Path(sys.argv[1])
original_sha256 = "1ca8557aed4260d2f4a742b9eda81d6f774e49d4b016ccc555dde40976c5a1a0"
patched_sha256 = "05ea8242fad53395207ad45591f6f5e07bbb74131d0282fe8fae7e9968ef1fa0"

if not source.is_file():
    raise SystemExit(f"MiniMax-M3 indexer source not found: {source}")

contents = source.read_bytes()
source_sha256 = hashlib.sha256(contents).hexdigest()
if source_sha256 == patched_sha256:
    print(f"MiniMax-M3 indexer grid patch already applied: {source}")
    raise SystemExit(0)
if source_sha256 != original_sha256:
    raise SystemExit(
        f"Refusing unexpected MiniMax-M3 indexer source: {source}, "
        f"sha256={source_sha256}"
    )

for old, new in (
    (b"    TARGET_GRID = 512\n", b"    TARGET_GRID = 4096\n"),
    (b"    TOPK_TARGET_GRID = 64\n", b"    TOPK_TARGET_GRID = 512\n"),
):
    if contents.count(old) != 1:
        raise SystemExit(f"Expected exactly one occurrence of {old!r} in {source}")
    contents = contents.replace(old, new, 1)

if hashlib.sha256(contents).hexdigest() != patched_sha256:
    raise SystemExit(f"Patched MiniMax-M3 indexer hash mismatch: {source}")

compile(contents, str(source), "exec")
source_stat = source.stat()
fd, temporary_path = tempfile.mkstemp(prefix=f".{source.name}.", dir=source.parent)
try:
    with os.fdopen(fd, "wb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary_path, stat.S_IMODE(source_stat.st_mode))
    if os.geteuid() == 0:
        os.chown(temporary_path, source_stat.st_uid, source_stat.st_gid)
    os.replace(temporary_path, source)
finally:
    if os.path.exists(temporary_path):
        os.unlink(temporary_path)

print(f"MiniMax-M3 indexer grid patch target: {source}")
print(f"MiniMax-M3 indexer grid patch original sha256: {original_sha256}")
print(f"MiniMax-M3 indexer grid patch verified sha256: {patched_sha256}")
print("MiniMax-M3 indexer grid patch verified: TARGET_GRID=4096, TOPK_TARGET_GRID=512")
PY
