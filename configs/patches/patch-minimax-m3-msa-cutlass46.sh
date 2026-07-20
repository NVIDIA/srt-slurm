#!/usr/bin/env bash
set -euo pipefail

# vLLM issue #49005: CUTLASS DSL 4.6 moved ThrMma out of cute.core.
PYTHON="${PYTHON:-python3}"
MSA_ROOT="${VLLM_MSA_CUTE_ROOT:-$("${PYTHON}" -c '
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm")
assert spec and spec.submodule_search_locations
print(Path(next(iter(spec.submodule_search_locations))) / "third_party/fmha_sm100/cute")
')}"

"${PYTHON}" - "${MSA_ROOT}" <<'PY'
from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path


root = Path(sys.argv[1])
old = b"cute.core.ThrMma"
new = b"cute.ThrMma"
expected_occurrences = 7

if not root.is_dir():
    raise SystemExit(f"MiniMax MSA CuteDSL source not found: {root}")

sources = sorted(root.rglob("*.py"))
old_count = sum(path.read_bytes().count(old) for path in sources)
new_count = sum(path.read_bytes().count(new) for path in sources)
if old_count == 0 and new_count == expected_occurrences:
    print(f"MiniMax MSA CUTLASS 4.6 compatibility patch already applied: {root}")
    raise SystemExit(0)
if old_count != expected_occurrences or new_count != 0:
    raise SystemExit(
        "Refusing unexpected MiniMax MSA ThrMma source state: "
        f"root={root}, old_count={old_count}, new_count={new_count}"
    )

patched_files = []
for source in sources:
    contents = source.read_bytes()
    if old not in contents:
        continue
    patched = contents.replace(old, new)
    compile(patched, str(source), "exec")
    source_stat = source.stat()
    fd, temporary_path = tempfile.mkstemp(prefix=f".{source.name}.", dir=source.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(patched)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, stat.S_IMODE(source_stat.st_mode))
        if os.geteuid() == 0:
            os.chown(temporary_path, source_stat.st_uid, source_stat.st_gid)
        os.replace(temporary_path, source)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    patched_files.append(source)

final_old_count = sum(path.read_bytes().count(old) for path in sources)
final_new_count = sum(path.read_bytes().count(new) for path in sources)
if final_old_count != 0 or final_new_count != expected_occurrences:
    raise SystemExit(
        "MiniMax MSA CUTLASS 4.6 patch verification failed: "
        f"old_count={final_old_count}, new_count={final_new_count}"
    )

print(f"MiniMax MSA CUTLASS 4.6 compatibility root: {root}")
print(f"MiniMax MSA CUTLASS 4.6 compatibility files patched: {len(patched_files)}")
print(f"MiniMax MSA CUTLASS 4.6 compatibility replacements: {final_new_count}")
PY
