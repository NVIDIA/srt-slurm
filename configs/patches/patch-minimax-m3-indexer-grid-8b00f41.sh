#!/usr/bin/env bash
set -euo pipefail

# Backport vLLM commit 8b00f4123776a47a6d8e315242ee5f0dd0b817cf.
# This is deliberately source-specific and fails if the installed vLLM no
# longer contains either the original or already-patched constants.
python - <<'PY'
from __future__ import annotations

from pathlib import Path

import vllm


source = (
    Path(vllm.__file__).resolve().parent
    / "models"
    / "minimax_m3"
    / "common"
    / "ops"
    / "index_topk.py"
)

if not source.is_file():
    raise SystemExit(f"MiniMax-M3 indexer source not found: {source}")

text = source.read_text()
replacements = (
    ("TARGET_GRID = 512", "TARGET_GRID = 4096"),
    ("TOPK_TARGET_GRID = 64", "TOPK_TARGET_GRID = 512"),
)

states: list[str] = []
for old, new in replacements:
    old_count = text.count(old)
    new_count = text.count(new)
    if old_count == 1 and new_count == 0:
        text = text.replace(old, new, 1)
        states.append(f"patched {old!r} -> {new!r}")
    elif old_count == 0 and new_count == 1:
        states.append(f"already patched: {new!r}")
    else:
        raise SystemExit(
            "Unexpected MiniMax-M3 indexer source while applying 8b00f41: "
            f"{old!r} count={old_count}, {new!r} count={new_count}, file={source}"
        )

source.write_text(text)
compiled = compile(text, str(source), "exec")
del compiled

verified = source.read_text()
for old, new in replacements:
    if old in verified or verified.count(new) != 1:
        raise SystemExit(f"Post-patch verification failed for {new!r} in {source}")

print(f"MiniMax-M3 indexer grid patch target: {source}")
for state in states:
    print(f"MiniMax-M3 indexer grid patch: {state}")
print("MiniMax-M3 indexer grid patch verified: TARGET_GRID=4096, TOPK_TARGET_GRID=512")
PY
