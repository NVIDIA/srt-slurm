# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent guidance (CLAUDE.md, AGENTS.md, REVIEW.md, .agents/skills) stays true to the code.

Every inline-code file path must exist, and every inline-code Python symbol must
still be defined somewhere in the repository. A failure names the file and the
stale reference: update the guidance to match the code (or delete the line).
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "srtctl"

# Paths that name user or job files, not files in this repository.
RUNTIME_PATHS = {"logs/", "perf_dashboard_bundle/", "srtslurm.yaml", "sbatch_script.sh"}
# Names from Python, third-party packages, or upstream engines.
EXTERNAL_NAMES = {"TypedDict", "RequestException", "MPI_Abort", "get_open_port", "isinstance"}

_FENCE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)
_INLINE = re.compile(r"`([^`\n]+)`")
_PATHLIKE = re.compile(r"^[\w.\-]+(/[\w.\-]*)+$|^[\w\-]+\.(py|md|yaml|yml|sh|toml)$")
_SYMBOL = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*(\(.*\))?$")


def agent_files() -> list[Path]:
    files = [REPO_ROOT / "CLAUDE.md", REPO_ROOT / "AGENTS.md", REPO_ROOT / "REVIEW.md"]
    files += sorted(REPO_ROOT.glob(".agents/skills/*/SKILL.md"))
    files += sorted(p for p in (REPO_ROOT / "src").rglob("AGENTS.md"))
    files += sorted((REPO_ROOT / "tests").rglob("AGENTS.md"))
    files += [REPO_ROOT / "docs" / "AGENTS.md"]
    return [f for f in files if f.exists()]


def inline_code(doc: Path) -> list[str]:
    return _INLINE.findall(_FENCE.sub("", doc.read_text()))


_DEFINITION_PATTERNS = (
    re.compile(r"(?:def|class)\s+(\w+)"),
    re.compile(r"^\s*(\w+)\s*[:=]", re.MULTILINE),
    re.compile(r"[\"'/{$](\w+)"),
)


@cache
def _source_files() -> tuple[Path, ...]:
    suffixes = {".py", ".sh", ".j2", ".yaml"}
    files = [p for p in (REPO_ROOT / "src").rglob("*") if p.suffix in suffixes]
    return (*files, *(REPO_ROOT / "tests").glob("*.py"), REPO_ROOT / "Makefile")


@cache
def _definitions() -> frozenset[str]:
    """Every name defined, assigned, or quoted in the sources and Makefile, plus module names."""
    names: set[str] = set()
    for path in _source_files():
        names.add(path.stem)
        text = path.read_text(errors="replace")
        for pattern in _DEFINITION_PATTERNS:
            names.update(pattern.findall(text))
    return frozenset(names)


def _path_exists(ref: str, doc: Path) -> bool:
    if "/" not in ref:
        return any(p.name == ref for p in _source_files()) or (REPO_ROOT / ref).exists() or (doc.parent / ref).exists()
    candidates = [REPO_ROOT / ref, SRC / ref, doc.parent / ref, REPO_ROOT / "tests" / ref]
    return any(c.exists() for c in candidates)


def _is_symbol(ref: str) -> bool:
    if not _SYMBOL.match(ref) or re.match(r"^_[A-Z_]+$", ref):
        return False
    head = ref.split("(", 1)[0]
    last = head.rsplit(".", 1)[-1]
    # Only names that look like code: snake_case with an underscore, CamelCase, or a call.
    return "_" in last or "(" in ref or bool(re.match(r"^[A-Z][a-z]+[A-Z]", last))


@pytest.mark.parametrize("doc", agent_files(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_referenced_paths_exist(doc: Path):
    stale = [
        ref
        for ref in inline_code(doc)
        if _PATHLIKE.match(ref) and ref not in RUNTIME_PATHS and not _path_exists(ref, doc)
    ]
    assert not stale, f"{doc.relative_to(REPO_ROOT)} references paths that do not exist: {sorted(set(stale))}"


@pytest.mark.parametrize("doc", agent_files(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_referenced_symbols_exist(doc: Path):
    defined = _definitions()
    stale = []
    for ref in inline_code(doc):
        if _PATHLIKE.match(ref) or not _is_symbol(ref):
            continue
        last = ref.split("(", 1)[0].rsplit(".", 1)[-1]
        if last not in defined and last not in EXTERNAL_NAMES:
            stale.append(ref)
    assert not stale, (
        f"{doc.relative_to(REPO_ROOT)} references names no longer defined in src/, tests/ or the Makefile: "
        f"{sorted(set(stale))}"
    )


def test_root_guide_fits_in_the_auto_loaded_budget():
    size = len((REPO_ROOT / "CLAUDE.md").read_bytes())
    assert size < 16 * 1024, (
        f"CLAUDE.md is {size} bytes; agents load only the first 16 KiB. Move subsystem detail into the "
        "AGENTS.md of the directory it describes or link to docs/."
    )


def test_every_agent_guide_is_listed_in_the_root_index():
    root = (REPO_ROOT / "CLAUDE.md").read_text()
    nested = [p for p in agent_files() if p.name == "AGENTS.md" and p.parent != REPO_ROOT]
    missing = [p.relative_to(REPO_ROOT).as_posix() for p in nested if p.relative_to(REPO_ROOT).as_posix() not in root]
    assert not missing, f"Add these to the 'Where to look' table in CLAUDE.md: {missing}"
