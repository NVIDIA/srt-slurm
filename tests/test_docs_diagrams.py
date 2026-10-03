# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagrams in Markdown are Mermaid, not hand-drawn boxes and arrows in a plain code block.

Directory trees (``├──``) and captured terminal output are not diagrams and are not flagged.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<lang>[\w-]*)[^\n]*\n(?P<body>.*?)^(?P=indent)```", re.MULTILINE | re.DOTALL
)
HAND_DRAWN = {
    "ASCII box": re.compile(r"\+-{3,}\+"),
    "vertical arrow": re.compile(r"^[\s|+]*\bv\b[\s|+v]*$"),
    "horizontal arrow": re.compile(r"(?:^|\s)-{2,}>|[─━]{2,}[>▶►]|[▶►▼▲◀→←↓↑]"),
    "connected boxes": re.compile(r"[┐┘│]─+[│┌└]|[│┤]─{2,}[│├]"),
}


def _markdown_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.md"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [REPO_ROOT / line for line in out.splitlines() if (REPO_ROOT / line).is_file()]


def _hand_drawn_blocks() -> list[str]:
    found = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in FENCE.finditer(text):
            if match["lang"] not in ("", "text", "txt", "plain", "plaintext"):
                continue
            for line in match["body"].splitlines():
                kind = next((name for name, pattern in HAND_DRAWN.items() if pattern.search(line)), None)
                if kind:
                    start = text[: match.start()].count("\n") + 1
                    found.append(f"{path.relative_to(REPO_ROOT)}:{start}: {kind}: {line.strip()[:80]}")
                    break
    return found


def test_markdown_diagrams_are_mermaid() -> None:
    found = _hand_drawn_blocks()
    assert not found, "Redraw these as a ```mermaid block (flowchart or sequenceDiagram):\n" + "\n".join(found)
