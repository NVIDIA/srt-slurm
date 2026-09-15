# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The in-package agent skill: one document teaching a coding agent how to drive srtctl.

``srtctl skill --target claude|codex|cursor`` installs it where that agent looks
for project skills, so a checkout can be handed to an agent without pasting
instructions. The text ships with the package (``SKILL.md`` beside this file)
and is the same for every target; only the destination path and, for Cursor,
the frontmatter differ.
"""

from __future__ import annotations

from pathlib import Path

SKILL_NAME = "srtctl"
SKILL_PATH = Path(__file__).with_name("SKILL.md")

# target -> path of the installed file, relative to the project root
SKILL_DESTINATIONS: dict[str, Path] = {
    "claude": Path(".claude") / "skills" / SKILL_NAME / "SKILL.md",
    "codex": Path(".codex") / "skills" / SKILL_NAME / "SKILL.md",
    "cursor": Path(".cursor") / "rules" / f"{SKILL_NAME}.mdc",
}


def skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def render_skill(target: str) -> str:
    """The skill as the target agent expects it (Cursor rules carry their own frontmatter)."""
    text = skill_text()
    if target == "cursor":
        body = text.split("---", 2)[2].lstrip("\n") if text.startswith("---") else text
        return (
            "---\ndescription: How to author, validate, submit, and read back srtctl jobs\nalwaysApply: false\n---\n\n"
            + body
        )
    return text


def install_skill(target: str, root: Path) -> Path:
    """Write the skill for ``target`` under ``root``; returns the written path."""
    if target not in SKILL_DESTINATIONS:
        raise ValueError(f"unknown skill target {target!r}; choose one of {', '.join(sorted(SKILL_DESTINATIONS))}")
    destination = root / SKILL_DESTINATIONS[target]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_skill(target), encoding="utf-8")
    return destination
