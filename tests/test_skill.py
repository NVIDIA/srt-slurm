# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The in-package agent skill and `srtctl skill`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from srtctl.cli import submit as submit_cli
from srtctl.skills import SKILL_DESTINATIONS, install_skill, render_skill, skill_text


def test_skill_ships_with_the_package_and_covers_the_workflow() -> None:
    text = skill_text()
    assert text.startswith("---\nname: srtctl\n")
    for needle in (
        "srtctl dry-run",
        "srtctl apply",
        "schema: 2",
        "roles:",
        "services:",
        "sweep_<job_id>.log",
        "srtctl-mcp",
    ):
        assert needle in text, needle


@pytest.mark.parametrize("target", sorted(SKILL_DESTINATIONS))
def test_install_writes_where_each_agent_looks(target: str, tmp_path: Path) -> None:
    written = install_skill(target, tmp_path)
    assert written == tmp_path / SKILL_DESTINATIONS[target]
    assert written.is_file()
    body = written.read_text()
    assert "# srtctl" in body
    if target == "cursor":
        assert body.startswith("---\ndescription:")
        assert "alwaysApply: false" in body
        assert "name: srtctl" not in body  # Cursor rules do not use the skill frontmatter
    else:
        assert body == skill_text()


def test_unknown_target_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown skill target"):
        install_skill("emacs", tmp_path)


def test_cli_installs_and_prints(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["srtctl", "skill", "--target", "claude", "--root", str(tmp_path)])
    submit_cli.main()
    assert (tmp_path / ".claude" / "skills" / "srtctl" / "SKILL.md").is_file()

    monkeypatch.setattr(sys, "argv", ["srtctl", "skill", "--target", "codex", "--print"])
    submit_cli.main()
    out = capsys.readouterr().out
    assert "# srtctl" in out
    assert render_skill("codex") in out
