# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One version, derived from the git tag: the runtime side and the release scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import srtctl
from srtctl.cli import submit as submit_cli
from srtctl.core.lockfile import LOCKFILE_VERSION
from srtctl.core.schema import SUPPORTED_SCHEMA_VERSIONS
from srtctl.version import FALLBACK_VERSION, package_version, version_info

SCRIPTS = Path(__file__).resolve().parents[1] / ".github" / "scripts"


def test_package_version_is_not_hardcoded() -> None:
    assert srtctl.__version__ == package_version()
    assert srtctl.__version__ != "0.3.0"
    assert srtctl.__version__  # never empty


def test_version_info_pairs_the_tool_with_its_protocols() -> None:
    info = version_info()
    assert info.version == package_version()
    assert info.recipe_schema_versions == tuple(SUPPORTED_SCHEMA_VERSIONS)
    assert info.lockfile_version == LOCKFILE_VERSION
    text = str(info)
    assert text.startswith(f"srtctl {info.version}")
    assert "recipe schema 1, 2" in text
    assert f"lockfile v{LOCKFILE_VERSION}" in text
    assert set(info.as_dict()) == {"version", "commit", "recipe_schema_versions", "lockfile_version"}


def test_cli_version_flag(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["srtctl", "--version"])
    with pytest.raises(SystemExit) as exc_info:
        submit_cli.main()
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == str(version_info())


def test_lockfile_context_stamps_the_version() -> None:
    from srtctl.core.lockfile import collect_slurm_context

    ctx = collect_slurm_context()
    assert ctx["srtctl_version"] == package_version()
    assert ctx["srtctl_version"] != FALLBACK_VERSION or "unknown" in ctx["srtctl_version"]


def _next(latest: str, title: str, body: str = "") -> tuple[str, str]:
    result = subprocess.run(
        ["bash", str(SCRIPTS / "next_version.sh"), latest, title, body],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip(), result.stderr.strip()


@pytest.mark.parametrize(
    ("latest", "title", "body", "expected", "kind"),
    [
        ("", "feat: first", "", "v1.0.0", "seed"),
        ("v1.0.103", "fix(sglang): pass the bootstrap port", "", "v1.0.104", "patch"),
        ("v1.0.103", "docs: 2.0 only", "", "v1.0.104", "patch"),
        ("v1.0.103", "feat(roles): add nodes: colocate", "", "v1.1.0", "minor"),
        ("v1.1.0", "feat: something", "", "v1.2.0", "minor"),
        ("v1.1.7", "feat!: drop the v1 layout", "", "v2.0.0", "major"),
        ("v1.1.7", "refactor(core)!: rename", "", "v2.0.0", "major"),
        ("v1.1.7", "fix: subtle", "Body\n\nBREAKING CHANGE: the lockfile format changed", "v2.0.0", "major"),
        ("v1.1.7.post2", "chore: noise", "", "v1.1.8", "patch"),
        ("v1.1.7-rc1", "chore: noise", "", "v1.1.8", "patch"),
    ],
)
def test_next_version_follows_the_pr_title(latest, title, body, expected, kind) -> None:
    assert _next(latest, title, body) == (expected, kind)
