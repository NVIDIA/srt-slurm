# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every example recipe's srun launch plan matches its golden file.

A failure here means the commands, environment, mounts or placement a recipe
launches changed. If the change is intended, run ``make snapshots`` and commit
the regenerated ``tests/snapshots/launch/`` files so the diff shows reviewers
exactly what moved on the cluster.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

from tests.launch_snapshots import REPO_ROOT, SNAPSHOT_DIR, example_recipes, render_launch_plan, snapshot_path

RECIPES = example_recipes()


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_launch_plan_matches_snapshot(recipe: Path):
    golden = snapshot_path(recipe)
    actual = render_launch_plan(recipe)
    assert golden.exists(), f"No launch snapshot for {recipe.name}; run `make snapshots` and commit {golden.name}"
    expected = golden.read_text()
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True), actual.splitlines(keepends=True), "snapshot", "actual", n=2
            )
        )
        pytest.fail(
            f"Launch plan for {recipe.relative_to(REPO_ROOT)} changed. If intended, run `make snapshots` "
            f"and commit the result.\n{diff[:6000]}"
        )


def test_every_runnable_example_runs_to_completion():
    for recipe in RECIPES:
        header = snapshot_path(recipe).read_text().splitlines()[:2]
        assert header[1] == "# exit_code: 0", f"{recipe.relative_to(REPO_ROOT)} fails under the mock orchestrator"


def test_no_orphan_snapshots():
    expected = {snapshot_path(r) for r in RECIPES}
    orphans = sorted(p.name for p in SNAPSHOT_DIR.glob("*.txt") if p not in expected)
    assert not orphans, f"Snapshots without a recipe: {orphans}; run `make snapshots`"
