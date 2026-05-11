# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for git state snapshots."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from srtctl.cli.submit import submit_single
from srtctl.core.git_state import GIT_STATE_FILENAME, GitSnapshotSource, write_git_state_snapshot


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)


def _init_repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.email", "srtctl@example.com")
    _git(path, "config", "user.name", "srtctl")
    (path / "tracked.txt").write_text("base\n")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-m", "base commit")
    return path


def _dirty_repo(path: Path) -> None:
    (path / "tracked.txt").write_text("base\nunstaged\n")
    (path / "staged.txt").write_text("staged\n")
    _git(path, "add", "staged.txt")
    (path / "untracked.txt").write_text("untracked\n")


def test_write_git_state_snapshot_includes_commits_and_dirty_changes(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _dirty_repo(repo)

    output = tmp_path / "git_state.txt"
    assert write_git_state_snapshot(output, [GitSnapshotSource("extra_mount:/workspace/repo", repo)])

    text = output.read_text()
    assert "Repository:" in text
    assert "extra_mount:/workspace/repo" in text
    assert "base commit" in text
    assert "## Staged diff" in text
    assert "staged.txt" in text
    assert "## Unstaged diff" in text
    assert "+unstaged" in text
    assert "## Untracked file contents" in text
    assert "untracked.txt" in text
    assert "+untracked" in text


def test_write_git_state_snapshot_redacts_remote_credentials(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _git(repo, "remote", "add", "origin", "https://YAMY1234:ghp_secret_token@github.com/YAMY1234/repo.git")

    output = tmp_path / "git_state.txt"
    assert write_git_state_snapshot(output, [GitSnapshotSource("repo", repo)])

    text = output.read_text()
    assert "ghp_secret_token" not in text
    assert "https://YAMY1234:<redacted>@github.com/YAMY1234/repo.git" in text


def test_submit_writes_git_state_for_extra_mount(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "extra-repo")
    _dirty_repo(repo)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    container = tmp_path / "container.sqsh"
    container.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "name": "git-state-test",
                "model": {"path": str(model_dir), "container": str(container), "precision": "fp8"},
                "resources": {
                    "gpu_type": "h100",
                    "gpus_per_node": 8,
                    "prefill_nodes": 1,
                    "prefill_workers": 1,
                    "decode_nodes": 1,
                    "decode_workers": 1,
                },
                "benchmark": {"type": "manual"},
                "extra_mount": [f"{repo}:/workspace/extra-repo"],
            },
            sort_keys=False,
        )
    )

    mock_result = MagicMock()
    mock_result.stdout = "Submitted batch job 99999"
    original_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list | tuple) and cmd and cmd[0] == "sbatch":
            return mock_result
        return original_run(cmd, *args, **kwargs)

    with (
        patch("subprocess.run", side_effect=fake_run),
        patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
        patch("srtctl.cli.submit.create_job_record"),
        patch("srtctl.cli.submit._assert_preflight_passed"),
        patch("srtctl.cli.submit.validate_setup"),
    ):
        submit_single(config_path=cfg, output_dir=tmp_path)

    text = (tmp_path / "99999" / "git_state.txt").read_text()
    assert "extra_mount:/workspace/extra-repo" in text
    assert "base commit" in text
    assert "staged.txt" in text
    assert "untracked.txt" in text


def test_submit_skips_git_state_without_extra_mount(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    container = tmp_path / "container.sqsh"
    container.write_text("fake")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "name": "no-extra-mount-test",
                "model": {"path": str(model_dir), "container": str(container), "precision": "fp8"},
                "resources": {
                    "gpu_type": "h100",
                    "gpus_per_node": 8,
                    "prefill_nodes": 1,
                    "prefill_workers": 1,
                    "decode_nodes": 1,
                    "decode_workers": 1,
                },
                "benchmark": {"type": "manual"},
            },
            sort_keys=False,
        )
    )

    mock_result = MagicMock()
    mock_result.stdout = "Submitted batch job 99999"
    original_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list | tuple) and cmd and cmd[0] == "sbatch":
            return mock_result
        return original_run(cmd, *args, **kwargs)

    with (
        patch("subprocess.run", side_effect=fake_run),
        patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
        patch("srtctl.cli.submit.create_job_record"),
        patch("srtctl.cli.submit._assert_preflight_passed"),
        patch("srtctl.cli.submit.validate_setup"),
    ):
        submit_single(config_path=cfg, output_dir=tmp_path)

    assert not (tmp_path / "99999" / GIT_STATE_FILENAME).exists()
