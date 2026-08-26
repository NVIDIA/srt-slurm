# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for git state snapshots."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.cli.submit import submit_single
from srtctl.core.git_state import (
    GIT_STATE_FILENAME,
    GitCommandOutcome,
    GitCommandResult,
    GitSnapshotSource,
    _format_git_failure,
    _format_repo_snapshot,
    _run_git,
    write_git_state_snapshot,
)


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


def test_status_skips_untracked_files_and_submodules(tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    with patch("srtctl.core.git_state.subprocess.run", return_value=completed) as run:
        _format_repo_snapshot(tmp_path, ["repo"], [tmp_path])

    commands = [call.args[0] for call in run.call_args_list]
    status_command = next(command for command in commands if "status" in command)
    assert "--untracked-files=no" in status_command
    assert "--ignore-submodules=all" in status_command


def test_timeout_marker_includes_command_and_duration(tmp_path: Path) -> None:
    with patch(
        "srtctl.core.git_state.subprocess.run",
        side_effect=subprocess.TimeoutExpired(["git", "status"], 60),
    ):
        result = _run_git(tmp_path, ["status", "--short"], timeout_seconds=60)

    assert result.outcome == GitCommandOutcome.TIMEOUT
    assert _format_git_failure(result) == "<git command timed out after 60s: git status --short>\n"


def test_nonzero_marker_includes_exit_code_and_redacts_stderr(tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess(
        [],
        128,
        stdout="",
        stderr="fatal: https://user:secret@github.com/example/repo.git unavailable",
    )
    with patch("srtctl.core.git_state.subprocess.run", return_value=completed):
        result = _run_git(tmp_path, ["remote", "-v"])

    marker = _format_git_failure(result)
    assert result.outcome == GitCommandOutcome.NONZERO
    assert "secret" not in result.stderr
    assert "exit 128" in marker
    assert "secret" not in marker
    assert "https://user:<redacted>@github.com/example/repo.git" in marker


def test_execution_error_is_distinct_and_redacted(tmp_path: Path) -> None:
    error = OSError("cannot use https://user:secret@github.com/example/repo.git")
    with patch("srtctl.core.git_state.subprocess.run", side_effect=error):
        result = _run_git(tmp_path, ["rev-parse", "HEAD"])

    marker = _format_git_failure(result)
    assert result.outcome == GitCommandOutcome.EXECUTION_ERROR
    assert "secret" not in result.error
    assert "could not execute" in marker
    assert "secret" not in marker
    assert "https://user:<redacted>@github.com/example/repo.git" in marker


def test_slow_command_log_uses_redacted_command(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    completed = subprocess.CompletedProcess([], 0, stdout="ok\n", stderr="")
    with (
        patch("srtctl.core.git_state.subprocess.run", return_value=completed),
        patch("srtctl.core.git_state.time.monotonic", side_effect=[100.0, 106.0]),
        caplog.at_level("INFO", logger="srtctl.core.git_state"),
    ):
        result = _run_git(
            tmp_path,
            ["remote", "get-url", "https://user:secret@github.com/example/repo.git"],
        )

    assert result.outcome == GitCommandOutcome.SUCCESS
    assert "Slow git command" in caplog.text
    assert "secret" not in caplog.text
    assert "https://user:<redacted>@github.com/example/repo.git" in caplog.text


def test_timeout_decodes_and_redacts_byte_streams(tmp_path: Path) -> None:
    timeout = subprocess.TimeoutExpired(
        ["git", "status"],
        60,
        output=b"partial output",
        stderr=b"https://user:secret@github.com/example/repo.git",
    )
    with patch("srtctl.core.git_state.subprocess.run", side_effect=timeout):
        result = _run_git(tmp_path, ["status"], timeout_seconds=60)

    assert result.stdout == "partial output"
    assert result.stderr == "https://user:<redacted>@github.com/example/repo.git"


def test_success_result_cannot_be_formatted_as_failure() -> None:
    result = GitCommandResult(command=("git", "status"), outcome=GitCommandOutcome.SUCCESS)

    with pytest.raises(ValueError, match="Cannot format successful git command"):
        _format_git_failure(result)


def test_untracked_timeout_is_not_rendered_as_clean(tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")

    def run_git(command, **_kwargs):
        if "ls-files" in command:
            raise subprocess.TimeoutExpired(command, 60)
        return completed

    with patch("srtctl.core.git_state.subprocess.run", side_effect=run_git):
        text = "".join(_format_repo_snapshot(tmp_path, ["repo"], [tmp_path]))

    untracked_section = text.split("## Untracked files\n", 1)[1]
    assert "timed out after 60s" in untracked_section
    assert not untracked_section.startswith("<none>")


def test_repository_budget_skips_remaining_commands(tmp_path: Path) -> None:
    with (
        patch("srtctl.core.git_state._REPO_TIMEOUT_S", 0),
        patch("srtctl.core.git_state.time.monotonic", return_value=100.0),
        patch("srtctl.core.git_state.subprocess.run") as run,
    ):
        text = "".join(_format_repo_snapshot(tmp_path, ["repo"], [tmp_path]))

    run.assert_not_called()
    assert text.count("repository budget was exhausted") == 8
    assert "## Untracked files\n<git command skipped" in text


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
