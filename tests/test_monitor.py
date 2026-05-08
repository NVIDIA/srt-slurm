# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that srtctl monitor data-gathering is read-only."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from srtctl.cli.monitor import _gather_all, _gather_job_info


def _make_job(outputs: Path, job_id: str) -> None:
    logs = outputs / job_id / "logs"
    logs.mkdir(parents=True)
    (logs / f"sweep_{job_id}.log").write_text("2026-04-27 10:35:18 [ERROR] Benchmark failed with exit code 1\n")


def test_gather_job_info_does_not_write_files(tmp_path: Path):
    _make_job(tmp_path, "99")
    before = {p: p.stat().st_mtime for p in tmp_path.rglob("*")}
    _gather_job_info("99", tmp_path, sq=None)
    after = {p: p.stat().st_mtime for p in tmp_path.rglob("*")}
    assert before == after


def test_gather_all_never_calls_subprocess(tmp_path: Path):
    _make_job(tmp_path, "99")
    with patch("srtctl.cli.monitor._squeue_jobs", return_value={}), patch("subprocess.run") as mock_run:
        _gather_all(tmp_path, include_all=True, seen_job_ids={"99"})
    mock_run.assert_not_called()
