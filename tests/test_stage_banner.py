# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stage banners benchmark scripts write into worker logs."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from srtctl.benchmarks.base import SCRIPTS_DIR

STAGE_BANNER_LIB = SCRIPTS_DIR / "lib" / "stage_banner.sh"

BANNER_LINE = re.compile(r"^======== \[\d{2}:\d{2}:\d{2}\] (?P<label>.+) ========$")


def _banner(log_dir: Path, label: str = "cc=1024 warmup begin") -> str:
    """Run stage_banner against a log dir and return what it wrote to stdout."""
    result = subprocess.run(
        ["bash", "-c", f'source "{STAGE_BANNER_LIB}"; stage_banner "{label}"'],
        env={"STAGE_BANNER_LOG_DIR": str(log_dir), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    """A job log dir as srtctl lays it out mid-run."""
    for name in (
        "node01_prefill_w0.out",
        "node02_decode_w0.out",
        "node03_agg_w0.out",
        "node01_frontend_0.out",
        "infra.out",
        "benchmark.out",
    ):
        (tmp_path / name).write_text("worker output\n")
    return tmp_path


def test_every_worker_log_gets_the_marker(log_dir):
    """Prefill, decode and aggregated workers are all annotated."""
    _banner(log_dir)

    for name in ("node01_prefill_w0.out", "node02_decode_w0.out", "node03_agg_w0.out"):
        assert "warmup begin" in (log_dir / name).read_text()


def test_only_worker_logs_are_touched(log_dir):
    """Frontend, infra and benchmark logs have their own structure already."""
    _banner(log_dir)

    for name in ("node01_frontend_0.out", "infra.out", "benchmark.out"):
        assert (log_dir / name).read_text() == "worker output\n"


def test_marker_carries_a_timestamp_and_the_label(log_dir):
    _banner(log_dir, "cc=2048 benchmark end")

    lines = (log_dir / "node01_prefill_w0.out").read_text().splitlines()
    matches = [BANNER_LINE.match(line) for line in lines]
    (match,) = [m for m in matches if m]

    assert match.group("label") == "cc=2048 benchmark end"


def test_marker_is_set_off_by_blank_lines(log_dir):
    """Standing out while scrolling is the whole point of the marker."""
    _banner(log_dir)

    lines = (log_dir / "node01_prefill_w0.out").read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if BANNER_LINE.match(line))

    assert lines[index - 1] == ""
    assert lines[index + 1] == ""


def test_existing_output_is_kept(log_dir):
    """Markers are appended to a live log, never written over it."""
    _banner(log_dir, "cc=1 warmup begin")
    _banner(log_dir, "cc=1 warmup end")

    content = (log_dir / "node01_prefill_w0.out").read_text()
    assert content.startswith("worker output\n")
    assert content.index("warmup begin") < content.index("warmup end")


def test_marker_falls_back_to_stdout_without_worker_logs(tmp_path):
    """A run with nothing to annotate must not swallow the marker."""
    stdout = _banner(tmp_path, "cc=8 benchmark begin")

    assert BANNER_LINE.match(stdout.strip())
    assert "cc=8 benchmark begin" in stdout


def test_bench_script_marks_both_phases_of_every_concurrency():
    """bench.sh brackets warmup and measured runs at each concurrency level."""
    source = (SCRIPTS_DIR / "sa-bench" / "bench.sh").read_text()

    banners = re.findall(r"stage_banner \"cc=\$\{concurrency\} (.+)\"", source)

    assert banners == ["warmup begin", "warmup end", "benchmark begin", "benchmark end"]
