# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual shell wrapper with a local stand-in for Nsight Systems."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress

import pytest

from srtctl.core.nsys_keepalive import keepalive_command


@pytest.fixture
def fake_nsys(tmp_path):
    script = tmp_path / "fake nsys"
    script.write_text(
        "#!"
        + sys.executable
        + "\n"
        + r"""
import json, os, signal, subprocess, sys, time
from pathlib import Path
root = Path(os.environ["FAKE_ROOT"])
rank = os.environ["SLURM_PROCID"]
mode = sys.argv[1]
if mode == "app":
    def finish(*_):
        reports = sorted(p.name for p in root.glob("*.nsys-rep"))
        (root / (rank + ".exit")).write_text(json.dumps(reports))
        sys.exit(0)
    signal.signal(signal.SIGTERM, finish)
    (root / (rank + ".started")).touch()
    while True:
        time.sleep(.02)
elif mode == "profile":
    # Record all processes so a failed test can clean up both sessions.
    (root / (rank + ".pid")).write_text(str(os.getpid()))
    time.sleep(float(os.environ.get("FAKE_START_DELAY", "0")))
    app = subprocess.Popen([sys.executable, __file__, "app"])
    sys.exit(app.wait())
elif mode == "stop":
    if os.environ.get("FAKE_STOP_FAIL"):
        sys.exit(3)
    time.sleep(float(os.environ.get("FAKE_STOP_DELAY", "0")))
    (root / (rank + ".nsys-rep")).write_text("fake report")
"""
    )
    script.chmod(0o755)
    return script


def wait_for(path, processes, timeout=8):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert all(p.poll() is None for p in processes), "wrapper exited before the fixture was ready"
        assert time.monotonic() < deadline, f"timed out waiting for {path}"
        time.sleep(0.02)


def launch(fake_nsys, tmp_path, rank, expected, **extra_env):
    env = {
        **os.environ,
        "FAKE_ROOT": str(tmp_path),
        "SLURM_PROCID": str(rank),
        "SRT_NSYS_REPORT_BARRIER_DIR": str(tmp_path / "barrier"),
        "SRT_NSYS_REPORT_EXPECTED": str(expected),
        "SRT_NSYS_REPORT_STOP_TIMEOUT": "3",
        **extra_env,
    }
    return subprocess.Popen(
        keepalive_command([str(fake_nsys), "profile"], app_exit_grace_secs=1),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def cleanup(processes, tmp_path):
    for path in tmp_path.glob("*.pid"):
        with suppress(ProcessLookupError):
            os.killpg(int(path.read_text()), signal.SIGKILL)
    for proc in processes:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate(timeout=5)


def test_all_rank_reports_finish_before_any_app_exits(fake_nsys, tmp_path):
    processes = [launch(fake_nsys, tmp_path, rank, 2, FAKE_STOP_DELAY=str(rank * 0.6)) for rank in range(2)]
    try:
        for rank in range(2):
            wait_for(tmp_path / f"{rank}.started", processes)
        for proc in processes:
            proc.terminate()
        for proc in processes:
            _, error = proc.communicate(timeout=10)
            assert proc.returncode == 0, error
            assert "2/2 ready; failed=0" in error
        for rank in range(2):
            assert json.loads((tmp_path / f"{rank}.exit").read_text()) == ["0.nsys-rep", "1.nsys-rep"]
    finally:
        cleanup(processes, tmp_path)


@pytest.mark.parametrize("stop_fails", [False, True])
def test_failed_or_missing_rank_does_not_hang_teardown(fake_nsys, tmp_path, stop_fails):
    proc = launch(fake_nsys, tmp_path, 0, 2, **({"FAKE_STOP_FAIL": "1"} if stop_fails else {}))
    try:
        wait_for(tmp_path / "0.started", [proc])
        proc.terminate()
        _, error = proc.communicate(timeout=10)
        assert proc.returncode == 1, error
        assert "failed=1" in error
        assert (tmp_path / "0.exit").exists()
    finally:
        cleanup([proc], tmp_path)


def test_term_during_profiler_startup_is_trapped(fake_nsys, tmp_path):
    # The profiler exists, but has not forked the application yet.
    proc = launch(fake_nsys, tmp_path, 0, 1, FAKE_START_DELAY=".5", FAKE_STOP_DELAY=".7")
    try:
        wait_for(tmp_path / "0.pid", [proc])
        proc.terminate()
        _, error = proc.communicate(timeout=10)
        assert proc.returncode == 0, error
        assert json.loads((tmp_path / "0.exit").read_text()) == ["0.nsys-rep"]
    finally:
        cleanup([proc], tmp_path)


def test_profiler_start_failure_is_propagated(tmp_path):
    missing = tmp_path / "not-installed"
    result = subprocess.run(
        keepalive_command([str(missing), "profile"]), capture_output=True, text=True, timeout=5, check=False
    )
    assert result.returncode != 0
