# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise acknowledged capture windows with real rank-wrapper processes."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress

import pytest

from srtctl.runtime_scripts import nsys_window


@pytest.fixture
def fake_nsys(tmp_path):
    tool = tmp_path / "fake nsys"
    tool.write_text(
        "#!"
        + sys.executable
        + "\n"
        + r"""
import json, os, signal, subprocess, sys, threading, time
from pathlib import Path
root = Path(os.environ["FAKE_ROOT"])
rank = os.environ.get("SLURM_PROCID", "0")
mode = sys.argv[1]
def option(name):
    return next(x.split("=", 1)[1] for x in sys.argv if x.startswith(name + "="))
if mode == "app":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    (root / (rank + ".app-ready")).touch()
    while True: time.sleep(.02)
elif mode == "launch":
    session = option("--session-new")
    # The real Nsight CLI forks its launcher from a helper thread.
    def launch():
        global app
        app = subprocess.Popen([sys.executable, __file__, "app"])
        (root / (session + ".pid")).write_text(str(app.pid))
        app.wait()
    thread = threading.Thread(target=launch)
    thread.start()
    thread.join()
    sys.exit(app.returncode)
else:
    session = option("--session")
    active = root / (session + ".active")
    if mode == "start":
        if os.environ.get("FAKE_START_FAIL"): sys.exit(3)
        active.write_text(option("--output"))
    elif mode == "stop":
        if os.environ.get("FAKE_STOP_FAIL"): sys.exit(3)
        time.sleep(float(os.environ.get("FAKE_STOP_DELAY", "0")))
        if os.environ.get("FAKE_EMPTY_REPORT"):
            Path(active.read_text() + ".nsys-rep").touch()
        else:
            Path(active.read_text() + ".nsys-rep").write_text("fake report")
        active.unlink()
    elif mode == "shutdown":
        try: os.kill(int((root / (session + ".pid")).read_text()), signal.SIGTERM)
        except ProcessLookupError: pass
"""
    )
    tool.chmod(0o755)
    return tool


def wait_for(path, processes, timeout=8):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert all(p.poll() is None for p in processes), "wrapper exited before readiness"
        assert time.monotonic() < deadline, f"missing {path}"
        time.sleep(0.02)


def start_workers(root, fake_nsys, *, count=2, expected=2, **extra):
    root.mkdir(exist_ok=True)
    nsys_window.write_json(root / "steps/group.json", {"ranks": expected})
    processes = []
    for rank in range(count):
        spec = {
            "control_dir": str(root),
            "step": "group",
            "ranks": expected,
            "nsys": str(fake_nsys),
            "output": str(root / "report_rank%q{SLURM_PROCID}"),
            "start_args": ["--sample=none"],
            "timeout": 3,
            "app_grace_secs": 1,
        }
        processes.append(
            subprocess.Popen(
                [sys.executable, nsys_window.__file__, "worker", "--spec", json.dumps(spec), "--", "application"],
                env={**os.environ, "FAKE_ROOT": str(root), "SLURM_PROCID": str(rank), **extra},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    for rank in range(count):
        wait_for(root / f"{rank}.app-ready", processes)
        wait_for(root / "ready" / f"group-{rank}.json", processes)
    return processes


def stop_workers(processes, root):
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    try:
        for proc in processes:
            proc.communicate(timeout=10)
    finally:
        for path in root.glob("*.pid"):
            with suppress(ProcessLookupError):
                os.kill(int(path.read_text()), signal.SIGKILL)
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
            proc.communicate(timeout=5)


def test_warmup_and_post_workload_are_outside_repeated_capture_windows(fake_nsys, tmp_path):
    root = tmp_path / "control"
    processes = start_workers(root, fake_nsys, FAKE_STOP_DELAY=".2")
    try:
        # Workers are running during warmup, with no collection active.
        assert not list(root.glob("*.active"))
        for window in (1, 2):
            nsys_window.request(root, "start", timeout=3)
            assert len(list(root.glob("*.active"))) == 2
            nsys_window.request(root, "stop", timeout=3)
            assert not list(root.glob("*.active"))
            assert len(list(root.glob(f"*_window{window:03d}.nsys-rep"))) == 2
            assert all(proc.poll() is None for proc in processes)
        nsys_window.finish(root, timeout=3)
        assert nsys_window.read_json(root / "client.json")["completed"] == 2
    finally:
        stop_workers(processes, root)
    assert all(proc.returncode == 0 for proc in processes)


@pytest.mark.parametrize("failure", ["FAKE_START_FAIL", "FAKE_STOP_FAIL", "FAKE_EMPTY_REPORT"])
def test_capture_failure_reaches_the_benchmark(fake_nsys, tmp_path, failure):
    root = tmp_path / "control"
    processes = start_workers(root, fake_nsys, count=1, expected=1, **{failure: "1"})
    try:
        with pytest.raises(RuntimeError, match="failed"):
            nsys_window.request(root, "start", timeout=3)
            nsys_window.request(root, "stop", timeout=3)
    finally:
        stop_workers(processes, root)
    assert processes[0].returncode != 0


def test_missing_rank_blocks_capture_before_workload(fake_nsys, tmp_path):
    root = tmp_path / "control"
    processes = start_workers(root, fake_nsys, count=1, expected=2)
    try:
        with pytest.raises(TimeoutError, match="group-1"):
            nsys_window.request(root, "start", timeout=0.2)
        assert not list(root.glob("*.active"))
    finally:
        stop_workers(processes, root)


def test_custom_client_must_start_and_finish_capture(fake_nsys, tmp_path):
    root = tmp_path / "control"
    processes = start_workers(root, fake_nsys, count=1, expected=1)
    try:
        with pytest.raises(RuntimeError, match="No nsys workload window"):
            nsys_window.finish(root, timeout=3)
        nsys_window.request(root, "start", timeout=3)
        with pytest.raises(RuntimeError, match="active nsys window"):
            nsys_window.finish(root, timeout=3)
        assert (root / "report_rank0_window001.nsys-rep").exists()
        assert processes[0].poll() is None
    finally:
        stop_workers(processes, root)


def test_teardown_flushes_active_capture_before_apps_exit(fake_nsys, tmp_path):
    root = tmp_path / "control"
    processes = start_workers(root, fake_nsys)
    nsys_window.request(root, "start", timeout=3)
    stop_workers(processes, root)
    assert len(list(root.glob("*_window001.nsys-rep"))) == 2
    assert all(proc.returncode == 0 for proc in processes)
