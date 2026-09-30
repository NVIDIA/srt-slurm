# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Donor diagnostics preserve failures and graceful shutdown."""

import json
import os
import runpy
import signal
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "src/srtctl/runtime_scripts/donor_diagnostics.py"
FUNCTIONS = runpy.run_path(str(SCRIPT))


def test_snapshot_reads_launch_context(tmp_path: Path, capsys) -> None:
    proc = tmp_path / "proc/self"
    proc.mkdir(parents=True)
    (proc / "cgroup").write_text("0::/\n")
    (proc / "limits").write_text("Max locked memory unlimited unlimited bytes\n")
    cg = tmp_path / "cgroup"
    cg.mkdir()
    (cg / "memory.current").write_text("123\n")
    (cg / "memory.max").write_text("456\n")
    (cg / "memory.events").write_text("oom 1\noom_kill 0\n")
    FUNCTIONS["memory_snapshot"]("running", proc.parent, cg)
    record = json.loads(capsys.readouterr().out.removeprefix("[donor-memory] "))
    assert record["cgroup"] == "0::/"
    assert record["memory"]["memory.current"] == "123"
    assert record["memory"]["memory.max"] == "456"
    assert record["memory"]["memory.events"] == "oom 1\noom_kill 0"
    assert record["memory"]["memory.peak"].startswith("unavailable:")


def test_kernel_messages_filtered(monkeypatch, capsys) -> None:
    def run(command, **kwargs):
        assert command == ["dmesg", "-T"]
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(command, 0, "ordinary message\nmlx5: MKEY allocation failure\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    FUNCTIONS["kernel_snapshot"]("after_exit")
    output = capsys.readouterr().out
    assert "mlx5: MKEY allocation failure" in output
    assert "ordinary message" not in output


def test_donor_failure_preserved_when_dmesg_unavailable() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), sys.executable, "-c", "import sys; sys.exit(7)"],
        env={**os.environ, "PATH": "/nonexistent"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 7
    assert '"phase": "before_start"' in result.stdout
    assert '"phase": "after_exit"' in result.stdout
    assert "[donor-kernel] unavailable:" in result.stdout


def test_sigterm_reaches_donor(tmp_path: Path) -> None:
    marker = tmp_path / "stopped"
    code = (
        "import signal, sys, time; from pathlib import Path; "
        f"signal.signal(signal.SIGTERM, lambda *_: (Path({str(marker)!r}).write_text('terminated'), sys.exit(0))); "
        "print('child ready', flush=True); time.sleep(30)"
    )
    process = subprocess.Popen(
        [sys.executable, str(SCRIPT), sys.executable, "-c", code],
        env={**os.environ, "PATH": "/nonexistent"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if line.strip() == "child ready":
                break
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=10)
        assert process.returncode == 0
        assert marker.read_text() == "terminated"
        assert '"phase": "after_exit"' in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
