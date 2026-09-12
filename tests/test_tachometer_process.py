# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual exec wrapper and scoped shutdown without a Slurm allocation."""

import json
import logging
import os
import stat
import subprocess
import sys
from unittest.mock import MagicMock, call

import pytest

from srtctl.core.processes import ProcessRegistry
from srtctl.core.tachometer_process import TachometerProcess, TachometerStep


def run_wrapper(step, *, job="12345", step_id="27", command=None):
    env = {**os.environ, "SLURM_JOB_ID": job, "SLURM_STEP_ID": step_id}
    return subprocess.run(
        step.command(command or [sys.executable, "-c", "pass"]),
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


def test_wrapper_records_private_identity_and_execs_unchanged_argv(tmp_path):
    step = TachometerStep.create(tmp_path, "12345")
    argument = "spaces ' quotes; $(exit 99)"
    command = [sys.executable, "-c", "import os,sys,json; print(json.dumps([os.getpid(),sys.argv[1:]]))", argument]
    with subprocess.Popen(
        step.command(command),
        env={**os.environ, "SLURM_JOB_ID": "12345", "SLURM_STEP_ID": "27"},
        stdout=subprocess.PIPE,
        text=True,
    ) as launched:
        output, _ = launched.communicate(timeout=5)
    assert launched.returncode == 0
    assert json.loads(output) == [launched.pid, [argument]]
    assert step.read() == "12345.27"
    assert stat.S_IMODE(step.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(step.path.stat().st_mode) == 0o600


@pytest.mark.parametrize("job,step_id", [("", "27"), ("99999", "27"), ("12345", "batch"), ("12345", "")])
def test_wrapper_rejects_wrong_allocation_or_non_numeric_step_before_exec(tmp_path, job, step_id):
    step = TachometerStep.create(tmp_path, "12345")
    result = run_wrapper(step, job=job, step_id=step_id, command=[sys.executable, "-c", "print('started')"])
    assert result.returncode != 0
    assert "started" not in result.stdout
    assert not step.path.exists()


def test_wrapper_never_overwrites_a_previous_handoff(tmp_path):
    step = TachometerStep.create(tmp_path, "12345")
    assert run_wrapper(step).returncode == 0
    assert run_wrapper(step, step_id="28").returncode != 0
    assert step.read() == "12345.27"
    other_launch = TachometerStep.create(tmp_path, "12345")
    assert other_launch.path != step.path
    assert other_launch.token != step.token


@pytest.fixture
def process(tmp_path):
    step = TachometerStep.create(tmp_path, "12345")
    assert run_wrapper(step).returncode == 0
    popen = MagicMock(spec=subprocess.Popen)
    popen.poll.return_value = None
    popen.wait.return_value = 0
    popen.pid = 123
    proc = TachometerProcess(
        name="tachometer", popen=popen, step=step, shutdown_grace_secs=45.0, critical=False, log_file=tmp_path / "out"
    )
    yield proc
    # Every path must keep signals away from the srun client.
    popen.terminate.assert_not_called()
    popen.kill.assert_not_called()
    popen.send_signal.assert_not_called()


@pytest.fixture
def signal_command(monkeypatch):
    runner = MagicMock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr("srtctl.core.tachometer_process.subprocess.run", runner)
    return runner


def test_scoped_term_waits_configured_grace_and_checks_status(process, signal_command, caplog):
    caplog.set_level(logging.INFO)
    process.terminate()
    signal_command.assert_called_once_with(
        ["scancel", "--signal=TERM", "12345.27"], capture_output=True, text=True, timeout=10.0, check=False
    )
    process.popen.wait.assert_called_once_with(timeout=45.0)
    assert "exited with status 0 after SIGTERM" in caplog.text
    assert "gracefully" not in caplog.text


def test_nonzero_exit_cannot_be_reported_as_success(process, signal_command, caplog):
    caplog.set_level(logging.INFO)
    process.popen.wait.return_value = 137
    process.terminate()
    assert signal_command.call_count == 1
    assert "exited with status 137 after SIGTERM" in caplog.text
    assert "status 0" not in caplog.text
    assert "gracefully" not in caplog.text


@pytest.mark.parametrize("returncode", [0, 137])
def test_already_exited_launcher_is_not_signalled(process, signal_command, caplog, returncode):
    process.popen.poll.return_value = returncode
    process.terminate()
    signal_command.assert_not_called()
    process.popen.wait.assert_not_called()
    if returncode:
        assert "already exited with status 137" in caplog.text


@pytest.mark.parametrize(
    "invalid", ["missing", "stale", "wrong-job", "batch", "symlink", "permissions", "directory", "fifo"]
)
def test_invalid_identity_leaves_launcher_and_other_steps_untouched(process, signal_command, caplog, invalid, tmp_path):
    step = process.step
    if invalid == "stale":
        step.path.write_text(f"12345 27 {'0' * 32}\n")
    elif invalid == "wrong-job":
        step.path.write_text(f"99999 27 {step.token}\n")
    elif invalid == "batch":
        step.path.write_text(f"12345 batch {step.token}\n")
    elif invalid == "permissions":
        step.path.chmod(0o644)
    elif invalid == "directory":
        step.path.parent.chmod(0o755)
    else:
        contents = step.path.read_text()
        step.path.unlink()
        if invalid == "symlink":
            target = tmp_path / "external"
            target.write_text(contents)
            target.chmod(0o600)
            step.path.symlink_to(target)
        elif invalid == "fifo":
            os.mkfifo(step.path, 0o600)
    process.terminate()
    signal_command.assert_not_called()
    process.popen.wait.assert_not_called()
    assert "Cannot safely stop Tachometer" in caplog.text


@pytest.mark.parametrize("failure", ["nonzero", "missing-command", "timeout"])
def test_failed_term_does_not_fall_back_to_client_signals(process, signal_command, caplog, failure):
    if failure == "nonzero":
        signal_command.return_value = subprocess.CompletedProcess([], 1, "", "permission denied")
    else:
        signal_command.side_effect = (
            FileNotFoundError("scancel") if failure == "missing-command" else subprocess.TimeoutExpired("scancel", 10)
        )
    process.terminate()
    assert signal_command.call_count == 1
    process.popen.wait.assert_not_called()
    assert "scancel TERM" in caplog.text


def test_timeout_escalates_only_same_step_after_grace(process, signal_command, caplog):
    events = []

    def signal(args, **kwargs):
        events.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    def wait(*, timeout):
        events.append(timeout)
        if timeout == 45.0:
            raise subprocess.TimeoutExpired("srun", timeout)
        return 137

    signal_command.side_effect = signal
    process.popen.wait.side_effect = wait
    process.terminate()
    assert events == [["scancel", "--signal=TERM", "12345.27"], 45.0, ["scancel", "--signal=KILL", "12345.27"], 10.0]
    assert "required SIGKILL" in caplog.text
    assert "gracefully" not in caplog.text


@pytest.mark.parametrize("failure", ["kill-failed", "reap-timeout"])
def test_escalation_failure_is_reported_without_client_fallback(process, signal_command, caplog, failure):
    process.popen.wait.side_effect = subprocess.TimeoutExpired("srun", 45)
    if failure == "kill-failed":
        signal_command.side_effect = [
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 1, "", "failed"),
        ]
    process.terminate()
    assert signal_command.call_count == 2
    assert "scancel KILL exited 1" in caplog.text if failure == "kill-failed" else "not reaped" in caplog.text


def test_registry_cleanup_uses_scoped_stop_and_saved_grace(process, signal_command):
    registry = ProcessRegistry(job_id="12345")
    registry.add_process(process)
    registry.cleanup()
    assert signal_command.call_args.args[0] == ["scancel", "--signal=TERM", "12345.27"]
    process.popen.wait.assert_called_once_with(timeout=45.0)


def test_repeated_cleanup_after_exit_does_not_signal_stale_step(process, signal_command):
    process.terminate()
    process.popen.poll.return_value = 0
    process.terminate()
    assert signal_command.call_count == 1
    assert process.popen.wait.call_args_list == [call(timeout=45.0)]
