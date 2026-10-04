# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProcessRegistry."""

import threading
from pathlib import Path
from subprocess import Popen, TimeoutExpired
from unittest.mock import MagicMock, patch

import pytest

from srtctl.core.processes import ManagedProcess, ProcessRegistry, start_process_monitor, terminate_and_reap


class TestManagedProcess:
    """Tests for ManagedProcess dataclass."""

    def test_managed_process_creation(self):
        """Test creating a ManagedProcess."""
        mock_popen = MagicMock(spec=Popen)
        mock_popen.poll.return_value = None
        mock_popen.pid = 12345

        mp = ManagedProcess(
            name="test_process",
            popen=mock_popen,
            log_file=Path("/tmp/test.log"),
            node="node0",
        )

        assert mp.name == "test_process"
        assert mp.node == "node0"

    def test_managed_process_exit_code(self):
        """Test exit_code property."""
        mock_popen = MagicMock(spec=Popen)
        mock_popen.poll.return_value = 1
        mock_popen.returncode = 1
        mock_popen.pid = 12345

        mp = ManagedProcess(
            name="test",
            popen=mock_popen,
            log_file=Path("/tmp/test.log"),
        )

        assert mp.exit_code == 1

    def test_terminate_does_not_raise_when_kill_wait_times_out(self):
        """A child that survives SIGKILL must not raise out of terminate()."""
        mock_popen = MagicMock(spec=Popen)
        mock_popen.poll.return_value = None
        mock_popen.wait.side_effect = TimeoutExpired(cmd="worker", timeout=1)

        mp = ManagedProcess(name="stuck", popen=mock_popen)

        mp.terminate(timeout=0.01)

        mock_popen.terminate.assert_called_once()
        mock_popen.kill.assert_called_once()


class TestTerminateAndReap:
    """Tests for the terminate_and_reap helper."""

    def test_already_exited_child_is_reported_reaped(self):
        mock_popen = MagicMock(spec=Popen)
        mock_popen.poll.return_value = 0

        outcome = terminate_and_reap(mock_popen)

        assert outcome.reaped is True
        assert outcome.force_killed is False
        mock_popen.terminate.assert_not_called()

    def test_graceful_terminate_is_reported_reaped(self):
        mock_popen = MagicMock(spec=Popen)
        mock_popen.poll.return_value = None
        mock_popen.wait.return_value = 0

        outcome = terminate_and_reap(mock_popen, terminate_timeout=0.01)

        assert outcome.reaped is True
        assert outcome.force_killed is False
        mock_popen.terminate.assert_called_once()
        mock_popen.kill.assert_not_called()

    def test_force_killed_child_is_reaped_but_not_graceful(self):
        mock_popen = MagicMock(spec=Popen)
        mock_popen.poll.return_value = None
        mock_popen.wait.side_effect = [TimeoutExpired(cmd="worker", timeout=1), -9]

        outcome = terminate_and_reap(mock_popen, terminate_timeout=0.01, kill_timeout=0.01)

        assert outcome.reaped is True
        assert outcome.force_killed is True
        mock_popen.terminate.assert_called_once()
        mock_popen.kill.assert_called_once()

    def test_unreapable_child_is_reported_not_reaped(self):
        mock_popen = MagicMock(spec=Popen)
        mock_popen.poll.return_value = None
        mock_popen.wait.side_effect = TimeoutExpired(cmd="worker", timeout=1)

        outcome = terminate_and_reap(mock_popen, terminate_timeout=0.01, kill_timeout=0.01)

        assert outcome.reaped is False
        assert outcome.force_killed is True
        mock_popen.terminate.assert_called_once()
        mock_popen.kill.assert_called_once()


class TestProcessRegistry:
    """Tests for ProcessRegistry."""

    def test_add_process(self):
        """Test adding a process to the registry."""
        registry = ProcessRegistry(job_id="test_job")

        mock_popen = MagicMock(spec=Popen)
        mock_popen.poll.return_value = None
        mock_popen.pid = 12345

        mp = ManagedProcess(
            name="worker_0",
            popen=mock_popen,
            log_file=Path("/tmp/test.log"),
        )

        registry.add_process(mp)
        # Just verify it doesn't error

    def test_add_processes(self):
        """Test adding multiple processes."""
        registry = ProcessRegistry(job_id="test_job")

        processes = {}
        for i in range(3):
            mock_popen = MagicMock(spec=Popen)
            mock_popen.poll.return_value = None
            mock_popen.pid = 12345 + i
            mp = ManagedProcess(
                name=f"worker_{i}",
                popen=mock_popen,
                log_file=Path(f"/tmp/test_{i}.log"),
            )
            processes[mp.name] = mp

        registry.add_processes(processes)
        # Just verify it doesn't error

    def test_check_failures_no_failures(self):
        """Test check_failures with no failures."""
        registry = ProcessRegistry(job_id="test_job")

        mock_popen = MagicMock(spec=Popen)
        mock_popen.poll.return_value = None  # Still running
        mock_popen.pid = 12345

        mp = ManagedProcess(
            name="worker_0",
            popen=mock_popen,
            log_file=Path("/tmp/test.log"),
            critical=True,
        )

        registry.add_process(mp)
        assert not registry.check_failures()

    def test_check_failures_with_failure(self):
        """Test check_failures detects failed process."""
        registry = ProcessRegistry(job_id="test_job")

        mock_popen = MagicMock(spec=Popen)
        mock_popen.poll.return_value = 1  # Failed
        mock_popen.returncode = 1
        mock_popen.pid = 12345

        mp = ManagedProcess(
            name="worker_0",
            popen=mock_popen,
            log_file=Path("/tmp/test.log"),
            critical=True,
        )

        registry.add_process(mp)
        assert registry.check_failures()

    def test_has_failures_reports_recorded_failures_without_rescanning(self):
        registry = ProcessRegistry(job_id="test_job")
        mock_popen = MagicMock(spec=Popen)
        mock_popen.poll.return_value = 1
        mock_popen.pid = 12345
        registry.add_process(ManagedProcess(name="worker_0", popen=mock_popen, critical=True))

        assert registry.has_failures is False  # nothing scanned yet
        assert registry.check_failures()
        assert registry.has_failures is True

        # A process that cleanup terminates afterwards is not counted until someone scans again.
        etcd = MagicMock(spec=Popen)
        etcd.poll.return_value = 143
        etcd.pid = 1
        registry.add_process(ManagedProcess(name="service_etcd", popen=etcd, critical=True))
        assert registry.has_failures is True
        registry.print_failure_details()  # only worker_0 is recorded

    def test_check_failures_skips_supervised_processes(self):
        """A step owned by the worker supervisor is its call to relaunch, until it hands the step back."""
        registry = ProcessRegistry(job_id="test_job")

        mock_popen = MagicMock(spec=Popen)
        mock_popen.poll.return_value = 1
        mock_popen.pid = 12345
        mp = ManagedProcess(name="decode_0_node0", popen=mock_popen, critical=True, supervised=True)
        registry.add_process(mp)

        assert not registry.check_failures()
        mp.supervised = False
        assert registry.check_failures()

    def test_pop_process(self):
        registry = ProcessRegistry(job_id="test_job")
        mock_popen = MagicMock(spec=Popen)
        mock_popen.poll.return_value = None
        mock_popen.pid = 12345
        mp = ManagedProcess(name="worker_0", popen=mock_popen)
        registry.add_process(mp)

        assert registry.pop_process("worker_0") is mp
        assert registry.pop_process("worker_0") is None
        assert registry.process_count == 0
        mock_popen.terminate.assert_not_called()

    def test_cleanup(self):
        """Test cleanup terminates all processes."""
        registry = ProcessRegistry(job_id="test_job")

        mock_popen = MagicMock(spec=Popen)
        mock_popen.poll.return_value = None  # Still running
        mock_popen.wait.return_value = 0
        mock_popen.pid = 12345

        mp = ManagedProcess(
            name="worker_0",
            popen=mock_popen,
            log_file=Path("/tmp/test.log"),
        )

        registry.add_process(mp)
        registry.cleanup()

        mock_popen.terminate.assert_called_once()


class TestTieredCleanup:
    """Cleanup signals a whole tier, waits for it, escalates, then moves to the next tier."""

    @staticmethod
    def _proc(name: str, events: list[str], *, tier: int = 0, hangs: bool = False, step: str | None = None):
        popen = MagicMock(spec=Popen)
        popen.pid = 1
        alive = {"running": True}
        popen.poll.side_effect = lambda: None if alive["running"] else 0

        def terminate():
            events.append(f"term:{name}")

        def wait(timeout=None):
            events.append(f"wait:{name}")
            if hangs:
                raise TimeoutExpired(cmd="srun", timeout=timeout)
            alive["running"] = False
            return 0

        def kill():
            events.append(f"kill:{name}")
            alive["running"] = False

        popen.terminate.side_effect = terminate
        popen.wait.side_effect = wait
        popen.kill.side_effect = kill
        return ManagedProcess(name=name, popen=popen, shutdown_tier=tier, step_name=step, terminate_timeout=1.0)

    def test_tiers_stop_in_order_and_a_tier_is_signalled_at_once(self):
        events: list[str] = []
        registry = ProcessRegistry(job_id="1")
        for proc in (
            self._proc("etcd", events, tier=2),
            self._proc("master", events, tier=1),
            self._proc("worker", events, tier=0),
            self._proc("frontend", events, tier=0),
        ):
            registry.add_process(proc)

        registry.cleanup()

        assert events == [
            # tier 0: SIGTERM to all (reverse registration order), then the waits
            "term:frontend",
            "term:worker",
            "wait:worker",
            "wait:frontend",
            # only then what the workers registered with
            "term:master",
            "wait:master",
            # the discovery plane last
            "term:etcd",
            "wait:etcd",
        ]

    def test_a_process_that_ignores_sigterm_is_killed_after_its_own_deadline(self):
        events: list[str] = []
        registry = ProcessRegistry(job_id="1")
        registry.add_process(self._proc("stuck", events, hangs=True))
        registry.add_process(self._proc("polite", events))

        registry.cleanup()

        assert events[:2] == ["term:polite", "term:stuck"]
        assert "kill:stuck" in events
        assert "kill:polite" not in events

    def test_steps_are_listed_once_and_signalled_through_slurm(self):
        events: list[str] = []
        registry = ProcessRegistry(job_id="1")
        registry.add_process(self._proc("decode_0_node1", events, step="decode_0_node1"))
        registry.add_process(self._proc("prefill_0_node0", events, step="prefill_0_node0"))
        steps = {"decode_0_node1": "1.5", "prefill_0_node0": "1.4"}
        with (
            patch("srtctl.core.processes.list_step_ids", return_value=steps) as listed,
            patch("srtctl.core.processes.signal_step", return_value=True) as signalled,
        ):
            registry.cleanup()

        listed.assert_called_once()
        assert [call.args[0] for call in signalled.call_args_list] == ["prefill_0_node0", "decode_0_node1"]
        assert all(call.kwargs["step_ids"] is steps for call in signalled.call_args_list)
        # Delivered through the step: srun itself is never SIGTERMed, only waited for.
        assert not any(event.startswith("term:") for event in events)
        assert events == ["wait:decode_0_node1", "wait:prefill_0_node0"]

    def test_without_slurm_tools_signal_step_declines_quietly(self):
        from srtctl.core.processes import signal_step

        with patch("srtctl.core.launcher.shutil.which", return_value=None):
            assert signal_step("anything") is False


def test_profile_wrapper_uses_task_only_step_signal():
    from srtctl.core.processes import signal_step

    with (
        patch("srtctl.core.launcher.shutil.which", return_value="/bin/scancel"),
        patch("srtctl.core.launcher.subprocess.run", return_value=MagicMock(returncode=0)) as run,
    ):
        assert signal_step("profiled", step_ids={"profiled": "123.4"}, full=False)
    assert run.call_args.args[0] == ["scancel", "--signal=TERM", "123.4"]


def test_profile_shutdown_metadata_survives_registry_rename():
    registry = ProcessRegistry(job_id="123")
    popen = MagicMock(spec=Popen)
    popen.poll.return_value = None
    popen.pid = 1234
    proc = ManagedProcess("original", popen, step_name="profiled", terminate_timeout=90, signal_full=False)
    registry.add_processes({"renamed": proc})
    with (
        patch("srtctl.core.processes.signal_step", return_value=True) as signal,
        patch("srtctl.core.processes.list_step_ids", return_value={"profiled": "123.4"}),
    ):
        registry.cleanup()
    signal.assert_called_once_with("profiled", "TERM", step_ids={"profiled": "123.4"}, full=False)
    assert 85 < popen.wait.call_args.kwargs["timeout"] <= 90


# The lines trtllm-llmapi-launch prints once the engine child of the rank-0 task has
# exited, and TRT-LLM's own terminal start-up line. The registry is agnostic; the
# backend supplies the table (tests/test_backend_protocol.py covers that side).
LAUNCHER_EXIT_PATTERNS = (r"^Rank\d+ Task exit code: (?!0$)\d+$", r"Failed to initialize executor")

BENIGN_LOG = (
    "Rank0 run mgmn leader node with mpi_world_size: 8\n[TRT-LLM] loading weights\nRank0 MPI Comm server exit code: 0\n"
)

# Dynamo prints this for every request the client cancels at EOS; it must never fail a run.
BENIGN_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "dynamo/trtllm/handlers.py", line 210, in generate\n'
    "    await response_stream.send(chunk)\n"
    "RuntimeError: response stream is closed\n"
    "[node1:12345] MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD\n"
)


def _live_step(log: Path, *, critical: bool = True, patterns: tuple[str, ...] = LAUNCHER_EXIT_PATTERNS):
    """A ManagedProcess whose srun is still running (poll() is None) and whose log is ``log``."""
    popen = MagicMock(spec=Popen)
    popen.poll.return_value = None
    popen.wait.return_value = 0
    popen.pid = 4242
    proc = ManagedProcess(
        name="decode_0_n1",
        popen=popen,
        log_file=log,
        node="n1",
        critical=critical,
        fatal_log_patterns=patterns,
    )
    return proc, popen


def _append(log: Path, text: str) -> None:
    with log.open("a", encoding="utf-8") as handle:
        handle.write(text)


class TestFatalLogMarkers:
    """A critical step whose log says its task died fails the run while srun is still alive.

    Background: a TRT-LLM worker step is one launcher task per GPU and the engine is a
    child of the rank-0 task only. When that child dies the launcher prints
    ``Rank0 Task exit code: <n>`` but the follower ranks stay blocked, so the srun step
    (and therefore ``popen.poll()``) reports nothing until the health window runs out.
    """

    def test_running_step_whose_log_reports_task_exit_is_a_failure(self, tmp_path: Path, caplog) -> None:
        log = tmp_path / "n1_decode_w0.out"
        log.write_text(BENIGN_LOG)
        proc, _ = _live_step(log)
        registry = ProcessRegistry(job_id="1")
        registry.add_process(proc)

        assert registry.check_failures() is False

        _append(log, "Rank0 Task exit code: 1\nRank0 MPI Comm server exit code: 0\n")
        with caplog.at_level("ERROR", logger="srtctl.core.processes"):
            assert registry.check_failures() is True

        message = " ".join(record.getMessage() for record in caplog.records)
        assert "decode_0_n1" in message
        assert "Rank0 Task exit code: 1" in message
        assert LAUNCHER_EXIT_PATTERNS[0] in message

    def test_executor_init_failure_is_a_failure(self, tmp_path: Path) -> None:
        log = tmp_path / "n1_decode_w0.out"
        log.write_text(BENIGN_LOG)
        proc, _ = _live_step(log)
        registry = ProcessRegistry(job_id="1")
        registry.add_process(proc)

        _append(log, "[TensorRT-LLM][ERROR] [executor][RANK 0] Failed to initialize executor\n")

        assert registry.check_failures() is True

    def test_ignores_exit_zero_and_benign_tracebacks(self, tmp_path: Path) -> None:
        log = tmp_path / "n1_decode_w0.out"
        log.write_text(BENIGN_LOG)
        proc, _ = _live_step(log)
        registry = ProcessRegistry(job_id="1")
        registry.add_process(proc)

        _append(log, "Rank0 Task exit code: 0\n" + BENIGN_TRACEBACK)

        assert registry.check_failures() is False
        assert registry.check_failures() is False

    def test_non_critical_process_markers_are_ignored(self, tmp_path: Path) -> None:
        log = tmp_path / "n1_decode_w0.out"
        log.write_text(BENIGN_LOG + "Rank0 Task exit code: 1\n")
        proc, _ = _live_step(log, critical=False)
        registry = ProcessRegistry(job_id="1")
        registry.add_process(proc)

        assert registry.check_failures() is False

    def test_marker_scan_reads_only_new_bytes_and_whole_lines(self, tmp_path: Path) -> None:
        log = tmp_path / "n1_decode_w0.out"
        log.write_bytes(BENIGN_LOG.encode() + b"\xff\xfe not utf-8 \xc3\n")
        proc, _ = _live_step(log)
        registry = ProcessRegistry(job_id="1")
        registry.add_process(proc)

        # Invalid UTF-8 is tolerated, and a marker split across two polls is judged
        # only once its newline has arrived.
        assert registry.check_failures() is False
        with log.open("ab") as handle:
            handle.write(b"Rank0 Task exit co")
        assert registry.check_failures() is False
        with log.open("ab") as handle:
            handle.write(b"de: 1\n")
        assert registry.check_failures() is True

        # Later markers do not register the same process twice.
        _append(log, "Rank3 Task exit code: 1\n")
        assert registry.check_failures() is True
        assert registry._failed_processes == ["decode_0_n1"]

    def test_missing_log_is_not_a_failure(self, tmp_path: Path) -> None:
        proc, _ = _live_step(tmp_path / "not-yet-created.out")
        registry = ProcessRegistry(job_id="1")
        registry.add_process(proc)

        assert registry.check_failures() is False

    def test_no_patterns_means_no_log_watch(self, tmp_path: Path) -> None:
        # The recipe kill switch (health_check.fatal_log_markers: false) hands the
        # registry an empty table; a marker in the log is then just a log line.
        log = tmp_path / "n1_decode_w0.out"
        log.write_text(BENIGN_LOG + "Rank0 Task exit code: 1\n")
        proc, _ = _live_step(log, patterns=())
        registry = ProcessRegistry(job_id="1")
        registry.add_process(proc)

        assert registry.check_failures() is False

    def test_extra_pattern_from_the_recipe_is_honoured(self, tmp_path: Path) -> None:
        log = tmp_path / "n1_decode_w0.out"
        log.write_text(BENIGN_LOG)
        proc, _ = _live_step(log, patterns=(*LAUNCHER_EXIT_PATTERNS, r"CUDA error: out of memory"))
        registry = ProcessRegistry(job_id="1")
        registry.add_process(proc)

        _append(log, "torch.OutOfMemoryError: CUDA error: out of memory\n")

        assert registry.check_failures() is True

    def test_failure_details_show_the_matched_line(self, tmp_path: Path, caplog) -> None:
        log = tmp_path / "n1_decode_w0.out"
        log.write_text(BENIGN_LOG + "Rank0 Task exit code: 1\n")
        proc, _ = _live_step(log)
        registry = ProcessRegistry(job_id="1")
        registry.add_process(proc)
        assert registry.check_failures() is True

        with caplog.at_level("ERROR", logger="srtctl.core.processes"):
            registry.print_failure_details(tail_lines=5)

        messages = [record.getMessage() for record in caplog.records]
        assert any("Rank0 Task exit code: 1" in message and "Process:" not in message for message in messages)
        assert any("--- Process: decode_0_n1 ---" in message for message in messages)

    def test_rename_on_registration_keeps_the_patterns(self, tmp_path: Path) -> None:
        log = tmp_path / "n1_decode_w0.out"
        log.write_text(BENIGN_LOG + "Rank0 Task exit code: 1\n")
        proc, _ = _live_step(log)
        registry = ProcessRegistry(job_id="1")
        registry.add_processes({"renamed": proc})

        assert registry.check_failures() is True
        assert registry._failed_processes == ["renamed"]


# The monitor thread ends itself with sys.exit(1) after setting stop_event; pytest reports that SystemExit.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_monitor_stops_the_run_on_a_marker(tmp_path: Path) -> None:
    """The monitor thread turns a marker into stop_event within a couple of polls and tears the step down.

    wait_for_model's own reaction to stop_event is covered by
    tests/test_health.py::TestWaitForModel::test_stop_event_aborts; this is the other half.
    """
    log = tmp_path / "n1_decode_w0.out"
    log.write_text(BENIGN_LOG)
    proc, popen = _live_step(log)
    registry = ProcessRegistry(job_id="1")
    registry.add_process(proc)
    stop_event = threading.Event()

    thread = start_process_monitor(stop_event, registry, poll_interval=0.01)
    assert not stop_event.wait(0.1)

    _append(log, "Rank0 Task exit code: 1\n")

    assert stop_event.wait(2.0), "monitor did not react to the fatal log marker"
    thread.join(2.0)
    assert not thread.is_alive()
    # No step name and no scancel here, so cleanup falls back to signalling srun directly.
    popen.terminate.assert_called()
