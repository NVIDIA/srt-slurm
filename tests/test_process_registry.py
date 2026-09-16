# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProcessRegistry."""

from pathlib import Path
from subprocess import Popen, TimeoutExpired
from unittest.mock import MagicMock

from srtctl.core.processes import ManagedProcess, ProcessRegistry, terminate_and_reap


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

        # exit_code comes from popen.poll()
        assert mp.exit_code == 1
        assert not mp.is_running

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


class TestCleanupGrace:
    """cleanup() signals every process first, then waits per-process up to terminate_timeout."""

    @staticmethod
    def _running(pid: int, exits_after_terminate: bool = True) -> MagicMock:
        popen = MagicMock(spec=Popen)
        popen.poll.return_value = None
        popen.pid = pid
        if exits_after_terminate:
            popen.wait.return_value = 0
        else:
            popen.wait.side_effect = [TimeoutExpired("x", 1), 0]  # survives SIGTERM, dies on SIGKILL
        return popen

    def test_signals_all_before_waiting(self):
        order: list[str] = []
        registry = ProcessRegistry(job_id="j")
        for i in range(3):
            popen = self._running(100 + i)
            popen.terminate.side_effect = lambda i=i: order.append(f"term{i}")
            popen.wait.side_effect = lambda timeout=None, i=i: order.append(f"wait{i}") or 0
            registry.add_process(ManagedProcess(name=f"p{i}", popen=popen))
        registry.cleanup()
        assert order[:3] == ["term0", "term1", "term2"]
        assert sorted(order[3:]) == ["wait0", "wait1", "wait2"]

    def test_waits_for_each_process_own_timeout(self):
        registry = ProcessRegistry(job_id="j")
        fast, slow = self._running(1), self._running(2)
        registry.add_process(ManagedProcess(name="fast", popen=fast))
        registry.add_process(ManagedProcess(name="slow", popen=slow, terminate_timeout=180.0))
        registry.cleanup()
        fast.wait.assert_called_once_with(timeout=10.0)
        slow.wait.assert_called_once_with(timeout=180.0)
        fast.kill.assert_not_called()
        slow.kill.assert_not_called()

    def test_kills_after_grace_expires(self):
        registry = ProcessRegistry(job_id="j")
        stubborn = self._running(3, exits_after_terminate=False)
        registry.add_process(ManagedProcess(name="stubborn", popen=stubborn, terminate_timeout=0.01))
        registry.cleanup()
        stubborn.terminate.assert_called_once()
        stubborn.kill.assert_called_once()
        assert stubborn.wait.call_count == 2

    def test_add_processes_keeps_terminate_timeout_when_renaming(self):
        registry = ProcessRegistry(job_id="j")
        popen = self._running(4)
        registry.add_processes({"renamed": ManagedProcess(name="orig", popen=popen, terminate_timeout=42.0)})
        assert registry._processes["renamed"].terminate_timeout == 42.0

    def test_terminate_defaults_to_own_timeout(self):
        popen = self._running(5)
        ManagedProcess(name="p", popen=popen, terminate_timeout=33.0).terminate()
        popen.wait.assert_called_once_with(timeout=33.0)


class TestGracefulStepSignal:
    """nsys-wrapped steps get SIGTERM via scancel --signal, not via their srun client."""

    @staticmethod
    def _running(pid: int) -> MagicMock:
        popen = MagicMock(spec=Popen)
        popen.poll.return_value = None
        popen.pid = pid
        popen.wait.return_value = 0
        return popen

    def test_named_long_grace_step_is_signalled_not_terminated(self):
        from unittest.mock import patch

        registry = ProcessRegistry(job_id="4242")
        worker = self._running(1)
        plain = self._running(2)
        registry.add_process(ManagedProcess(name="decode_0_n1", popen=worker, terminate_timeout=600.0, step_name="decode_0_n1"))
        registry.add_process(ManagedProcess(name="etcd", popen=plain))
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            res = MagicMock()
            res.returncode = 0
            res.stderr = ""
            res.stdout = "4242.3 decode_0_n1\n4242.2 bash\n" if cmd[0] == "squeue" else ""
            return res

        with patch("srtctl.core.processes.subprocess.run", side_effect=fake_run):
            registry.cleanup()
        assert ["scancel", "--signal=TERM", "4242.3"] in calls
        worker.terminate.assert_not_called()  # step finished on its own within the grace
        worker.wait.assert_called_once_with(timeout=600.0)
        plain.terminate.assert_called_once()

    def test_falls_back_to_srun_terminate_when_step_missing(self):
        from unittest.mock import patch

        registry = ProcessRegistry(job_id="4242")
        worker = self._running(1)
        registry.add_process(ManagedProcess(name="decode_0_n1", popen=worker, terminate_timeout=600.0, step_name="decode_0_n1"))

        def fake_run(cmd, **kwargs):
            res = MagicMock()
            res.returncode = 0
            res.stderr = ""
            res.stdout = "4242.2 bash\n"  # no step with our name
            return res

        with patch("srtctl.core.processes.subprocess.run", side_effect=fake_run):
            registry.cleanup()
        worker.terminate.assert_called_once()

    def test_escalates_when_signalled_step_outlives_grace(self):
        from unittest.mock import patch

        registry = ProcessRegistry(job_id="4242")
        worker = self._running(1)
        worker.wait.side_effect = [TimeoutExpired("x", 1), TimeoutExpired("x", 1), 0]  # grace, post-terminate, post-kill
        registry.add_process(ManagedProcess(name="decode_0_n1", popen=worker, terminate_timeout=0.01, step_name="decode_0_n1"))

        def fake_run(cmd, **kwargs):
            res = MagicMock()
            res.returncode = 0
            res.stderr = ""
            res.stdout = "4242.3 decode_0_n1\n"
            return res

        # terminate_timeout 0.01 is not "raised", so this one goes the plain route; raise it to exercise the step path
        registry._processes["decode_0_n1"].terminate_timeout = 11.0
        with patch("srtctl.core.processes.subprocess.run", side_effect=fake_run):
            registry.cleanup()
        worker.terminate.assert_called_once()
        worker.kill.assert_called_once()

    def test_default_grace_processes_never_touch_slurm(self):
        from unittest.mock import patch

        registry = ProcessRegistry(job_id="4242")
        popen = self._running(1)
        registry.add_process(ManagedProcess(name="w", popen=popen, step_name="w"))  # default 10 s grace
        with patch("srtctl.core.processes.subprocess.run") as run:
            registry.cleanup()
        run.assert_not_called()
        popen.terminate.assert_called_once()
