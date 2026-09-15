# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProcessRegistry."""

from pathlib import Path
from subprocess import Popen, TimeoutExpired
from unittest.mock import MagicMock, patch

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

        # exit_code comes from popen.returncode
        assert mock_popen.returncode == 1

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

        with patch("srtctl.core.processes.shutil.which", return_value=None):
            assert signal_step("anything") is False
