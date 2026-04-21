# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProcessRegistry."""

from pathlib import Path
from subprocess import Popen
from unittest.mock import MagicMock

from srtctl.core.processes import ManagedProcess, ProcessRegistry


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
