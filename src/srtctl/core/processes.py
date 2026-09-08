# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Process registry for managing and monitoring spawned processes.

This module provides lifecycle management for srun processes, including:
- Process registration and tracking
- Health monitoring via background thread
- Graceful cleanup on exit or failure
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TerminationOutcome:
    """How local process termination completed."""

    reaped: bool
    force_killed: bool


def terminate_and_reap(
    popen: subprocess.Popen, *, terminate_timeout: float = 10.0, kill_timeout: float = 5.0
) -> TerminationOutcome:
    """Terminate, then kill, while preserving whether SIGKILL was required."""
    if popen.poll() is not None:
        return TerminationOutcome(reaped=True, force_killed=False)
    popen.terminate()
    try:
        popen.wait(timeout=terminate_timeout)
        return TerminationOutcome(reaped=True, force_killed=False)
    except subprocess.TimeoutExpired:
        logger.warning("Process did not terminate, killing...")
    popen.kill()
    try:
        popen.wait(timeout=kill_timeout)
        return TerminationOutcome(reaped=True, force_killed=True)
    except subprocess.TimeoutExpired:
        logger.error("Process was not reaped after SIGKILL")
        return TerminationOutcome(reaped=False, force_killed=True)


@dataclass
class ManagedProcess:
    """A process managed by the registry.

    Attributes:
        name: Human-readable process name (e.g., "prefill_0", "decode_1")
        popen: The subprocess.Popen object
        log_file: Path to the process log file
        node: Node hostname where the process runs
        critical: If True, failure triggers full cleanup
        terminate_timeout: Seconds to wait after SIGTERM before SIGKILL on
            cleanup. Processes that flush state on SIGTERM (tachometer
            compacting parquet) need more than the default.
        step_name: The Slurm step name this srun was launched with
            (``start_srun_process(step_name=...)``). When set, ``terminate()``
            delivers SIGTERM to the task with ``scancel --signal=TERM --full``
            on that step, because SIGTERM to the srun process itself only
            aborts the step and the task is SIGKILLed without warning.
    """

    name: str
    popen: subprocess.Popen
    log_file: Path | None = None
    node: str | None = None
    critical: bool = True
    terminate_timeout: float = 10.0
    step_name: str | None = None

    @property
    def is_running(self) -> bool:
        """Check if process is still running."""
        return self.popen.poll() is None

    @property
    def exit_code(self) -> int | None:
        """Get exit code if process has exited, None otherwise."""
        return self.popen.poll()

    def terminate(self, timeout: float | None = None) -> None:
        """Terminate the process gracefully (SIGTERM, then SIGKILL after ``timeout`` or ``terminate_timeout``).

        With a ``step_name`` the SIGTERM goes to the Slurm step's task via
        ``scancel --signal``; the srun process is only SIGTERMed as a fallback.
        """
        if not self.is_running:
            return

        wait = self.terminate_timeout if timeout is None else timeout
        if self.step_name and signal_step(self.step_name, "TERM"):
            try:
                self.popen.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Step %s (%s) did not exit %.0fs after SIGTERM; terminating srun", self.step_name, self.name, wait
                )
        outcome = terminate_and_reap(self.popen, terminate_timeout=wait, kill_timeout=5)
        if not outcome.reaped:
            logger.error("Process %s was not reaped after SIGKILL", self.name)


# Type alias for named process collections
NamedProcesses = dict[str, ManagedProcess]


def find_step_id(step_name: str, job_id: str | None = None) -> str | None:
    """The ``<job>.<step>`` id of the running step named ``step_name`` in this job, or None."""
    job_id = job_id or os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if not job_id:
        return None
    try:
        result = subprocess.run(
            ["squeue", "--steps", f"--jobs={job_id}", "--noheader", "--format=%i %j"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("squeue --steps failed while looking up step %s: %s", step_name, exc)
        return None
    if result.returncode != 0:
        logger.warning(
            "squeue --steps exited %d looking up step %s: %s", result.returncode, step_name, result.stderr.strip()
        )
        return None
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == step_name:
            return parts[0]
    return None


def signal_step(step_name: str, sig: str = "TERM") -> bool:
    """Send ``sig`` to every process of the Slurm step named ``step_name``; True when delivered.

    ``srun`` turns a SIGTERM aimed at itself into a step abort that SIGKILLs the
    task, so a process that must flush on SIGTERM (tachometer compacting its
    parquet, an engine shutting down cleanly) has to be signalled through Slurm:
    ``scancel --signal=<sig> --full <job>.<step>``.
    """
    step_id = find_step_id(step_name)
    if step_id is None:
        logger.warning("No running step named %s found; falling back to signalling srun", step_name)
        return False
    try:
        result = subprocess.run(
            ["scancel", f"--signal={sig}", "--full", step_id], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("scancel --signal=%s %s failed: %s", sig, step_id, exc)
        return False
    if result.returncode != 0:
        logger.warning("scancel --signal=%s %s exited %d: %s", sig, step_id, result.returncode, result.stderr.strip())
        return False
    logger.info("Sent SIG%s to step %s (%s)", sig, step_id, step_name)
    return True


class ProcessRegistry:
    """Registry for managing multiple processes with health monitoring.

    Features:
    - Tracks all spawned processes by name
    - Background thread monitors for unexpected exits
    - Graceful cleanup on signal or failure
    - Detailed failure reporting with log tails

    Usage:
        registry = ProcessRegistry(job_id="12345")
        registry.add_process(managed_proc)
        # ... run workload ...
        if registry.check_failures():
            registry.cleanup()
    """

    def __init__(self, job_id: str):
        """Initialize the registry.

        Args:
            job_id: SLURM job ID for logging
        """
        self.job_id = job_id
        self._processes: dict[str, ManagedProcess] = {}
        self._lock = threading.Lock()
        self._failed_processes: list[str] = []

    def add_process(self, process: ManagedProcess) -> None:
        """Add a process to the registry.

        Args:
            process: ManagedProcess to track
        """
        with self._lock:
            if process.name in self._processes:
                logger.warning("Replacing existing process '%s' in registry", process.name)
            self._processes[process.name] = process
            logger.debug("Registered process: %s (pid=%d)", process.name, process.popen.pid)

    def add_processes(self, processes: NamedProcesses) -> None:
        """Add multiple processes to the registry.

        Args:
            processes: Dict mapping names to ManagedProcess objects
        """
        for name, proc in processes.items():
            # Ensure the name matches
            if proc.name != name:
                proc = ManagedProcess(
                    name=name,
                    popen=proc.popen,
                    log_file=proc.log_file,
                    node=proc.node,
                    critical=proc.critical,
                )
            self.add_process(proc)

    def check_failures(self) -> bool:
        """Check if any critical process has failed.

        Returns:
            True if any critical process has exited with non-zero code
        """
        with self._lock:
            for name, proc in self._processes.items():
                if proc.critical and not proc.is_running:
                    exit_code = proc.exit_code
                    if exit_code != 0 and name not in self._failed_processes:
                        self._failed_processes.append(name)
                        logger.error(
                            "Critical process '%s' exited with code %d",
                            name,
                            exit_code,
                        )

            return len(self._failed_processes) > 0

    def cleanup(self) -> None:
        """Terminate all registered processes."""
        with self._lock:
            logger.info("Cleaning up %d processes...", len(self._processes))
            for name, proc in self._processes.items():
                if proc.is_running:
                    logger.debug("Terminating process: %s", name)
                    try:
                        proc.terminate()
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Failed to terminate %s: %s", name, e)

    def print_failure_details(self, tail_lines: int = 50) -> None:
        """Print detailed failure information including log tails.

        Args:
            tail_lines: Number of lines to show from each failed process log
        """
        if not self._failed_processes:
            return

        logger.error("=" * 60)
        logger.error("FAILURE DETAILS")
        logger.error("=" * 60)

        with self._lock:
            for name in self._failed_processes:
                proc = self._processes.get(name)
                if not proc:
                    continue

                logger.error("\n--- Process: %s ---", name)
                logger.error("Exit code: %s", proc.exit_code)
                logger.error("Node: %s", proc.node or "unknown")
                logger.error("Log file: %s", proc.log_file or "none")

                # Tail the log file if available
                if proc.log_file and proc.log_file.exists():
                    try:
                        lines = proc.log_file.read_text().splitlines()
                        if lines:
                            logger.error("\nLast %d lines of log:", tail_lines)
                            for line in lines[-tail_lines:]:
                                logger.error("  %s", line)
                    except Exception as e:  # noqa: BLE001
                        logger.error("Could not read log file: %s", e)

        logger.error("=" * 60)

    def get_process(self, name: str) -> ManagedProcess | None:
        """Get a process by name."""
        with self._lock:
            return self._processes.get(name)

    def get_all_processes(self) -> dict[str, ManagedProcess]:
        """Get a copy of all registered processes."""
        with self._lock:
            return dict(self._processes)

    @property
    def process_count(self) -> int:
        """Get the number of registered processes."""
        with self._lock:
            return len(self._processes)


def setup_signal_handlers(
    stop_event: threading.Event,
    registry: ProcessRegistry,
) -> None:
    """Setup signal handlers for graceful shutdown.

    Args:
        stop_event: Event to signal shutdown
        registry: ProcessRegistry to cleanup on signal
    """

    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.warning("Received signal %s, initiating cleanup...", sig_name)
        stop_event.set()
        registry.cleanup()
        sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def start_process_monitor(
    stop_event: threading.Event,
    registry: ProcessRegistry,
    poll_interval: float = 2.0,
) -> threading.Thread:
    """Start a background thread that monitors for process failures.

    Args:
        stop_event: Event that signals the monitor to stop
        registry: ProcessRegistry to monitor
        poll_interval: Seconds between checks

    Returns:
        The monitoring thread (already started)
    """

    def monitor_loop():
        while not stop_event.is_set():
            if registry.check_failures():
                logger.error("Critical process failure detected!")
                stop_event.set()
                registry.cleanup()
                sys.exit(1)
            time.sleep(poll_interval)

    thread = threading.Thread(
        target=monitor_loop,
        daemon=True,
        name="ProcessMonitor",
    )
    thread.start()
    return thread
