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


_DEFAULT_TERMINATE_TIMEOUT = 10.0


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
        terminate_timeout: Seconds ``cleanup()`` waits after SIGTERM before SIGKILL. Raise
            it for processes that must flush state on exit (an nsys-wrapped worker whose
            capture range is still open writes its report only after the engine exits).
        step_name: Name of the SLURM job step (``srun --job-name``). When set together with
            a raised ``terminate_timeout``, ``cleanup()`` delivers SIGTERM to the step's
            processes (``scancel --signal=TERM <jobid>.<stepid>``) instead of to the srun
            client: srun reacts to SIGTERM with "forcing job termination", after which
            slurmstepd SIGKILLs the tasks within seconds, so nsys never gets to write its
            report (hecate 596299). Signalled directly, nsys stops, writes the report and
            exits, and srun then ends normally.
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
        """Terminate the process gracefully, then kill if needed.

        ``timeout`` defaults to ``self.terminate_timeout``.
        """
        if not self.is_running:
            return

        if timeout is None:
            timeout = self.terminate_timeout
        outcome = terminate_and_reap(self.popen, terminate_timeout=timeout, kill_timeout=5)
        if not outcome.reaped:
            logger.error("Process %s was not reaped after SIGKILL", self.name)


# Type alias for named process collections
NamedProcesses = dict[str, ManagedProcess]


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
                    terminate_timeout=proc.terminate_timeout,
                    step_name=proc.step_name,
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

    def _step_ids_by_name(self) -> dict[str, list[str]]:
        """Map step names to step ids for this job (``squeue -s``); empty on any failure."""
        try:
            out = subprocess.run(
                ["squeue", "-s", "-j", str(self.job_id), "-h", "-o", "%i %j"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError) as e:  # noqa: PERF203
            logger.warning("squeue -s failed; falling back to signalling srun clients: %s", e)
            return {}
        steps: dict[str, list[str]] = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2:
                steps.setdefault(parts[1], []).append(parts[0])
        return steps

    def _signal_steps(self, procs: list[ManagedProcess]) -> set[str]:
        """SIGTERM the processes of each named step; return the names that were signalled."""
        if not procs:
            return set()
        steps = self._step_ids_by_name()
        signalled: set[str] = set()
        for proc in procs:
            step_ids = steps.get(proc.step_name or "", [])
            if not step_ids:
                logger.warning(
                    "No running step named %r for %s; signalling its srun instead", proc.step_name, proc.name
                )
                continue
            ok = True
            for step_id in step_ids:
                try:
                    res = subprocess.run(
                        ["scancel", "--signal=TERM", step_id], capture_output=True, text=True, timeout=30, check=False
                    )
                    if res.returncode != 0:
                        ok = False
                        logger.warning("scancel --signal=TERM %s failed: %s", step_id, res.stderr.strip())
                except (OSError, subprocess.SubprocessError) as e:  # noqa: PERF203
                    ok = False
                    logger.warning("scancel --signal=TERM %s failed: %s", step_id, e)
            if ok:
                logger.info(
                    "Sent SIGTERM to step %s (%s); waiting up to %.0fs for it to finish (nsys report flush)",
                    ",".join(step_ids),
                    proc.name,
                    proc.terminate_timeout,
                )
                signalled.add(proc.name)
        return signalled

    def cleanup(self) -> None:
        """Terminate all registered processes.

        Three phases so the grace periods overlap instead of adding up:

        1. Processes with a ``step_name`` and a raised ``terminate_timeout`` (nsys-wrapped
           workers / frontend) get SIGTERM delivered to their step's processes via
           ``scancel --signal=TERM``. Signalling the srun client instead would make srun
           "force job termination" and slurmstepd would SIGKILL the tasks within seconds,
           before nsys can write its report.
        2. Everything else (and any step that could not be signalled) gets ``popen.terminate()``.
        3. Wait for each process up to its own ``terminate_timeout``; on expiry escalate:
           SIGTERM to the srun client if it only had the step signal so far, then SIGKILL.
        """
        with self._lock:
            logger.info("Cleaning up %d processes...", len(self._processes))
            running: list[ManagedProcess] = [p for p in self._processes.values() if p.is_running]
            graceful = [
                p
                for p in running
                if isinstance(p.step_name, str)
                and p.step_name
                and isinstance(p.terminate_timeout, int | float)
                and p.terminate_timeout > _DEFAULT_TERMINATE_TIMEOUT
            ]
            signalled = self._signal_steps(graceful)
            for proc in running:
                if proc.name in signalled:
                    continue
                logger.debug("Terminating process: %s", proc.name)
                try:
                    proc.popen.terminate()
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to terminate %s: %s", proc.name, e)
            for proc in running:
                try:
                    proc.popen.wait(timeout=proc.terminate_timeout)
                    continue
                except subprocess.TimeoutExpired:
                    pass
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to reap %s: %s", proc.name, e)
                    continue
                if proc.name in signalled:
                    logger.warning(
                        "Step of %s did not finish within %.0fs of SIGTERM, terminating its srun...",
                        proc.name,
                        proc.terminate_timeout,
                    )
                    try:
                        proc.popen.terminate()
                        proc.popen.wait(timeout=_DEFAULT_TERMINATE_TIMEOUT)
                        continue
                    except subprocess.TimeoutExpired:
                        pass
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Failed to terminate %s: %s", proc.name, e)
                logger.warning("Process %s did not exit, killing...", proc.name)
                try:
                    proc.popen.kill()
                    proc.popen.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error("Process %s was not reaped after SIGKILL", proc.name)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to kill %s: %s", proc.name, e)

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
