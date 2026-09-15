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
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Routers and frontends drain in-flight requests on SIGTERM; more than the default 10s, less than an engine.
FRONTEND_TERMINATE_TIMEOUT_SECONDS = 20.0


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
    # Cleanup stops tier 0 first and waits for it before moving on: workers and
    # frontends (0) deregister from the Mooncake master (1) and etcd/NATS (2)
    # while those are still up, instead of hanging on a plane that is gone.
    shutdown_tier: int = 0
    _stopped_via_step: bool = field(default=False, init=False, repr=False)
    _stop_deadline: float | None = field(default=None, init=False, repr=False)

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

    def request_stop(self, step_ids: dict[str, str] | None = None) -> None:
        """Deliver SIGTERM without waiting (phase one of a group shutdown); ``await_stop`` finishes it.

        ``step_ids`` is one ``squeue --steps`` listing shared by the whole group so
        a job with dozens of workers does not query Slurm dozens of times.
        """
        if not self.is_running:
            return
        self._stopped_via_step = bool(self.step_name) and signal_step(self.step_name or "", "TERM", step_ids=step_ids)
        if not self._stopped_via_step:
            self.popen.terminate()
        self._stop_deadline = time.monotonic() + self.terminate_timeout

    def await_stop(self) -> None:
        """Wait out this process's own deadline after ``request_stop``, then escalate to SIGKILL."""
        if not self.is_running:
            return
        deadline = self._stop_deadline if self._stop_deadline is not None else time.monotonic()
        try:
            self.popen.wait(timeout=max(0.0, deadline - time.monotonic()))
            return
        except subprocess.TimeoutExpired:
            pass
        if self._stopped_via_step:
            logger.warning(
                "Step %s (%s) did not exit %.0fs after SIGTERM; terminating srun",
                self.step_name,
                self.name,
                self.terminate_timeout,
            )
            outcome = terminate_and_reap(self.popen, terminate_timeout=5, kill_timeout=5)
        else:
            logger.warning("Process %s did not exit %.0fs after SIGTERM, killing...", self.name, self.terminate_timeout)
            self.popen.kill()
            try:
                self.popen.wait(timeout=5)
                outcome = TerminationOutcome(reaped=True, force_killed=True)
            except subprocess.TimeoutExpired:
                outcome = TerminationOutcome(reaped=False, force_killed=True)
        if not outcome.reaped:
            logger.error("Process %s was not reaped after SIGKILL", self.name)


# Type alias for named process collections
NamedProcesses = dict[str, ManagedProcess]


def list_step_ids(job_id: str | None = None) -> dict[str, str] | None:
    """``{step name: <job>.<step>}`` for the running steps of this job; None when Slurm cannot be asked."""
    job_id = job_id or os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if not job_id or shutil.which("squeue") is None:
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
        logger.warning("squeue --steps failed: %s", exc)
        return None
    if result.returncode != 0:
        logger.warning("squeue --steps exited %d: %s", result.returncode, result.stderr.strip())
        return None
    steps: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            steps.setdefault(parts[1], parts[0])
    return steps


def find_step_id(step_name: str, job_id: str | None = None) -> str | None:
    """The ``<job>.<step>`` id of the running step named ``step_name`` in this job, or None."""
    steps = list_step_ids(job_id)
    return steps.get(step_name) if steps else None


def signal_step(step_name: str, sig: str = "TERM", *, step_ids: dict[str, str] | None = None) -> bool:
    """Send ``sig`` to every process of the Slurm step named ``step_name``; True when delivered.

    ``srun`` turns a SIGTERM aimed at itself into a step abort that SIGKILLs the
    task, so a process that must flush on SIGTERM (tachometer compacting its
    parquet, an engine shutting down cleanly) has to be signalled through Slurm:
    ``scancel --signal=<sig> --full <job>.<step>``. ``step_ids`` is a listing from
    ``list_step_ids`` to reuse instead of querying again.
    """
    if shutil.which("scancel") is None:
        return False  # not under Slurm (tests, the mock): the caller signals srun directly
    step_id = step_ids.get(step_name) if step_ids is not None else find_step_id(step_name)
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
        """Stop every registered process: SIGTERM to a whole tier at once, wait, escalate, next tier.

        Within a tier the signal goes out in reverse registration order without
        waiting in between, so a job with many workers finishes cleanup in about
        one ``terminate_timeout`` rather than one per process. Tiers run lowest
        first (workers and frontends, then the Mooncake master, then etcd/NATS),
        and a tier is fully stopped before the next is signalled, so nothing is
        left deregistering from a plane that has already gone away.
        """
        with self._lock:
            running = [proc for proc in self._processes.values() if proc.is_running]
            logger.info("Cleaning up %d processes (%d running)...", len(self._processes), len(running))
            step_ids = list_step_ids() if any(proc.step_name for proc in running) else None
            for tier in sorted({proc.shutdown_tier for proc in running}):
                group = [proc for proc in running if proc.shutdown_tier == tier and proc.is_running]
                for proc in reversed(group):
                    logger.debug("Stopping process: %s", proc.name)
                    try:
                        proc.request_stop(step_ids)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Failed to signal %s: %s", proc.name, e)
                for proc in group:
                    try:
                        proc.await_stop()
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Failed to stop %s: %s", proc.name, e)

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
