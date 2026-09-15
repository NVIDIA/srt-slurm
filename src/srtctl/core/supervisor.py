# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker supervisor: relaunch a worker endpoint that exits mid-run.

The process monitor (:func:`srtctl.core.processes.start_process_monitor`) is a
circuit breaker: the first critical exit tears the job down. ``roles.<role>.critical:
false`` makes it ignore the exit instead. This module is the third option, a
reconcile loop in the Kubernetes sense: the desired state is the worker topology
the orchestrator computed at start (``backend_processes``), the observed state is
the process registry, and every monitor tick the supervisor relaunches whatever
has gone missing, subject to the role's :class:`~srtctl.core.schema.RestartPolicy`.

The unit of restart is the endpoint, not the process. A multi-node worker whose
ranks run as separate steps (SGLang TP across nodes) cannot survive losing one
rank, so the surviving siblings are stopped and the whole endpoint comes back
together. Relaunch is in place: same nodes, same GPUs, same ports, so a Dynamo
frontend sees the replacement register under a fresh instance id once the old
lease expires, and a static router sees the same URL come back.

Bookkeeping: every restart is appended to ``worker_restarts.json`` in the log
directory (picked up by the lockfile), the previous life's log is rotated to
``<log>.out.<n>`` so the crash log survives, and the relaunched step carries an
``_r<n>`` suffix so a lingering Slurm step of the same name is never signalled by
mistake.

What the supervisor does not do: reschedule onto another node (the allocation
has no spare, and only a discovery-based frontend could follow the new address),
or detect a worker that is alive but not serving. Both build on this loop.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from srtctl.core.processes import ManagedProcess, ProcessRegistry, list_step_ids
from srtctl.core.readiness import probe_http

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from srtctl.core.schema import RestartPolicy
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)

WORKER_RESTARTS_FILENAME = "worker_restarts.json"

# (mode, index): one logical worker, possibly spanning several processes/nodes.
EndpointKey = tuple[str, int]


class WorkerLauncher(Protocol):
    """What the supervisor needs from the worker stage to bring an endpoint back."""

    def relaunch_endpoint(self, endpoint_processes: list[Process], *, attempt: int) -> list[ManagedProcess]:
        """Launch every step of the endpoint again; ``attempt`` is 1 for the first relaunch."""
        ...

    def worker_ready_probe(self, endpoint_processes: list[Process]) -> tuple[str, int] | None:
        """``(host, port)`` whose ``GET /health`` returns 200 once the endpoint serves, or None to skip."""
        ...


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class RestartEvent:
    """One relaunch of one endpoint, from the exit that triggered it to readiness."""

    worker: str  # name of the step whose exit triggered the relaunch
    mode: str
    index: int
    node: str | None
    attempt: int  # 1 for the first relaunch of this endpoint
    exit_code: int | None
    exited_at: str
    # scheduled -> relaunched -> ready | not_ready; or exhausted / launch_failed / stopped.
    outcome: str = "scheduled"
    backoff_seconds: float = 0.0
    relaunched_at: str | None = None
    ready_at: str | None = None
    ready_seconds: float | None = None
    crash_log: str | None = None


@dataclass
class _EndpointState:
    key: EndpointKey
    processes: list[Process]
    policy: RestartPolicy
    names: list[str]  # registry names of the endpoint's current steps
    restarts: int = 0
    due_at: float | None = None  # monotonic time of the pending relaunch
    pending: RestartEvent | None = None
    ready_probe: tuple[str, int] | None = None
    ready_deadline: float | None = None
    next_probe_at: float = 0.0
    launched_at: float | None = None
    settled: bool = False  # nothing more for the supervisor to do with this endpoint

    @property
    def label(self) -> str:
        return f"{self.key[0]}_{self.key[1]}"


class WorkerSupervisor:
    """Relaunch exited worker endpoints according to each role's restart policy.

    ``reconcile()`` is called from the process monitor thread every tick. It is
    cheap when nothing has happened (one ``poll()`` per tracked step) and does
    all of its work under one lock, so ``track()`` from the main thread and the
    monitor never interleave.
    """

    def __init__(
        self,
        *,
        registry: ProcessRegistry,
        stop_event: threading.Event,
        launcher: WorkerLauncher,
        log_dir: Path | None = None,
        ready_timeout: float = 1800.0,
        probe_interval: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.registry = registry
        self.stop_event = stop_event
        self.launcher = launcher
        self.log_dir = log_dir
        self.ready_timeout = ready_timeout
        # Every failed probe lands as a 503 error line in the worker's own log
        # (Dynamo's system server logs it), so do not probe on every 2 s tick.
        self.probe_interval = probe_interval
        self._clock = clock
        self._lock = threading.RLock()
        self._endpoints: dict[EndpointKey, _EndpointState] = {}
        self._events: list[RestartEvent] = []

    # ------------------------------------------------------------------ setup

    def track(
        self,
        groups: dict[EndpointKey, list[Process]],
        worker_names: Iterable[str],
        policy_for: Callable[[str], RestartPolicy],
    ) -> None:
        """Adopt the endpoints of roles with a restart policy; others are left to the registry.

        ``worker_names`` are the registry names ``start_all_workers`` produced
        (``<mode>_<index>_<node>``); the ones belonging to a tracked endpoint are
        flagged ``supervised`` so ``check_failures`` leaves their exits to us.
        """
        names = list(worker_names)
        with self._lock:
            for key, processes in groups.items():
                mode, index = key
                policy = policy_for(mode)
                if not policy.enabled:
                    continue
                prefix = f"{mode}_{index}_"
                mine = [name for name in names if name.startswith(prefix)]
                if not mine:
                    logger.warning("Worker supervisor: no registered steps found for %s_%d", mode, index)
                    continue
                for name in mine:
                    proc = self.registry.get_process(name)
                    if proc is not None:
                        proc.supervised = True
                self._endpoints[key] = _EndpointState(key=key, processes=list(processes), policy=policy, names=mine)
                logger.info(
                    "Worker supervisor: %s_%d restart policy %s (max %d, backoff %.0fs..%.0fs)",
                    mode,
                    index,
                    policy.policy,
                    policy.max_restarts,
                    policy.backoff_seconds,
                    policy.max_backoff_seconds,
                )

    # -------------------------------------------------------------- reporting

    @property
    def events(self) -> list[RestartEvent]:
        with self._lock:
            return list(self._events)

    @property
    def total_restarts(self) -> int:
        with self._lock:
            return sum(state.restarts for state in self._endpoints.values())

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_restarts": sum(state.restarts for state in self._endpoints.values()),
                "endpoints": {
                    state.label: {
                        "policy": state.policy.policy,
                        "restarts": state.restarts,
                        "max_restarts": state.policy.max_restarts,
                        "exhausted": state.settled and state.restarts >= state.policy.max_restarts,
                    }
                    for state in self._endpoints.values()
                    if state.restarts or state.settled
                },
                "events": [asdict(event) for event in self._events],
            }

    def summary_line(self) -> str | None:
        """One log line for the end of the job, or None when nothing was restarted."""
        with self._lock:
            restarted = [state for state in self._endpoints.values() if state.restarts]
            if not restarted:
                return None
            parts = [f"{state.label} x{state.restarts}" for state in restarted]
            return f"Worker restarts: {sum(s.restarts for s in restarted)} ({', '.join(parts)})"

    def _write(self) -> None:
        if self.log_dir is None:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / WORKER_RESTARTS_FILENAME
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.summary(), indent=2) + "\n")
            tmp.replace(path)
        except OSError as exc:
            logger.debug("Could not write %s: %s", WORKER_RESTARTS_FILENAME, exc)

    # -------------------------------------------------------------- reconcile

    def reconcile(self) -> None:
        """One pass: schedule, relaunch, or give up on every tracked endpoint."""
        with self._lock:
            if self.stop_event.is_set():
                return
            now = self._clock()
            for state in self._endpoints.values():
                if state.settled:
                    continue
                if state.due_at is not None:
                    if now >= state.due_at:
                        self._relaunch(state, now)
                    continue
                self._observe(state, now)

    def _observe(self, state: _EndpointState, now: float) -> None:
        procs = [proc for proc in (self.registry.get_process(name) for name in state.names) if proc is not None]
        exited = [proc for proc in procs if not proc.is_running]
        if not exited:
            self._probe_ready(state, now)
            return

        trigger = exited[0]
        exit_code = trigger.exit_code
        if not state.policy.restarts_on(exit_code):
            # A clean exit under on-failure: not ours to relaunch. Hand the
            # endpoint back so the role's critical flag has the final say.
            logger.info(
                "Worker %s exited with code %s; restart policy %s does not relaunch it",
                trigger.name,
                exit_code,
                state.policy.policy,
            )
            self._release(state, procs)
            return

        if state.restarts >= state.policy.max_restarts:
            logger.error(
                "Worker %s exited with code %s after %d restart(s); max_restarts=%d reached, giving up",
                trigger.name,
                exit_code,
                state.restarts,
                state.policy.max_restarts,
            )
            self._events.append(
                RestartEvent(
                    worker=trigger.name,
                    mode=state.key[0],
                    index=state.key[1],
                    node=trigger.node,
                    attempt=state.restarts + 1,
                    exit_code=exit_code,
                    exited_at=_now_iso(),
                    outcome="exhausted",
                    crash_log=str(trigger.log_file) if trigger.log_file else None,
                )
            )
            self._release(state, procs)
            self._write()
            return

        # Siblings of a multi-step endpoint cannot carry on without the lost rank.
        running = [proc for proc in procs if proc.is_running]
        if running:
            logger.warning(
                "Worker %s exited; stopping %d sibling step(s) of %s before relaunch",
                trigger.name,
                len(running),
                state.label,
            )
            step_ids = list_step_ids() if any(proc.step_name for proc in running) else None
            for proc in running:
                proc.request_stop(step_ids)
            for proc in running:
                proc.await_stop()

        state.restarts += 1
        delay = state.policy.backoff(state.restarts)
        state.due_at = now + delay
        state.ready_deadline = None
        state.pending = RestartEvent(
            worker=trigger.name,
            mode=state.key[0],
            index=state.key[1],
            node=trigger.node,
            attempt=state.restarts,
            exit_code=exit_code,
            exited_at=_now_iso(),
            backoff_seconds=delay,
        )
        self._events.append(state.pending)
        logger.warning(
            "Worker %s exited with code %s; relaunching %s in %.0fs (restart %d/%d)",
            trigger.name,
            exit_code,
            state.label,
            delay,
            state.restarts,
            state.policy.max_restarts,
        )
        self._write()

    def _release(self, state: _EndpointState, procs: list[ManagedProcess]) -> None:
        """Stop supervising: the registry's critical semantics apply to these steps again."""
        for proc in procs:
            proc.supervised = False
        state.settled = True

    def _relaunch(self, state: _EndpointState, now: float) -> None:
        if self.stop_event.is_set():
            return
        old = [proc for proc in (self.registry.pop_process(name) for name in state.names) if proc is not None]
        for proc in old:
            rotated = self._rotate_log(proc.log_file, state.restarts)
            if rotated is not None:
                proc.log_file = rotated
                if state.pending is not None and state.pending.worker == proc.name:
                    state.pending.crash_log = str(rotated)

        try:
            replacements = self.launcher.relaunch_endpoint(state.processes, attempt=state.restarts)
        except Exception:
            # Put the exited steps back unsupervised so the role's critical flag
            # decides the run's fate; nothing is running for this endpoint now.
            logger.exception("Relaunch of %s failed; leaving it to the registry", state.label)
            for proc in old:
                proc.supervised = False
                self.registry.add_process(proc)
            if state.pending is not None:
                state.pending.outcome = "launch_failed"
            state.due_at = None
            state.settled = True
            self._write()
            return

        for proc in replacements:
            proc.supervised = True
            self.registry.add_process(proc)
        state.names = [proc.name for proc in replacements]
        state.due_at = None
        state.launched_at = now
        state.ready_probe = self.launcher.worker_ready_probe(state.processes)
        state.ready_deadline = now + self.ready_timeout if state.ready_probe else None
        state.next_probe_at = now + self.probe_interval  # give the step a moment before the first probe
        if state.pending is not None:
            state.pending.relaunched_at = _now_iso()
            state.pending.outcome = "relaunched"
        logger.warning(
            "Relaunched %s as %s (restart %d/%d)%s",
            state.label,
            ", ".join(state.names),
            state.restarts,
            state.policy.max_restarts,
            f"; waiting up to {self.ready_timeout:.0f}s for /health" if state.ready_probe else "",
        )
        self._write()

    def _probe_ready(self, state: _EndpointState, now: float) -> None:
        if state.ready_deadline is None or state.ready_probe is None or now < state.next_probe_at:
            return
        state.next_probe_at = now + self.probe_interval
        host, port = state.ready_probe
        if probe_http(host, port, "/health", 200, request_timeout=2.0):
            state.ready_deadline = None
            elapsed = now - (state.launched_at or now)
            if state.pending is not None:
                state.pending.ready_at = _now_iso()
                state.pending.ready_seconds = round(elapsed, 1)
                state.pending.outcome = "ready"
            logger.info("Relaunched %s is serving again after %.0fs (%s:%d/health)", state.label, elapsed, host, port)
            self._write()
        elif now >= state.ready_deadline:
            state.ready_deadline = None
            if state.pending is not None:
                state.pending.outcome = "not_ready"
            logger.warning(
                "Relaunched %s has not answered %s:%d/health within %.0fs; still running, leaving it up",
                state.label,
                host,
                port,
                self.ready_timeout,
            )
            self._write()

    @staticmethod
    def _rotate_log(log_file: Path | None, life: int) -> Path | None:
        """Move ``<log>.out`` to ``<log>.out.<life>`` so the next life writes a fresh file."""
        if log_file is None or not log_file.exists():
            return None
        target = log_file.with_name(f"{log_file.name}.{life}")
        try:
            log_file.replace(target)
        except OSError as exc:
            logger.debug("Could not rotate %s: %s", log_file, exc)
            return None
        return target


def load_worker_restarts(log_dir: Path) -> dict[str, Any] | None:
    """Load ``worker_restarts.json`` from a log directory; None when absent or unreadable."""
    try:
        data = json.loads((log_dir / WORKER_RESTARTS_FILENAME).read_text())
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) and data.get("total_restarts") else None
