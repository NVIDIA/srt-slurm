# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent chaos timeline runner executed alongside the benchmark stage."""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from srtctl.chaos.actions import ACTION_HANDLERS, StalledProcess, resume_stalled_processes
from srtctl.chaos.targets import WorkerTarget, resolve_worker_targets

if TYPE_CHECKING:
    from srtctl.backends.base import WorkerMode
    from srtctl.core.processes import ProcessRegistry
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ChaosConfig, ChaosEventConfig, ChaosScheduleConfig, SrtConfig
    from srtctl.core.supervisor import EndpointKey, WorkerSupervisor
    from srtctl.core.topology import Process


logger = logging.getLogger(__name__)

CHAOS_EVENTS_FILENAME = "chaos_events.json"


def sample_schedule_delay(schedule: ChaosScheduleConfig, rng: random.Random) -> float:
    """Draw the next inter-fault interval (seconds) with mean ``mean_interval_seconds`` and floor ``min_interval_seconds``."""
    floor = schedule.min_interval_seconds
    tail_mean = schedule.mean_interval_seconds - floor
    if tail_mean <= 0.0:
        return floor
    return floor + rng.expovariate(1.0 / tail_mean)


def _now_utc() -> tuple[str, int]:
    now_ns = time.time_ns()
    iso = (
        datetime.fromtimestamp(now_ns / 1e9, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )
    return iso, now_ns


@dataclass
class ChaosEventRecord:
    """Serialized record of one chaos event written into ``chaos_events.json``."""

    index: int
    name: str
    type: str
    role: str
    workers: int | list[int] | str
    status: str  # "completed" | "skipped" | "failed"
    started_at: str
    started_at_ns: int
    ended_at: str | None = None
    ended_at_ns: int | None = None
    targets: list[str] = field(default_factory=list)
    selection: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ChaosContext:
    """Live runtime handles passed to chaos action handlers."""

    def __init__(
        self,
        *,
        config: SrtConfig,
        runtime: RuntimeContext,
        registry: ProcessRegistry,
        supervisor: WorkerSupervisor | None,
        endpoint_groups: dict[EndpointKey, list[Process]],
        stop_event: threading.Event,
        done_event: threading.Event,
        rng: random.Random,
        leashed: bool = False,
        min_healthy: int = 0,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.registry = registry
        self.supervisor = supervisor
        self.endpoint_groups = endpoint_groups
        self.stop_event = stop_event
        self.done_event = done_event
        self.rng = rng
        self.leashed = leashed
        self.min_healthy = min_healthy
        self._stall_lock = threading.Lock()
        self._active_stalls: list[StalledProcess] = []
        self._killed_steps: set[str] = set()

    def should_stop(self) -> bool:
        return self.stop_event.is_set() or self.done_event.is_set()

    def wait(self, seconds: float) -> bool:
        """Sleep up to ``seconds``; returns True if interrupted early by stop/done."""
        if seconds <= 0:
            return self.should_stop()
        deadline = time.monotonic() + seconds
        while not self.should_stop():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self.done_event.wait(timeout=min(0.25, remaining))
        return True

    def active_step_names(self, target: WorkerTarget) -> list[str]:
        """Current registry step names for ``target``, accounting for prior supervisor relaunches."""
        if self.supervisor is not None:
            supervised_names = self.supervisor.active_names(target.key)
            if supervised_names:
                return supervised_names
        return list(target.step_names)

    def candidates_for_role(self, role: str) -> list[WorkerTarget]:
        """Build sorted ``WorkerTarget`` entries for every endpoint of ``role``."""
        from srtctl.cli.mixins.worker_stage import worker_step_name

        targets: list[WorkerTarget] = []
        for key, procs in self.endpoint_groups.items():
            mode, index, engine = key
            if mode != role or not procs:
                continue
            default_names = tuple(
                worker_step_name(mode, index, p.node, engine_id=engine, dp_rank=p.dp_rank) for p in procs
            )
            step_names = (
                tuple(self.supervisor.active_names(key) or default_names)
                if self.supervisor is not None
                else default_names
            )
            targets.append(
                WorkerTarget(
                    key=key,
                    mode=cast("WorkerMode", mode),
                    index=index,
                    processes=tuple(procs),
                    step_names=step_names,
                )
            )
        targets.sort(key=lambda t: (t.index, t.key[2]))
        return targets

    def mark_step_killed(self, step_name: str) -> None:
        """Record a step as signalled for termination before srun finishes exiting."""
        with self._stall_lock:
            self._killed_steps.add(step_name)

    def healthy_worker_count(self, role: str) -> int:
        """Count endpoints of ``role`` whose steps are running, not killed/frozen, and ready in the supervisor."""
        with self._stall_lock:
            frozen_workers = {s.worker for s in self._active_stalls}
            killed_steps = set(self._killed_steps)
        healthy = 0
        for target in self.candidates_for_role(role):
            if target.label in frozen_workers:
                continue
            if self.supervisor is not None and not self.supervisor.is_endpoint_ready(target.key):
                continue
            live_names = self.active_step_names(target)
            if not live_names or any(name in killed_steps for name in live_names):
                continue
            procs = [self.registry.get_process(name) for name in live_names]
            if all(proc is not None and proc.is_running for proc in procs):
                healthy += 1
        return healthy

    def register_stall(self, stalled: StalledProcess) -> None:
        with self._stall_lock:
            self._active_stalls.append(stalled)

    def unregister_stalls(self, resumed: list[StalledProcess]) -> None:
        keys = {(s.node, s.pid) for s in resumed}
        with self._stall_lock:
            self._active_stalls = [s for s in self._active_stalls if (s.node, s.pid) not in keys]

    def drain_active_stalls(self) -> list[StalledProcess]:
        with self._stall_lock:
            stalled = list(self._active_stalls)
            self._active_stalls.clear()
            return stalled


class ChaosRunner:
    """Executes ``config.chaos.events`` sequentially on a background thread while the benchmark runs."""

    def __init__(
        self,
        *,
        chaos: ChaosConfig,
        config: SrtConfig,
        runtime: RuntimeContext,
        registry: ProcessRegistry,
        supervisor: WorkerSupervisor | None,
        endpoint_groups: dict[EndpointKey, list[Process]],
        stop_event: threading.Event,
        scraper: Callable[[str], str | None] | None = None,
    ) -> None:
        self._chaos = chaos
        self._done_event = threading.Event()
        self._scraper = scraper
        self._rng = random.Random(chaos.seed)
        self._ctx = ChaosContext(
            config=config,
            runtime=runtime,
            registry=registry,
            supervisor=supervisor,
            endpoint_groups=endpoint_groups,
            stop_event=stop_event,
            done_event=self._done_event,
            rng=self._rng,
            leashed=chaos.leashed,
            min_healthy=chaos.min_healthy,
        )
        self._lock = threading.Lock()
        self._records: list[ChaosEventRecord] = []
        self._thread: threading.Thread | None = None

    @property
    def records(self) -> list[ChaosEventRecord]:
        with self._lock:
            return list(self._records)

    @property
    def events_path(self) -> Path:
        return self._ctx.runtime.log_dir / CHAOS_EVENTS_FILENAME

    def start(self) -> None:
        """Start the background chaos timeline thread."""
        if not self._chaos.enabled or self._thread is not None:
            return
        self._write()
        self._thread = threading.Thread(target=self._run, name="ChaosTimeline", daemon=True)
        self._thread.start()
        logger.info("Chaos timeline started (%d event(s) -> %s)", len(self._chaos.events), self.events_path)

    def stop(self, *, timeout: float = 30.0) -> None:
        """Signal the timeline to stop, wait for clean exit, and resume any paused processes."""
        self._done_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Chaos timeline thread still active after %.0fs", timeout)
        remaining = self._ctx.drain_active_stalls()
        if remaining:
            logger.info("Chaos teardown: resuming %d stalled process(es)", len(remaining))
            resume_stalled_processes(remaining, self._ctx, tag="teardown")
        self._write()

    def _run(self) -> None:
        started = time.monotonic()
        try:
            next_idx = 0
            for event in self._chaos.events:
                if event.after_seconds > 0 and self._ctx.wait(event.after_seconds):
                    self._record_skipped(next_idx, event, "stopped before scheduled time")
                    next_idx += 1
                    continue
                if self._ctx.should_stop():
                    self._record_skipped(next_idx, event, "stopped before execution")
                    next_idx += 1
                    continue
                self._execute_event(next_idx, event)
                next_idx += 1

            schedule = self._chaos.schedule
            if schedule is not None and not self._ctx.should_stop():
                # initial_delay_seconds counts from benchmark start, not from the last scripted event.
                warmup_left = started + schedule.initial_delay_seconds - time.monotonic()
                if warmup_left > 0 and self._ctx.wait(warmup_left):
                    return
                fired = 0
                while not self._ctx.should_stop():
                    if schedule.max_events is not None and fired >= schedule.max_events:
                        break
                    ev = schedule.to_event(name=f"soak_{fired}")
                    self._execute_event(next_idx, ev)
                    next_idx += 1
                    fired += 1
                    if schedule.max_events is not None and fired >= schedule.max_events:
                        break
                    delay = sample_schedule_delay(schedule, self._rng)
                    logger.info(
                        "Chaos schedule: next %s on %s in %.1fs (mean=%.1fs, min=%.1fs)",
                        schedule.type,
                        schedule.role,
                        delay,
                        schedule.mean_interval_seconds,
                        schedule.min_interval_seconds,
                    )
                    if self._ctx.wait(delay):
                        break
        finally:
            # Untimed freezes stay paused until benchmark teardown. stop() resumes them, but a
            # pause registered after its drain (join timed out mid-event) is resumed here.
            if self._ctx.should_stop():
                late = self._ctx.drain_active_stalls()
                if late:
                    resume_stalled_processes(late, self._ctx, tag="late")
            self._write()

    def _new_record(self, idx: int, event: ChaosEventConfig, status: str) -> ChaosEventRecord:
        iso, ns = _now_utc()
        rec = ChaosEventRecord(
            index=idx,
            name=event.name or f"{event.type}_{idx}",
            type=event.type,
            role=event.role,
            workers="all" if event.type == "rolling_restart" else event.workers,
            status=status,
            started_at=iso,
            started_at_ns=ns,
        )
        with self._lock:
            self._records.append(rec)
        return rec

    def _record_skipped(self, idx: int, event: ChaosEventConfig, reason: str) -> None:
        rec = self._new_record(idx, event, "skipped")
        rec.ended_at, rec.ended_at_ns = rec.started_at, rec.started_at_ns
        rec.details = {"reason": reason}
        self._write()

    def _execute_event(self, idx: int, event: ChaosEventConfig) -> None:
        rec = self._new_record(idx, event, "running")
        self._write()

        try:
            candidates = self._ctx.candidates_for_role(event.role)
            if event.type == "rolling_restart":
                chosen = candidates
                sel_info: dict[str, Any] = {"workers": "all", "chosen": [t.label for t in chosen]}
            else:
                chosen, sel_info = resolve_worker_targets(
                    event,
                    candidates,
                    config=self._ctx.config,
                    runtime=self._ctx.runtime,
                    hot_metric=self._chaos.hot_metric,
                    rng=self._rng,
                    scraper=self._scraper,
                )
            rec.targets = [t.label for t in chosen]
            rec.selection = sel_info
            if not chosen:
                raise RuntimeError(f"No worker endpoints matched for role={event.role!r} workers={event.workers!r}")
            if event.type != "rolling_restart" and self._ctx.min_healthy > 0:
                healthy = self._ctx.healthy_worker_count(event.role)
                if healthy - len(chosen) < self._ctx.min_healthy:
                    logger.warning(
                        "Chaos[%d] %s skipped by min_healthy guard (healthy=%d, targets=%d, min_healthy=%d)",
                        idx,
                        rec.name,
                        healthy,
                        len(chosen),
                        self._ctx.min_healthy,
                    )
                    rec.status = "skipped"
                    rec.details = {
                        "reason": "min_healthy_guard",
                        "healthy": healthy,
                        "min_healthy": self._ctx.min_healthy,
                        "requested_targets": len(chosen),
                    }
                    return
            handler = ACTION_HANDLERS[event.type]
            rec.details = handler(event, chosen, self._ctx, idx)
            rec.status = "leashed" if self._ctx.leashed else "completed"
        except Exception as exc:
            logger.exception("Chaos event %d (%s) failed", idx, rec.name)
            rec.status = "failed"
            rec.error = str(exc)
        finally:
            ended_iso, ended_ns = _now_utc()
            rec.ended_at = ended_iso
            rec.ended_at_ns = ended_ns
            self._write()

    def _write(self) -> None:
        payload = {
            "leashed": self._chaos.leashed,
            "min_healthy": self._chaos.min_healthy,
            "seed": self._chaos.seed,
            "hot_metric": self._chaos.hot_metric,
            "events": [asdict(rec) for rec in self.records],
        }
        try:
            self._ctx.runtime.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.events_path
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n")
            tmp.replace(path)
        except OSError as exc:
            logger.debug("Could not write %s: %s", CHAOS_EVENTS_FILENAME, exc)
