# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The restart unit is one engine of a worker, not the worker.

Under vLLM shadow engine recovery (``engine.failover``) a worker runs engine 0
and standby engines on the same GPUs, each its own step. When the serving engine
dies the shadow takes over; relaunching "the endpoint" would stop the engine that
just took over. So the supervisor keys its units by ``(mode, index, engine)`` and
matches steps by exact name.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen
from unittest.mock import MagicMock

from srtctl.cli.mixins.worker_stage import worker_step_name
from srtctl.core.processes import ManagedProcess, ProcessRegistry
from srtctl.core.schema import RestartPolicy
from srtctl.core.supervisor import WorkerSupervisor
from srtctl.core.topology import Process

ALWAYS = RestartPolicy(policy="always", max_restarts=3, backoff_seconds=10, max_backoff_seconds=40)


@dataclass(frozen=True)
class EngineProcess(Process):
    """A Process with the ``engine_id`` shadow engine recovery adds; 0 is the usual engine."""

    engine_id: int = 0


def _engine(engine_id: int) -> EngineProcess:
    return EngineProcess(
        node="node-b",
        gpu_indices=frozenset({3}),
        sys_port=7500 + engine_id,
        http_port=6100 + engine_id,
        endpoint_mode="agg",  # type: ignore[arg-type]
        endpoint_index=0,
        engine_id=engine_id,
    )


class FakeStep:
    def __init__(self, name: str) -> None:
        self.code: int | None = None
        popen = MagicMock(spec=Popen)
        popen.pid = 4242
        popen.poll.side_effect = lambda: self.code
        popen.wait.side_effect = self._wait
        popen.terminate.side_effect = lambda: self.exit(-15)
        popen.kill.side_effect = lambda: self.exit(-9)
        self.popen = popen
        self.managed = ManagedProcess(name=name, popen=popen, node="node-b", step_name=name, terminate_timeout=1.0)

    def exit(self, code: int) -> None:
        self.code = code

    def _wait(self, timeout=None):
        if self.code is None:
            self.code = 0
        return self.code


class EngineLauncher:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []  # (engine_id, attempt)
        self.steps: dict[str, FakeStep] = {}

    def relaunch_endpoint(self, endpoint_processes: list[Process], *, attempt: int) -> list[ManagedProcess]:
        out = []
        for process in endpoint_processes:
            engine = getattr(process, "engine_id", 0)
            self.calls.append((engine, attempt))
            name = worker_step_name(process.endpoint_mode, process.endpoint_index, process.node, attempt, engine)
            step = FakeStep(name)
            self.steps[name] = step
            out.append(step.managed)
        return out

    def worker_ready_probe(self, endpoint_processes: list[Process]) -> tuple[str, int] | None:
        return None


def test_step_names_carry_the_engine_before_the_attempt() -> None:
    assert worker_step_name("agg", 0, "node-b") == "agg_0_node-b"
    assert worker_step_name("agg", 0, "node-b", engine_id=1) == "agg_0_node-b_e1"
    assert worker_step_name("agg", 0, "node-b", 2, 1) == "agg_0_node-b_e1_r2"


def test_each_engine_of_a_worker_is_its_own_restart_unit(tmp_path: Path) -> None:
    now = 1000.0
    registry = ProcessRegistry(job_id="1")
    launcher = EngineLauncher()
    supervisor = WorkerSupervisor(
        registry=registry,
        stop_event=threading.Event(),
        launcher=launcher,
        log_dir=tmp_path,
        probe_interval=0.0,
        clock=lambda: now,
    )
    e0, e1 = _engine(0), _engine(1)
    steps = {name: FakeStep(name) for name in ("agg_0_node-b", "agg_0_node-b_e1")}
    for step in steps.values():
        registry.add_process(step.managed)
    supervisor.track({("agg", 0, 0): [e0], ("agg", 0, 1): [e1]}, steps.keys(), lambda mode: ALWAYS)

    # Engine 0 (serving) dies; the shadow has the lock now and must be left alone.
    steps["agg_0_node-b"].exit(137)
    supervisor.reconcile()
    steps["agg_0_node-b_e1"].popen.terminate.assert_not_called()
    assert steps["agg_0_node-b_e1"].code is None
    event = supervisor.events[-1]
    assert (event.worker, event.engine, event.attempt) == ("agg_0_node-b", 0, 1)

    now += 10
    supervisor.reconcile()
    assert launcher.calls == [(0, 1)]
    assert list(launcher.steps) == ["agg_0_node-b_r1"]
    assert registry.get_process("agg_0_node-b_e1") is steps["agg_0_node-b_e1"].managed
    assert supervisor.summary()["endpoints"]["agg_0"]["restarts"] == 1

    # Later the promoted engine 1 dies; only its unit relaunches, labelled with its engine.
    steps["agg_0_node-b_e1"].exit(137)
    supervisor.reconcile()
    now += 10
    supervisor.reconcile()
    assert launcher.calls == [(0, 1), (1, 1)]
    assert "agg_0_node-b_e1_r1" in launcher.steps
    assert supervisor.summary()["endpoints"]["agg_0_e1"]["restarts"] == 1
    assert supervisor.events[-1].engine == 1


def test_two_tuple_keys_still_mean_engine_zero(tmp_path: Path) -> None:
    registry = ProcessRegistry(job_id="1")
    supervisor = WorkerSupervisor(registry=registry, stop_event=threading.Event(), launcher=EngineLauncher())
    step = FakeStep("agg_0_node-b")
    registry.add_process(step.managed)
    supervisor.track({("agg", 0): [_engine(0)]}, ["agg_0_node-b", "agg_0_node-b_e1"], lambda mode: ALWAYS)
    assert step.managed.supervised is True
    assert supervisor.summary_line() is None
