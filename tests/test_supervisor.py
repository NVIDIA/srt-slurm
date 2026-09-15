# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the worker supervisor behind ``roles.<role>.restart``."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from subprocess import Popen
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.core.processes import ManagedProcess, ProcessRegistry, start_process_monitor
from srtctl.core.schema import RestartPolicy
from srtctl.core.supervisor import WORKER_RESTARTS_FILENAME, WorkerSupervisor, load_worker_restarts
from srtctl.core.topology import Process

ON_FAILURE = RestartPolicy(policy="on-failure", max_restarts=3, backoff_seconds=10, max_backoff_seconds=40)


def _process(mode: str = "decode", index: int = 1, node: str = "node-b", rank: int = 0) -> Process:
    return Process(
        node=node,
        gpu_indices=frozenset(range(4)),
        sys_port=9000 + index,
        http_port=8000 + index,
        endpoint_mode=mode,  # type: ignore[arg-type]
        endpoint_index=index,
        node_rank=rank,
    )


class FakeStep:
    """A ManagedProcess over a scriptable popen: alive until ``exit(code)`` is called."""

    def __init__(self, name: str, *, node: str = "node-b", log_file: Path | None = None, critical: bool = True):
        self.code: int | None = None
        popen = MagicMock(spec=Popen)
        popen.pid = 4242
        popen.poll.side_effect = lambda: self.code
        popen.wait.side_effect = self._wait
        popen.terminate.side_effect = lambda: self.exit(-15)
        popen.kill.side_effect = lambda: self.exit(-9)
        self.popen = popen
        self.managed = ManagedProcess(
            name=name,
            popen=popen,
            log_file=log_file,
            node=node,
            critical=critical,
            step_name=name,
            terminate_timeout=1.0,
        )

    def exit(self, code: int) -> None:
        self.code = code

    def _wait(self, timeout=None):
        if self.code is None:
            self.code = 0
        return self.code


class FakeLauncher:
    """Stands in for the worker stage: records relaunches and hands back fresh steps."""

    def __init__(self, *, fail: bool = False, probe: tuple[str, int] | None = None) -> None:
        self.fail = fail
        self.probe = probe
        self.calls: list[tuple[tuple[str, int], int]] = []
        self.steps: dict[str, FakeStep] = {}

    def relaunch_endpoint(self, endpoint_processes: list[Process], *, attempt: int) -> list[ManagedProcess]:
        leader = endpoint_processes[0]
        self.calls.append(((leader.endpoint_mode, leader.endpoint_index), attempt))
        if self.fail:
            raise RuntimeError("srun exploded")
        out = []
        for process in endpoint_processes:
            name = f"{process.endpoint_mode}_{process.endpoint_index}_{process.node}_r{attempt}"
            step = FakeStep(name, node=process.node)
            self.steps[name] = step
            out.append(step.managed)
        return out

    def worker_ready_probe(self, endpoint_processes: list[Process]) -> tuple[str, int] | None:
        return self.probe


class Harness:
    """One tracked endpoint, a fake clock, and the real registry + supervisor."""

    def __init__(
        self,
        tmp_path: Path,
        policy: RestartPolicy,
        *,
        processes: list[Process] | None = None,
        launcher: FakeLauncher | None = None,
        ready_timeout: float = 100.0,
        probe_interval: float = 0.0,
        critical: bool = True,
    ) -> None:
        self.now = 1000.0
        self.log_dir = tmp_path
        self.registry = ProcessRegistry(job_id="1")
        self.stop = threading.Event()
        self.launcher = launcher or FakeLauncher()
        self.supervisor = WorkerSupervisor(
            registry=self.registry,
            stop_event=self.stop,
            launcher=self.launcher,
            log_dir=tmp_path,
            ready_timeout=ready_timeout,
            probe_interval=probe_interval,
            clock=lambda: self.now,
        )
        self.processes = processes or [_process()]
        self.steps: dict[str, FakeStep] = {}
        for process in self.processes:
            name = f"{process.endpoint_mode}_{process.endpoint_index}_{process.node}"
            log = tmp_path / f"{process.node}_{process.endpoint_mode}_w{process.endpoint_index}.out"
            log.write_text(f"life one of {name}\n")
            step = FakeStep(name, node=process.node, log_file=log, critical=critical)
            self.steps[name] = step
            self.registry.add_process(step.managed)
        leader = self.processes[0]
        self.key = (leader.endpoint_mode, leader.endpoint_index)
        self.supervisor.track({self.key: list(self.processes)}, self.steps.keys(), lambda mode: policy)

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def tick(self) -> None:
        self.supervisor.reconcile()

    def on_disk(self) -> dict:
        return json.loads((self.log_dir / WORKER_RESTARTS_FILENAME).read_text())


class TestRestartPolicy:
    def test_default_is_off_and_never_relaunches(self):
        policy = RestartPolicy()
        assert policy.enabled is False
        assert policy.restarts_on(1) is False
        assert policy.restarts_on(0) is False

    def test_on_failure_and_always(self):
        on_failure = RestartPolicy(policy="on-failure")
        assert on_failure.enabled
        assert on_failure.restarts_on(1) is True
        assert on_failure.restarts_on(-9) is True
        assert on_failure.restarts_on(None) is True  # an unknown exit is not a clean one
        assert on_failure.restarts_on(0) is False
        assert RestartPolicy(policy="always").restarts_on(0) is True

    def test_backoff_doubles_and_caps(self):
        assert [ON_FAILURE.backoff(n) for n in (1, 2, 3, 4)] == [10, 20, 40, 40]
        assert RestartPolicy(backoff_seconds=0, max_backoff_seconds=0).backoff(5) == 0

    def test_rejects_nonsense(self):
        with pytest.raises(ValidationError, match="max_restarts"):
            RestartPolicy(max_restarts=-1)
        with pytest.raises(ValidationError, match="backoff_seconds must be 0"):
            RestartPolicy(backoff_seconds=-1)
        with pytest.raises(ValidationError, match="max_backoff_seconds"):
            RestartPolicy(backoff_seconds=30, max_backoff_seconds=5)


def test_never_policy_leaves_the_worker_to_the_registry(tmp_path: Path) -> None:
    h = Harness(tmp_path, RestartPolicy())
    step = h.steps["decode_1_node-b"]
    assert step.managed.supervised is False

    step.exit(1)
    h.tick()

    assert h.launcher.calls == []
    assert h.registry.check_failures() is True  # today's behavior: a critical exit fails the run
    assert h.supervisor.summary_line() is None


def test_on_failure_schedules_then_relaunches_after_the_backoff(tmp_path: Path) -> None:
    h = Harness(tmp_path, ON_FAILURE)
    old = h.steps["decode_1_node-b"]
    assert old.managed.supervised is True

    old.exit(1)
    h.tick()

    # Scheduled, not yet relaunched: the exited step stays registered but is not a failure.
    assert h.launcher.calls == []
    assert h.registry.check_failures() is False
    assert h.registry.get_process("decode_1_node-b") is old.managed
    (event,) = h.supervisor.events
    assert (event.outcome, event.exit_code, event.attempt, event.backoff_seconds) == ("scheduled", 1, 1, 10)

    h.advance(9)
    h.tick()
    assert h.launcher.calls == []

    h.advance(1)
    h.tick()
    assert h.launcher.calls == [(("decode", 1), 1)]
    replacement = h.registry.get_process("decode_1_node-b_r1")
    assert replacement is not None and replacement.supervised is True
    assert h.registry.get_process("decode_1_node-b") is None

    # The first life's log moved aside; the relaunch writes the canonical path fresh.
    rotated = tmp_path / "node-b_decode_w1.out.1"
    assert rotated.read_text() == "life one of decode_1_node-b\n"
    assert not (tmp_path / "node-b_decode_w1.out").exists()
    assert old.managed.log_file == rotated

    on_disk = h.on_disk()
    assert on_disk["total_restarts"] == 1
    assert on_disk["events"][0]["outcome"] == "relaunched"
    assert on_disk["events"][0]["crash_log"] == str(rotated)
    assert on_disk["events"][0]["relaunched_at"] is not None
    assert on_disk["endpoints"]["decode_1"] == {
        "policy": "on-failure",
        "restarts": 1,
        "max_restarts": 3,
        "exhausted": False,
    }
    assert h.supervisor.summary_line() == "Worker restarts: 1 (decode_1 x1)"


def test_backoff_grows_across_restarts_of_the_same_endpoint(tmp_path: Path) -> None:
    h = Harness(tmp_path, ON_FAILURE)
    h.steps["decode_1_node-b"].exit(1)
    h.tick()
    h.advance(10)
    h.tick()

    h.launcher.steps["decode_1_node-b_r1"].exit(137)
    h.tick()
    assert h.supervisor.events[-1].backoff_seconds == 20
    h.advance(19)
    h.tick()
    assert len(h.launcher.calls) == 1
    h.advance(1)
    h.tick()
    assert h.launcher.calls[-1] == (("decode", 1), 2)
    assert (tmp_path / "node-b_decode_w1.out.1").exists()
    assert h.supervisor.total_restarts == 2


def test_a_clean_exit_under_on_failure_is_handed_back(tmp_path: Path) -> None:
    h = Harness(tmp_path, ON_FAILURE)
    step = h.steps["decode_1_node-b"]

    step.exit(0)
    h.tick()

    assert h.launcher.calls == []
    assert step.managed.supervised is False
    assert h.supervisor.events == []
    assert h.registry.check_failures() is False  # exit 0 is no failure for the registry either
    h.advance(100)
    h.tick()
    assert h.launcher.calls == []  # settled for good


def test_always_relaunches_a_clean_exit(tmp_path: Path) -> None:
    h = Harness(tmp_path, RestartPolicy(policy="always", backoff_seconds=0, max_backoff_seconds=0))
    h.steps["decode_1_node-b"].exit(0)
    h.tick()  # schedules (due now)
    h.tick()  # relaunches
    assert h.launcher.calls == [(("decode", 1), 1)]
    assert h.supervisor.events[0].exit_code == 0


def test_exhausted_restarts_return_the_endpoint_to_critical_semantics(tmp_path: Path) -> None:
    policy = RestartPolicy(policy="on-failure", max_restarts=1, backoff_seconds=0, max_backoff_seconds=0)
    h = Harness(tmp_path, policy)
    h.steps["decode_1_node-b"].exit(1)
    h.tick()
    h.tick()
    assert h.launcher.calls == [(("decode", 1), 1)]
    relaunched = h.launcher.steps["decode_1_node-b_r1"]
    assert h.registry.check_failures() is False

    relaunched.exit(1)
    h.tick()

    assert len(h.launcher.calls) == 1
    assert relaunched.managed.supervised is False
    assert h.supervisor.events[-1].outcome == "exhausted"
    assert h.supervisor.events[-1].attempt == 2
    assert h.on_disk()["endpoints"]["decode_1"]["exhausted"] is True
    assert h.registry.check_failures() is True  # critical (the default): the monitor tears the job down
    h.tick()
    assert len(h.launcher.calls) == 1  # settled


def test_exhausted_restarts_of_a_non_critical_role_do_not_fail_the_run(tmp_path: Path) -> None:
    policy = RestartPolicy(policy="on-failure", max_restarts=1, backoff_seconds=0, max_backoff_seconds=0)
    h = Harness(tmp_path, policy, critical=False)
    h.steps["decode_1_node-b"].exit(1)
    h.tick()
    h.tick()
    # The launcher builds replacements with the registry's default; mirror the role's flag.
    relaunched = h.launcher.steps["decode_1_node-b_r1"]
    relaunched.managed.critical = False
    relaunched.exit(1)
    h.tick()
    assert relaunched.managed.supervised is False
    assert h.registry.check_failures() is False


def test_a_multi_node_endpoint_restarts_as_a_unit(tmp_path: Path) -> None:
    processes = [_process(node="node-a", rank=0), _process(node="node-b", rank=1)]
    h = Harness(tmp_path, ON_FAILURE, processes=processes)
    leader, follower = h.steps["decode_1_node-a"], h.steps["decode_1_node-b"]

    follower.exit(1)
    with patch("srtctl.core.processes.shutil.which", return_value=None):  # no scancel: SIGTERM srun directly
        h.tick()

    # The surviving rank cannot carry on without its peer; it was stopped so the endpoint comes back whole.
    leader.popen.terminate.assert_called_once()
    assert leader.code == -15
    assert h.supervisor.events[-1].worker == "decode_1_node-b"

    h.advance(10)
    h.tick()
    assert h.launcher.calls == [(("decode", 1), 1)]
    assert sorted(h.launcher.steps) == ["decode_1_node-a_r1", "decode_1_node-b_r1"]
    assert h.registry.get_process("decode_1_node-a") is None
    assert h.registry.get_process("decode_1_node-b") is None
    assert (tmp_path / "node-a_decode_w1.out.1").exists()
    assert (tmp_path / "node-b_decode_w1.out.1").exists()


def test_the_stop_event_cancels_a_pending_relaunch(tmp_path: Path) -> None:
    h = Harness(tmp_path, ON_FAILURE)
    h.steps["decode_1_node-b"].exit(1)
    h.tick()
    h.stop.set()
    h.advance(10)
    h.tick()
    assert h.launcher.calls == []


def test_a_failed_launch_hands_the_exited_steps_back(tmp_path: Path) -> None:
    h = Harness(tmp_path, ON_FAILURE, launcher=FakeLauncher(fail=True))
    old = h.steps["decode_1_node-b"]
    old.exit(1)
    h.tick()
    h.advance(10)
    h.tick()

    assert h.launcher.calls == [(("decode", 1), 1)]
    assert h.registry.get_process("decode_1_node-b") is old.managed
    assert old.managed.supervised is False
    assert h.supervisor.events[-1].outcome == "launch_failed"
    assert h.registry.check_failures() is True


def test_readiness_probe_records_when_the_replacement_serves(tmp_path: Path) -> None:
    h = Harness(tmp_path, ON_FAILURE, launcher=FakeLauncher(probe=("10.0.0.2", 9001)), ready_timeout=60)
    h.steps["decode_1_node-b"].exit(1)
    h.tick()
    h.advance(10)
    h.tick()  # relaunched at t=1010

    with patch("srtctl.core.supervisor.probe_http", return_value=False) as probe:
        h.advance(5)
        h.tick()
    probe.assert_called_once_with("10.0.0.2", 9001, "/health", 200, request_timeout=2.0)
    assert h.supervisor.events[-1].outcome == "relaunched"

    with patch("srtctl.core.supervisor.probe_http", return_value=True):
        h.advance(7)
        h.tick()
    event = h.supervisor.events[-1]
    assert event.outcome == "ready"
    assert event.ready_seconds == 12.0
    assert event.ready_at is not None
    assert h.on_disk()["events"][-1]["outcome"] == "ready"

    with patch("srtctl.core.supervisor.probe_http") as probe:
        h.tick()
    probe.assert_not_called()  # ready once; no more probing


def test_readiness_probe_gives_up_after_the_timeout_but_leaves_the_worker_up(tmp_path: Path) -> None:
    h = Harness(tmp_path, ON_FAILURE, launcher=FakeLauncher(probe=("10.0.0.2", 9001)), ready_timeout=60)
    h.steps["decode_1_node-b"].exit(1)
    h.tick()
    h.advance(10)
    h.tick()

    with patch("srtctl.core.supervisor.probe_http", return_value=False):
        h.advance(60)
        h.tick()

    assert h.supervisor.events[-1].outcome == "not_ready"
    replacement = h.registry.get_process("decode_1_node-b_r1")
    assert replacement is not None and replacement.is_running
    with patch("srtctl.core.supervisor.probe_http") as probe:
        h.tick()
    probe.assert_not_called()


def test_readiness_probe_is_rate_limited(tmp_path: Path) -> None:
    """Each failed probe lands as a 503 line in the worker log (16 per relaunch on sa-b200 at a
    2 s tick), so the supervisor probes at most once per probe_interval, and not right away."""
    launcher = FakeLauncher(probe=("10.0.0.2", 9001))
    h = Harness(tmp_path, ON_FAILURE, launcher=launcher, ready_timeout=100, probe_interval=10)
    h.steps["decode_1_node-b"].exit(1)
    h.tick()
    h.advance(10)
    h.tick()  # relaunched at t=1010

    with patch("srtctl.core.supervisor.probe_http", return_value=False) as probe:
        for _ in range(5):  # t=1012..1020: ticks every 2 s
            h.advance(2)
            h.tick()
    assert probe.call_count == 1  # only the tick at t=1020 probed
    with patch("srtctl.core.supervisor.probe_http", return_value=True) as probe:
        h.advance(2)
        h.tick()  # t=1022: inside the interval, no probe
        h.advance(8)
        h.tick()  # t=1030: probes and finds it ready
    assert probe.call_count == 1
    assert h.supervisor.events[-1].outcome == "ready"
    assert h.supervisor.events[-1].ready_seconds == 20.0


def test_no_health_port_means_no_probe(tmp_path: Path) -> None:
    h = Harness(tmp_path, ON_FAILURE)  # FakeLauncher() has probe=None
    h.steps["decode_1_node-b"].exit(1)
    h.tick()
    h.advance(10)
    h.tick()
    with patch("srtctl.core.supervisor.probe_http") as probe:
        h.advance(1)
        h.tick()
    probe.assert_not_called()
    assert h.supervisor.events[-1].outcome == "relaunched"


def test_track_skips_roles_without_a_policy_and_matches_step_names_exactly(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="srtctl.core.supervisor")
    registry = ProcessRegistry(job_id="1")
    supervisor = WorkerSupervisor(registry=registry, stop_event=threading.Event(), launcher=FakeLauncher())
    policies = {"prefill": RestartPolicy(), "decode": ON_FAILURE}
    steps = {name: FakeStep(name) for name in ("prefill_0_node-a", "decode_1_node-b", "decode_10_node-c")}
    for step in steps.values():
        registry.add_process(step.managed)

    supervisor.track(
        {
            ("prefill", 0): [_process("prefill", 0, "node-a")],
            ("decode", 1): [_process()],
            ("agg", 0): [_process("agg", 0, "node-z")],  # no steps registered for it
        },
        steps.keys(),
        policies.__getitem__ if False else (lambda mode: policies.get(mode, ON_FAILURE)),
    )

    assert steps["prefill_0_node-a"].managed.supervised is False  # policy never
    assert steps["decode_1_node-b"].managed.supervised is True
    assert steps["decode_10_node-c"].managed.supervised is False  # prefix decode_1_ must not swallow decode_10_
    assert "no registered steps found for agg_0" in caplog.text
    assert supervisor.summary()["endpoints"] == {}  # nothing restarted yet


def test_monitor_thread_runs_reconcile_and_survives_an_exception() -> None:
    registry = ProcessRegistry(job_id="1")
    stop = threading.Event()
    calls: list[int] = []

    def reconcile() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        if len(calls) >= 3:
            stop.set()

    thread = start_process_monitor(stop, registry, poll_interval=0.01, reconcile=reconcile)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(calls) >= 3


def test_load_worker_restarts_ignores_missing_or_empty_records(tmp_path: Path) -> None:
    assert load_worker_restarts(tmp_path) is None
    (tmp_path / WORKER_RESTARTS_FILENAME).write_text(json.dumps({"total_restarts": 0, "events": []}))
    assert load_worker_restarts(tmp_path) is None
    (tmp_path / WORKER_RESTARTS_FILENAME).write_text(json.dumps({"total_restarts": 2, "events": []}))
    assert load_worker_restarts(tmp_path) == {"total_restarts": 2, "events": []}
    (tmp_path / WORKER_RESTARTS_FILENAME).write_text("not json")
    assert load_worker_restarts(tmp_path) is None
