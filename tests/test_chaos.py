# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit and mock-orchestrator tests for concurrent chaos fault injection."""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import replace
from pathlib import Path
from subprocess import Popen
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.chaos import (
    CHAOS_EVENTS_FILENAME,
    ChaosRunner,
    WorkerTarget,
    parse_prometheus_metric,
    resolve_worker_targets,
    sample_schedule_delay,
)
from srtctl.chaos.actions import _shares_node
from srtctl.core.config import load_config
from srtctl.core.processes import ManagedProcess, ProcessRegistry
from srtctl.core.schema import ChaosConfig, ChaosEventConfig, ChaosScheduleConfig, SrtConfig
from srtctl.core.supervisor import WorkerSupervisor
from srtctl.core.topology import Process
from srtctl.mock import MockOptions, run_mock_sweep


def _base_recipe(**overrides) -> dict:
    doc = {
        "schema": 2,
        "name": "chaos-unit-test",
        "model": {"path": "/models/test", "container": "/containers/test.sqsh", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8},
        "frontend": {"type": "dynamo", "enable_multiple_frontends": False},
        "engine": {"type": "mocker", "engine_type": "vllm", "speedup_ratio": 100},
        "roles": {"agg": {"nodes": 1, "workers": 2}},
        "benchmark": {"type": "custom", "command": "echo ok"},
    }
    doc.update(overrides)
    return doc


def _fake_node_helper(payload: dict, pids: tuple[int, ...] = (777, 778)) -> dict:
    """Stand-in for the node helper: discovery lists ``pids``; signalling reports ``T`` after SIGSTOP."""
    if "pids" not in payload:
        return {"matches": [{"pid": pid, "cmd": payload["pattern"]} for pid in pids], "errors": []}
    state = "T" if payload["signal"] == "STOP" else "S"
    return {"signalled": payload["pids"], "verified_states": {str(p): state for p in payload["pids"]}, "errors": []}


def _load_dict(doc: dict) -> SrtConfig:
    return SrtConfig.Schema().load(doc)


def _proc(mode: str = "agg", index: int = 0, node: str = "node0", sys_port: int = 9000) -> Process:
    return Process(
        node=node,
        gpu_indices=frozenset({index}),
        sys_port=sys_port + index,
        http_port=8000 + index,
        endpoint_mode=mode,  # type: ignore[arg-type]
        endpoint_index=index,
        node_rank=0,
    )


class _FakeStep:
    def __init__(self, name: str, *, node: str = "node0", critical: bool = True):
        self.code: int | None = None
        popen = MagicMock(spec=Popen)
        popen.pid = 5000 + abs(hash(name)) % 1000
        popen.poll.side_effect = lambda: self.code
        popen.wait.side_effect = self._wait
        popen.terminate.side_effect = lambda: self.exit(-15)
        popen.kill.side_effect = lambda: self.exit(-9)
        self.popen = popen
        self.managed = ManagedProcess(
            name=name,
            popen=popen,
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


class _FakeLauncher:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, int], int]] = []
        self.steps: dict[str, _FakeStep] = {}

    def relaunch_endpoint(self, endpoint_processes: list[Process], *, attempt: int) -> list[ManagedProcess]:
        leader = endpoint_processes[0]
        self.calls.append(((leader.endpoint_mode, leader.endpoint_index), attempt))
        out = []
        for process in endpoint_processes:
            name = f"{process.endpoint_mode}_{process.endpoint_index}_{process.node}_r{attempt}"
            step = _FakeStep(name, node=process.node)
            self.steps[name] = step
            out.append(step.managed)
        return out

    def worker_ready_probe(self, endpoint_processes: list[Process]) -> tuple[str, int] | None:
        return None


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestChaosSchema:
    def test_kill_rejected_with_observability_nsys(self):
        with pytest.raises(ValidationError, match="observability.nsys"):
            _load_dict(
                _base_recipe(
                    benchmark={"type": "sa-bench", "isl": 128, "osl": 128, "concurrencies": "4"},
                    observability={"enabled": True},
                    chaos={"events": [{"type": "kill", "role": "agg"}]},
                )
            )

    def test_main_alias_rejected_for_native_workers(self):
        with pytest.raises(ValidationError, match="process='main'"):
            _load_dict(
                _base_recipe(
                    frontend={"type": "sglang-router"},
                    engine={"type": "sglang"},
                    chaos={"events": [{"type": "kill", "role": "agg", "process": "main"}]},
                )
            )

    def test_valid_chaos_events_load(self):
        cfg = _load_dict(
            _base_recipe(
                chaos={
                    "seed": 123,
                    "events": [
                        {
                            "after_seconds": 10,
                            "type": "kill",
                            "role": "agg",
                            "workers": "hottest",
                            "respawn": True,
                            "wait_ready": True,
                        },
                        {
                            "after_seconds": 5,
                            "type": "kill",
                            "role": "agg",
                            "workers": [0, 1],
                            "process": "rank",
                            "rank": 1,
                            "signal": "SIGKILL",
                        },
                        {
                            "after_seconds": 5,
                            "type": "freeze",
                            "role": "agg",
                            "workers": "random:2",
                            "duration_seconds": 10,
                        },
                        {
                            "after_seconds": 5,
                            "type": "rolling_restart",
                            "role": "agg",
                            "max_unavailable": 1,
                            "graceful": True,
                        },
                    ],
                }
            )
        )
        assert cfg.chaos.enabled is True
        assert cfg.chaos.seed == 123
        assert len(cfg.chaos.events) == 4
        assert cfg.chaos.events[1].normalized_signal == "KILL"
        assert cfg.chaos.events[2].effective_process == "engine"
        assert cfg.chaos.events[2].parse_workers() == ("random", (), 2)

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError, match="has no workers"):
            _load_dict(_base_recipe(chaos={"events": [{"type": "kill", "role": "prefill"}]}))

    def test_out_of_range_worker_index_rejected(self):
        with pytest.raises(ValidationError, match="out of range"):
            _load_dict(_base_recipe(chaos={"events": [{"type": "kill", "role": "agg", "workers": [2]}]}))

    def test_invalid_workers_selector_rejected(self):
        with pytest.raises(ValidationError, match="workers"):
            _load_dict(_base_recipe(chaos={"events": [{"type": "kill", "role": "agg", "workers": "hottest:0"}]}))

    def test_wait_ready_without_restart_or_respawn_rejected(self):
        with pytest.raises(ValidationError, match="wait_ready"):
            _load_dict(_base_recipe(chaos={"events": [{"type": "kill", "role": "agg", "wait_ready": True}]}))

    def test_invalid_signal_rejected(self):
        with pytest.raises(ValidationError, match="not a recognized POSIX signal"):
            _load_dict(_base_recipe(chaos={"events": [{"type": "kill", "role": "agg", "signal": "BOGUS"}]}))

    def test_negative_after_seconds_rejected(self):
        with pytest.raises(ValidationError, match="after_seconds"):
            _load_dict(_base_recipe(chaos={"events": [{"type": "kill", "role": "agg", "after_seconds": -1}]}))

    def test_unsupported_backend_alias_requires_explicit_process_substring(self):
        doc = _base_recipe(
            engine="tokenspeed",
            frontend={"type": "dynamo", "enable_multiple_frontends": False},
            roles={"agg": {"nodes": 1, "workers": 1, "gpus": 8}},
            chaos={"events": [{"type": "freeze", "role": "agg"}]},
        )
        with pytest.raises(ValidationError, match="explicit cmdline substring"):
            _load_dict(doc)

        doc["chaos"]["events"][0]["process"] = "my_custom_worker"
        cfg = _load_dict(doc)
        assert cfg.chaos.events[0].effective_process == "my_custom_worker"

    def test_min_healthy_must_be_less_than_role_workers(self):
        with pytest.raises(ValidationError, match="min_healthy"):
            _load_dict(_base_recipe(chaos={"min_healthy": 2, "events": [{"type": "kill", "role": "agg"}]}))

    def test_schedule_validation(self):
        with pytest.raises(ValidationError, match="cannot exceed mean_interval_seconds"):
            _load_dict(
                _base_recipe(
                    chaos={
                        "schedule": {
                            "mean_interval_seconds": 30,
                            "min_interval_seconds": 60,
                            "role": "agg",
                        }
                    }
                )
            )
        cfg = _load_dict(
            _base_recipe(
                chaos={
                    "leashed": True,
                    "min_healthy": 1,
                    "seed": 5,
                    "schedule": {
                        "mean_interval_seconds": 120,
                        "min_interval_seconds": 60,
                        "max_events": 3,
                        "role": "agg",
                    },
                }
            )
        )
        assert cfg.chaos.enabled is True
        assert cfg.chaos.leashed is True
        assert cfg.chaos.min_healthy == 1
        assert cfg.chaos.schedule is not None
        assert cfg.chaos.schedule.max_events == 3


# ---------------------------------------------------------------------------
# Target selection & Prometheus metric parsing
# ---------------------------------------------------------------------------


class TestTargetSelection:
    def test_parse_prometheus_metric_sums_labeled_series(self):
        text = (
            "# HELP dynamo_component_inflight_requests Inflight\n"
            "# TYPE dynamo_component_inflight_requests gauge\n"
            'dynamo_component_inflight_requests{model="m",dp="0"} 4\n'
            'dynamo_component_inflight_requests{model="m",dp="1"} 7.5\n'
            "dynamo_component_inflight_requests_total 999\n"
        )
        assert parse_prometheus_metric(text, "dynamo_component_inflight_requests") == 11.5
        assert parse_prometheus_metric(text, "missing_metric") is None

    def test_hottest_picks_busiest_workers(self):
        cfg = _load_dict(_base_recipe(roles={"agg": {"nodes": 1, "workers": 3}}))
        runtime = SimpleNamespace(network_interface=None)
        candidates = [
            WorkerTarget(("agg", i, 0), "agg", i, (_proc("agg", i, f"node{i}"),), (f"agg_{i}_node{i}",))
            for i in range(3)
        ]
        metrics_by_host = {
            "10.0.0.0": "dynamo_component_inflight_requests 2\n",
            "10.0.0.1": "dynamo_component_inflight_requests 18\n",
            "10.0.0.2": "dynamo_component_inflight_requests 9\n",
        }
        event = ChaosEventConfig(type="kill", role="agg", workers="hottest:2")
        with patch("srtctl.chaos.targets.get_hostname_ip", side_effect=lambda node, _iface=None: f"10.0.0.{node[-1]}"):
            chosen, info = resolve_worker_targets(
                event,
                candidates,
                config=cfg,
                runtime=runtime,  # type: ignore[arg-type]
                hot_metric="dynamo_component_inflight_requests",
                rng=random.Random(0),
                scraper=lambda url: next((txt for ip, txt in metrics_by_host.items() if ip in url), None),
            )
        assert [t.index for t in chosen] == [1, 2]
        assert info["fallback"] is False
        assert info["scores"]["agg_1"] == 18.0

    def test_hottest_falls_back_to_index_order_when_metrics_absent(self):
        cfg = _load_dict(_base_recipe())
        runtime = SimpleNamespace(network_interface=None)
        candidates = [
            WorkerTarget(("agg", i, 0), "agg", i, (_proc("agg", i, f"node{i}"),), (f"agg_{i}_node{i}",))
            for i in range(2)
        ]
        event = ChaosEventConfig(type="kill", role="agg", workers="hottest")
        with patch("srtctl.chaos.targets.get_hostname_ip", return_value="10.0.0.1"):
            chosen, info = resolve_worker_targets(
                event,
                candidates,
                config=cfg,
                runtime=runtime,  # type: ignore[arg-type]
                hot_metric="dynamo_component_inflight_requests",
                rng=random.Random(0),
                scraper=lambda _url: None,
            )
        assert [t.index for t in chosen] == [0]
        assert info["fallback"] is True

    def test_random_selection_respects_seed(self):
        cfg = _load_dict(_base_recipe(roles={"agg": {"nodes": 1, "workers": 4}}))
        runtime = SimpleNamespace(network_interface=None)
        candidates = [
            WorkerTarget(("agg", i, 0), "agg", i, (_proc("agg", i, f"node{i}"),), (f"agg_{i}_node{i}",))
            for i in range(4)
        ]
        event = ChaosEventConfig(type="kill", role="agg", workers="random")
        a, _ = resolve_worker_targets(
            event,
            candidates,
            config=cfg,
            runtime=runtime,  # type: ignore[arg-type]
            hot_metric="m",
            rng=random.Random(99),
        )
        b, _ = resolve_worker_targets(
            event,
            candidates,
            config=cfg,
            runtime=runtime,  # type: ignore[arg-type]
            hot_metric="m",
            rng=random.Random(99),
        )
        assert a == b


# ---------------------------------------------------------------------------
# ProcessRegistry per-target critical + WorkerSupervisor integration
# ---------------------------------------------------------------------------


class TestFaultToleranceAndActions:
    def test_kill_marks_only_target_non_critical(self, tmp_path: Path):
        cfg = _load_dict(_base_recipe())
        runtime = SimpleNamespace(job_id="100", log_dir=tmp_path, network_interface=None)
        registry = ProcessRegistry(job_id="100")
        p0, p1 = _proc("agg", 0, "node0"), _proc("agg", 1, "node0")
        s0, s1 = _FakeStep("agg_0_node0", node="node0"), _FakeStep("agg_1_node0", node="node0")
        registry.add_process(s0.managed)
        registry.add_process(s1.managed)

        chaos = ChaosConfig(events=[ChaosEventConfig(type="kill", role="agg", workers=0)])
        runner = ChaosRunner(
            chaos=chaos,
            config=cfg,
            runtime=runtime,  # type: ignore[arg-type]
            registry=registry,
            supervisor=None,
            endpoint_groups={("agg", 0, 0): [p0], ("agg", 1, 0): [p1]},
            stop_event=threading.Event(),
        )
        with (
            patch("srtctl.chaos.actions.list_step_ids", return_value=None),
            patch("srtctl.chaos.actions.signal_step", return_value=False),
        ):
            runner.start()
            runner.stop(timeout=5.0)

        # Target exited without triggering a critical failure; surviving worker stays critical.
        assert not s0.managed.is_running
        assert s0.managed.critical is False
        assert s1.managed.critical is True
        assert registry.check_failures() is False

        # An unintended crash on the surviving worker still fails the job.
        s1.exit(137)
        assert registry.check_failures() is True

        recorded = json.loads((tmp_path / CHAOS_EVENTS_FILENAME).read_text())
        assert len(recorded["events"]) == 1
        assert recorded["events"][0]["status"] == "completed"
        assert recorded["events"][0]["targets"] == ["agg_0"]

    def test_freeze_and_subprocess_kill_with_step_scoping(self, tmp_path: Path):
        cfg = _load_dict(_base_recipe())
        runtime = SimpleNamespace(job_id="101", log_dir=tmp_path, network_interface=None)
        registry = ProcessRegistry(job_id="101")
        p0 = _proc("agg", 0, "node0")
        s0 = _FakeStep("agg_0_node0", node="node0")
        registry.add_process(s0.managed)

        calls: list[dict] = []

        def fake_node_python(node, payload, *, log_dir, tag, het_group=None, timeout=30.0):
            calls.append({"node": node, "tag": tag, "payload": payload})
            return _fake_node_helper(payload)

        chaos = ChaosConfig(
            events=[
                ChaosEventConfig(type="freeze", role="agg", process="engine", rank=1, duration_seconds=0.05),
                ChaosEventConfig(type="kill", role="agg", process="rank", rank=1, signal="KILL"),
            ]
        )
        runner = ChaosRunner(
            chaos=chaos,
            config=cfg,
            runtime=runtime,  # type: ignore[arg-type]
            registry=registry,
            supervisor=None,
            endpoint_groups={("agg", 0, 0): [p0]},
            stop_event=threading.Event(),
        )
        with (
            patch("srtctl.chaos.actions.list_step_ids", return_value={"agg_0_node0": "101.3"}),
            patch("srtctl.chaos.actions.run_node_python", side_effect=fake_node_python),
        ):
            runner.start()
            deadline = time.monotonic() + 5.0
            while len(runner.records) < 2 or any(r.status == "running" for r in runner.records):
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            runner.stop(timeout=5.0)

        signal_calls = [c["payload"] for c in calls if "pids" in c["payload"]]
        assert [c["signal"] for c in signal_calls] == ["STOP", "CONT", "KILL"]
        assert all(c["pids"] == [778] for c in signal_calls)  # rank 1 of the discovered PIDs
        assert calls[0]["payload"] == {"pattern": "dynamo.mocker", "step_num": "3", "allow_unscoped": True}
        assert all(r.status == "completed" for r in runner.records)

    def test_unscoped_process_match_only_when_node_not_shared(self):
        p0, p1, p2 = _proc("agg", 0, "node0"), _proc("agg", 1, "node0"), _proc("agg", 2, "node1")
        ctx = SimpleNamespace(endpoint_groups={("agg", 0, 0): [p0], ("agg", 1, 0): [p1], ("agg", 2, 0): [p2]})
        target0 = WorkerTarget(("agg", 0, 0), "agg", 0, (p0,), ("agg_0_node0",))
        target2 = WorkerTarget(("agg", 2, 0), "agg", 2, (p2,), ("agg_2_node1",))
        assert _shares_node(target0, "node0", ctx)  # type: ignore[arg-type]
        assert not _shares_node(target2, "node1", ctx)  # type: ignore[arg-type]

    def _runner(self, tmp_path: Path, registry: ProcessRegistry, groups: dict, events: list, **kwargs) -> ChaosRunner:
        return ChaosRunner(
            chaos=ChaosConfig(events=events, **kwargs),
            config=_load_dict(_base_recipe()),
            runtime=SimpleNamespace(job_id="105", log_dir=tmp_path, network_interface=None),  # type: ignore[arg-type]
            registry=registry,
            supervisor=None,
            endpoint_groups=groups,
            stop_event=threading.Event(),
        )

    @staticmethod
    def _wait_records(runner: ChaosRunner, count: int) -> None:
        deadline = time.monotonic() + 5.0
        while len(runner.records) < count or any(r.status == "running" for r in runner.records):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)

    def test_process_kill_with_no_match_leaves_worker_critical(self, tmp_path: Path):
        registry = ProcessRegistry(job_id="105")
        step = _FakeStep("agg_0_node0", node="node0")
        registry.add_process(step.managed)
        calls: list[dict] = []

        def fake(node, payload, **_kwargs):
            calls.append(payload)
            return _fake_node_helper(payload, pids=(777,))

        runner = self._runner(
            tmp_path,
            registry,
            {("agg", 0, 0): [_proc("agg", 0, "node0")]},
            [ChaosEventConfig(type="kill", role="agg", process="rank", rank=1)],
        )
        with (
            patch("srtctl.chaos.actions.list_step_ids", return_value=None),
            patch("srtctl.chaos.actions.run_node_python", side_effect=fake),
        ):
            runner.start()
            self._wait_records(runner, 1)
            runner.stop(timeout=5.0)

        assert runner.records[0].status == "failed"
        assert all("pids" not in c for c in calls)
        managed = registry.get_process("agg_0_node0")
        assert managed is not None and managed.critical is True
        assert runner._ctx.healthy_worker_count("agg") == 1

    def test_freeze_resolves_every_target_before_pausing(self, tmp_path: Path):
        registry = ProcessRegistry(job_id="105")
        for i in (0, 1):
            registry.add_process(_FakeStep(f"agg_{i}_node{i}", node=f"node{i}").managed)
        calls: list[tuple[str, dict]] = []

        def fake(node, payload, **_kwargs):
            calls.append((node, payload))
            return _fake_node_helper(payload, pids=(777,) if node == "node0" else ())

        runner = self._runner(
            tmp_path,
            registry,
            {("agg", 0, 0): [_proc("agg", 0, "node0")], ("agg", 1, 0): [_proc("agg", 1, "node1")]},
            [ChaosEventConfig(type="freeze", role="agg", workers="all", duration_seconds=5.0)],
        )
        with (
            patch("srtctl.chaos.actions.list_step_ids", return_value=None),
            patch("srtctl.chaos.actions.run_node_python", side_effect=fake),
        ):
            runner.start()
            self._wait_records(runner, 1)
            runner.stop(timeout=5.0)

        assert runner.records[0].status == "failed"
        assert not any(payload.get("signal") == "STOP" for _, payload in calls)

    def test_rank_all_spans_every_node_of_a_multinode_worker(self, tmp_path: Path):
        registry = ProcessRegistry(job_id="105")
        registry.add_process(_FakeStep("agg_0_node0", node="node0").managed)
        registry.add_process(_FakeStep("agg_0_node1", node="node1").managed)
        leader, follower = _proc("agg", 0, "node0"), replace(_proc("agg", 0, "node1"), node_rank=1)
        stops: list[tuple[str, list[int]]] = []

        def fake(node, payload, **_kwargs):
            if payload.get("signal") == "STOP":
                stops.append((node, payload["pids"]))
            return _fake_node_helper(payload)

        runner = self._runner(
            tmp_path,
            registry,
            {("agg", 0, 0): [leader, follower]},
            [ChaosEventConfig(type="freeze", role="agg", process="engine", rank="all", duration_seconds=0.01)],
        )
        with (
            patch("srtctl.chaos.actions.list_step_ids", return_value={"agg_0_node0": "105.1", "agg_0_node1": "105.2"}),
            patch("srtctl.chaos.actions.run_node_python", side_effect=fake),
        ):
            runner.start()
            self._wait_records(runner, 1)
            runner.stop(timeout=5.0)

        assert runner.records[0].status == "completed"
        assert stops == [("node0", [777, 778]), ("node1", [777, 778])]

    def test_rolling_restart_stops_when_replacement_not_ready(self, tmp_path: Path):
        registry = ProcessRegistry(job_id="105")
        steps = [_FakeStep(f"agg_{i}_node{i}", node=f"node{i}") for i in (0, 1)]
        for step in steps:
            registry.add_process(step.managed)
        supervisor = MagicMock()
        supervisor.active_names.return_value = None
        supervisor.ensure_relaunch.return_value = 0
        supervisor.relaunch_status.return_value = (True, "not_ready")
        runner = ChaosRunner(
            chaos=ChaosConfig(events=[ChaosEventConfig(type="rolling_restart", role="agg", timeout_seconds=5.0)]),
            config=_load_dict(_base_recipe()),
            runtime=SimpleNamespace(job_id="105", log_dir=tmp_path, network_interface=None),  # type: ignore[arg-type]
            registry=registry,
            supervisor=supervisor,
            endpoint_groups={("agg", 0, 0): [_proc("agg", 0, "node0")], ("agg", 1, 0): [_proc("agg", 1, "node1")]},
            stop_event=threading.Event(),
        )
        with (
            patch("srtctl.chaos.actions.list_step_ids", return_value=None),
            patch("srtctl.chaos.actions.signal_step", return_value=True),
        ):
            runner.start()
            self._wait_records(runner, 1)
            runner.stop(timeout=5.0)

        assert runner.records[0].status == "failed"
        assert "agg_0 replacement not_ready" in (runner.records[0].error or "")
        assert supervisor.ensure_relaunch.call_count == 1  # agg_1 was never taken down

    def test_untimed_freeze_holds_until_teardown(self, tmp_path: Path):
        registry = ProcessRegistry(job_id="105")
        registry.add_process(_FakeStep("agg_0_node0", node="node0").managed)
        conts: list[list[int]] = []

        def fake(node, payload, **_kwargs):
            if payload.get("signal") == "CONT":
                conts.append(payload["pids"])
            return _fake_node_helper(payload, pids=(777,))

        runner = self._runner(
            tmp_path,
            registry,
            {("agg", 0, 0): [_proc("agg", 0, "node0")]},
            [ChaosEventConfig(type="freeze", role="agg")],
        )
        with (
            patch("srtctl.chaos.actions.list_step_ids", return_value=None),
            patch("srtctl.chaos.actions.run_node_python", side_effect=fake),
        ):
            runner.start()
            self._wait_records(runner, 1)
            assert runner._thread is not None
            runner._thread.join(timeout=5.0)
            assert conts == []  # timeline finished; the freeze still holds
            runner.stop(timeout=5.0)

        assert conts == [[777]]

    def test_kill_wait_ready_and_rolling_restart_via_supervisor(self, tmp_path: Path):
        cfg = _load_dict(_base_recipe())
        runtime = SimpleNamespace(job_id="102", log_dir=tmp_path, network_interface=None)
        registry = ProcessRegistry(job_id="102")
        stop = threading.Event()
        launcher = _FakeLauncher()
        supervisor = WorkerSupervisor(
            registry=registry,
            stop_event=stop,
            launcher=launcher,
            log_dir=tmp_path,
            probe_interval=0.0,
        )
        p0, p1 = _proc("agg", 0, "node0"), _proc("agg", 1, "node0")
        s0, s1 = _FakeStep("agg_0_node0", node="node0"), _FakeStep("agg_1_node0", node="node0")
        registry.add_process(s0.managed)
        registry.add_process(s1.managed)

        reconcile_done = threading.Event()

        def reconcile_loop():
            while not reconcile_done.is_set():
                supervisor.reconcile()
                time.sleep(0.02)

        t = threading.Thread(target=reconcile_loop, daemon=True)
        t.start()

        chaos = ChaosConfig(
            events=[
                ChaosEventConfig(
                    type="kill", role="agg", workers=0, respawn=True, wait_ready=True, timeout_seconds=10.0
                ),
                ChaosEventConfig(
                    type="rolling_restart",
                    role="agg",
                    max_unavailable=1,
                    graceful=True,
                    timeout_seconds=10.0,
                ),
            ]
        )
        runner = ChaosRunner(
            chaos=chaos,
            config=cfg,
            runtime=runtime,  # type: ignore[arg-type]
            registry=registry,
            supervisor=supervisor,
            endpoint_groups={("agg", 0, 0): [p0], ("agg", 1, 0): [p1]},
            stop_event=stop,
        )
        try:
            with (
                patch("srtctl.chaos.actions.list_step_ids", return_value=None),
                patch("srtctl.chaos.actions.signal_step", return_value=False),
            ):
                runner.start()
                deadline = time.monotonic() + 5.0
                while len(runner.records) < 2 or any(r.status == "running" for r in runner.records):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)
                runner.stop(timeout=5.0)
        finally:
            reconcile_done.set()
            t.join(timeout=2.0)

        assert len(runner.records) == 2
        kill_rec, roll_rec = runner.records
        assert kill_rec.status == "completed"
        assert kill_rec.details["wait_ready_timed_out"] is False
        assert kill_rec.details["recovered"] == [{"worker": "agg_0", "outcome": "relaunched"}]
        assert roll_rec.status == "completed"
        assert roll_rec.details["total"] == 2
        assert len(roll_rec.details["restarted"]) == 2
        assert roll_rec.details["timed_out"] is False
        # agg_0 restarted twice (kill + rolling_restart), agg_1 once (rolling_restart).
        assert registry.get_process("agg_0_node0_r2") is not None
        assert registry.get_process("agg_1_node0_r1") is not None

    def test_leashed_mode_logs_targets_without_signalling(self, tmp_path: Path):
        cfg = _load_dict(_base_recipe())
        runtime = SimpleNamespace(job_id="103", log_dir=tmp_path, network_interface=None)
        registry = ProcessRegistry(job_id="103")
        p0, p1 = _proc("agg", 0, "node0"), _proc("agg", 1, "node0")
        s0, s1 = _FakeStep("agg_0_node0", node="node0"), _FakeStep("agg_1_node0", node="node0")
        registry.add_process(s0.managed)
        registry.add_process(s1.managed)

        chaos = ChaosConfig(
            leashed=True,
            events=[
                ChaosEventConfig(type="kill", role="agg", workers=0),
                ChaosEventConfig(type="rolling_restart", role="agg", max_unavailable=1),
            ],
        )
        runner = ChaosRunner(
            chaos=chaos,
            config=cfg,
            runtime=runtime,  # type: ignore[arg-type]
            registry=registry,
            supervisor=None,
            endpoint_groups={("agg", 0, 0): [p0], ("agg", 1, 0): [p1]},
            stop_event=threading.Event(),
        )
        with (
            patch("srtctl.chaos.actions.list_step_ids", return_value=None),
            patch("srtctl.chaos.actions.signal_step", return_value=False),
        ):
            runner.start()
            deadline = time.monotonic() + 5.0
            while len(runner.records) < 2 or any(r.status == "running" for r in runner.records):
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            runner.stop(timeout=5.0)

        # Neither worker was terminated or marked non-critical.
        assert s0.managed.is_running is True
        assert s1.managed.is_running is True
        assert s0.managed.critical is True
        assert [r.status for r in runner.records] == ["leashed", "leashed"]
        assert all(r.details.get("leashed") is True for r in runner.records)

    def test_min_healthy_guard_skips_kill_when_capacity_at_floor(self, tmp_path: Path):
        cfg = _load_dict(_base_recipe())
        runtime = SimpleNamespace(job_id="104", log_dir=tmp_path, network_interface=None)
        registry = ProcessRegistry(job_id="104")
        p0, p1 = _proc("agg", 0, "node0"), _proc("agg", 1, "node0")
        s0, s1 = _FakeStep("agg_0_node0", node="node0"), _FakeStep("agg_1_node0", node="node0")
        registry.add_process(s0.managed)
        registry.add_process(s1.managed)

        # Kill worker 0 (2 -> 1 >= min_healthy=1), then attempt to kill worker 1 (1 -> 0 < 1 -> skipped).
        chaos = ChaosConfig(
            min_healthy=1,
            events=[
                ChaosEventConfig(type="kill", role="agg", workers=0),
                ChaosEventConfig(type="kill", role="agg", workers=1),
            ],
        )
        runner = ChaosRunner(
            chaos=chaos,
            config=cfg,
            runtime=runtime,  # type: ignore[arg-type]
            registry=registry,
            supervisor=None,
            endpoint_groups={("agg", 0, 0): [p0], ("agg", 1, 0): [p1]},
            stop_event=threading.Event(),
        )
        with (
            patch("srtctl.chaos.actions.list_step_ids", return_value=None),
            patch("srtctl.chaos.actions.signal_step", return_value=False),
        ):
            runner.start()
            deadline = time.monotonic() + 5.0
            while len(runner.records) < 2 or any(r.status == "running" for r in runner.records):
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            runner.stop(timeout=5.0)

        assert not s0.managed.is_running
        assert s1.managed.is_running is True
        assert [r.status for r in runner.records] == ["completed", "skipped"]
        assert runner.records[1].details["reason"] == "min_healthy_guard"

    def test_min_healthy_guard_blocks_overlap_before_srun_reaps(self, tmp_path: Path):
        """Even if scancel was delivered and local srun has not exited yet, min_healthy counts the target down."""
        cfg = _load_dict(_base_recipe())
        runtime = SimpleNamespace(job_id="104b", log_dir=tmp_path, network_interface=None)
        registry = ProcessRegistry(job_id="104b")
        stop = threading.Event()
        supervisor = WorkerSupervisor(
            registry=registry,
            stop_event=stop,
            launcher=_FakeLauncher(),
            log_dir=tmp_path,
            probe_interval=0.0,
        )
        p0, p1 = _proc("agg", 0, "node0"), _proc("agg", 1, "node0")
        s0, s1 = _FakeStep("agg_0_node0", node="node0"), _FakeStep("agg_1_node0", node="node0")
        registry.add_process(s0.managed)
        registry.add_process(s1.managed)

        chaos = ChaosConfig(
            min_healthy=1,
            events=[
                ChaosEventConfig(type="kill", role="agg", workers=0, respawn=True),
                ChaosEventConfig(type="kill", role="agg", workers=1, respawn=True),
            ],
        )
        runner = ChaosRunner(
            chaos=chaos,
            config=cfg,
            runtime=runtime,  # type: ignore[arg-type]
            registry=registry,
            supervisor=supervisor,
            endpoint_groups={("agg", 0, 0): [p0], ("agg", 1, 0): [p1]},
            stop_event=stop,
        )
        # Simulate signal_step succeeding (scancel delivered) while local srun popen.poll() is still None.
        with (
            patch("srtctl.chaos.actions.list_step_ids", return_value={"agg_0_node0": "104.2"}),
            patch("srtctl.chaos.actions.signal_step", return_value=True),
        ):
            runner.start()
            deadline = time.monotonic() + 5.0
            while len(runner.records) < 2 or any(r.status == "running" for r in runner.records):
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            runner.stop(timeout=5.0)

        assert [r.status for r in runner.records] == ["completed", "skipped"]
        assert runner.records[1].details["reason"] == "min_healthy_guard"

    def test_stochastic_schedule_fires_up_to_max_events_and_respects_floor(self, tmp_path: Path):
        sched = ChaosScheduleConfig(mean_interval_seconds=0.04, min_interval_seconds=0.02, max_events=3, role="agg")
        rng = random.Random(11)
        draws = [sample_schedule_delay(sched, rng) for _ in range(50)]
        assert all(d >= 0.02 for d in draws)

        cfg = _load_dict(_base_recipe())
        runtime = SimpleNamespace(job_id="105", log_dir=tmp_path, network_interface=None)
        registry = ProcessRegistry(job_id="105")
        p0, p1 = _proc("agg", 0, "node0"), _proc("agg", 1, "node0")
        s0, s1 = _FakeStep("agg_0_node0", node="node0"), _FakeStep("agg_1_node0", node="node0")
        registry.add_process(s0.managed)
        registry.add_process(s1.managed)

        chaos = ChaosConfig(
            leashed=True,
            seed=11,
            schedule=ChaosScheduleConfig(
                mean_interval_seconds=0.02,
                min_interval_seconds=0.01,
                max_events=3,
                role="agg",
                workers="random",
            ),
        )
        runner = ChaosRunner(
            chaos=chaos,
            config=cfg,
            runtime=runtime,  # type: ignore[arg-type]
            registry=registry,
            supervisor=None,
            endpoint_groups={("agg", 0, 0): [p0], ("agg", 1, 0): [p1]},
            stop_event=threading.Event(),
        )
        with (
            patch("srtctl.chaos.actions.list_step_ids", return_value=None),
            patch("srtctl.chaos.actions.signal_step", return_value=False),
        ):
            runner.start()
            deadline = time.monotonic() + 5.0
            while len(runner.records) < 3 or any(r.status == "running" for r in runner.records):
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.02)
            runner.stop(timeout=5.0)

        assert len(runner.records) == 3
        assert [r.name for r in runner.records] == ["soak_0", "soak_1", "soak_2"]
        assert all(r.status == "leashed" for r in runner.records)


# ---------------------------------------------------------------------------
# End-to-end mock orchestrator acceptance test
# ---------------------------------------------------------------------------


def test_mock_sweep_runs_chaos_concurrently_and_writes_timeline(tmp_path: Path):
    recipe_path = tmp_path / "chaos-recipe.yaml"
    doc = _base_recipe(
        model={"path": "hf:fake/mock-model", "container": "nvcr.io/fake:latest", "precision": "fp8"},
        roles={"agg": {"nodes": 1, "workers": 2, "restart": {"policy": "on-failure", "backoff_seconds": 0}}},
        chaos={
            "seed": 1,
            "events": [
                {
                    "name": "kill-agg-0",
                    "after_seconds": 0.0,
                    "type": "kill",
                    "role": "agg",
                    "workers": 0,
                    "respawn": True,
                },
                {
                    "name": "freeze-agg-1",
                    "type": "freeze",
                    "role": "agg",
                    "workers": 1,
                    "duration_seconds": 0.05,
                },
            ],
        },
    )
    recipe_path.write_text(yaml.safe_dump(doc, sort_keys=False))

    loaded = load_config(recipe_path)
    assert loaded.chaos.enabled is True

    out_dir = tmp_path / "run_out"
    rc = run_mock_sweep(
        config_path=recipe_path,
        output_dir=out_dir,
        job_id="99901",
        options=MockOptions(
            child_duration_s=0.6,
            phase_pause_s=0.0,
            nodelist=("mock-node-01",),
        ),
    )
    assert rc == 0

    events_file = out_dir / "logs" / CHAOS_EVENTS_FILENAME
    assert events_file.exists()
    data = json.loads(events_file.read_text())
    assert [ev["name"] for ev in data["events"]] == ["kill-agg-0", "freeze-agg-1"]
    assert [ev["status"] for ev in data["events"]] == ["completed", "completed"]
    ev = data["events"][0]
    assert ev["targets"] == ["agg_0"]
    assert ev["started_at_ns"] <= ev["ended_at_ns"]
