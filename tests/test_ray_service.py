# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``services[].type: ray``: head and worker commands, per-instance readiness, and the fleet gate."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.services import ServiceConfig, ServiceLaunchContext, get_service_kind
from srtctl.services.config import HttpProbe, LogProbe
from srtctl.services.ray import RayService

IPS = {"node0": "10.0.0.10", "node1": "10.0.0.11", "node2": "10.0.0.12", "node3": "10.0.0.13"}
SRUN = "srtctl.cli.mixins.service_stage.start_srun_process"
WAIT = "srtctl.cli.mixins.service_stage.wait_until_ready"
STAGE_HOST_IP = "srtctl.cli.mixins.service_stage.get_hostname_ip"
KIND_HOST_IP = "srtctl.services.ray.get_hostname_ip"

RAY_JOB = {
    "schema": 2,
    "name": "ray-test",
    "model": {"path": "hf:fake/mock-model", "container": "nvcr.io/fake:latest", "precision": "bf16"},
    "resources": {"gpu_type": "b200", "gpus_per_node": 8},
    "frontend": {"type": "none"},
    "services": [{"name": "train", "type": "ray", "nodes": 3}],
    "benchmark": {"type": "custom", "command": "echo driver"},
    "observability": {"tachometer": {"enabled": False}},
}


def _load(services: list[dict] | None = None) -> SrtConfig:
    data = yaml.safe_load(yaml.dump(RAY_JOB))
    if services is not None:
        data["services"] = services
    return SrtConfig.Schema().load(data)


def _runtime(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(
        job_id="12345",
        run_name="ray-run",
        nodes=Nodes(
            head="node1", bench="node1", infra="node1", worker=(), pools={"train": ("node1", "node2", "node3")}
        ),
        head_node_ip=IPS["node1"],
        infra_node_ip=IPS["node1"],
        log_dir=tmp_path,
        model_path=Path("/model"),
        container_image=Path("/miles.sqsh"),
        gpus_per_node=8,
        network_interface="bond0",
        container_mounts={Path("/data"): Path("/data")},
        environment={},
    )


def _ctx(runtime: RuntimeContext, node: str, index: int) -> ServiceLaunchContext:
    return ServiceLaunchContext(
        runtime=runtime, node=node, node_ip=IPS[node], node_id=index, index=index, role="workers"
    )


def _service(**overrides) -> ServiceConfig:
    return ServiceConfig(name="train", type="ray", **{"nodes": 3, **overrides})


# --- kind -----------------------------------------------------------------------


def test_ray_kind_defaults() -> None:
    kind = get_service_kind("ray")
    assert isinstance(kind, RayService)
    svc = _service()
    assert svc.effective_start == "before_workers"
    assert svc.effective_critical is True
    assert svc.effective_placement == "workers"
    assert kind.default_readiness_ports == ()


def test_head_command(tmp_path: Path) -> None:
    kind = get_service_kind("ray")
    cmd = kind.build_command(_service(), _ctx(_runtime(tmp_path), "node1", 0))
    assert cmd[:3] == ["ray", "start", "--head"]
    assert "--port=6379" in cmd
    assert "--dashboard-host=0.0.0.0" in cmd
    assert "--dashboard-port=8265" in cmd
    assert "--node-ip-address=10.0.0.11" in cmd
    assert "--num-gpus=8" in cmd
    assert "--block" in cmd


def test_worker_command_joins_the_head_and_execs(tmp_path: Path) -> None:
    kind = get_service_kind("ray")
    with patch(KIND_HOST_IP, side_effect=lambda host, iface=None: IPS[host]):
        cmd = kind.build_command(_service(), _ctx(_runtime(tmp_path), "node2", 1))
    assert cmd[:2] == ["bash", "-c"]
    script = cmd[2]
    assert "/dev/tcp/10.0.0.11/6379" in script, "waits for the head GCS before joining"
    assert "exec ray start --address=10.0.0.11:6379" in script
    assert "--node-ip-address=10.0.0.12" in script
    assert "--num-gpus=8" in script
    assert "--block" in script


def test_options_and_args_flow_into_both_commands(tmp_path: Path) -> None:
    kind = get_service_kind("ray")
    svc = _service(options={"port": 6400, "dashboard_port": 8300, "num_gpus": 4}, args=["--object-store-memory=1"])
    runtime = _runtime(tmp_path)
    head = kind.build_command(svc, _ctx(runtime, "node1", 0))
    assert "--port=6400" in head and "--dashboard-port=8300" in head and "--num-gpus=4" in head
    assert "--object-store-memory=1" in head
    with patch(KIND_HOST_IP, side_effect=lambda host, iface=None: IPS[host]):
        worker = kind.build_command(svc, _ctx(runtime, "node3", 2))[2]
    assert "--address=10.0.0.11:6400" in worker and "--num-gpus=4" in worker and "--object-store-memory=1" in worker
    assert kind.default_environment(svc, _ctx(runtime, "node1", 0))["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_default_environment(tmp_path: Path) -> None:
    env = get_service_kind("ray").default_environment(_service(), _ctx(_runtime(tmp_path), "node1", 0))
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert env["RAY_memory_monitor_refresh_ms"] == "0"


def test_readiness_differs_by_role(tmp_path: Path) -> None:
    kind = get_service_kind("ray")
    runtime = _runtime(tmp_path)
    head = kind.readiness(_service(), _ctx(runtime, "node1", 0))
    worker = kind.readiness(_service(), _ctx(runtime, "node2", 1))
    assert head is not None and isinstance(head.probe, HttpProbe)
    assert (head.probe.port, head.probe.path, head.probe.status) == (8265, "/api/version", 200)
    assert worker is not None and isinstance(worker.probe, LogProbe)
    assert worker.probe.pattern == "Ray runtime started"


def test_preview_command_for_dry_run() -> None:
    cmd = _service().preview_command()
    assert cmd[:3] == ["ray", "start", "--head"]
    assert "--node-ip-address=<node_ip>" in cmd


# --- validation -----------------------------------------------------------------


def test_ray_loads_in_a_services_only_job() -> None:
    config = _load()
    (svc,) = config.services
    assert svc.type == "ray" and svc.effective_placement == "workers"
    assert svc.nodes == 3 and config.total_nodes == 3


def test_ray_rejects_role_placement_command_and_bad_options() -> None:
    with pytest.raises(ValidationError, match="must be placed on head, workers, all"):
        _load([{"name": "train", "type": "ray", "placement": {"node": "prefill"}}])
    with pytest.raises(ValidationError, match="builds its own"):
        _load([{"name": "train", "type": "ray", "command": ["ray", "start"]}])
    with pytest.raises(ValidationError, match="options.port must be a positive integer"):
        _load([{"name": "train", "type": "ray", "options": {"port": "6379"}}])
    with pytest.raises(ValidationError, match="does not understand"):
        _load([{"name": "train", "type": "ray", "options": {"shm": 1}}])


# --- fleet gate -----------------------------------------------------------------


def _response(summary: list[dict]) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"result": True, "msg": "", "data": {"summary": summary}}
    return response


def test_alive_nodes_counts_alive_raylets_and_tolerates_a_dead_dashboard() -> None:
    summary = [{"raylet": {"state": "ALIVE"}}, {"raylet": {"state": "DEAD"}}, {"raylet": {"state": "ALIVE"}}]
    with patch("srtctl.services.ray.requests.get", return_value=_response(summary)):
        assert RayService.alive_nodes("http://h:8265/nodes?view=summary") == 2
    import requests

    with patch("srtctl.services.ray.requests.get", side_effect=requests.exceptions.ConnectionError()):
        assert RayService.alive_nodes("http://h:8265/nodes?view=summary") == 0


def _procs(n: int) -> list[MagicMock]:
    procs = []
    for i in range(n):
        proc = MagicMock()
        proc.is_running = True
        proc.node = f"node{i + 1}"
        procs.append(proc)
    return procs


def test_fleet_gate_waits_until_every_node_is_alive(tmp_path: Path) -> None:
    kind = get_service_kind("ray")
    with (
        patch(KIND_HOST_IP, side_effect=lambda host, iface=None: IPS[host]),
        patch.object(RayService, "alive_nodes", side_effect=[1, 2, 3]) as alive,
        patch("srtctl.services.ray.time.sleep") as sleep,
    ):
        kind.wait_fleet_ready(_service(), _runtime(tmp_path), _procs(3))
    assert alive.call_count == 3
    assert sleep.call_count == 2


def test_fleet_gate_is_a_noop_for_a_single_node_cluster(tmp_path: Path) -> None:
    with patch.object(RayService, "alive_nodes") as alive:
        get_service_kind("ray").wait_fleet_ready(_service(), _runtime(tmp_path), _procs(1))
    alive.assert_not_called()


def test_fleet_gate_fails_fast_when_a_raylet_step_dies(tmp_path: Path) -> None:
    procs = _procs(3)
    procs[2].is_running = False
    procs[2].exit_code = 1
    procs[2].log_file = Path("/logs/service_train_node3.out")
    with (
        patch(KIND_HOST_IP, side_effect=lambda host, iface=None: IPS[host]),
        patch.object(RayService, "alive_nodes", return_value=2),
        pytest.raises(RuntimeError, match="node3 exited"),
    ):
        get_service_kind("ray").wait_fleet_ready(_service(), _runtime(tmp_path), procs)


def test_fleet_gate_times_out(tmp_path: Path) -> None:
    clock = iter([0.0, 0.0, 1e9])
    with (
        patch(KIND_HOST_IP, side_effect=lambda host, iface=None: IPS[host]),
        patch.object(RayService, "alive_nodes", return_value=1),
        patch("srtctl.services.ray.time.monotonic", side_effect=lambda: next(clock)),
        patch("srtctl.services.ray.time.sleep"),
        pytest.raises(RuntimeError, match="only 1/3 Ray nodes alive"),
    ):
        get_service_kind("ray").wait_fleet_ready(_service(), _runtime(tmp_path), _procs(3))


# --- stage ------------------------------------------------------------------------


def test_stage_launches_head_then_workers_with_per_instance_probes(tmp_path: Path) -> None:
    orchestrator = SweepOrchestrator(config=_load(), runtime=_runtime(tmp_path))
    popen = MagicMock()
    popen.poll.return_value = None
    with (
        patch(SRUN, return_value=popen) as srun,
        patch(WAIT, return_value=True) as wait,
        patch(STAGE_HOST_IP, side_effect=lambda host, iface=None: IPS[host]),
        patch(KIND_HOST_IP, side_effect=lambda host, iface=None: IPS[host]),
        patch.object(RayService, "wait_fleet_ready") as fleet,
    ):
        procs = orchestrator.start_services("before_workers")

    assert len(procs) == 3
    calls = srun.call_args_list
    assert calls[0].kwargs["nodelist"] == ["node1"]
    assert calls[0].kwargs["command"][:3] == ["ray", "start", "--head"]
    for call, node in zip(calls[1:], ("node2", "node3"), strict=True):
        assert call.kwargs["nodelist"] == [node]
        assert call.kwargs["command"][:2] == ["bash", "-c"]
        assert "--address=10.0.0.11:6379" in call.kwargs["command"][2]
    for call in calls:
        assert call.kwargs["container_image"] == "/miles.sqsh"
        assert call.kwargs["container_mounts"] == {Path("/data"): Path("/data")}
        assert call.kwargs["env_to_set"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
        assert call.kwargs["step_name"].startswith("service_train")
    # Head probed over http, workers over their logs; then one fleet gate for the service.
    probes = [call.args[0] for call in wait.call_args_list]
    assert isinstance(probes[0], HttpProbe)
    assert all(isinstance(p, LogProbe) for p in probes[1:])
    fleet.assert_called_once()
    assert len(fleet.call_args.args[2]) == 3
    assert all(proc.critical for proc in procs)


def test_head_is_the_first_node_of_the_pool_next_to_engine_roles(tmp_path: Path) -> None:
    nodes = Nodes(head="node0", bench="node0", infra="node0", worker=("node0",), pools={"train": ("node1", "node2")})
    runtime = dataclasses.replace(_runtime(tmp_path), nodes=nodes)
    assert RayService.head_node(_service(), runtime) == "node1", "the engine node is not part of the cluster"
