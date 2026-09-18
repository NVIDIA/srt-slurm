# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``placement.per: worker``: a service instance per engine worker, in that worker's device view."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Process
from srtctl.services import ServiceConfig, ServicePlacementConfig
from srtctl.services.registry import ServiceLaunchContext

SRUN = "srtctl.cli.mixins.service_stage.start_srun_process"
WAIT = "srtctl.cli.mixins.service_stage.wait_until_ready"
HOST_IP = "srtctl.cli.mixins.service_stage.get_hostname_ip"

HEAD = """
name: per-worker-test
model:
  path: /model
  container: /job.sqsh
  precision: bf16
resources:
  gpu_type: b200
  gpus_per_node: 8
  agg_nodes: 1
  agg_workers: 2
  gpus_per_agg: 1
backend:
  type: sglang
frontend:
  type: sglang-router
benchmark:
  type: manual
observability:
  tachometer:
    enabled: false
"""


def _load(services_yaml: str) -> SrtConfig:
    return SrtConfig.Schema().load(yaml.safe_load(HEAD + services_yaml))


def _runtime(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(
        job_id="777",
        run_name="per-worker-test",
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node1",)),
        head_node_ip="10.0.0.10",
        infra_node_ip="10.0.0.10",
        log_dir=tmp_path,
        model_path=Path("/model"),
        container_image=Path("/job.sqsh"),
        gpus_per_node=8,
        network_interface="eth0",
        container_mounts={},
        environment={},
    )


def _proc() -> MagicMock:
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    return proc


# --- schema -----------------------------------------------------------------------------


def test_per_defaults_to_node_and_accepts_worker() -> None:
    assert ServicePlacementConfig(node="agg").per == "node"
    assert ServicePlacementConfig(node="agg", per="worker").per == "worker"
    service = ServiceConfig(name="w", command=["true"], placement=ServicePlacementConfig(node="agg", per="worker"))
    assert service.effective_per == "worker"
    assert ServiceConfig(name="n", command=["true"]).effective_per == "node"


@pytest.mark.parametrize(
    ("placement", "message"),
    [
        ({"node": "head", "per": "worker"}, "needs placement.node in prefill, decode, agg, workers"),
        ({"node": "all", "per": "worker"}, "needs placement.node in"),
        ({"pool": "train", "per": "worker"}, "not to a pool"),
        ({"node": "agg", "per": "rank"}, "placement.per must be one of node, worker"),
    ],
)
def test_per_worker_placement_rules(placement: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ServicePlacementConfig(**placement)


def test_per_worker_service_rules() -> None:
    with pytest.raises(ValidationError, match="cannot own a pool"):
        _load(
            "services:\n  - name: w\n    command: [/bin/true]\n    nodes: 1\n    placement: {node: workers, per: worker}\n"
        )
    with pytest.raises(ValidationError, match="readiness must be a log probe"):
        _load(
            "services:\n  - name: w\n    command: [/bin/true]\n    placement: {node: agg, per: worker}\n"
            "    readiness: {port: 9000}\n"
        )
    _load(
        "services:\n  - name: w\n    command: [/bin/true]\n    placement: {node: agg, per: worker}\n"
        "    readiness: {log: {pattern: up}}\n"
    )


def test_template_vars_describe_the_attached_worker() -> None:
    process = Process("node1", frozenset({2, 3}), 7500, 6100, "decode", 4, node_rank=1)
    ctx = ServiceLaunchContext(
        runtime=_runtime(Path("/tmp")),
        node="node1",
        node_ip="10.0.0.11",
        node_id=0,
        index=0,
        role="decode",
        process=process,
    )
    values = ctx.template_vars()
    assert values["worker_role"] == "decode"
    assert values["worker_index"] == "4"
    assert values["worker_node_rank"] == "1"
    assert values["worker_gpus"] == "2,3"
    assert values["worker_gpu_count"] == "2"
    assert "worker_role" not in ServiceLaunchContext.preview().template_vars()


# --- launch -----------------------------------------------------------------------------


def test_one_instance_per_worker_pinned_to_its_gpus(tmp_path: Path) -> None:
    config = _load(
        """
services:
  - name: gpu-watch
    command: [nvidia-smi, dmon, -i, "{worker_gpus}", --tag, "{worker_role}-{worker_index}"]
    placement:
      node: agg
      per: worker
    start: before_workers
    env:
      WATCH_GPUS: "{worker_gpu_count}"
"""
    )
    orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))
    with patch(SRUN, return_value=_proc()) as srun, patch(WAIT, return_value=True), patch(HOST_IP, return_value="ip"):
        procs = orchestrator.start_services("before_workers")

    assert [p.name for p in procs] == ["service_gpu-watch_agg_0_node1", "service_gpu-watch_agg_1_node1"]
    calls = [call.kwargs for call in srun.call_args_list]
    assert calls[0]["command"] == ["nvidia-smi", "dmon", "-i", "0", "--tag", "agg-0"]
    assert calls[1]["command"] == ["nvidia-smi", "dmon", "-i", "1", "--tag", "agg-1"]
    assert calls[0]["env_to_set"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert calls[1]["env_to_set"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert calls[0]["env_to_set"]["WATCH_GPUS"] == "1"
    assert calls[0]["step_name"] == "service_gpu-watch_agg_0_node1"
    assert calls[1]["output"] == str(tmp_path / "service_gpu-watch_agg_1_node1.out")
    assert procs[0].node == "node1" and procs[0].shutdown_tier == 1


def test_full_node_workers_are_not_pinned(tmp_path: Path) -> None:
    config = SrtConfig.Schema().load(
        yaml.safe_load(
            HEAD.replace("agg_workers: 2", "agg_workers: 1").replace("gpus_per_agg: 1", "gpus_per_agg: 8")
            + "services:\n  - name: w\n    command: [/bin/true]\n    placement: {node: workers, per: worker}\n"
            "    start: before_workers\n"
        )
    )
    orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))
    with patch(SRUN, return_value=_proc()) as srun, patch(WAIT, return_value=True), patch(HOST_IP, return_value="ip"):
        procs = orchestrator.start_services("before_workers")
    assert [p.name for p in procs] == ["service_w_agg_0_node1"]
    assert "CUDA_VISIBLE_DEVICES" not in srun.call_args.kwargs["env_to_set"]


def test_per_node_services_are_unchanged(tmp_path: Path) -> None:
    config = _load(
        "services:\n  - name: n\n    command: [/bin/true]\n    placement: {node: agg}\n    start: before_workers\n"
    )
    orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))
    with patch(SRUN, return_value=_proc()) as srun, patch(WAIT, return_value=True), patch(HOST_IP, return_value="ip"):
        procs = orchestrator.start_services("before_workers")
    assert [p.name for p in procs] == ["service_n"]
    assert "CUDA_VISIBLE_DEVICES" not in srun.call_args.kwargs["env_to_set"]
