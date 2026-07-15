# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for managed standalone Mooncake Store services."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Endpoint
from srtctl.ports import MOONCAKE_HTTP_METADATA_PORT, MOONCAKE_MASTER_PORT


def _load_config(standalone: str) -> SrtConfig:
    raw = yaml.safe_load(
        f"""
name: mooncake-standalone-test
model:
  path: /model
  container: /job-image.sqsh
  precision: fp4
resources:
  prefill_nodes: 1
  decode_nodes: 2
  prefill_workers: 1
  decode_workers: 2
  gpus_per_prefill: 8
  gpus_per_decode: 8
  gpus_per_node: 8
  gpu_type: b200
backend:
  type: sglang
  mooncake_kv_store:
    container: /mooncake-master.sqsh
{standalone}
  sglang_config:
    prefill:
      disaggregation-transfer-backend: mooncake
    decode:
      disaggregation-transfer-backend: mooncake
"""
    )
    return SrtConfig.Schema().load(raw)


def _runtime(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(
        job_id="12345",
        run_name="test-run",
        nodes=Nodes(
            head="node0",
            bench="node0",
            infra="node0",
            worker=("node1", "node2", "node3"),
        ),
        head_node_ip="10.0.0.10",
        infra_node_ip="10.0.0.10",
        log_dir=tmp_path,
        model_path=Path("/model"),
        container_image=Path("/job-image.sqsh"),
        gpus_per_node=8,
        network_interface="eth0",
        container_mounts={},
        environment={},
    )


STANDALONE_YAML = """    standalone:
      container: /mooncake-store.sqsh
      command: [python, -m, mooncake.mooncake_store_service]
      args: [--port, "8800", --label, "{role}-{node_id}"]
      env:
        MOONCAKE_PROTOCOL: rdma
        MOONCAKE_DEVICE: mlx5_0,mlx5_1
        MOONCAKE_MASTER: ignored:9999
        MOONCAKE_EXTRA_CONFIG: '{"prefetch_timeout_base": 4}'
      placements:
        prefill:
          env:
            MOONCAKE_GLOBAL_SEGMENT_SIZE: 100gb
          args: [--placement, prefill]
        decode:
          env:
            MOONCAKE_GLOBAL_SEGMENT_SIZE: 400gb
          args: [--placement, decode]
      preamble: "echo starting-{role}-on-{node}"
      cpus_per_task: 8
      cpu_bind: none
      srun_options:
        exclusive: ""
      health_check:
        port: 8800
        timeout_seconds: 90
"""

def test_standalone_config_loads_from_yaml() -> None:
    config = _load_config(STANDALONE_YAML)
    standalone = config.backend.mooncake_kv_store.standalone

    assert standalone is not None
    assert standalone.command == ("python", "-m", "mooncake.mooncake_store_service")
    assert standalone.args[:2] == ("--port", "8800")
    assert standalone.placements["prefill"].env["MOONCAKE_GLOBAL_SEGMENT_SIZE"] == "100gb"
    assert standalone.placements["decode"].env["MOONCAKE_GLOBAL_SEGMENT_SIZE"] == "400gb"
    assert standalone.health_check is not None
    assert standalone.health_check.port == 8800


def test_mooncake_master_uses_fixed_args_without_nof(tmp_path: Path) -> None:
    config = _load_config("")
    orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))
    proc = MagicMock()
    proc.poll.return_value = None
    with (
        patch("srtctl.cli.do_sweep.start_srun_process", return_value=proc) as mock_srun,
        patch("srtctl.cli.do_sweep.wait_for_port", return_value=True),
    ):
        orchestrator.start_mooncake_master(MagicMock())

    command = mock_srun.call_args.kwargs["command"]
    assert "--eviction_high_watermark_ratio=0.9" in command
    assert "--default_kv_lease_ttl=10000" in command
    assert "--rpc_thread_num=16" in command
    assert not any(arg.startswith("--nof_eviction_high_watermark_ratio") for arg in command)


def test_standalone_rejects_unknown_placement() -> None:
    with pytest.raises(ValueError, match="invalid mooncake standalone placements"):
        _load_config(
            """    standalone:
      placements:
        storage:
          env: {}
"""
        )


def test_vllm_standalone_config_loads_from_yaml() -> None:
    kv_config = '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
    raw = yaml.safe_load(
        f"""
name: vllm-mooncake-standalone-test
model: {{path: /model, container: /image, precision: fp4}}
resources:
  prefill_nodes: 1
  decode_nodes: 1
  prefill_workers: 1
  decode_workers: 1
  gpu_type: b200
backend:
  type: vllm
  mooncake_kv_store:
    standalone:
      placements:
        prefill:
          env: {{MOONCAKE_GLOBAL_SEGMENT_SIZE: 100gb}}
        decode:
          env: {{MOONCAKE_GLOBAL_SEGMENT_SIZE: 200gb}}
  vllm_config:
    prefill:
      kv-transfer-config: '{kv_config}'
    decode:
      kv-transfer-config: '{kv_config}'
"""
    )
    config = SrtConfig.Schema().load(raw)
    assert config.backend.mooncake_kv_store.standalone is not None
    assert config.backend.mooncake_kv_store.standalone.placements["decode"].env == {
        "MOONCAKE_GLOBAL_SEGMENT_SIZE": "200gb"
    }


def test_standalone_launches_once_per_selected_physical_node(tmp_path: Path) -> None:
    orchestrator = SweepOrchestrator(config=_load_config(STANDALONE_YAML), runtime=_runtime(tmp_path))
    registry = MagicMock()
    mock_processes = []

    def make_process(**_kwargs):
        proc = MagicMock()
        proc.poll.return_value = None
        mock_processes.append(proc)
        return proc

    node_ips = {"node1": "10.0.0.11", "node2": "10.0.0.12", "node3": "10.0.0.13"}
    with (
        patch("srtctl.cli.do_sweep.get_hostname_ip", side_effect=lambda node, _interface: node_ips[node]),
        patch("srtctl.cli.do_sweep.start_srun_process", side_effect=make_process) as mock_srun,
        patch("srtctl.cli.do_sweep.wait_for_port", return_value=True) as mock_wait,
    ):
        managed = orchestrator.start_mooncake_store_services(registry)

    assert len(managed) == 3
    assert mock_srun.call_count == 3
    assert mock_wait.call_count == 3
    assert registry.add_process.call_count == 3

    calls = {call.kwargs["nodelist"][0]: call.kwargs for call in mock_srun.call_args_list}
    prefill = calls["node1"]
    decode = calls["node2"]

    assert prefill["container_image"] == "/mooncake-store.sqsh"
    assert prefill["command"] == [
        "python",
        "-m",
        "mooncake.mooncake_store_service",
        "--port",
        "8800",
        "--label",
        "prefill-0",
        "--placement",
        "prefill",
    ]
    assert prefill["env_to_set"]["MOONCAKE_LOCAL_HOSTNAME"] == "10.0.0.11"
    assert prefill["env_to_set"]["MOONCAKE_GLOBAL_SEGMENT_SIZE"] == "100gb"
    assert prefill["env_to_set"]["MOONCAKE_EXTRA_CONFIG"] == '{"prefetch_timeout_base": 4}'
    assert decode["env_to_set"]["MOONCAKE_GLOBAL_SEGMENT_SIZE"] == "400gb"
    assert prefill["env_to_set"]["MOONCAKE_MASTER"] == f"10.0.0.10:{MOONCAKE_MASTER_PORT}"
    assert prefill["env_to_set"]["MOONCAKE_TE_META_DATA_SERVER"] == (
        f"http://10.0.0.10:{MOONCAKE_HTTP_METADATA_PORT}/metadata"
    )
    assert prefill["bash_preamble"] == "echo starting-prefill-on-node1"
    assert prefill["cpus_per_task"] == 8
    assert prefill["cpu_bind"] == "none"
    assert prefill["srun_options"] == {"exclusive": ""}


def test_standalone_deduplicates_identical_colocated_roles(tmp_path: Path) -> None:
    standalone = """    standalone:
      placements:
        prefill: &shared
          env:
            MOONCAKE_GLOBAL_SEGMENT_SIZE: 100gb
        decode: *shared
"""
    orchestrator = SweepOrchestrator(config=_load_config(standalone), runtime=_runtime(tmp_path))
    orchestrator.__dict__["endpoints"] = [
        Endpoint(mode="prefill", index=0, nodes=("node1",)),
        Endpoint(mode="decode", index=0, nodes=("node1",)),
    ]
    proc = MagicMock()
    proc.poll.return_value = None

    with (
        patch("srtctl.cli.do_sweep.get_hostname_ip", return_value="10.0.0.11"),
        patch("srtctl.cli.do_sweep.start_srun_process", return_value=proc) as mock_srun,
        patch("srtctl.cli.do_sweep.wait_for_port", return_value=True),
    ):
        managed = orchestrator.start_mooncake_store_services()

    assert len(managed) == 1
    assert mock_srun.call_count == 1
    assert "prefill_decode" in next(iter(managed))


def test_standalone_rejects_conflicting_colocated_roles(tmp_path: Path) -> None:
    orchestrator = SweepOrchestrator(config=_load_config(STANDALONE_YAML), runtime=_runtime(tmp_path))
    orchestrator.__dict__["endpoints"] = [
        Endpoint(mode="prefill", index=0, nodes=("node1",)),
        Endpoint(mode="decode", index=0, nodes=("node1",)),
    ]

    with pytest.raises(ValueError, match="co-located roles"):
        orchestrator.start_mooncake_store_services()


def test_standalone_health_failure_terminates_started_services(tmp_path: Path) -> None:
    orchestrator = SweepOrchestrator(config=_load_config(STANDALONE_YAML), runtime=_runtime(tmp_path))
    proc = MagicMock()
    proc.poll.return_value = None

    with (
        patch("srtctl.cli.do_sweep.get_hostname_ip", return_value="10.0.0.11"),
        patch("srtctl.cli.do_sweep.start_srun_process", return_value=proc),
        patch("srtctl.cli.do_sweep.wait_for_port", return_value=False),
        pytest.raises(RuntimeError, match="failed to start"),
    ):
        orchestrator.start_mooncake_store_services()

    proc.terminate.assert_called_once()
