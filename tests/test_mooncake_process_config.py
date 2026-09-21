# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical placement, not process-local CUDA numbering, selects store devices."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

import pytest

from srtctl.backends.vllm import VLLMMooncakeKVStoreConfig, VLLMProtocol, VLLMServerConfig
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.mixins.worker_stage import WorkerStageMixin
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import (
    DynamoConfig,
    ModelConfig,
    ProfilingConfig,
    ProfilingPhaseConfig,
    ResourceConfig,
    SrtConfig,
)
from srtctl.core.topology import Process


def process(gpus, node="n0"):
    return Process(node, frozenset(gpus), 7500, 6100, "decode", 0)


def backend(devices=()):
    return VLLMProtocol(
        mooncake_kv_store=VLLMMooncakeKVStoreConfig(
            device_names_by_gpu=list(devices),
            store_config={"device_name": "shared", "global_segment_size": "170GB"},
        )
    )


def test_default_unchanged():
    b = backend()
    assert b.build_mooncake_process_config(process([0]), "infra", 4) is None
    assert b.build_mooncake_store_config("infra")["device_name"] == "shared"
    assert VLLMProtocol().build_mooncake_process_config(process([0]), "infra", 4) is None


@pytest.mark.parametrize("gpus,expected", [([2], "h2"), ([0, 1], "h0,h1"), ([2, 3], "h2,h3"), ([0, 2], "h0,h2")])
def test_physical_gpu_subsets(gpus, expected):
    b = backend(["h0", "h1", "h2", "h3"])
    filename, payload = b.build_mooncake_process_config(process(gpus), "infra", 4)
    assert filename == "mooncake_store_config_gpu" + "-".join(map(str, sorted(gpus))) + ".json"
    assert payload["device_name"] == expected
    assert payload["global_segment_size"] == "170GB"
    assert payload["master_server_address"] == "infra:8700"
    assert b.mooncake_kv_store.store_config["device_name"] == "shared"
    assert b.build_mooncake_process_config(process(gpus, node="n1"), "infra", 4) == (filename, payload)


def test_shared_hca_deduplicated():
    b = backend(["h0", "h0", "h1", "h1"])
    assert b.build_mooncake_process_config(process([0, 1]), "infra", 4)[1]["device_name"] == "h0"


def test_rendered_configs_match_worker_environment(tmp_path):
    workers = [process([0, 1]), process([2, 3]), process([2, 3], node="n1")]
    b = backend(["h0", "h1", "h2", "h3"])
    runtime = SimpleNamespace(log_dir=tmp_path, container_log_dir=Path("/logs"), infra_node_ip="infra", gpus_per_node=4)
    context = SimpleNamespace(config=SimpleNamespace(backend=b), backend=b, runtime=runtime, backend_processes=workers)
    SweepOrchestrator._write_mooncake_store_config(context)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "mooncake_store_config.json",
        "mooncake_store_config_gpu0-1.json",
        "mooncake_store_config_gpu2-3.json",
    ]
    for worker, expected in zip(workers, ["h0,h1", "h2,h3", "h2,h3"], strict=True):
        env = b.get_mooncake_worker_env("infra", worker.node)
        WorkerStageMixin._apply_mooncake_process_config(context, worker, env)
        payload = json.loads((tmp_path / env["MOONCAKE_CONFIG_PATH"].split("/")[-1]).read_text())
        assert payload["device_name"] == expected
        assert payload["global_segment_size"] == "170GB"
        assert not any(k.startswith("SRT_MOONCAKE") for k in env)


def test_default_writer_and_worker_keep_shared_config(tmp_path):
    b = backend()
    context = SimpleNamespace(
        config=SimpleNamespace(backend=b),
        backend=b,
        runtime=SimpleNamespace(log_dir=tmp_path, infra_node_ip="infra", gpus_per_node=4),
    )
    SweepOrchestrator._write_mooncake_store_config(context)
    assert [p.name for p in tmp_path.iterdir()] == ["mooncake_store_config.json"]
    env = b.get_mooncake_worker_env("infra", "node")
    before = dict(env)
    WorkerStageMixin._apply_mooncake_process_config(context, process([2]), env)
    assert env == before


@pytest.mark.parametrize("disaggregated", [False, True], ids=["agg", "disagg"])
@pytest.mark.parametrize("mapped", [False, True], ids=["shared", "process-local"])
@pytest.mark.parametrize("capture_scope", [None, "selected", "all"], ids=["no-profiling", "selected", "all"])
def test_worker_launch_uses_rendered_config(
    tmp_path: Path, disaggregated: bool, mapped: bool, capture_scope: Literal["selected", "all"] | None
) -> None:
    """Protect the real writer-to-srun wiring, including profiling selection."""
    roles = ["prefill", "decode"] if disaggregated else ["aggregated"]
    args = {"tensor-parallel-size": 2, "kv-transfer-config": '{"kv_connector":"MooncakeStoreConnector"}'}
    b = replace(
        backend(["h0", "h1", "h2", "h3"] if mapped else []),
        vllm_config=VLLMServerConfig(**dict.fromkeys(roles, args)),
    )
    profiling = ProfilingConfig()
    if capture_scope is not None:
        phase = ProfilingPhaseConfig(start_step=2, stop_step=5, capture_scope=capture_scope, worker_index=1)
        profiling = ProfilingConfig(type="nsys", nsys_library_paths=["/host/lib64"], **dict.fromkeys(roles, phase))
    resources = (
        ResourceConfig(gpus_per_node=4, prefill_nodes=2, prefill_workers=4, decode_nodes=2, decode_workers=4)
        if disaggregated
        else ResourceConfig(gpus_per_node=4, agg_nodes=2, agg_workers=4)
    )
    config = SrtConfig(
        name="mooncake-launch-test",
        model=ModelConfig(path="/model", container="/container.sqsh", precision="bf16"),
        resources=resources,
        backend=b,
        dynamo=DynamoConfig(install=False),
        profiling=profiling,
    )
    runtime = RuntimeContext(
        job_id="12345",
        run_name=config.name,
        nodes=Nodes(
            head="n0", bench="n0", infra="n0", worker=("n0", "n1", "n2", "n3") if disaggregated else ("n0", "n1")
        ),
        head_node_ip="192.0.2.1",
        infra_node_ip="192.0.2.1",
        log_dir=tmp_path,
        model_path=Path("/model"),
        container_image=Path("/container.sqsh"),
        container_mounts={tmp_path: Path("/logs")},
        gpus_per_node=4,
        network_interface=None,
    )
    orchestrator = SweepOrchestrator(config, runtime)
    workers = orchestrator.backend_processes
    assert [sorted(worker.gpu_indices) for worker in workers] == [[0, 1], [2, 3]] * (4 if disaggregated else 2)
    orchestrator._write_mooncake_store_config()

    with (
        patch("srtctl.core.slurm.get_hostname_ip", return_value="192.0.2.2"),
        patch("srtctl.cli.mixins.worker_stage.get_hostname_ip", return_value="192.0.2.2"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as srun,
    ):
        for worker in workers:
            orchestrator.start_worker(worker, [worker])

    assert srun.call_count == len(workers)
    for worker, call in zip(workers, srun.call_args_list, strict=True):
        launch = call.kwargs
        env = launch["env_to_set"]
        gpu_ids = "-".join(map(str, sorted(worker.gpu_indices)))
        filename = f"mooncake_store_config_gpu{gpu_ids}.json" if mapped else "mooncake_store_config.json"
        assert env["MOONCAKE_CONFIG_PATH"] == f"/logs/{filename}"
        assert launch["container_mounts"][tmp_path] == Path("/logs")
        assert launch["nodelist"] == [worker.node]
        payload = json.loads((tmp_path / filename).read_text())
        expected_devices = ",".join(f"h{i}" for i in sorted(worker.gpu_indices)) if mapped else "shared"
        assert payload["device_name"] == expected_devices
        assert payload["global_segment_size"] == "170GB"
        assert payload["master_server_address"] == "192.0.2.1:8700"

        profiled = capture_scope == "all" or (capture_scope == "selected" and worker.endpoint_index == 1)
        assert (env.get("PROFILE_TYPE") == "nsys") == profiled
        assert (launch["command"][0] == profiling.nsys_binary) == profiled
        assert ("--profiler-config" in launch["command"]) == profiled
        assert ("/host/lib64" in launch["bash_preamble"]) == profiled


@pytest.mark.parametrize(
    "devices,gpus",
    [
        (["h0"], [0]),
        (["h0", "", "h2", "h3"], [0]),
        (["h0,h1", "h1", "h2", "h3"], [0]),
        (["h0", "h1", "h2", "h3"], [4]),
        (["h0", "h1", "h2", "h3"], []),
        (["h0", "h1", "h2", "h3"], [-1]),
    ],
)
def test_invalid_mapping_fails(devices, gpus):
    with pytest.raises(ValueError):
        backend(devices).build_mooncake_process_config(process(gpus), "infra", 4)
