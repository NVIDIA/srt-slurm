# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical placement, not process-local CUDA numbering, selects store devices."""

import json
from types import SimpleNamespace

import pytest

from srtctl.backends.vllm import VLLMMooncakeKVStoreConfig, VLLMProtocol
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.mixins.worker_stage import WorkerStageMixin
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
    runtime = SimpleNamespace(log_dir=tmp_path, infra_node_ip="infra", gpus_per_node=4)
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
