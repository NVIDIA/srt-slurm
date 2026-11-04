# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared P/D node sizing and launch placement across engines."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends import (
    AtomBackend,
    MockerBackend,
    SGLangBackend,
    TileRTBackend,
    TokenSpeedBackend,
    TRTLLMBackend,
    VLLMBackend,
)
from srtctl.cli.submit import generate_minimal_sbatch_script
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep

ENGINE_FRONTENDS = [("sglang", "sglang-router"), ("vllm", "vllm-router"), ("trtllm", "trtllm_serve")]
EXPECTED = [
    ("prefill", 0, ("node1",), frozenset({0, 1})),
    ("decode", 0, ("node1",), frozenset({2, 3})),
    ("decode", 1, ("node2",), frozenset({0, 1})),
    ("decode", 2, ("node2",), frozenset({2, 3})),
]


def recipe(engine="sglang", frontend="sglang-router"):
    return {
        "schema": 2,
        "name": "disagg-packing",
        "model": {"path": "hf:fake/mock-model", "container": "fake.sqsh", "precision": "bf16"},
        "resources": {"gpu_type": "gb300", "gpus_per_node": 4, "pack_prefill_decode": True},
        "engine": engine,
        "frontend": {"type": frontend, "enable_multiple_frontends": False},
        "roles": {
            "prefill": {"nodes": 1, "workers": 1, "gpus": 2},
            "decode": {"nodes": 2, "workers": 3, "gpus": 2},
        },
        "benchmark": {"type": "custom", "command": "echo done"},
    }


def load(tmp_path, data):
    path = tmp_path / "packing.yaml"
    path.write_text(yaml.safe_dump(data))
    return SrtConfig.from_yaml(path), path


@pytest.mark.parametrize(
    "backend", [AtomBackend, MockerBackend, SGLangBackend, TileRTBackend, TokenSpeedBackend, TRTLLMBackend, VLLMBackend]
)
def test_all_engine_allocators_pack_1p3d(backend):
    endpoints = backend().allocate_endpoints(
        num_prefill=1,
        num_decode=3,
        num_agg=0,
        gpus_per_prefill=2,
        gpus_per_decode=2,
        gpus_per_agg=0,
        gpus_per_node=4,
        available_nodes=("node1", "node2"),
        allow_prefill_decode_colocation=True,
    )
    assert [(e.mode, e.index, e.nodes, e.gpu_indices) for e in endpoints] == EXPECTED


@pytest.mark.parametrize(("engine", "frontend"), ENGINE_FRONTENDS)
@pytest.mark.parametrize("packing", [False, True])
def test_node_count_and_sbatch_match_placement(tmp_path, engine, frontend, packing):
    data = recipe(engine, frontend)
    data["resources"]["pack_prefill_decode"] = packing
    config, path = load(tmp_path, data)
    assert config.total_nodes == (2 if packing else 3)
    with patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None):
        script = generate_minimal_sbatch_script(config=config, config_path=path)
    assert f"#SBATCH --nodes={config.total_nodes}" in script
    nodes = ("node1", "node2") if packing else ("node1", "node2", "node3")
    endpoints = config.allocate_worker_endpoints(nodes)
    if packing:
        assert [(e.mode, e.index, e.nodes, e.gpu_indices) for e in endpoints] == EXPECTED
    else:
        assert not set(endpoints[0].nodes) & {n for e in endpoints[1:] for n in e.nodes}


@pytest.mark.parametrize(("engine", "frontend"), ENGINE_FRONTENDS)
def test_fragmentation_is_included_in_node_count(tmp_path, engine, frontend):
    data = recipe(engine, frontend)
    data["roles"]["prefill"]["gpus"] = 3
    data["roles"]["decode"]["gpus"] = 3
    config, _ = load(tmp_path, data)
    # 12 GPUs still need four nodes: two 3-GPU workers cannot share a 4-GPU node.
    assert config.engine_node_count == 4
    endpoints = config.allocate_worker_endpoints([f"n{i}" for i in range(4)])
    assert len({n for e in endpoints for n in e.nodes}) == 4


@pytest.mark.parametrize(("engine", "frontend"), ENGINE_FRONTENDS)
def test_multinode_workers_keep_their_engine_layout(tmp_path, engine, frontend):
    data = recipe(engine, frontend)
    data["roles"]["decode"].update(workers=1, gpus=6)
    config, _ = load(tmp_path, data)
    # MPI can use the partial prefill node; other engines need equal-sized ranks
    # on two fresh nodes. Both must size the allocation from their real layout.
    assert config.engine_node_count == (2 if engine == "trtllm" else 3)
    endpoints = config.allocate_worker_endpoints([f"n{i}" for i in range(config.engine_node_count)])
    occupied = set()
    for endpoint in endpoints:
        assert endpoint.total_gpus == (2 if endpoint.mode == "prefill" else 6)
        for rank, node in enumerate(endpoint.nodes):
            for gpu in endpoint.gpus_on_node(rank):
                assert (node, gpu) not in occupied
                occupied.add((node, gpu))
    assert len(occupied) == 8


def test_packing_with_role_engines(tmp_path):
    data = recipe()
    data.pop("engine")
    for role in data["roles"].values():
        role["engine"] = "sglang"
    config, _ = load(tmp_path, data)
    assert config.engine_node_count == 2
    assert [
        (e.mode, e.index, e.nodes, e.gpu_indices) for e in config.allocate_worker_endpoints(("node1", "node2"))
    ] == EXPECTED


def test_pools_and_dedicated_nodes_remain_outside_worker_pool(tmp_path):
    data = recipe()
    data["frontend"]["placement"] = {"node": "dedicated"}
    data["services"] = [{"name": "extra", "type": "generic", "nodes": 2, "command": ["sleep", "infinity"]}]
    config, path = load(tmp_path, data)
    assert config.engine_node_count == 2
    assert config.total_nodes == 4  # two engine nodes plus two service nodes
    with patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None):
        script = generate_minimal_sbatch_script(config=config, config_path=path)
    assert "#SBATCH --nodes=5" in script  # dedicated frontend is additional


@pytest.mark.parametrize(("engine", "frontend"), ENGINE_FRONTENDS)
def test_mock_launches_use_packed_nodes_and_gpu_masks(tmp_path, monkeypatch, engine, frontend):
    monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "absent.yaml"))
    config, path = load(tmp_path, recipe(engine, frontend))
    launches = []
    assert (
        run_mock_sweep(
            config_path=path,
            output_dir=tmp_path / "output",
            job_id="4242",
            options=MockOptions(
                nodelist=("node1", "node2"), phase_pause_s=0, child_duration_s=0, on_srun=launches.append
            ),
        )
        == 0
    )
    workers = [c for c in launches if str(c.get("step_name", "")).startswith(("prefill_", "decode_"))]
    assert len(workers) == 4
    assert [tuple(c["nodelist"]) for c in workers] == [("node1",), ("node1",), ("node2",), ("node2",)]
    masks = ["0,1", "2,3", "0,1", "2,3"]
    if config.backend.should_set_visible_devices():
        assert [c["env_to_set"]["CUDA_VISIBLE_DEVICES"] for c in workers] == masks
    else:
        for call, mask in zip(workers, masks, strict=True):
            command = call["command"]
            assert command[command.index("--device-ids") + 1] == mask


@pytest.mark.parametrize(
    "change",
    [
        {"resources": {"spread_workers": True}},
        {"resources": {"het_jobs": True}},
        {"roles": {"decode": {"nodes": "colocate"}}},
        {"roles": {"prefill": {"gpu_indices": [0, 1]}}},
        {"roles": {"prefill": None}},
    ],
)
def test_incompatible_layouts_fail_before_submission(tmp_path, change):
    data = deepcopy(recipe())
    for group, fields in change.items():
        for key, value in fields.items():
            if value is None:
                data[group].pop(key)
            elif isinstance(value, dict):
                data[group][key].update(value)
            else:
                data[group][key] = value
    with pytest.raises(ValidationError, match="resources.pack_prefill_decode"):
        load(tmp_path, data)


def test_cluster_default_het_rejected_at_render(tmp_path):
    config, path = load(tmp_path, recipe())
    with (
        patch(
            "srtctl.cli.submit.get_srtslurm_setting", side_effect=lambda k, *a: True if k == "use_het_jobs" else None
        ),
        pytest.raises(ValueError, match="resources.pack_prefill_decode"),
    ):
        generate_minimal_sbatch_script(config=config, config_path=path)
