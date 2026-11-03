from copy import deepcopy

import pytest

from srtctl.core.schema import SrtConfig, Topology


def config(roles, *, engine="sglang", **resources):
    return {
        "schema": 2,
        "name": "inferred-nodes",
        "model": {"path": "hf:org/model", "container": "engine:tag", "precision": "fp8"},
        "engine": engine,
        "resources": {"gpus_per_node": 8, **resources},
        "roles": roles,
        "benchmark": {"type": "custom", "command": "true"},
        "telemetry": {"enabled": False},
    }


@pytest.mark.parametrize(
    "roles,resources,nodes,sizes",
    [
        ({"agg": {"workers": 3, "gpus": 4}}, {}, 2, [4, 4, 4]),
        ({"agg": {"workers": 3, "gpus": 5}}, {}, 3, [5, 5, 5]),
        ({"agg": {"workers": 1, "gpus": 16}}, {}, 2, [16]),
        ({"agg": {"workers": 3, "gpus": 4}}, {"spread_workers": True}, 3, [4, 4, 4]),
        ({"agg": {"workers": 3, "gpus": 2, "gpu_indices": [[0, 2], [1, 3]]}}, {}, 2, [2, 2, 2]),
        ({"prefill": {"workers": 1, "gpus": 2}, "decode": {"workers": 2, "gpus": 3}}, {}, 2, [2, 3, 3]),
        (
            {"prefill": {"workers": 1, "gpus": 2}, "decode": {"workers": 2, "gpus": 3, "nodes": "colocate"}},
            {},
            1,
            [2, 3, 3],
        ),
        (
            {"prefill": {"workers": 1, "gpus": 4}, "decode": {"workers": 1, "gpus": 16, "nodes": "colocate"}},
            {},
            3,
            [4, 16],
        ),
    ],
)
def test_worker_shape_sizes_the_actual_allocation(roles, resources, nodes, sizes):
    recipe = config(roles, **resources)
    fragment = {key: recipe[key] for key in ("roles", "resources", "engine")}
    assert Topology.from_recipe(fragment, gpus_per_node=8).total_nodes == nodes
    loaded = SrtConfig.Schema().load(recipe)
    assert loaded.total_nodes == nodes
    endpoints = loaded.allocate_worker_endpoints([f"node-{i}" for i in range(nodes)])
    assert [endpoint.total_gpus for endpoint in endpoints] == sizes
    occupied = {}
    for endpoint in endpoints:
        for rank, node in enumerate(endpoint.nodes):
            used = occupied.setdefault(node, set())
            gpus = endpoint.gpus_on_node(rank)
            assert not used & gpus
            used.update(gpus)


@pytest.mark.parametrize("engine", ["sglang", "vllm", "trtllm"])
def test_explicit_reservations_are_preserved(engine):
    recipe = config({"agg": {"workers": 1, "gpus": 4, "nodes": 3}}, engine=engine)
    assert Topology.from_recipe(recipe, gpus_per_node=8).total_nodes == 3
    assert SrtConfig.Schema().load(recipe).total_nodes == 3


def test_shape_planning_uses_the_backends_automatic_colocation():
    recipe = config(
        {"prefill": {"workers": 1, "gpus": 2}, "decode": {"workers": 1, "gpus": 2}},
        engine={"type": "vllm", "allow_prefill_decode_colocation": True},
    )
    assert Topology.from_recipe(recipe, gpus_per_node=8).total_nodes == 1
    assert SrtConfig.Schema().load(recipe).total_nodes == 1


def test_full_job_adds_service_nodes_to_the_inferred_worker_nodes():
    recipe = config({"agg": {"workers": 3, "gpus": 4}})
    recipe["services"] = [{"type": "generic", "name": "helper", "nodes": 1, "command": ["sleep", "infinity"]}]
    loaded = SrtConfig.Schema().load(recipe)
    assert loaded.engine_node_count == 2
    assert loaded.total_nodes == 3


def test_trt_inference_uses_its_mpi_packing():
    recipe = config({"agg": {"workers": 2, "gpus": 12}}, engine="trtllm")
    loaded = SrtConfig.Schema().load(recipe)
    assert Topology.from_recipe(recipe, gpus_per_node=8).total_nodes == 3
    assert loaded.total_nodes == 3
    assert [e.total_gpus for e in loaded.allocate_worker_endpoints(["n0", "n1", "n2"])] == [12, 12]


def test_fragment_planning_does_not_change_the_input():
    recipe = config({"agg": {"workers": 3, "gpus": 4}})
    original = deepcopy(recipe)
    assert Topology.from_recipe(recipe, gpus_per_node=8).total_nodes == 2
    assert recipe == original


@pytest.mark.parametrize("engine", ["sglang", "trtllm"])
def test_colocated_full_node_decode_starts_after_partial_prefill(engine):
    recipe = config(
        {"prefill": {"workers": 1, "gpus": 4}, "decode": {"workers": 1, "gpus": 16, "nodes": "colocate"}},
        engine=engine,
    )
    loaded = SrtConfig.Schema().load(recipe)
    assert loaded.total_nodes == 3
    prefill, decode = loaded.allocate_worker_endpoints(["n0", "n1", "n2"])
    assert prefill.nodes == ("n0",)
    assert decode.nodes == ("n1", "n2")
    assert [decode.gpus_on_node(i) for i in range(2)] == [frozenset(range(8))] * 2


def test_inference_rejects_an_unsized_role():
    recipe = config({"prefill": {"workers": 1}, "decode": {"workers": 1, "gpus": 4}})
    with pytest.raises(ValueError, match="deriving nodes requires explicit workers and gpus"):
        Topology.from_recipe(recipe, gpus_per_node=8).total_nodes
