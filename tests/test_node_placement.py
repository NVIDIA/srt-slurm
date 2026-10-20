# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for orchestrator/benchmark node placement (first_decode / last_decode).

The trtllm-serve path can run the disaggregated orchestrator on the first GEN
node and the benchmark client on the last GEN node, off the CTX/head node.
"""

import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
import yaml

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Process, ordered_decode_leader_nodes, placed_node


def _proc(node: str, mode: str, idx: int, rank: int = 0) -> Process:
    return Process(
        node=node,
        gpu_indices=frozenset({0}),
        sys_port=0,
        http_port=6100,
        endpoint_mode=mode,
        endpoint_index=idx,
        node_rank=rank,
    )


# --- pure helpers -----------------------------------------------------------


def test_ordered_decode_leader_nodes_orders_by_index_and_dedups():
    procs = [
        _proc("p0", "prefill", 0),
        _proc("g2", "decode", 3),
        _proc("g0", "decode", 1),
        _proc("g0", "decode", 1, rank=1),  # non-leader, same node -> ignored
        _proc("g1", "decode", 2),
    ]
    assert ordered_decode_leader_nodes(procs) == ["g0", "g1", "g2"]


def test_placed_node_head_ignores_processes():
    # head never looks at the process list (empty list is fine).
    assert placed_node([], "head", "p0", kind="x") == "p0"


def test_placed_node_first_and_last_decode():
    procs = [_proc("p0", "prefill", 0), _proc("g0", "decode", 1), _proc("g1", "decode", 2)]
    assert placed_node(procs, "first_decode", "p0", kind="x") == "g0"
    assert placed_node(procs, "last_decode", "p0", kind="x") == "g1"


def test_placed_node_invalid_value_raises():
    with pytest.raises(ValueError, match="invalid"):
        placed_node([], "second_decode", "p0", kind="frontend.orchestrator_placement")


def test_placed_node_no_decode_workers_raises():
    procs = [_proc("p0", "prefill", 0)]
    with pytest.raises(ValueError, match="no decode workers"):
        placed_node(procs, "first_decode", "p0", kind="x")


# --- integration with SweepOrchestrator -------------------------------------


def _config(
    *,
    orchestrator_placement="head",
    client_placement="head",
    frontend_type="trtllm_serve",
    enable_multiple_frontends=False,
) -> SrtConfig:
    data = {
        "schema": 2,
        "name": "test",
        "model": {"path": "/models/test", "container": "test.sqsh", "precision": "fp4"},
        "resources": {"gpu_type": "gb300", "gpus_per_node": 4},
        "engine": "trtllm",
        "roles": {
            "prefill": {"nodes": 1, "workers": 1, "gpus": 4},
            "decode": {"nodes": 2, "workers": 2, "gpus": 4},
        },
        "frontend": {
            "type": frontend_type,
            "enable_multiple_frontends": enable_multiple_frontends,
            "placement": {"node": orchestrator_placement},
        },
        "benchmark": {"type": "custom", "command": "true", "placement": {"node": client_placement}},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        path = f.name
    return SrtConfig.from_yaml(Path(path))


def _runtime(nodes: list[str]) -> RuntimeContext:
    return RuntimeContext(
        job_id="1",
        run_name="test",
        nodes=Nodes(head=nodes[0], bench=nodes[0], infra=nodes[0], worker=tuple(nodes)),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=Path("/tmp/logs"),
        model_path=Path("/models/test"),
        container_image=Path("/c.sqsh"),
        gpus_per_node=4,
        network_interface=None,
        container_mounts={},
        environment={},
    )


_PROCS = [_proc("p0", "prefill", 0), _proc("g0", "decode", 1), _proc("g1", "decode", 2)]


def test_frontend_default_head_does_not_touch_processes():
    orch = SweepOrchestrator(config=_config(), runtime=_runtime(["p0", "g0", "g1"]))
    # No backend_processes mock: head placement must not evaluate the property.
    topo = orch._compute_frontend_topology()
    assert topo.frontend_nodes == ["p0"]


def test_public_api_node_on_head_does_not_require_frontend_topology():
    orch = SweepOrchestrator(config=_config(), runtime=_runtime(["p0", "g0", "g1"]))
    with (
        patch.object(type(orch), "backend_processes", new_callable=PropertyMock, return_value=_PROCS),
        patch.object(orch, "_compute_frontend_topology", side_effect=AssertionError("topology is not needed")),
    ):
        assert orch._public_api_node() == "p0"


def test_frontend_orchestrator_on_first_decode():
    orch = SweepOrchestrator(
        config=_config(orchestrator_placement="first_decode"), runtime=_runtime(["p0", "g0", "g1"])
    )
    with patch.object(type(orch), "backend_processes", new_callable=PropertyMock, return_value=_PROCS):
        topo = orch._compute_frontend_topology()
    assert topo.frontend_nodes == ["g0"]


def test_benchmark_client_on_last_decode_and_orchestrator_first_decode():
    orch = SweepOrchestrator(
        config=_config(orchestrator_placement="first_decode", client_placement="last_decode"),
        runtime=_runtime(["p0", "g0", "g1"]),
    )
    with patch.object(type(orch), "backend_processes", new_callable=PropertyMock, return_value=_PROCS):
        assert orch._compute_frontend_topology().frontend_nodes == ["g0"]
        assert orch._benchmark_node() == "g1"


@pytest.mark.parametrize("placement", ["first_decode", "last_decode"])
def test_public_api_node_uses_nginx_head_with_non_head_frontend_placement(placement):
    orch = SweepOrchestrator(
        config=_config(
            orchestrator_placement=placement,
            frontend_type="dynamo",
            enable_multiple_frontends=True,
        ),
        runtime=_runtime(["p0", "g0", "g1"]),
    )
    with patch.object(type(orch), "backend_processes", new_callable=PropertyMock, return_value=_PROCS):
        assert orch._public_api_node() == "p0"


def test_benchmark_env_injects_frontend_host():
    from unittest.mock import MagicMock

    orch = SweepOrchestrator(
        config=_config(orchestrator_placement="first_decode", client_placement="last_decode"),
        runtime=_runtime(["p0", "g0", "g1"]),
    )
    runner = MagicMock()
    runner.name = "custom"
    with (
        patch.object(type(orch), "backend_processes", new_callable=PropertyMock, return_value=_PROCS),
        patch("srtctl.cli.mixins.benchmark_stage.get_hostname_ip", side_effect=lambda n, *a, **k: f"ip-{n}"),
    ):
        env = orch._get_benchmark_env(runner)
    assert env["SRT_FRONTEND_HOST"] == "ip-g0"
    assert env["SRT_FRONTEND_PORT"] == "8000"


def _agg_config(*, engine="vllm", frontend_type="vllm", frontend_placement="head") -> SrtConfig:
    data = {
        "schema": 2,
        "name": "test",
        "model": {"path": "/models/test", "container": "test.sqsh", "precision": "fp4"},
        "resources": {"gpu_type": "b200", "gpus_per_node": 8},
        "engine": engine,
        "roles": {"agg": {"nodes": 2, "workers": 1}},
        "frontend": {
            "type": frontend_type,
            "enable_multiple_frontends": False,
            "placement": {"node": frontend_placement},
        },
        "benchmark": {"type": "custom", "command": "true"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        path = f.name
    return SrtConfig.from_yaml(Path(path))


_AGG_PROCS = [
    Process(
        node="g0",
        gpu_indices=frozenset(range(8)),
        sys_port=8081,
        http_port=0,
        endpoint_mode="agg",
        endpoint_index=0,
        node_rank=0,
    ),
    Process(
        node="g1",
        gpu_indices=frozenset(range(8)),
        sys_port=8082,
        http_port=0,
        endpoint_mode="agg",
        endpoint_index=0,
        node_rank=1,
    ),
]


def test_public_api_node_for_direct_vllm_agg_uses_leader_not_head():
    """Direct vLLM serve binds the public port on the agg endpoint leader."""
    orch = SweepOrchestrator(config=_agg_config(), runtime=_runtime(["p0", "g0", "g1"]))
    with patch.object(type(orch), "backend_processes", new_callable=PropertyMock, return_value=_AGG_PROCS):
        assert orch._compute_frontend_topology().frontend_nodes == ["p0"]
        assert orch._public_api_node() == "g0"


def test_public_api_node_for_direct_trtllm_agg_uses_leader_not_dedicated_head():
    runtime = replace(
        _runtime(["p0", "g0", "g1"]),
        nodes=Nodes(head="p0", bench="p0", infra="p0", worker=("g0", "g1")),
    )
    orch = SweepOrchestrator(
        config=_agg_config(engine="trtllm", frontend_type="trtllm_serve", frontend_placement="dedicated"),
        runtime=runtime,
    )
    with patch.object(type(orch), "backend_processes", new_callable=PropertyMock, return_value=_AGG_PROCS):
        assert orch._compute_frontend_topology().frontend_nodes == ["p0"]
        assert orch._public_api_node() == "g0"
