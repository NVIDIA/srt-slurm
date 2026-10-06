# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone router launch, readiness and engine-publisher allocation contracts."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.core.schema import SrtConfig
from srtctl.core.topology import NodePortAllocator
from srtctl.frontends import SGLRouterFrontend
from srtctl.mock import MockOptions, run_mock_sweep

RECIPE = Path("examples/sglang/sgl-router-disagg.yaml")


def test_standalone_router_launches_stock_binary_and_allocated_publishers(tmp_path):
    launches = []
    result = run_mock_sweep(
        config_path=RECIPE,
        job_id="42042",
        output_dir=tmp_path / "outputs" / "42042",
        options=MockOptions(on_srun=launches.append),
    )
    assert result == 0
    router = next(launch for launch in launches if launch.get("step_name") == "sgl_router_0")
    command = router["command"]
    assert command[0] == "/usr/local/bin/sgl-router"
    assert router["use_bash_wrapper"] is False
    assert "--pd-disaggregation" not in command and "--prefill" not in command
    assert command[command.index("--model-id") + 1] == "Qwen/Qwen3-0.6B"
    assert command.count("--worker-urls") == 1
    assert "--disable-input-ids-forwarding" in command
    ports = []
    for launch in launches:
        command = launch["command"]
        if "sglang.launch_server" not in command:
            continue
        kv = json.loads(command[command.index("--kv-events-config") + 1])
        ports.extend([kv["endpoint"], kv["replay_endpoint"], command[command.index("--load-publish-endpoint") + 1]])
    assert len(ports) == len(set(ports)) == 9


@pytest.mark.parametrize("missing_worker,unhealthy", [(False, False), (True, False), (False, True)])
def test_readyz_alone_does_not_satisfy_full_worker_readiness(missing_worker, unhealthy):
    ready = MagicMock()
    metrics = MagicMock()
    metrics.text = (
        f'sgl_router_workers{{mode="prefill"}} {1 if missing_worker else 2}\n'
        'sgl_router_workers{mode="decode"} 1\n'
        'sgl_router_workers{mode="plain"} 0\n'
        'sgl_router_worker_health{worker_url="http://p0"} 1\n'
        f'sgl_router_worker_health{{worker_url="http://d0"}} {0 if unhealthy else 1}\n'
    )
    if not missing_worker:
        metrics.text += 'sgl_router_worker_health{worker_url="http://p1"} 1\n'
    with patch("srtctl.frontends.sgl_router.requests.get", side_effect=[ready, metrics]):
        result = SGLRouterFrontend().probe_ready("localhost", 8000, 2, 1, None)
    assert result.ready == (not missing_worker and not unhealthy)


@pytest.mark.parametrize("dp_args", [{"data-parallel-size": 4}, {"attention-data-parallel-size": 4}])
def test_colocated_publishers_reserve_every_dp_rank(dp_args):
    config = SrtConfig.from_yaml(RECIPE)
    roles = {mode: replace(role, args={**role.args, **dp_args}) for mode, role in config.roles.items()}
    backend = replace(config.backend, roles=roles)
    endpoints = config.allocate_worker_endpoints(["node0"])
    processes = backend.endpoints_to_processes(endpoints, port_allocator=NodePortAllocator())
    ports = [
        port + rank
        for process in processes
        for port in (process.kv_events_port, process.kv_replay_port, process.load_publish_port)
        for rank in range(4)
    ]
    assert len(ports) == len(set(ports)) == 36
