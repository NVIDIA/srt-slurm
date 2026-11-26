# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from marshmallow import ValidationError

from srtctl.backends import AtomBackend, VLLMBackend
from srtctl.core.schema import FrontendConfig, ResourceConfig, RestartPolicy, RoleConfig, SrtConfig
from srtctl.core.topology import Process
from srtctl.frontends import AtomeshFrontend


def _config(**overrides) -> SrtConfig:
    fields = {
        "name": "atomesh-vllm",
        "model": {"path": "model", "container": "image", "precision": "bf16"},
        "resources": ResourceConfig(gpu_type="mi355x", gpus_per_node=8),
        "roles": {mode: RoleConfig(nodes=1, workers=1, gpus=1) for mode in ("prefill", "decode")},
        "frontend": FrontendConfig(type="atomesh", enable_multiple_frontends=False),
        "engine": VLLMBackend(connector="mooncake"),
    }
    return SrtConfig(**(fields | overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"engine": VLLMBackend(connector="nixl")}, "Mooncake on both"),
        ({"health_check": None}, "requires health_check"),
        ({"roles": {"agg": RoleConfig(nodes=1, workers=1, gpus=1)}}, "prefill/decode topology"),
        (
            {"frontend": FrontendConfig(type="atomesh", enable_multiple_frontends=True)},
            "enable_multiple_frontends: false",
        ),
        (
            {"frontend": FrontendConfig(type="atomesh", enable_multiple_frontends=False, args={"backend": "atom"})},
            "managed by srtctl",
        ),
    ],
)
def test_rejects_unsupported_vllm_config(overrides: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _config(**overrides)


@pytest.mark.parametrize(
    ("role", "settings", "message"),
    [
        ("prefill", {"restart": RestartPolicy(policy="on-failure")}, "prefill restart: never"),
        ("prefill", {"args": {"data-parallel-size": 2}}, "prefill data-parallel-size: 1"),
        (
            "decode",
            {"args": {"data-parallel-size": 2, "data-parallel-external-lb": True}},
            "decode internal load balancing",
        ),
        ("decode", {"args": {"connector": "nixl"}}, "Mooncake on both"),
    ],
)
def test_rejects_unsafe_worker_topology(role: str, settings: dict, message: str) -> None:
    roles = {mode: RoleConfig(nodes=1, workers=1, gpus=2) for mode in ("prefill", "decode")}
    roles[role] = replace(roles[role], **settings)
    with pytest.raises(ValidationError, match=message):
        _config(roles=roles)


def test_rejects_mixed_atom_and_vllm_roles() -> None:
    roles = {
        "prefill": RoleConfig(nodes=1, workers=1, gpus=1, engine=VLLMBackend()),
        "decode": RoleConfig(nodes=1, workers=1, gpus=1, engine=AtomBackend()),
    }
    with pytest.raises(ValidationError, match="same backend type"):
        _config(engine=None, roles=roles)


def test_mooncake_accepts_colocated_pp_prefill_and_tp_decode() -> None:
    config = _config(
        roles={
            "prefill": RoleConfig(
                nodes=1,
                workers=1,
                gpus=4,
                args={"tensor-parallel-size": 1, "pipeline-parallel-size": 4},
            ),
            "decode": RoleConfig(
                nodes="colocate",
                workers=1,
                gpus=4,
                args={"tensor-parallel-size": 4, "pipeline-parallel-size": 1},
            ),
        }
    )
    assert config.total_nodes == 1
    assert config.backend._get_parallel_size("prefill", "pipeline-parallel-size") == 4
    assert config.backend._get_parallel_size("decode", "tensor-parallel-size") == 4


def test_mooncake_accepts_decode_dp_ep_behind_one_api() -> None:
    config = _config(
        roles={
            "prefill": RoleConfig(nodes=1, workers=1, gpus=4, args={"pipeline-parallel-size": 4}),
            "decode": RoleConfig(
                nodes="colocate",
                workers=1,
                gpus=4,
                args={"data-parallel-size": 4, "enable-expert-parallel": True},
            ),
        }
    )
    assert config.total_nodes == 1
    assert config.backend._get_parallel_size("decode", "data-parallel-size") == 4
    assert not AtomeshFrontend().expands_node_local_dp


def test_vllm_bootstrap_health_and_metrics_follow_allocated_workers() -> None:
    config = _config()
    backend = config.backend
    frontend = AtomeshFrontend()
    processes = [
        Process("p0", frozenset({0}), 7500, 6100, "prefill", 0, bootstrap_port=7303),
        Process("d0", frozenset({1}), 7501, 6101, "decode", 0),
    ]
    with patch("srtctl.frontends.static_router.get_hostname_ip", return_value="10.0.0.1"):
        workers = frontend.collect_workers(backend, processes)
        health = frontend.get_backend_health_urls(backend, processes)
    command = frontend.build_router_command(workers, "0.0.0.0", 8000, backend)
    command.extend(frontend.get_managed_frontend_args(config, backend, processes))
    assert command == [
        "atomesh",
        "launch",
        "--pd-disaggregation",
        "--prefill",
        "http://10.0.0.1:6100",
        "7303",
        "--decode",
        "http://10.0.0.1:6101",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--backend",
        "vllm",
    ]
    assert health == ["http://10.0.0.1:6100/health", "http://10.0.0.1:6101/health"]
    assert frontend.worker_metrics_port(processes[0], SimpleNamespace()) == 6100
    assert frontend.worker_metrics_path(backend, "prefill") == "/metrics"
    with pytest.raises(ValueError, match="allocated Mooncake bootstrap port"):
        frontend.worker_bootstrap_port(backend, replace(processes[0], bootstrap_port=None))
