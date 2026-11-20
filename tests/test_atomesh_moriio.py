# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static MoRI-IO READ: real command generation, topology and failure gates."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from marshmallow import ValidationError

from srtctl.backends import VLLMBackend
from srtctl.core.schema import FrontendConfig, RestartPolicy, RoleConfig
from srtctl.core.topology import Process
from srtctl.frontends import AtomeshFrontend
from srtctl.ports import VLLM_DISCOVERY_PORT
from tests.test_atomesh_frontend import _config


def _moriio(**overrides):
    return _config(**({"engine": VLLMBackend(connector="moriio")} | overrides))


def test_static_worker_urls_have_no_bootstrap_or_discovery_port():
    config = _moriio()
    frontend = AtomeshFrontend()
    workers = [
        Process(
            "node",
            frozenset({i}),
            7500 + i,
            6100 + i,
            role,
            0,
            moriio_handshake_port=6300 + i * 8,
            moriio_notify_port=6400 + i * 8,
        )
        for i, role in enumerate(("prefill", "decode"))
    ]
    with patch("srtctl.frontends.static_router.get_hostname_ip", return_value="10.0.0.1"):
        endpoints = frontend.collect_workers(config.backend, workers)
        health = frontend.get_backend_health_urls(config.backend, workers)
    cmd = frontend.build_router_command(endpoints, "0.0.0.0", 8000, config.backend)
    cmd += frontend.get_managed_frontend_args(config, config.backend, workers)
    assert cmd[cmd.index("--prefill") + 1 : cmd.index("--decode")] == ["http://10.0.0.1:6100"]
    assert cmd[-4:] == ["--backend", "vllm", "--vllm-kv-connector", "moriio-read"]
    assert health == ["http://10.0.0.1:6100/health", "http://10.0.0.1:6101/health"]


@pytest.mark.parametrize("nested", [False, True])
def test_moriio_metadata_disables_proxy_and_preserves_offload(nested):
    target = {"kv_connector": "MoRIIOConnector", "kv_connector_extra_config": {"read_mode": True}}
    payload = target
    if nested:
        payload = {
            "kv_connector": "MultiConnector",
            "kv_connector_extra_config": {
                "connectors": [
                    {"kv_connector": "MultiConnector", "kv_connector_extra_config": {"connectors": [target]}},
                    {"kv_connector": "OffloadingConnector", "kv_connector_extra_config": {"cpu_bytes_to_use": 123}},
                ]
            },
        }
    config = _moriio(
        roles={
            mode: RoleConfig(nodes=1, workers=1, gpus=1, args={"kv-transfer-config": payload})
            for mode in ("prefill", "decode")
        }
    )
    worker = Process(
        "node", frozenset({0}), 7500, 6100, "prefill", 0, moriio_handshake_port=6300, moriio_notify_port=6400
    )
    runtime = SimpleNamespace(head_node_ip="10.0.0.1", network_interface="ib0")
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.2"):
        result = json.loads(config.backend.kv_transfer_config("prefill", worker, runtime, register_with_router=False))
    leaf = result
    if nested:
        assert (
            result["kv_connector_extra_config"]["connectors"][1]
            == payload["kv_connector_extra_config"]["connectors"][1]
        )
        leaf = result["kv_connector_extra_config"]["connectors"][0]["kv_connector_extra_config"]["connectors"][0]
    assert leaf["kv_role"] == "kv_producer"
    assert leaf["kv_connector_extra_config"] == {
        "read_mode": True,
        "proxy_ip": "",
        "proxy_ping_port": str(VLLM_DISCOVERY_PORT),
        "http_port": "6100",
        "host_ip": "10.0.0.2",
        "handshake_port": "6300",
        "notify_port": "6400",
    }
    assert target["kv_connector_extra_config"] == {"read_mode": True}


@pytest.mark.parametrize("read_mode", [False, "true", 1])
def test_rejects_write_or_ambiguous_mode(read_mode):
    with pytest.raises(ValidationError, match="read_mode"):
        payload = {"kv_connector": "MoRIIOConnector", "kv_connector_extra_config": {"read_mode": read_mode}}
        _moriio(
            roles={
                mode: RoleConfig(nodes=1, workers=1, gpus=1, args={"kv-transfer-config": payload})
                for mode in ("prefill", "decode")
            }
        )


@pytest.mark.parametrize(
    "role,changes,message",
    [
        ("decode", {"args": {"connector": "mooncake"}}, "both prefill and decode"),
        ("decode", {"args": {"data-parallel-size": 2}}, "data-parallel-size: 1"),
        ("decode", {"args": {"tensor-parallel-size": 2}}, "same tensor-parallel-size"),
        ("prefill", {"restart": RestartPolicy(policy="on-failure")}, "restart: never"),
        ("decode", {"restart": RestartPolicy(policy="on-failure")}, "restart: never"),
    ],
)
def test_rejects_unsupported_topology(role, changes, message):
    roles = {mode: RoleConfig(nodes=1, workers=1, gpus=2) for mode in ("prefill", "decode")}
    roles[role] = replace(roles[role], **changes)
    with pytest.raises(ValidationError, match=message):
        _moriio(roles=roles)


def test_router_protocol_is_managed():
    with pytest.raises(ValidationError, match="managed by srtctl"):
        _moriio(
            frontend=FrontendConfig(
                type="atomesh", enable_multiple_frontends=False, args={"vllm-kv-connector": "mooncake"}
            )
        )


def test_mock_launch_uses_static_read_and_allocated_ports(tmp_path, monkeypatch):
    import shlex
    from pathlib import Path

    import yaml

    from srtctl.mock import MockOptions, run_mock_sweep

    monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "absent.yaml"))
    recipe = yaml.safe_load(Path("examples/vllm/vllm-router-moriio-disagg.yaml").read_text())
    recipe["frontend"] = {"type": "atomesh", "enable_multiple_frontends": False, "container_image": "router-image"}
    recipe["model"] = {"path": "hf:fake/mock-model", "container": "worker-image", "precision": "bf16"}
    recipe["observability"] = {"enabled": False, "tachometer": {"enabled": False}}
    recipe["benchmark"] = {"type": "custom", "command": "echo static-read"}
    path = tmp_path / "mock.yaml"
    path.write_text(yaml.safe_dump(recipe))
    calls = []
    assert (
        run_mock_sweep(
            config_path=path,
            output_dir=tmp_path / "out",
            job_id="42043",
            options=MockOptions(
                child_duration_s=0, phase_pause_s=0, on_srun=calls.append, nodelist=("n0", "n1", "n2", "n3")
            ),
        )
        == 0
    )
    workers = [call for call in calls if str(call.get("step_name", "")).startswith(("prefill_", "decode_"))]
    assert len(workers) == 2
    ports = set()
    for call in workers:
        argv = call["command"]
        if isinstance(argv, str):
            argv = shlex.split(argv)
        payload = json.loads(argv[argv.index("--kv-transfer-config") + 1])
        extra = payload["kv_connector_extra_config"]
        assert extra["proxy_ip"] == "" and extra["read_mode"] is True
        for key in ("handshake_port", "notify_port"):
            assert extra[key] not in ports
            ports.add(extra[key])
    router = next(call for call in calls if call.get("step_name") == "atomesh_0")
    assert "moriio-read" in str(router["command"])
