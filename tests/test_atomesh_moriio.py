# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static MoRI-IO READ/WRITE: command generation, topology and failure gates."""

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
@pytest.mark.parametrize("read_mode", [False, True])
def test_moriio_metadata_disables_proxy_and_preserves_offload(nested, read_mode):
    target = {"kv_connector": "MoRIIOConnector", "kv_connector_extra_config": {"read_mode": read_mode}}
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
        "read_mode": read_mode,
        "proxy_ip": "",
        "proxy_ping_port": str(VLLM_DISCOVERY_PORT),
        "http_port": "6100",
        "host_ip": "10.0.0.2",
        "handshake_port": "6300",
        "notify_port": "6400",
    }
    assert target["kv_connector_extra_config"] == {"read_mode": read_mode}


@pytest.mark.parametrize("read_mode", ["true", 1])
def test_rejects_ambiguous_mode(read_mode):
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


def test_static_write_pp4_dep4_uses_allocated_peer_ports_without_discovery():
    modes = {}
    for mode in ("prefill", "decode"):
        leaf = {"kv_connector": "MoRIIOConnector", "kv_connector_extra_config": {"read_mode": False}}
        payload = leaf
        if mode == "prefill":
            payload = {
                "kv_connector": "MultiConnector",
                "kv_connector_extra_config": {
                    "connectors": [
                        leaf,
                        {
                            "kv_connector": "SimpleCPUOffloadConnector",
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {"cpu_bytes_to_use": 123456},
                        },
                    ]
                },
            }
        args = {"tensor-parallel-size": 1, "kv-transfer-config": payload}
        args["pipeline-parallel-size" if mode == "prefill" else "data-parallel-size"] = 4
        modes[mode] = RoleConfig(nodes=1 if mode == "prefill" else "colocate", workers=1, gpus=4, args=args)
    config = _moriio(roles=modes)
    workers = [
        Process(
            "node",
            frozenset(range(i * 4, i * 4 + 4)),
            7500 + i,
            6100 + i,
            mode,
            0,
            moriio_handshake_port=6300 + i * 8,
            moriio_notify_port=6400 + i * 8,
        )
        for i, mode in enumerate(("prefill", "decode"))
    ]
    runtime = SimpleNamespace(head_node_ip="10.0.0.1", network_interface="ib0")
    with (
        patch("srtctl.frontends.static_router.get_hostname_ip", return_value="10.0.0.1"),
        patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.2"),
    ):
        args = AtomeshFrontend().get_managed_frontend_args(config, config.backend, workers, runtime)
    assert args[:4] == ["--backend", "vllm", "--vllm-kv-connector", "moriio-write"]
    peers = json.loads(args[args.index("--vllm-moriio-peers") + 1])
    assert [p["dp_size"] for p in peers] == [1, 4]
    assert [p["handshake_port"] for p in peers] == [6300, 6308]
    assert [p["notify_port"] for p in peers] == [6400, 6408]
    assert [p["url"] for p in peers] == ["http://10.0.0.1:6100", "http://10.0.0.1:6101"]
    assert all(p["host"] == "10.0.0.2" and p["tp_size"] == 1 for p in peers)
    with pytest.raises(ValueError, match="allocated runtime ports"):
        AtomeshFrontend().get_managed_frontend_args(config, config.backend, workers)


@pytest.mark.parametrize(
    "mode, overrides, gpus, message",
    [
        ("prefill", {"data-parallel-size": 2}, 8, "prefill data-parallel-size: 1"),
        ("decode", {"pipeline-parallel-size": 2}, 8, "decode PP=1"),
        ("decode", {"data-parallel-external-lb": True}, 4, "internal DP load balancing"),
        ("decode", {"data-parallel-size": 16}, 16, "DP pool to fit on one node"),
    ],
)
def test_static_write_rejects_unrepresentable_peer_pools(mode, overrides, gpus, message):
    roles = {}
    for role in ("prefill", "decode"):
        args = {
            "tensor-parallel-size": 1,
            "pipeline-parallel-size": 4 if role == "prefill" else 1,
            "data-parallel-size": 1 if role == "prefill" else 4,
            "kv-transfer-config": {
                "kv_connector": "MoRIIOConnector",
                "kv_connector_extra_config": {"read_mode": False},
            },
        }
        if role == mode:
            args.update(overrides)
        roles[role] = RoleConfig(nodes=1, workers=1, gpus=gpus if role == mode else 4, args=args)
    with pytest.raises(ValidationError, match=message):
        _moriio(roles=roles)


def test_mock_launch_static_write_preserves_pp_dep_and_offloader(tmp_path, monkeypatch):
    import shlex
    from pathlib import Path

    import yaml

    from srtctl.mock import MockOptions, run_mock_sweep

    monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "absent.yaml"))
    recipe = yaml.safe_load(Path("examples/vllm/vllm-router-moriio-disagg.yaml").read_text())
    recipe["frontend"] = {"type": "atomesh", "enable_multiple_frontends": False, "container_image": "router-image"}
    recipe["model"] = {"path": "hf:fake/mock-model", "container": "worker-image", "precision": "bf16"}
    recipe["observability"] = {"enabled": False, "tachometer": {"enabled": False}}
    recipe["benchmark"] = {"type": "custom", "command": "echo static-write"}
    for mode in ("prefill", "decode"):
        leaf = {
            "kv_connector": "MoRIIOConnector",
            "kv_role": "kv_producer" if mode == "prefill" else "kv_consumer",
            "kv_connector_extra_config": {"read_mode": False},
        }
        kv = (
            leaf
            if mode == "decode"
            else {
                "kv_connector": "MultiConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "connectors": [
                        leaf,
                        {
                            "kv_connector": "SimpleCPUOffloadConnector",
                            "kv_role": "kv_both",
                            "kv_connector_extra_config": {"cpu_bytes_to_use": 123456},
                        },
                    ]
                },
            }
        )
        args = {"tensor-parallel-size": 1, "kv-transfer-config": kv}
        args["pipeline-parallel-size" if mode == "prefill" else "data-parallel-size"] = 4
        recipe["roles"][mode] = {"nodes": 1 if mode == "prefill" else "colocate", "workers": 1, "gpus": 4, "args": args}
    path = tmp_path / "write.yaml"
    path.write_text(yaml.safe_dump(recipe))
    calls = []
    assert (
        run_mock_sweep(
            config_path=path,
            output_dir=tmp_path / "out",
            job_id="42104",
            options=MockOptions(child_duration_s=0, phase_pause_s=0, on_srun=calls.append, nodelist=("n0", "n1")),
        )
        == 0
    )
    actual = {}
    for call in calls:
        step = str(call.get("step_name", ""))
        cmd = call.get("command", [])
        argv = shlex.split(cmd) if isinstance(cmd, str) else cmd
        if step.startswith(("prefill_", "decode_")):
            mode = "prefill" if step.startswith("prefill_") else "decode"
            kv = json.loads(argv[argv.index("--kv-transfer-config") + 1])
            if mode == "prefill":
                children = kv["kv_connector_extra_config"]["connectors"]
                assert children[1]["kv_connector_extra_config"]["cpu_bytes_to_use"] == 123456
                kv = children[0]
            extra = kv["kv_connector_extra_config"]
            assert extra["read_mode"] is False and extra["proxy_ip"] == ""
            actual[mode] = extra
        if step == "atomesh_0":
            router = argv
    assert set(actual) == {"prefill", "decode"}
    assert router[router.index("--vllm-kv-connector") + 1] == "moriio-write"
    peers = json.loads(router[router.index("--vllm-moriio-peers") + 1])
    for peer, mode in zip(peers, ("prefill", "decode"), strict=True):
        assert peer["host"] == actual[mode]["host_ip"]
        assert peer["handshake_port"] == int(actual[mode]["handshake_port"])
        assert peer["notify_port"] == int(actual[mode]["notify_port"])
        assert peer["dp_size"] == (1 if mode == "prefill" else 4)
