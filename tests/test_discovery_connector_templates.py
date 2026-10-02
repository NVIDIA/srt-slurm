# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render real worker commands for discovery connectors with CPU offload."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from srtctl.backends import VLLMProtocol, VLLMServerConfig
from srtctl.backends.vllm import WorkerMode
from srtctl.core.topology import Process


def _template() -> dict[str, Any]:
    return {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MoRIIOConnector",
                    "kv_role": "kv_producer",
                    "kv_load_failure_policy": "fail",
                    "kv_connector_extra_config": {"qp_per_transfer": 8, "backend": "rdma"},
                },
                {
                    "kv_connector": "SimpleCPUOffloadConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {"cpu_bytes_to_use": 1024, "lazy_offload": False},
                },
            ]
        },
    }


def _command(
    template: Any,
    mode: WorkerMode = "prefill",
    connector: str = "moriio",
    *,
    key: str = "kv-transfer-config",
    extra_args: dict[str, Any] | None = None,
) -> list[str]:
    backend = VLLMProtocol(
        connector=connector,
        vllm_config=VLLMServerConfig(**{mode: {"tensor-parallel-size": 2, key: template, **(extra_args or {})}}),
    )
    process = Process(
        "node1",
        frozenset({0, 1}),
        7500,
        6101,
        mode,
        0,
        moriio_handshake_port=26008,
        moriio_notify_port=27008,
    )
    runtime = SimpleNamespace(
        model_path=Path("/weights"),
        is_hf_model=False,
        frontend_port=8000,
        network_interface="rdma-test",
        request_plane="nats",
        head_node_ip="10.0.0.1",
    )
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.2"):
        return backend.build_worker_command(process, [process], runtime, frontend_type="vllm-router")


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("key", ["kv-transfer-config", "kv_transfer_config"])
def test_offload_template_receives_discovery_topology_without_mutating_input(encoded: bool, key: str) -> None:
    template = _template()
    original = copy.deepcopy(template)
    command = _command(json.dumps(template) if encoded else template, key=key)
    assert command.count("--kv-transfer-config") == 1
    result = json.loads(command[command.index("--kv-transfer-config") + 1])
    mori, cpu = result["kv_connector_extra_config"]["connectors"]
    assert mori["kv_connector_extra_config"] == {
        "qp_per_transfer": 8,
        "backend": "rdma",
        "read_mode": True,
        "proxy_ip": "10.0.0.1",
        "proxy_ping_port": "36367",
        "host_ip": "10.0.0.2",
        "http_port": "6101",
        "handshake_port": "26008",
        "notify_port": "27008",
    }
    assert mori["kv_role"] == "kv_producer"
    assert result["kv_load_failure_policy"] == mori["kv_load_failure_policy"] == "fail"
    assert cpu == {
        "kv_connector": "SimpleCPUOffloadConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"cpu_bytes_to_use": 1024, "lazy_offload": False},
    }
    assert template == original


@pytest.mark.parametrize("key", ["kv-transfer-config", "kv_transfer_config"])
def test_direct_consumer_keeps_transfer_options_and_receives_allocated_ports(key: str) -> None:
    template = {
        "kv_connector": "MoRIIOConnector",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {"qp_per_transfer": 4},
    }
    command = _command(template, "decode", key=key)
    assert command.count("--kv-transfer-config") == 1
    result = json.loads(command[command.index("--kv-transfer-config") + 1])
    assert result["kv_role"] == "kv_consumer"
    assert result["kv_load_failure_policy"] == "fail"
    assert result["kv_connector_extra_config"]["qp_per_transfer"] == 4
    assert result["kv_connector_extra_config"]["notify_port"] == "27008"


@pytest.mark.parametrize(
    "field,value",
    [
        ("host_ip", "wrong-host"),
        ("http_port", "1"),
        ("http_port", 6102),
        ("http_port", 6101.0),
        ("http_port", True),
        ("read_mode", False),
    ],
)
def test_conflicting_discovery_bindings_fail_before_launch(field: str, value: Any) -> None:
    template = _template()
    template["kv_connector_extra_config"]["connectors"][0]["kv_connector_extra_config"][field] = value
    with pytest.raises(ValueError, match=field):
        _command(template)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong-role", "not-object"])
def test_ambiguous_or_invalid_discovery_template_fails(kind: str) -> None:
    template = _template()
    children = template["kv_connector_extra_config"]["connectors"]
    if kind == "missing":
        children.pop(0)
    elif kind == "duplicate":
        children.append(copy.deepcopy(children[0]))
    elif kind == "wrong-role":
        children[0]["kv_role"] = "kv_consumer"
    else:
        template = "[]"
    with pytest.raises((ValueError, TypeError)):
        _command(template)


def test_nested_template_binds_only_the_discovery_child() -> None:
    template = {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"connectors": [_template()]},
    }
    original = copy.deepcopy(template)
    command = _command(template)
    result = json.loads(command[command.index("--kv-transfer-config") + 1])
    chain = result["kv_connector_extra_config"]["connectors"][0]
    children = chain["kv_connector_extra_config"]["connectors"]
    assert children[0]["kv_connector_extra_config"]["http_port"] == "6101"
    assert children[1] == _template()["kv_connector_extra_config"]["connectors"][1]
    assert "host_ip" not in result["kv_connector_extra_config"]
    assert "host_ip" not in chain["kv_connector_extra_config"]
    assert template == original


@pytest.mark.parametrize("integer_ports", [False, True])
def test_matching_explicit_bindings_are_accepted(integer_ports: bool) -> None:
    expected = {
        "proxy_ip": "10.0.0.1",
        "proxy_ping_port": "36367",
        "host_ip": "10.0.0.2",
        "http_port": "6101",
        "handshake_port": "26008",
        "notify_port": "27008",
        "read_mode": True,
    }
    extra = dict(expected)
    if integer_ports:
        extra.update(proxy_ping_port=36367, http_port=6101, handshake_port=26008, notify_port=27008)
    template = {"kv_connector": "MoRIIOConnector", "kv_connector_extra_config": extra}
    original = copy.deepcopy(template)
    command = _command(template)
    result = json.loads(command[command.index("--kv-transfer-config") + 1])
    assert result["kv_connector_extra_config"] == expected
    assert template == original


def test_duplicate_discovery_argument_spellings_are_rejected() -> None:
    template = _template()
    with pytest.raises(ValueError, match="both kv-transfer-config and kv_transfer_config"):
        _command(template, extra_args={"kv_transfer_config": template})


@pytest.mark.parametrize(
    "extra,error",
    [
        ([], "kv_connector_extra_config must be a JSON object"),
        ({"connectors": {}}, "connectors must be a list"),
        ({"connectors": [None]}, "connectors must be JSON objects"),
    ],
)
def test_malformed_connector_chain_fails(extra: Any, error: str) -> None:
    template = _template()
    template["kv_connector_extra_config"] = extra
    with pytest.raises(TypeError, match=error):
        _command(template)


def test_non_discovery_explicit_config_keeps_existing_precedence() -> None:
    template = json.dumps({"kv_connector": "CustomConnector", "kv_role": "kv_both"})
    command = _command(template, connector="nixl")
    assert command.count("--kv-transfer-config") == 1
    assert command[command.index("--kv-transfer-config") + 1] == template
