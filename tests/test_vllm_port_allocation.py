# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-process VLLM_PORT assignment (rendezvous EADDRINUSE avoidance)."""

from srtctl.backends.vllm import VLLMProtocol, VLLMServerConfig
from srtctl.core.topology import Endpoint, NodePortAllocator, Process
from srtctl.ports import (
    VLLM_PORT_BASE,
    VLLM_PORT_STRIDE,
)


def _colocated_decode_endpoints(count: int) -> list[Endpoint]:
    """``count`` TP1 decode workers sharing one node."""
    return [
        Endpoint(mode="decode", index=index, nodes=("node0",), gpu_indices=frozenset({index}), gpus_per_node=8)
        for index in range(count)
    ]


def test_vllm_port_is_unique_per_process_with_stride():
    """Co-located workers get distinct VLLM_PORT bases spaced by the full stride."""
    backend = VLLMProtocol()
    processes = backend.endpoints_to_processes(_colocated_decode_endpoints(3), port_allocator=NodePortAllocator())

    ports = [int(backend.get_process_environment(process)["VLLM_PORT"]) for process in processes]

    assert ports == [
        VLLM_PORT_BASE,
        VLLM_PORT_BASE + VLLM_PORT_STRIDE,
        VLLM_PORT_BASE + 2 * VLLM_PORT_STRIDE,
    ]
    # Distinct and a full stride apart, so per-process get_open_port() scan
    # ranges cannot overlap.
    assert len(set(ports)) == len(ports)
    assert all(ports[i + 1] - ports[i] == VLLM_PORT_STRIDE for i in range(len(ports) - 1))


def test_vllm_port_comes_from_the_allocation_not_the_system_port():
    """A process built without an allocated scan range sets no VLLM_PORT; nothing is derived from sys_port."""
    process = Process("node0", frozenset({0}), 9999, 0, "decode", 0)

    assert "VLLM_PORT" not in VLLMProtocol().get_process_environment(process)


def test_discovery_connector_workers_get_listeners_instead_of_a_scan_range():
    """Each colocated TP4 worker needs four handshake and four notify ports."""
    backend = VLLMProtocol(connector="moriio", vllm_config=VLLMServerConfig(decode={"tensor-parallel-size": 4}))
    endpoints = [
        Endpoint(mode="decode", index=0, nodes=("node0",), gpu_indices=frozenset({0, 1, 2, 3}), gpus_per_node=8),
        Endpoint(mode="decode", index=1, nodes=("node0",), gpu_indices=frozenset({4, 5, 6, 7}), gpus_per_node=8),
    ]

    processes = backend.endpoints_to_processes(
        endpoints,
        port_allocator=NodePortAllocator(bases={"moriio_handshake": 12000, "moriio_notify": 13000}),
        frontend_type="vllm-router",
    )

    assert [p.moriio_handshake_port for p in processes] == [12000, 12004]
    assert [p.moriio_notify_port for p in processes] == [13000, 13004]
    assert all(p.vllm_scan_port is None for p in processes)
    for process in processes:
        env = backend.get_process_environment(process)
        assert "VLLM_PORT" not in env
        assert "VLLM_NIXL_SIDE_CHANNEL_PORT" not in env


def test_discovery_listeners_avoid_linux_ephemeral_ports():
    """MoRI's bind(0) listeners must not acquire a later fixed handshake/notify port."""
    backend = VLLMProtocol(connector="moriio", vllm_config=VLLMServerConfig(decode={"tensor-parallel-size": 4}))
    endpoint = Endpoint(mode="decode", index=0, nodes=("node0",), gpu_indices=frozenset({0, 1, 2, 3}))
    process = backend.endpoints_to_processes([endpoint], frontend_type="vllm-router")[0]

    # Default Linux ephemeral range, also observed on the failing MI355X run.
    for base in (process.moriio_handshake_port, process.moriio_notify_port):
        assert base is not None
        assert base + 3 < 32768 or base > 60999


def test_role_override_selects_the_discovery_connector_per_mode():
    """roles.decode.args.connector overrides engine.connector for that role only."""
    backend = VLLMProtocol(connector="nixl", vllm_config=VLLMServerConfig(decode={"connector": "moriio"}))

    assert backend.connector_for_mode("prefill") == "nixl"
    assert backend.connector_for_mode("decode") == "moriio"
    assert backend.discovers_workers() is True
    row = backend.kv_connector_for_mode("decode")
    assert row is not None and row.discovery is True
