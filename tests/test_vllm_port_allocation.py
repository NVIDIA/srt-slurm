# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-process VLLM_PORT assignment (rendezvous EADDRINUSE avoidance)."""

from srtctl.backends.vllm import VLLMProtocol
from srtctl.core.topology import Endpoint, NodePortAllocator, Process
from srtctl.ports import VLLM_PORT_BASE, VLLM_PORT_STRIDE


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
