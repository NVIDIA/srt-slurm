# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-process VLLM_PORT assignment (rendezvous EADDRINUSE avoidance)."""

from srtctl.backends.vllm import VLLMProtocol
from srtctl.core.topology import Process
from srtctl.ports import DYN_SYSTEM_PORT_BASE, VLLM_PORT_BASE, VLLM_PORT_STRIDE


def _process(sys_port: int) -> Process:
    """A minimal vLLM worker process (nixl_port=None so no host lookup runs)."""
    return Process(
        node="node0",
        gpu_indices=frozenset({0}),
        sys_port=sys_port,
        http_port=0,
        endpoint_mode="decode",
        endpoint_index=0,
    )


def test_vllm_port_is_unique_per_process_with_stride():
    """Co-located workers get distinct VLLM_PORT bases spaced by the full stride."""
    backend = VLLMProtocol()

    envs = [backend.get_process_environment(_process(DYN_SYSTEM_PORT_BASE + i)) for i in range(3)]
    ports = [int(env["VLLM_PORT"]) for env in envs]

    assert ports == [
        VLLM_PORT_BASE,
        VLLM_PORT_BASE + VLLM_PORT_STRIDE,
        VLLM_PORT_BASE + 2 * VLLM_PORT_STRIDE,
    ]
    # Distinct and a full stride apart, so per-process get_open_port() scan
    # ranges cannot overlap.
    assert len(set(ports)) == len(ports)
    assert all(ports[i + 1] - ports[i] == VLLM_PORT_STRIDE for i in range(len(ports) - 1))


def test_vllm_port_clamps_when_sys_port_below_anchor():
    """A sys_port below the anchor must not produce a negative offset."""
    backend = VLLMProtocol()

    env = backend.get_process_environment(_process(DYN_SYSTEM_PORT_BASE - 10))

    assert env["VLLM_PORT"] == str(VLLM_PORT_BASE)
