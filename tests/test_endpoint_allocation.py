# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for endpoint allocation logic."""

import pytest

from srtctl.core.topology import (
    NodePortAllocator,
    allocate_endpoints,
    allocate_endpoints_het,
    endpoints_to_processes,
)
from srtctl.ports import (
    DYN_SYSTEM_PORT_BASE,
    KV_EVENTS_PORT_BASE,
    SGLANG_BOOTSTRAP_PORT_BASE,
    SGLANG_HTTP_PORT_BASE,
    SGLANG_HTTP_PORT_STRIDE,
    VLLM_NIXL_PORT_BASE,
)


class TestAllocateEndpoints:
    """Tests for allocate_endpoints function."""

    def test_multiple_endpoints_per_node(self):
        """Test multiple endpoints sharing a single node."""
        # 2 prefill endpoints, 2 GPUs each, 4 GPUs per node -> both on node0
        # 2 decode endpoints, 2 GPUs each, 4 GPUs per node -> both on node1
        endpoints = allocate_endpoints(
            num_prefill=2,
            num_decode=2,
            num_agg=0,
            gpus_per_prefill=2,
            gpus_per_decode=2,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0", "node1"),
        )

        assert len(endpoints) == 4

        # Check prefill endpoints - both should be on the SAME node
        prefill_eps = [e for e in endpoints if e.mode == "prefill"]
        assert len(prefill_eps) == 2
        assert prefill_eps[0].nodes[0] == prefill_eps[1].nodes[0] == "node0"
        assert prefill_eps[0].total_gpus == 2
        assert prefill_eps[1].total_gpus == 2
        # They should have different GPU indices
        assert prefill_eps[0].gpu_indices != prefill_eps[1].gpu_indices
        assert prefill_eps[0].gpu_indices == frozenset({0, 1})
        assert prefill_eps[1].gpu_indices == frozenset({2, 3})

        # Check decode endpoints - both should be on the SAME node (node1)
        decode_eps = [e for e in endpoints if e.mode == "decode"]
        assert len(decode_eps) == 2
        assert decode_eps[0].nodes[0] == decode_eps[1].nodes[0] == "node1"
        assert decode_eps[0].gpu_indices == frozenset({0, 1})
        assert decode_eps[1].gpu_indices == frozenset({2, 3})

    def test_full_node_endpoints(self):
        """Test endpoints that use full nodes."""
        endpoints = allocate_endpoints(
            num_prefill=2,
            num_decode=2,
            num_agg=0,
            gpus_per_prefill=4,
            gpus_per_decode=4,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0", "node1", "node2", "node3"),
        )

        assert len(endpoints) == 4

        # Each endpoint should use a full node
        for ep in endpoints:
            assert ep.total_gpus == 4
            assert len(ep.nodes) == 1

    def test_multi_node_endpoints(self):
        """Test endpoints that span multiple nodes."""
        # 1 prefill worker, 8 GPUs, 4 GPUs per node -> spans 2 nodes
        endpoints = allocate_endpoints(
            num_prefill=1,
            num_decode=1,
            num_agg=0,
            gpus_per_prefill=8,
            gpus_per_decode=8,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0", "node1", "node2", "node3"),
        )

        assert len(endpoints) == 2

        # Each endpoint should span 2 nodes
        for ep in endpoints:
            assert len(ep.nodes) == 2
            assert ep.total_gpus == 8

    def test_insufficient_gpus(self):
        """Test that we raise an error when there are insufficient GPUs."""
        # This should raise an IndexError when trying to access nodes that don't exist
        with pytest.raises((ValueError, IndexError)):
            allocate_endpoints(
                num_prefill=2,
                num_decode=2,
                num_agg=0,
                gpus_per_prefill=8,
                gpus_per_decode=8,
                gpus_per_agg=8,
                gpus_per_node=4,
                available_nodes=("node0", "node1"),  # Only 8 GPUs total, need 32
            )

    def test_single_endpoint_single_gpu(self):
        """Test edge case: single endpoint with single GPU."""
        endpoints = allocate_endpoints(
            num_prefill=1,
            num_decode=0,
            num_agg=0,
            gpus_per_prefill=1,
            gpus_per_decode=1,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0",),
        )

        assert len(endpoints) == 1
        assert endpoints[0].mode == "prefill"
        assert endpoints[0].total_gpus == 1

    def test_aggregated_mode(self):
        """Test aggregated mode (no disaggregation)."""
        endpoints = allocate_endpoints(
            num_prefill=0,
            num_decode=0,
            num_agg=2,
            gpus_per_prefill=4,
            gpus_per_decode=4,
            gpus_per_agg=4,
            gpus_per_node=4,
            available_nodes=("node0", "node1"),
        )

        assert len(endpoints) == 2
        for ep in endpoints:
            assert ep.mode == "agg"
            assert ep.total_gpus == 4

    def test_spread_workers_partial_node(self):
        """spread_workers=True forces each partial-node worker onto its own node."""
        endpoints = allocate_endpoints(
            num_prefill=1,
            num_decode=2,
            num_agg=0,
            gpus_per_prefill=1,
            gpus_per_decode=2,
            gpus_per_agg=0,
            gpus_per_node=4,
            available_nodes=("node0", "node1", "node2"),
            spread_workers=True,
        )

        decode_eps = [e for e in endpoints if e.mode == "decode"]
        assert len(decode_eps) == 2
        # Without spread_workers both decode workers would land on node1.
        assert decode_eps[0].nodes == ("node1",)
        assert decode_eps[1].nodes == ("node2",)
        assert decode_eps[0].gpu_indices == frozenset({0, 1})
        assert decode_eps[1].gpu_indices == frozenset({0, 1})

    def test_spread_workers_default_packs(self):
        """spread_workers=False (default) packs partial-node workers onto the same node."""
        endpoints = allocate_endpoints(
            num_prefill=0,
            num_decode=2,
            num_agg=0,
            gpus_per_prefill=0,
            gpus_per_decode=2,
            gpus_per_agg=0,
            gpus_per_node=4,
            available_nodes=("node0", "node1"),
        )

        decode_eps = [e for e in endpoints if e.mode == "decode"]
        assert decode_eps[0].nodes == ("node0",)
        assert decode_eps[1].nodes == ("node0",)

    def test_prefill_decode_never_share_node_partial_allocation(self):
        """Test that prefill and decode workers are never colocated on the same node.

        This tests the bug fix for the case where:
        - 3 prefill workers with 2 GPUs each on 4-GPU nodes
        - prefill 0: node0 (GPUs 0,1)
        - prefill 1: node0 (GPUs 2,3) - node full, advance
        - prefill 2: node1 (GPUs 0,1) - partial allocation

        Without the fix, decode workers would start at node1, sharing with prefill 2.
        With the fix, decode workers should start at node2.
        """
        endpoints = allocate_endpoints(
            num_prefill=3,
            num_decode=1,
            num_agg=0,
            gpus_per_prefill=2,
            gpus_per_decode=32,  # 8 nodes * 4 GPUs/node
            gpus_per_agg=0,
            gpus_per_node=4,
            available_nodes=tuple(f"node{i}" for i in range(11)),  # Plenty of nodes
        )

        # Get prefill and decode endpoints
        prefill_eps = [e for e in endpoints if e.mode == "prefill"]
        decode_eps = [e for e in endpoints if e.mode == "decode"]

        assert len(prefill_eps) == 3
        assert len(decode_eps) == 1

        # Collect all nodes used by prefill workers
        prefill_nodes = set()
        for ep in prefill_eps:
            prefill_nodes.update(ep.nodes)

        # Collect all nodes used by decode workers
        decode_nodes = set()
        for ep in decode_eps:
            decode_nodes.update(ep.nodes)

        # Critical assertion: prefill and decode nodes must not overlap
        overlap = prefill_nodes & decode_nodes
        assert len(overlap) == 0, f"Prefill and decode workers share nodes: {overlap}"

        # Verify expected allocation
        assert prefill_eps[0].nodes == ("node0",)
        assert prefill_eps[0].gpu_indices == frozenset({0, 1})
        assert prefill_eps[1].nodes == ("node0",)
        assert prefill_eps[1].gpu_indices == frozenset({2, 3})
        assert prefill_eps[2].nodes == ("node1",)
        assert prefill_eps[2].gpu_indices == frozenset({0, 1})

        # Decode should start at node2, not node1
        assert decode_eps[0].nodes[0] == "node2"

    def test_prefill_decode_never_share_node_single_partial_prefill(self):
        """Test prefill/decode separation when only one prefill worker uses partial node."""
        endpoints = allocate_endpoints(
            num_prefill=1,
            num_decode=1,
            num_agg=0,
            gpus_per_prefill=2,
            gpus_per_decode=4,
            gpus_per_agg=0,
            gpus_per_node=4,
            available_nodes=("node0", "node1"),
        )

        prefill_eps = [e for e in endpoints if e.mode == "prefill"]
        decode_eps = [e for e in endpoints if e.mode == "decode"]

        # Prefill on node0 (GPUs 0,1), decode should be on node1 (not sharing node0)
        assert prefill_eps[0].nodes == ("node0",)
        assert prefill_eps[0].gpu_indices == frozenset({0, 1})

        assert decode_eps[0].nodes == ("node1",)
        assert decode_eps[0].gpu_indices == frozenset({0, 1, 2, 3})

    def test_prefill_workers_per_node_packs_multiple_on_same_node(self):
        """prefill_workers_per_node=2 packs 2 prefill workers per node."""
        endpoints = allocate_endpoints(
            num_prefill=4,
            num_decode=0,
            num_agg=0,
            gpus_per_prefill=4,
            gpus_per_decode=0,
            gpus_per_agg=0,
            gpus_per_node=8,
            available_nodes=("node0", "node1"),
            prefill_nodes=2,
            prefill_workers_per_node=2,
        )

        prefill_eps = [e for e in endpoints if e.mode == "prefill"]
        assert len(prefill_eps) == 4
        assert prefill_eps[0].nodes == ("node0",)
        assert prefill_eps[0].gpu_indices == frozenset({0, 1, 2, 3})
        assert prefill_eps[1].nodes == ("node0",)
        assert prefill_eps[1].gpu_indices == frozenset({4, 5, 6, 7})
        assert prefill_eps[2].nodes == ("node1",)
        assert prefill_eps[2].gpu_indices == frozenset({0, 1, 2, 3})
        assert prefill_eps[3].nodes == ("node1",)
        assert prefill_eps[3].gpu_indices == frozenset({4, 5, 6, 7})

    def test_prefill_workers_per_node_validation_rejects_mismatch(self):
        """prefill_workers_per_node raises if required_nodes != prefill_nodes."""
        with pytest.raises(ValueError, match="prefill placement needs"):
            allocate_endpoints(
                num_prefill=4,
                num_decode=0,
                num_agg=0,
                gpus_per_prefill=4,
                gpus_per_decode=0,
                gpus_per_agg=0,
                gpus_per_node=8,
                available_nodes=tuple(f"node{i}" for i in range(4)),
                prefill_nodes=3,
                prefill_workers_per_node=2,
            )

    def test_spread_decode_workers_one_per_node(self):
        """spread_decode_workers=True places each partial-node decode worker on its own node."""
        endpoints = allocate_endpoints(
            num_prefill=0,
            num_decode=3,
            num_agg=0,
            gpus_per_prefill=0,
            gpus_per_decode=4,
            gpus_per_agg=0,
            gpus_per_node=8,
            available_nodes=("node0", "node1", "node2"),
            spread_decode_workers=True,
        )

        decode_eps = [e for e in endpoints if e.mode == "decode"]
        assert len(decode_eps) == 3
        assert decode_eps[0].nodes == ("node0",)
        assert decode_eps[0].gpu_indices == frozenset({0, 1, 2, 3})
        assert decode_eps[1].nodes == ("node1",)
        assert decode_eps[1].gpu_indices == frozenset({0, 1, 2, 3})
        assert decode_eps[2].nodes == ("node2",)
        assert decode_eps[2].gpu_indices == frozenset({0, 1, 2, 3})

    def test_spread_decode_workers_false_packs(self):
        """spread_decode_workers=False (default) packs decode workers onto the same node."""
        endpoints = allocate_endpoints(
            num_prefill=0,
            num_decode=2,
            num_agg=0,
            gpus_per_prefill=0,
            gpus_per_decode=4,
            gpus_per_agg=0,
            gpus_per_node=8,
            available_nodes=("node0",),
            spread_decode_workers=False,
        )

        decode_eps = [e for e in endpoints if e.mode == "decode"]
        assert len(decode_eps) == 2
        assert decode_eps[0].nodes == ("node0",)
        assert decode_eps[1].nodes == ("node0",)
        assert decode_eps[0].gpu_indices == frozenset({0, 1, 2, 3})
        assert decode_eps[1].gpu_indices == frozenset({4, 5, 6, 7})


class TestEndpointsToProcesses:
    """Tests for endpoints_to_processes function."""

    def test_process_construction(self):
        """Test that endpoints_to_processes creates correct process mappings."""
        endpoints = allocate_endpoints(
            num_prefill=1,
            num_decode=1,
            num_agg=0,
            gpus_per_prefill=2,
            gpus_per_decode=2,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0",),
        )

        processes = endpoints_to_processes(endpoints, base_sys_port=8081)

        # SGLang creates one process per node
        assert len(processes) == 2

        # Check all sys_ports are unique
        ports = [p.sys_port for p in processes]
        assert len(ports) == len(set(ports)), "All processes should have unique sys_ports"

        # Check http_ports are unique per node (both on node0, so should differ)
        http_ports = [p.http_port for p in processes]
        assert len(http_ports) == len(set(http_ports)), "Processes on same node should have unique http_ports"

    def test_multi_node_process_construction(self):
        """Test process construction for multi-node endpoints."""
        endpoints = allocate_endpoints(
            num_prefill=1,
            num_decode=0,
            num_agg=0,
            gpus_per_prefill=8,
            gpus_per_decode=4,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0", "node1"),
        )

        processes = endpoints_to_processes(endpoints, base_sys_port=8081)

        # Multi-node endpoint should create one process per node
        assert len(processes) == 2
        nodes = [p.node for p in processes]
        assert "node0" in nodes
        assert "node1" in nodes

        # Only leader gets http_port, child gets 0
        leader = [p for p in processes if p.is_leader][0]
        assert leader.http_port == SGLANG_HTTP_PORT_BASE
        assert leader.bootstrap_port == SGLANG_BOOTSTRAP_PORT_BASE  # prefill gets bootstrap port

        child = [p for p in processes if not p.is_leader][0]
        assert child.http_port == 0
        # All processes in prefill endpoint share the same bootstrap port
        assert child.bootstrap_port == leader.bootstrap_port

    def test_cuda_visible_devices(self):
        """Test that CUDA_VISIBLE_DEVICES is set correctly for each process."""
        endpoints = allocate_endpoints(
            num_prefill=2,
            num_decode=0,
            num_agg=0,
            gpus_per_prefill=2,
            gpus_per_decode=2,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0",),
        )

        processes = endpoints_to_processes(endpoints, base_sys_port=8081)

        # Each process should have correct GPU indices
        for p in processes:
            assert len(p.gpu_indices) == 2
            # Check cuda_visible_devices is formatted correctly
            assert "," in p.cuda_visible_devices or p.cuda_visible_devices.isdigit()

    def test_kv_events_port_allocation(self):
        """Test that kv_events_port is allocated for all nodes (not just leaders)."""
        endpoints = allocate_endpoints(
            num_prefill=2,
            num_decode=2,
            num_agg=0,
            gpus_per_prefill=2,
            gpus_per_decode=2,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0", "node1"),
        )

        processes = endpoints_to_processes(endpoints, base_sys_port=8081)

        # All processes should have kv_events_port (each node publishes independently)
        kv_ports = [p.kv_events_port for p in processes]
        assert all(port is not None for port in kv_ports), "All processes should have kv_events_port"
        assert len(kv_ports) == len(set(kv_ports)), "All kv_events_ports should be globally unique"

        # Ports should be sequential starting from the centralized default
        # With 2 prefill + 2 decode workers, each on single node = 4 processes = 4 ports
        assert sorted(kv_ports) == [
            KV_EVENTS_PORT_BASE,
            KV_EVENTS_PORT_BASE + 1,
            KV_EVENTS_PORT_BASE + 2,
            KV_EVENTS_PORT_BASE + 3,
        ]

    def test_kv_events_port_same_node_unique(self):
        """Test kv_events_port is unique even when workers share a node."""
        # 2 prefill workers on same node
        endpoints = allocate_endpoints(
            num_prefill=2,
            num_decode=0,
            num_agg=0,
            gpus_per_prefill=2,
            gpus_per_decode=2,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0",),
        )

        processes = endpoints_to_processes(endpoints, base_sys_port=8081)

        # Both on node0, both should have unique ports
        assert len(processes) == 2
        assert processes[0].node == processes[1].node == "node0"
        assert processes[0].kv_events_port != processes[1].kv_events_port
        assert processes[0].kv_events_port == KV_EVENTS_PORT_BASE
        assert processes[1].kv_events_port == KV_EVENTS_PORT_BASE + 1

    def test_nixl_port_allocation(self):
        """Test NIXL ports are allocated globally unique from the default."""
        from srtctl.core.topology import Endpoint

        endpoints = [
            Endpoint(
                mode="prefill",
                index=0,
                nodes=("node0",),
                gpu_indices=frozenset(range(8)),
                gpus_per_node=8,
            ),
            Endpoint(
                mode="decode",
                index=0,
                nodes=("node1",),
                gpu_indices=frozenset(range(8)),
                gpus_per_node=8,
            ),
        ]

        processes = endpoints_to_processes(endpoints)

        # Each process should have a unique NIXL port
        nixl_ports = [p.nixl_port for p in processes]
        assert all(port is not None for port in nixl_ports), "All processes should have nixl_port"
        assert len(nixl_ports) == len(set(nixl_ports))  # All unique
        assert min(nixl_ports) == VLLM_NIXL_PORT_BASE  # Starts at base
        assert nixl_ports == [VLLM_NIXL_PORT_BASE, VLLM_NIXL_PORT_BASE + 1]  # Sequential


class TestDefaultPorts:
    """Tests for centralized default port values."""

    def test_allocator_uses_centralized_defaults(self):
        allocator = NodePortAllocator()
        assert allocator.base_http_port == SGLANG_HTTP_PORT_BASE
        assert allocator.base_bootstrap_port == SGLANG_BOOTSTRAP_PORT_BASE
        assert allocator.base_kv_events_port == KV_EVENTS_PORT_BASE
        assert allocator.base_nixl_port == VLLM_NIXL_PORT_BASE

    def test_http_ports_use_centralized_stride(self):
        allocator = NodePortAllocator()
        assert allocator.next_http_port("node0") == SGLANG_HTTP_PORT_BASE
        assert allocator.next_http_port("node0") == SGLANG_HTTP_PORT_BASE + SGLANG_HTTP_PORT_STRIDE
        assert allocator.next_http_port("node1") == SGLANG_HTTP_PORT_BASE

    def test_endpoints_to_processes_uses_default_sys_port(self):
        endpoints = allocate_endpoints(
            num_prefill=1,
            num_decode=1,
            num_agg=0,
            gpus_per_prefill=2,
            gpus_per_decode=2,
            gpus_per_agg=8,
            gpus_per_node=4,
            available_nodes=("node0",),
        )

        processes = endpoints_to_processes(endpoints)

        assert [p.sys_port for p in processes] == [DYN_SYSTEM_PORT_BASE, DYN_SYSTEM_PORT_BASE + 1]


class TestAllocateEndpointsHet:
    """Per-side heterogeneous-job allocation."""

    def test_prefill_and_decode_isolated_to_own_pools(self):
        """Prefill workers only land on prefill_nodes; decode only on decode_nodes."""
        # Asymmetric: 12 prefill nodes + 10 decode nodes (the 48+40 case)
        prefill_nodes = tuple(f"p-{i:02d}" for i in range(12))
        decode_nodes = tuple(f"d-{i:02d}" for i in range(10))

        endpoints = allocate_endpoints_het(
            num_prefill=12,
            gpus_per_prefill=4,
            prefill_nodes=prefill_nodes,
            num_decode=10,
            gpus_per_decode=4,
            decode_nodes=decode_nodes,
            gpus_per_node=4,
        )

        prefill_eps = [e for e in endpoints if e.mode == "prefill"]
        decode_eps = [e for e in endpoints if e.mode == "decode"]
        assert len(prefill_eps) == 12
        assert len(decode_eps) == 10

        # Side isolation: no prefill endpoint lands on a decode node, and vice versa.
        for ep in prefill_eps:
            for node in ep.nodes:
                assert node in prefill_nodes, f"prefill ep on decode node {node}"
        for ep in decode_eps:
            for node in ep.nodes:
                assert node in decode_nodes, f"decode ep on prefill node {node}"

    def test_het_group_tagged_on_endpoints(self):
        prefill_nodes = ("p0", "p1")
        decode_nodes = ("d0", "d1")
        endpoints = allocate_endpoints_het(
            num_prefill=2,
            gpus_per_prefill=4,
            prefill_nodes=prefill_nodes,
            num_decode=2,
            gpus_per_decode=4,
            decode_nodes=decode_nodes,
            gpus_per_node=4,
        )
        for ep in endpoints:
            if ep.mode == "prefill":
                assert ep.het_group == 0
            elif ep.mode == "decode":
                assert ep.het_group == 1

    def test_het_group_propagates_to_processes(self):
        endpoints = allocate_endpoints_het(
            num_prefill=1,
            gpus_per_prefill=4,
            prefill_nodes=("p0",),
            num_decode=1,
            gpus_per_decode=4,
            decode_nodes=("d0",),
            gpus_per_node=4,
        )
        processes = endpoints_to_processes(endpoints)
        for proc in processes:
            if proc.endpoint_mode == "prefill":
                assert proc.het_group == 0
            elif proc.endpoint_mode == "decode":
                assert proc.het_group == 1

    def test_multi_node_prefill_worker_stays_in_prefill_pool(self):
        # Single prefill worker with TP8 (2 nodes) — confirm it pulls from prefill_nodes only.
        prefill_nodes = ("p0", "p1", "p2", "p3")
        decode_nodes = ("d0", "d1")
        endpoints = allocate_endpoints_het(
            num_prefill=1,
            gpus_per_prefill=8,
            prefill_nodes=prefill_nodes,
            num_decode=2,
            gpus_per_decode=4,
            decode_nodes=decode_nodes,
            gpus_per_node=4,
        )
        prefill_ep = next(e for e in endpoints if e.mode == "prefill")
        assert len(prefill_ep.nodes) == 2
        for node in prefill_ep.nodes:
            assert node in prefill_nodes
