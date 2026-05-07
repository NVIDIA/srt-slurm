# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for endpoint allocation logic."""

import pytest

from srtctl.core.topology import (
    NodePortAllocator,
    allocate_endpoints,
    compute_port_jitter,
    endpoints_to_processes,
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
        assert leader.http_port == 30000
        assert leader.bootstrap_port == 31000  # prefill gets bootstrap port

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

        # Ports should be sequential starting from 5550
        # With 2 prefill + 2 decode workers, each on single node = 4 processes = 4 ports
        assert sorted(kv_ports) == [5550, 5551, 5552, 5553]

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
        assert processes[0].kv_events_port == 5550
        assert processes[1].kv_events_port == 5551

    def test_nixl_port_allocation(self):
        """Test NIXL ports are allocated globally unique starting at 6550."""
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
        assert min(nixl_ports) == 6550  # Starts at base
        assert nixl_ports == [6550, 6551]  # Sequential


class TestPortJitter:
    """Tests for per-job port jitter that WARs the SGLang ZMQ rpc_port collision."""

    def test_compute_port_jitter_deterministic(self):
        # Same job id always returns the same offset.
        assert compute_port_jitter("144826") == compute_port_jitter("144826")
        # Mapping: ((id % 5000) * 2) + 1
        assert compute_port_jitter("12345") == ((12345 % 5000) * 2) + 1

    def test_compute_port_jitter_non_numeric_falls_back(self):
        assert compute_port_jitter("not-a-number") == 0
        assert compute_port_jitter("") == 0
        assert compute_port_jitter(None) == 0

    def test_compute_port_jitter_always_odd(self):
        """Critical: jitter must be odd so rpc_port (port+236) stays odd.

        Linux ``__inet_hash_connect`` allocates ports for outbound connect()
        in two passes — even ports first, odd ports only after the entire
        even pool is exhausted. NIXL/UCX outbound connections during prefill
        engine init grab even ports preferentially. If our planned rpc_port
        is even AND inside the kernel ephemeral range
        (/proc/sys/net/ipv4/ip_local_port_range, default 32768-60999), the
        kernel snatches it before SGLang can bind.
        """
        for jid in ("0", "1", "2", "144826", "144827", "145866", "145867", "9999", "10000", "99999"):
            j = compute_port_jitter(jid)
            assert j % 2 == 1, f"jitter for job {jid} must be odd, got {j}"

    def test_compute_port_jitter_range(self):
        for jid in ("0", "9999", "10000", "10001", "144826"):
            j = compute_port_jitter(jid)
            # Odd values in [1, 9999].
            assert 1 <= j <= 9999

    def test_from_job_id_shifts_bases(self):
        a = NodePortAllocator.from_job_id("12345")
        # 12345 % 5000 = 2345 -> jitter = 2345*2 + 1 = 4691
        assert a.base_http_port == 30000 + 4691
        assert a.base_bootstrap_port == 31000 + 4691
        # kv_events / nixl unaffected (they're already globally unique).
        assert a.base_kv_events_port == 5550
        assert a.base_nixl_port == 6550

    def test_from_job_id_none_means_no_jitter(self):
        a = NodePortAllocator.from_job_id(None)
        assert a.base_http_port == 30000
        assert a.base_bootstrap_port == 31000

    def test_from_job_id_rpc_port_is_odd(self):
        """rpc_port (= http_port + 236) must be odd to avoid the kernel
        even-port-first connect() pool — see test_compute_port_jitter_always_odd."""
        # Sample the originally-failing run + the second-wave failing runs.
        for jid in ("144826", "144827", "145866", "145867", "145868", "145872", "145874", "145876"):
            a = NodePortAllocator.from_job_id(jid)
            assert (a.base_http_port + 236) % 2 == 1, f"rpc_port for {jid} must be odd"

    def test_endpoints_to_processes_with_jittered_allocator(self):
        """Jittered allocator shifts http_port + bootstrap_port; sys_port unchanged."""
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

        # job_id 144826 -> jitter = ((144826 % 5000) * 2) + 1 = 9653
        allocator = NodePortAllocator.from_job_id("144826")
        expected_jitter = 9653
        processes = endpoints_to_processes(endpoints, base_sys_port=8081, port_allocator=allocator)

        leaders = [p for p in processes if p.is_leader]
        assert len(leaders) == 2
        for leader in leaders:
            assert leader.http_port >= 30000 + expected_jitter
        # SGLang's ZMQ rpc_port = http_port + 236; verify it's no longer 30236
        # AND that it's odd.
        prefill = [p for p in processes if p.endpoint_mode == "prefill"][0]
        assert prefill.http_port + 236 != 30236
        assert (prefill.http_port + 236) % 2 == 1
        assert prefill.bootstrap_port == 31000 + expected_jitter

    def test_different_job_ids_produce_different_ports(self):
        """The originally-failing 8 jobs land on 8 distinct rpc_ports."""
        failing_job_ids = ["144826", "144827", "144828", "144829", "144831", "144832", "144835", "144836"]
        rpc_ports = set()
        for jid in failing_job_ids:
            allocator = NodePortAllocator.from_job_id(jid)
            rpc_ports.add(allocator.base_http_port + 236)
        assert len(rpc_ports) == 8

    def test_second_wave_failures_now_get_odd_ports(self):
        """The 145866-145876 wave failed because they had even rpc_ports.

        Under the new odd-jitter formula they all map to odd rpc_ports — the
        property that should keep them out of the kernel's even-first
        connect() pool.
        """
        second_wave = ["145866", "145868", "145872", "145874", "145876"]
        for jid in second_wave:
            allocator = NodePortAllocator.from_job_id(jid)
            rpc_port = allocator.base_http_port + 236
            assert rpc_port % 2 == 1, f"job {jid} must now have odd rpc_port, got {rpc_port}"
