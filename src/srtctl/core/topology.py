# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Endpoint and Process dataclasses for worker topology.

This module replaces the bash array math in Jinja templates with typed Python:

Before (bash):
    for i in $(seq 0 $((PREFILL_WORKERS - 1))); do
        leader_idx=$((WORKER_NODE_OFFSET + i * PREFILL_NODES_PER_WORKER))
        prefill_leaders[$i]=$leader_idx
    done

After (Python):
    endpoints = allocate_endpoints(config, nodes)
    for endpoint in endpoints:
        print(f"{endpoint.mode} worker {endpoint.index} on {endpoint.nodes}")
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

# Worker mode type
WorkerMode = Literal["prefill", "decode", "agg"]


@dataclass
class NodePortAllocator:
    """Allocates unique ports per node to avoid conflicts.

    When multiple workers share a node (e.g., 2 decode workers with 4 GPUs each
    on an 8-GPU node), they need unique ports. This allocator tracks port
    assignments per node and hands out the next available port.

    Port ranges (non-overlapping):
        - kv_events_port: 5550+  (global) - ZMQ port for kv-events publishing
        - nixl_port:      6550+  (global) - NIXL side channel for KV transfers (vLLM)
        - dp_rpc_port:    13345+ (per node) - DP coordination port (vLLM data-parallel)
        - http_port:      30000+ (per node) - HTTP serving port
        - bootstrap_port: 31000+ (per node) - P/D coordination port (prefill only)

    Example:
        allocator = NodePortAllocator()

        # Two workers on same node get different ports
        port1 = allocator.next_http_port("node0")  # 30000
        port2 = allocator.next_http_port("node0")  # 30001

        # Different node starts fresh
        port3 = allocator.next_http_port("node1")  # 30000
    """

    base_http_port: int = 30000
    base_bootstrap_port: int = 31000
    base_kv_events_port: int = 5550
    base_nixl_port: int = 6550  # NIXL side channel ports (must not overlap with kv_events)
    base_dp_rpc_port: int = 13345  # DP coordination port for vLLM data-parallel

    _http_ports: dict[str, int] = field(default_factory=dict, repr=False)
    _bootstrap_ports: dict[str, int] = field(default_factory=dict, repr=False)
    _dp_rpc_ports: dict[str, int] = field(default_factory=dict, repr=False)
    _next_kv_events_port: int = field(default=0, repr=False)  # Global counter
    _next_nixl_port: int = field(default=0, repr=False)  # Global counter for NIXL

    def next_http_port(self, node: str) -> int:
        """Get next available HTTP port for a node."""
        if node not in self._http_ports:
            self._http_ports[node] = self.base_http_port
        port = self._http_ports[node]
        self._http_ports[node] += 1000
        return port

    def next_bootstrap_port(self, node: str) -> int:
        """Get next available bootstrap port for a node (prefill only)."""
        if node not in self._bootstrap_ports:
            self._bootstrap_ports[node] = self.base_bootstrap_port
        port = self._bootstrap_ports[node]
        self._bootstrap_ports[node] += 1
        return port

    def next_kv_events_port(self) -> int:
        """Get next available kv-events ZMQ port (globally unique across all nodes)."""
        if self._next_kv_events_port == 0:
            self._next_kv_events_port = self.base_kv_events_port
        port = self._next_kv_events_port
        self._next_kv_events_port += 1
        return port

    def next_nixl_port(self) -> int:
        """Get next available NIXL side channel port (globally unique across all nodes)."""
        if self._next_nixl_port == 0:
            self._next_nixl_port = self.base_nixl_port
        port = self._next_nixl_port
        self._next_nixl_port += 1
        return port

    def next_dp_rpc_port(self, node: str) -> int:
        """Get next available DP RPC port for a node.

        When multiple DP endpoints share a node, each needs a unique
        data-parallel-rpc-port to avoid bind collisions.
        """
        if node not in self._dp_rpc_ports:
            self._dp_rpc_ports[node] = self.base_dp_rpc_port
        port = self._dp_rpc_ports[node]
        self._dp_rpc_ports[node] += 1
        return port


@dataclass(frozen=True)
class Endpoint:
    """A logical worker endpoint (serving unit).

    An endpoint represents one logical worker that may span multiple nodes.
    For example, a prefill worker with TP=16 on a cluster with 8 GPUs/node
    would span 2 nodes.

    Attributes:
        mode: Worker mode ("prefill", "decode", or "agg")
        index: Zero-based index within the mode (e.g., prefill worker 0, 1, 2)
        nodes: Tuple of node hostnames this endpoint uses
        gpu_indices: Set of GPU indices used on each node (e.g., {0,1,2,3,4,5,6,7})
        gpus_per_node: Number of GPUs per node in the cluster
    """

    mode: WorkerMode
    index: int
    nodes: tuple[str, ...]
    gpu_indices: frozenset[int] = field(default_factory=lambda: frozenset(range(8)))
    gpus_per_node: int = 8

    @property
    def leader_node(self) -> str:
        """The first node in the endpoint (used for distributed init)."""
        return self.nodes[0]

    @property
    def num_nodes(self) -> int:
        """Number of nodes this endpoint spans."""
        return len(self.nodes)

    @property
    def total_gpus(self) -> int:
        """Total GPUs used by this endpoint across all nodes."""
        return self.num_nodes * len(self.gpu_indices)

    @property
    def is_multi_node(self) -> bool:
        """Whether this endpoint spans multiple nodes."""
        return self.num_nodes > 1


@dataclass(frozen=True)
class Process:
    """A physical process within an endpoint.

    For most backends, there's one Process per node within an Endpoint.
    This dataclass holds the per-process configuration needed for srun.

    Attributes:
        node: The node hostname this process runs on
        gpu_indices: GPU indices visible to this process
        sys_port: DYN_SYSTEM_PORT for this process
        http_port: HTTP serving port for this process (avoids conflicts on same node)
        bootstrap_port: P/D coordination port (only for prefill leaders)
        kv_events_port: ZMQ port for kv-events publishing (all worker leaders)
        nixl_port: NIXL side channel port for KV transfers (vLLM only)
        endpoint_mode: The mode of the parent endpoint
        endpoint_index: The index of the parent endpoint
        node_rank: Rank within the endpoint (0 for leader)
    """

    node: str
    gpu_indices: frozenset[int]
    sys_port: int
    http_port: int
    endpoint_mode: WorkerMode
    endpoint_index: int
    node_rank: int = 0
    bootstrap_port: int | None = None
    kv_events_port: int | None = None
    nixl_port: int | None = None
    dp_rpc_port: int | None = None

    @property
    def is_leader(self) -> bool:
        """Whether this is the leader process for the endpoint."""
        return self.node_rank == 0

    @property
    def cuda_visible_devices(self) -> str:
        """CUDA_VISIBLE_DEVICES string for this process."""
        return ",".join(str(i) for i in sorted(self.gpu_indices))


def allocate_endpoints(
    num_prefill: int,
    num_decode: int,
    num_agg: int,
    gpus_per_prefill: int,
    gpus_per_decode: int,
    gpus_per_agg: int,
    gpus_per_node: int,
    available_nodes: Sequence[str],
) -> list[Endpoint]:
    """Allocate endpoints to nodes based on GPU requirements.

    This is the core allocation logic that replaces bash array math.

    Args:
        num_prefill: Number of prefill workers
        num_decode: Number of decode workers
        num_agg: Number of aggregated workers
        gpus_per_prefill: GPUs per prefill worker
        gpus_per_decode: GPUs per decode worker
        gpus_per_agg: GPUs per agg worker
        gpus_per_node: GPUs available per node
        available_nodes: List of available node hostnames

    Returns:
        List of Endpoint objects with node assignments

    Example:
        # 2 prefill workers with 8 GPUs each, 4 decode workers with 4 GPUs each
        # on 4 nodes with 8 GPUs/node
        endpoints = allocate_endpoints(
            num_prefill=2, num_decode=4, num_agg=0,
            gpus_per_prefill=8, gpus_per_decode=4, gpus_per_agg=0,
            gpus_per_node=8, available_nodes=["node1", "node2", "node3", "node4"]
        )
        # Results:
        # - prefill_0 on node1 (8 GPUs)
        # - prefill_1 on node2 (8 GPUs)
        # - decode_0 on node3 (GPUs 0-3)
        # - decode_1 on node3 (GPUs 4-7)
        # - decode_2 on node4 (GPUs 0-3)
        # - decode_3 on node4 (GPUs 4-7)
    """
    endpoints: list[Endpoint] = []
    node_idx = 0
    gpu_offset = 0  # Track GPU offset within current node

    def allocate_worker(mode: WorkerMode, index: int, gpus_needed: int) -> Endpoint:
        """Allocate a single worker endpoint."""
        nonlocal node_idx, gpu_offset

        if gpus_needed <= 0:
            raise ValueError(f"gpus_needed must be positive, got {gpus_needed}")

        # Calculate how many nodes this worker spans
        nodes_per_worker = (gpus_needed + gpus_per_node - 1) // gpus_per_node

        # For multi-node workers, start fresh on node boundary
        if (nodes_per_worker > 1 or gpus_needed == gpus_per_node) and gpu_offset > 0:
            node_idx += 1
            gpu_offset = 0

        # Collect nodes for this worker
        worker_nodes = []
        for _ in range(nodes_per_worker):
            if node_idx >= len(available_nodes):
                raise ValueError(f"Not enough nodes: need node {node_idx}, but only {len(available_nodes)} available")
            worker_nodes.append(available_nodes[node_idx])
            node_idx += 1

        # Determine GPU indices (full node for multi-node, or specific range for single)
        if nodes_per_worker > 1:
            gpu_indices = frozenset(range(gpus_per_node))
            gpu_offset = 0
        else:
            # Single node: might be partial or full
            if gpu_offset + gpus_needed > gpus_per_node:
                # Doesn't fit, move to next node
                node_idx += 1
                if node_idx > len(available_nodes):
                    raise ValueError("Not enough nodes for GPU allocation")
                worker_nodes = [available_nodes[node_idx - 1]]
                gpu_offset = 0

            gpu_indices = frozenset(range(gpu_offset, gpu_offset + gpus_needed))
            gpu_offset += gpus_needed

            # If we filled the node, move to next
            if gpu_offset >= gpus_per_node:
                node_idx += 1
                gpu_offset = 0
            else:
                # Still on same node, rewind node_idx for next partial worker
                node_idx -= len(worker_nodes) - 1 if len(worker_nodes) > 1 else 0

        # Fix: for single-node workers staying on same node
        if nodes_per_worker == 1 and gpu_offset > 0 and gpu_offset < gpus_per_node:
            # We're still on the same node, don't increment
            pass
        elif nodes_per_worker == 1 and gpu_offset == 0:
            # We moved to a new node after filling previous
            pass

        return Endpoint(
            mode=mode,
            index=index,
            nodes=tuple(worker_nodes),
            gpu_indices=gpu_indices,
            gpus_per_node=gpus_per_node,
        )

    # Reset for cleaner allocation
    node_idx = 0
    gpu_offset = 0

    # Simpler allocation: each worker gets nodes sequentially
    def allocate_workers_simple(mode: WorkerMode, count: int, gpus_per_worker: int) -> list[Endpoint]:
        nonlocal node_idx, gpu_offset
        result = []

        nodes_per_worker = (gpus_per_worker + gpus_per_node - 1) // gpus_per_node

        for i in range(count):
            if nodes_per_worker >= 1 and gpus_per_worker >= gpus_per_node:
                # Multi-node or full-node worker
                worker_nodes = tuple(available_nodes[node_idx + j] for j in range(nodes_per_worker))
                node_idx += nodes_per_worker

                result.append(
                    Endpoint(
                        mode=mode,
                        index=i,
                        nodes=worker_nodes,
                        gpu_indices=frozenset(range(gpus_per_node)),
                        gpus_per_node=gpus_per_node,
                    )
                )
            else:
                # Partial node worker - pack multiple on same node
                if gpu_offset + gpus_per_worker > gpus_per_node:
                    node_idx += 1
                    gpu_offset = 0

                worker_node = available_nodes[node_idx]
                gpu_indices = frozenset(range(gpu_offset, gpu_offset + gpus_per_worker))
                gpu_offset += gpus_per_worker

                if gpu_offset >= gpus_per_node:
                    node_idx += 1
                    gpu_offset = 0

                result.append(
                    Endpoint(
                        mode=mode,
                        index=i,
                        nodes=(worker_node,),
                        gpu_indices=gpu_indices,
                        gpus_per_node=gpus_per_node,
                    )
                )

        return result

    # Allocate in order: prefill, decode, agg
    if num_prefill > 0:
        endpoints.extend(allocate_workers_simple("prefill", num_prefill, gpus_per_prefill))

    # When there's a partial allocation on the current node (gpu_offset > 0) and
    # there are more nodes available, advance to ensure prefill and decode don't
    # share a node. This prevents the bug where a multi-node decode worker overlaps
    # with a partial-node prefill worker.
    # When there are no more nodes (decode_nodes=0 config), allow sharing.
    if num_decode > 0:
        if gpu_offset > 0 and (node_idx + 1) < len(available_nodes):
            node_idx += 1
            gpu_offset = 0
        endpoints.extend(allocate_workers_simple("decode", num_decode, gpus_per_decode))

    if num_agg > 0:
        endpoints.extend(allocate_workers_simple("agg", num_agg, gpus_per_agg))

    return endpoints


def endpoints_to_processes(
    endpoints: list[Endpoint],
    base_sys_port: int = 8081,
    port_allocator: NodePortAllocator | None = None,
) -> list[Process]:
    """Convert endpoints to physical processes.

    For SGLang, we create one process per node in each endpoint.
    Each process gets a unique DYN_SYSTEM_PORT and ports from the allocator.

    Ports are assigned per-node to avoid conflicts when multiple workers
    share a node (e.g., 2 decode workers with 4 GPUs each on an 8-GPU node).

    Args:
        endpoints: List of Endpoint objects
        base_sys_port: Starting port for DYN_SYSTEM_PORT assignment
        port_allocator: NodePortAllocator for HTTP/bootstrap ports (created if None)

    Returns:
        List of Process objects
    """
    processes: list[Process] = []
    current_sys_port = base_sys_port

    if port_allocator is None:
        port_allocator = NodePortAllocator()

    for endpoint in endpoints:
        # Allocate bootstrap port once per prefill endpoint (shared by all processes)
        leader_node = endpoint.nodes[0]
        endpoint_bootstrap_port = (
            port_allocator.next_bootstrap_port(leader_node) if endpoint.mode == "prefill" else None
        )

        for node_rank, node in enumerate(endpoint.nodes):
            is_leader = node_rank == 0

            # Only leaders need http_port (for router to connect to)
            http_port = port_allocator.next_http_port(node) if is_leader else 0

            # Allocate kv_events port for each node in the endpoint (globally unique)
            # Each node publishes KV events independently
            node_kv_events_port = port_allocator.next_kv_events_port()

            # Allocate NIXL side channel port (globally unique, used by vLLM)
            node_nixl_port = port_allocator.next_nixl_port()

            processes.append(
                Process(
                    node=node,
                    gpu_indices=endpoint.gpu_indices,
                    sys_port=current_sys_port,
                    http_port=http_port,
                    endpoint_mode=endpoint.mode,
                    endpoint_index=endpoint.index,
                    node_rank=node_rank,
                    bootstrap_port=endpoint_bootstrap_port,
                    kv_events_port=node_kv_events_port,
                    nixl_port=node_nixl_port,
                )
            )
            current_sys_port += 1

    return processes
