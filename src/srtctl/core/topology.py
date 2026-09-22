# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from srtctl.ports import (
    BOOTSTRAP_PORTS,
    DYN_SYSTEM_PORT_BASE,
    HTTP_PORTS,
    KV_EVENTS_PORTS,
    KVBM_ZMQ_PORTS,
    NIXL_PORTS,
    SIDECAR_GRPC_PORTS,
    SYS_PORTS,
    PortKind,
)

# Worker mode type
WorkerMode = Literal["prefill", "decode", "agg"]


@dataclass
class NodePortAllocator:
    """Hands out every port a worker process binds, one counter per port kind.

    A ``PortKind`` (``srtctl.ports``) says where its range starts, how far apart
    consecutive allocations sit, and whether its counter is per node (the port is
    bound on that node only: HTTP, bootstrap, DP RPC, dist-init) or global (a side
    channel that peers on other nodes address: system, KV events, NIXL, NCCL).
    Two workers sharing a node therefore never collide, and global kinds never
    repeat across the job.

    Allocation happens once, in ``endpoints_to_processes``, and the results ride
    on ``Process``; command builders and stages read those fields and never
    derive one port from another. ``bases`` overrides a kind's first port by
    kind name: the system-status base for tests, the Dynamo sidecar gRPC base
    from ``dynamo.sidecar_port``.

    Example:
        allocator = NodePortAllocator()
        allocator.next(HTTP_PORTS, "node0")  # 6100
        allocator.next(HTTP_PORTS, "node0")  # 6132: a second worker on the node
        allocator.next(HTTP_PORTS, "node1")  # 6100: per node, so node1 starts over
        allocator.next(NIXL_PORTS, size=4)   # 5400, and the next NIXL allocation is 5404
    """

    bases: dict[str, int] = field(default_factory=dict)
    _next: dict[tuple[str, str | None], int] = field(default_factory=dict, repr=False)

    def next(self, kind: PortKind, node: str | None = None, size: int = 1) -> int:
        """Reserve ``size`` consecutive slots of ``kind`` and return the first port.

        ``size > 1`` is for engines that add a rank offset to the port they are
        given (vLLM adds ``data_parallel_rank`` to its NIXL side-channel port and
        opens one KV-event publisher per local DP rank), so the next allocation
        starts past the whole block.
        """
        if size < 1:
            raise ValueError(f"{kind.name}: size must be at least 1, got {size}")
        if kind.per_node and node is None:
            raise ValueError(f"{kind.name} ports are allocated per node; pass the node")
        key = (kind.name, node if kind.per_node else None)
        ordinal = self._next.get(key, 0)
        self._next[key] = ordinal + size
        port = self.bases.get(kind.name, kind.base) + ordinal * kind.stride
        last = port + (size - 1) * kind.stride
        if last > 65535:
            raise ValueError(f"{kind.name} port range exhausted at {port}..{last}")
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
    # SLURM heterogeneous-job component index (0=prefill side, 1=decode side).
    # None when the job is non-het — callers that pass this to srun treat None
    # as "omit --het-group".
    het_group: int | None = None

    # Optional per-node allocation for workers spanning partially occupied nodes.
    node_gpu_indices: tuple[frozenset[int], ...] = ()

    def gpus_on_node(self, node_rank: int) -> frozenset[int]:
        return self.node_gpu_indices[node_rank] if self.node_gpu_indices else self.gpu_indices

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
        return sum(len(self.gpus_on_node(i)) for i in range(self.num_nodes))

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
        engine_id: Which engine of the worker this is. 0 is the one every job has;
            under ``backend.failover`` (vLLM shadow engine recovery) engines 1.. are
            the standbys, sharing node, GPUs and node_rank with engine 0 but with
            their own ports and their own srun step.
        kvbm_zmq_port: KVBM leader ZMQ pub port (ack is the next port); the leader's is used
        sidecar_grpc_port: Dynamo sidecar gRPC listener, allocated when the job runs sidecars
        nccl_port: SGLang local TP rendezvous port, one per server process
        dist_init_port: SGLang multi-node dist-init port; the same value on every process of an endpoint
        vllm_scan_port: first port of this vLLM process's private ``get_open_port()`` scan range
        trtllm_dist_init_port: TRT-LLM torch.distributed bootstrap port; the leader's is used

    Every port is allocated by ``NodePortAllocator`` in ``endpoints_to_processes``;
    consumers read these fields and never derive one port from another.
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
    # Inherited from the parent Endpoint when the job is heterogeneous.
    het_group: int | None = None
    engine_id: int = 0
    kvbm_zmq_port: int | None = None
    sidecar_grpc_port: int | None = None
    nccl_port: int | None = None
    dist_init_port: int | None = None
    vllm_scan_port: int | None = None
    trtllm_dist_init_port: int | None = None

    @property
    def is_leader(self) -> bool:
        """Whether this is the leader process for the endpoint."""
        return self.node_rank == 0

    @property
    def engine_suffix(self) -> str:
        """Step-name and log-name suffix that tells a shadow engine apart from engine 0 (``""`` for it)."""
        return f"_e{self.engine_id}" if self.engine_id else ""

    @property
    def cuda_visible_devices(self) -> str:
        """CUDA_VISIBLE_DEVICES string for this process."""
        return ",".join(str(i) for i in sorted(self.gpu_indices))


def ordered_decode_leader_nodes(processes: list[Process]) -> list[str]:
    """Distinct decode (GEN) worker-leader node hostnames, ordered by worker index.

    Reads endpoint_mode from the physical process topology, so it works for both
    heterogeneous and monolithic jobs (unlike Nodes.decode_group, which is only
    populated for het jobs).
    """
    leaders = sorted(
        (p for p in processes if p.endpoint_mode == "decode" and p.is_leader),
        key=lambda p: p.endpoint_index,
    )
    nodes: list[str] = []
    for p in leaders:
        if p.node not in nodes:
            nodes.append(p.node)
    return nodes


def placed_node(processes: list[Process], placement: str, head: str, *, kind: str) -> str:
    """Resolve a node-placement spec to a concrete node hostname.

    placement:
        "head"         -> the head node (default; unchanged behavior)
        "first_decode" -> first decode/GEN worker-leader node
        "last_decode"  -> last decode/GEN worker-leader node
    ``kind`` is the config-field name, used only for error messages.
    """
    if placement == "head":
        return head
    if placement in ("first_decode", "last_decode"):
        decode = ordered_decode_leader_nodes(processes)
        if not decode:
            raise ValueError(f"{kind}={placement!r} but no decode workers were found")
        return decode[0] if placement == "first_decode" else decode[-1]
    raise ValueError(f"{kind}={placement!r} is invalid (expected head|first_decode|last_decode)")


def allocate_endpoints(
    num_prefill: int,
    num_decode: int,
    num_agg: int,
    gpus_per_prefill: int,
    gpus_per_decode: int,
    gpus_per_agg: int,
    gpus_per_node: int,
    available_nodes: Sequence[str],
    spread_workers: bool = False,
    allow_prefill_decode_colocation: bool = False,
    pack_multinode_workers: bool = False,
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
        spread_workers: If True, place each partial-node worker on its own
            node instead of packing multiple onto the same node. Requires the
            caller to reserve enough nodes (one per worker per mode).
        pack_multinode_workers: Preserve exact per-node GPU allocations for MPI
            workers, allowing adjacent workers to share a partially filled node.
        allow_prefill_decode_colocation: If True, decode workers may use
            remaining GPUs on a node already used by prefill workers.

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
            if pack_multinode_workers:
                if gpus_per_worker <= 0:
                    raise ValueError("GPUs per worker must be positive")
                if gpus_per_worker <= gpus_per_node and gpu_offset + gpus_per_worker > gpus_per_node:
                    node_idx += 1
                    gpu_offset = 0
                worker_nodes = []
                node_gpus = []
                remaining = gpus_per_worker
                while remaining:
                    if node_idx >= len(available_nodes):
                        raise ValueError("Not enough nodes for GPU allocation")
                    take = min(remaining, gpus_per_node - gpu_offset)
                    worker_nodes.append(available_nodes[node_idx])
                    node_gpus.append(frozenset(range(gpu_offset, gpu_offset + take)))
                    remaining -= take
                    gpu_offset += take
                    if gpu_offset == gpus_per_node:
                        node_idx += 1
                        gpu_offset = 0
                # MPI model mappings expect full nodes before a partial tail.
                allocations = sorted(zip(worker_nodes, node_gpus, strict=True), key=lambda item: -len(item[1]))
                worker_nodes, node_gpus = zip(*allocations, strict=True)
                result.append(
                    Endpoint(
                        mode=mode,
                        index=i,
                        nodes=tuple(worker_nodes),
                        gpu_indices=node_gpus[0],
                        node_gpu_indices=tuple(node_gpus),
                        gpus_per_node=gpus_per_node,
                    )
                )
                if spread_workers and gpu_offset:
                    node_idx += 1
                    gpu_offset = 0
            elif nodes_per_worker >= 1 and gpus_per_worker >= gpus_per_node:
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

                if gpu_offset >= gpus_per_node or spread_workers:
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

    # By default, when there's a partial allocation on the current node
    # (gpu_offset > 0) and there are more nodes available, advance to ensure
    # prefill and decode don't share a node. This prevents the bug where a
    # multi-node decode worker overlaps with a partial-node prefill worker.
    # When there are no more nodes (decode_nodes=0 config), allow sharing.
    if num_decode > 0:
        if not allow_prefill_decode_colocation and gpu_offset > 0 and (node_idx + 1) < len(available_nodes):
            node_idx += 1
            gpu_offset = 0
        endpoints.extend(allocate_workers_simple("decode", num_decode, gpus_per_decode))

    if num_agg > 0:
        endpoints.extend(allocate_workers_simple("agg", num_agg, gpus_per_agg))

    return endpoints


def allocate_endpoints_het(
    *,
    num_prefill: int,
    gpus_per_prefill: int,
    prefill_nodes: Sequence[str],
    num_decode: int,
    gpus_per_decode: int,
    decode_nodes: Sequence[str],
    gpus_per_node: int,
    pack_multinode_workers: bool = False,
) -> list[Endpoint]:
    """Allocate endpoints for a SLURM heterogeneous job.

    Prefill workers come from ``prefill_nodes`` (het component 0); decode
    workers come from ``decode_nodes`` (het component 1). Side pools are
    independent — no gpu-offset bleed across sides — so SLURM places each side
    inside its own topology segment.

    Each returned Endpoint is tagged with ``het_group`` (0 for prefill, 1 for
    decode) for downstream srun ``--het-group=`` threading.

    Aggregated mode is unsupported under het and rejected at config validation.
    """
    prefill_eps = allocate_endpoints(
        num_prefill=num_prefill,
        num_decode=0,
        num_agg=0,
        gpus_per_prefill=gpus_per_prefill,
        gpus_per_decode=gpus_per_decode,
        gpus_per_agg=0,
        gpus_per_node=gpus_per_node,
        available_nodes=prefill_nodes,
        pack_multinode_workers=pack_multinode_workers,
    )
    decode_eps = allocate_endpoints(
        num_prefill=0,
        num_decode=num_decode,
        num_agg=0,
        gpus_per_prefill=gpus_per_prefill,
        gpus_per_decode=gpus_per_decode,
        gpus_per_agg=0,
        gpus_per_node=gpus_per_node,
        available_nodes=decode_nodes,
        pack_multinode_workers=pack_multinode_workers,
    )
    # Endpoint is frozen; re-emit with het_group set.
    tagged: list[Endpoint] = []
    for ep in prefill_eps:
        tagged.append(
            Endpoint(
                mode=ep.mode,
                index=ep.index,
                nodes=ep.nodes,
                gpu_indices=ep.gpu_indices,
                node_gpu_indices=ep.node_gpu_indices,
                gpus_per_node=ep.gpus_per_node,
                het_group=0,
            )
        )
    for ep in decode_eps:
        tagged.append(
            Endpoint(
                mode=ep.mode,
                index=ep.index,
                nodes=ep.nodes,
                gpu_indices=ep.gpu_indices,
                node_gpu_indices=ep.node_gpu_indices,
                gpus_per_node=ep.gpus_per_node,
                het_group=1,
            )
        )
    return tagged


def port_allocator_for(
    port_allocator: NodePortAllocator | None,
    base_sys_port: int = DYN_SYSTEM_PORT_BASE,
) -> NodePortAllocator:
    """The allocator a topology builder uses: the caller's, or a fresh one whose system base is ``base_sys_port``."""
    if port_allocator is not None:
        return port_allocator
    return NodePortAllocator(bases={SYS_PORTS.name: base_sys_port})


def endpoints_to_processes(
    endpoints: list[Endpoint],
    base_sys_port: int = DYN_SYSTEM_PORT_BASE,
    port_allocator: NodePortAllocator | None = None,
    engines_per_process: int = 1,
    sidecar_grpc: bool = False,
) -> list[Process]:
    """Convert endpoints to physical processes, one per node of each endpoint.

    Every port a process binds is allocated here through ``port_allocator``:
    the system-status port, the leader's HTTP port, the prefill bootstrap port,
    the KV-events publisher, the NIXL side channel, the KVBM ZMQ pair, and the
    sidecar gRPC listener when the job runs Dynamo sidecars. Backends add their
    engine-specific kinds on top (``Process.with_ports``).

    Args:
        endpoints: List of Endpoint objects
        base_sys_port: System-status base when no allocator is passed
        port_allocator: The job's allocator (created if None)
        engines_per_process: Engines launched per (endpoint, node). 1 is the usual
            layout; vLLM shadow engine recovery asks for ``1 + shadows``, and every
            engine of a node then gets its own Process (same GPUs and node_rank,
            distinct ports, ``engine_id`` 0..n-1), emitted engine 0 first.
        sidecar_grpc: Allocate a Dynamo sidecar gRPC port for every process.

    Returns:
        List of Process objects
    """
    if engines_per_process < 1:
        raise ValueError(f"engines_per_process must be at least 1, got {engines_per_process}")
    allocator = port_allocator_for(port_allocator, base_sys_port)
    processes: list[Process] = []

    for endpoint in endpoints:
        # Allocate bootstrap ports once per prefill endpoint (shared by all of an
        # engine's processes); each engine of a worker binds its own.
        leader_node = endpoint.nodes[0]
        endpoint_bootstrap_ports = [
            allocator.next(BOOTSTRAP_PORTS, leader_node) if endpoint.mode == "prefill" else None
            for _ in range(engines_per_process)
        ]

        for node_rank, node in enumerate(endpoint.nodes):
            is_leader = node_rank == 0

            for engine_id in range(engines_per_process):
                processes.append(
                    Process(
                        node=node,
                        gpu_indices=endpoint.gpus_on_node(node_rank),
                        sys_port=allocator.next(SYS_PORTS),
                        # Only leaders serve HTTP (for a router to connect to).
                        http_port=allocator.next(HTTP_PORTS, node) if is_leader else 0,
                        endpoint_mode=endpoint.mode,
                        endpoint_index=endpoint.index,
                        node_rank=node_rank,
                        bootstrap_port=endpoint_bootstrap_ports[engine_id],
                        # Every process publishes KV events and opens a NIXL side channel of its own.
                        kv_events_port=allocator.next(KV_EVENTS_PORTS),
                        nixl_port=allocator.next(NIXL_PORTS),
                        het_group=endpoint.het_group,
                        engine_id=engine_id,
                        kvbm_zmq_port=allocator.next(KVBM_ZMQ_PORTS),
                        sidecar_grpc_port=allocator.next(SIDECAR_GRPC_PORTS) if sidecar_grpc else None,
                    )
                )

    return processes
