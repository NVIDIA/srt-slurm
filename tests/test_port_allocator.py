# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every port a worker binds comes from NodePortAllocator, and no two listeners collide.

The unit tests pin the allocator's contract; the corpus test builds the process
topology of every topology example under ``examples/`` and asserts that no two
processes on a node share a port and that job-global kinds never repeat.
"""

from collections import Counter
from pathlib import Path

import pytest

from srtctl.core.schema import SrtConfig
from srtctl.core.topology import NodePortAllocator, Process
from srtctl.ports import (
    BOOTSTRAP_PORTS,
    DIST_INIT_PORTS,
    HTTP_PORTS,
    KV_EVENTS_PORTS,
    NCCL_PORTS,
    NIXL_PORTS,
    SIDECAR_GRPC_PORTS,
    SYS_PORTS,
    TRTLLM_DIST_INIT_PORTS,
    VLLM_SCAN_PORTS,
    PortKind,
)

TOPOLOGY_EXAMPLE_DIRS = ("examples/sglang", "examples/vllm", "examples/trtllm", "examples/mocker")


class TestNodePortAllocator:
    def test_per_node_kinds_restart_on_every_node(self):
        allocator = NodePortAllocator()
        assert allocator.next(HTTP_PORTS, "node0") == HTTP_PORTS.base
        assert allocator.next(HTTP_PORTS, "node0") == HTTP_PORTS.base + HTTP_PORTS.stride
        assert allocator.next(HTTP_PORTS, "node1") == HTTP_PORTS.base
        assert allocator.next(BOOTSTRAP_PORTS, "node1") == BOOTSTRAP_PORTS.base

    def test_global_kinds_never_repeat_across_nodes(self):
        allocator = NodePortAllocator()
        assert [allocator.next(SYS_PORTS) for _ in range(3)] == [SYS_PORTS.base + i for i in range(3)]
        assert allocator.next(VLLM_SCAN_PORTS) == VLLM_SCAN_PORTS.base
        assert allocator.next(VLLM_SCAN_PORTS) == VLLM_SCAN_PORTS.base + VLLM_SCAN_PORTS.stride

    def test_a_block_reserves_size_slots(self):
        allocator = NodePortAllocator()
        assert allocator.next(NIXL_PORTS, size=4) == NIXL_PORTS.base
        assert allocator.next(NIXL_PORTS) == NIXL_PORTS.base + 4
        assert allocator.next(KV_EVENTS_PORTS, size=2) == KV_EVENTS_PORTS.base
        assert allocator.next(KV_EVENTS_PORTS) == KV_EVENTS_PORTS.base + 2

    def test_per_node_kinds_need_a_node(self):
        with pytest.raises(ValueError, match="allocated per node"):
            NodePortAllocator().next(HTTP_PORTS)

    def test_bases_override_a_kind_by_name(self):
        allocator = NodePortAllocator(bases={SIDECAR_GRPC_PORTS.name: 60000, SYS_PORTS.name: 9000})
        assert allocator.next(SIDECAR_GRPC_PORTS) == 60000
        assert allocator.next(SYS_PORTS) == 9000
        assert allocator.next(NCCL_PORTS) == NCCL_PORTS.base

    def test_range_exhaustion_is_an_error(self):
        allocator = NodePortAllocator(bases={"tiny": 65535})
        tiny = PortKind("tiny", 65535)
        assert allocator.next(tiny) == 65535
        with pytest.raises(ValueError, match="exhausted"):
            allocator.next(tiny)

    def test_every_kind_has_its_own_range(self):
        kinds = [
            SYS_PORTS,
            HTTP_PORTS,
            BOOTSTRAP_PORTS,
            KV_EVENTS_PORTS,
            NIXL_PORTS,
            NCCL_PORTS,
            DIST_INIT_PORTS,
            VLLM_SCAN_PORTS,
            TRTLLM_DIST_INIT_PORTS,
            SIDECAR_GRPC_PORTS,
        ]
        assert len({kind.name for kind in kinds}) == len(kinds)
        assert len({kind.base for kind in kinds}) == len(kinds)


def _example_processes(config: SrtConfig) -> list[Process]:
    """The process topology srtctl would launch for a recipe, on synthetic nodes."""
    resources = config.resources
    nodes = [f"node-{index}" for index in range(config.total_nodes)]
    endpoints = config.backend.allocate_endpoints(
        num_prefill=resources.num_prefill,
        num_decode=resources.num_decode,
        num_agg=resources.num_agg,
        gpus_per_prefill=resources.gpus_per_prefill,
        gpus_per_decode=resources.gpus_per_decode,
        gpus_per_agg=resources.gpus_per_agg,
        gpus_per_node=resources.gpus_per_node,
        available_nodes=nodes,
        spread_workers=resources.spread_workers,
    )
    return config.backend.endpoints_to_processes(
        endpoints,
        port_allocator=NodePortAllocator(bases={SIDECAR_GRPC_PORTS.name: config.dynamo.sidecar_port}),
        frontend_type=config.frontend.type,
        dynamo_sidecar=config.dynamo.sidecar,
    )


def _example_recipes() -> list[Path]:
    return sorted(path for example_dir in TOPOLOGY_EXAMPLE_DIRS for path in Path(example_dir).rglob("*.yaml"))


@pytest.mark.parametrize("recipe", _example_recipes(), ids=lambda path: str(path))
def test_example_topologies_bind_no_port_twice(recipe: Path):
    """Across every process of a recipe, nothing on one node shares a port and global kinds never repeat."""
    config = SrtConfig.from_yaml(recipe)
    processes = _example_processes(config)
    assert processes, recipe

    # Listeners each process binds on its own node. dist_init_port is shared by
    # the processes of one endpoint on purpose, and nixl_port by its DP ranks, so
    # they are checked per endpoint below rather than per process.
    per_node: Counter[tuple[str, str, int]] = Counter()
    for process in processes:
        ports = {
            "sys": process.sys_port,
            "http": process.http_port or None,
            "bootstrap": process.bootstrap_port,
            "kv_events": process.kv_events_port,
            "kvbm_zmq": process.kvbm_zmq_port,
            "sidecar_grpc": process.sidecar_grpc_port,
            "nccl": process.nccl_port,
            "vllm_scan": process.vllm_scan_port,
            "trtllm_dist_init": process.trtllm_dist_init_port,
        }
        for kind, port in ports.items():
            if port is not None:
                per_node[(process.node, kind, port)] += 1
    duplicates = {key: count for key, count in per_node.items() if count > 1}
    assert not duplicates, f"{recipe}: ports bound twice on one node: {duplicates}"

    # Global kinds: one value per process across the whole job.
    for kind in ("sys_port", "kv_events_port", "nccl_port", "vllm_scan_port", "trtllm_dist_init_port"):
        values = [getattr(process, kind) for process in processes if getattr(process, kind) is not None]
        assert len(values) == len(set(values)), f"{recipe}: {kind} repeats across the job"

    # Per-endpoint kinds: one value per endpoint, distinct across endpoints.
    per_endpoint_nixl = {(p.endpoint_mode, p.endpoint_index, p.engine_id): p.nixl_port for p in processes}
    nixl_values = [port for port in per_endpoint_nixl.values() if port is not None]
    assert len(nixl_values) == len(set(nixl_values)), f"{recipe}: nixl_port repeats across endpoints"
    dist_init_by_leader = Counter(
        (p.node, p.dist_init_port)
        for p in processes
        if p.is_leader and p.dist_init_port is not None and p.engine_id == 0
    )
    assert all(count == 1 for count in dist_init_by_leader.values()), f"{recipe}: dist_init_port repeats on a node"
