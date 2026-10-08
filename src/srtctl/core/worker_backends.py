# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Role-aware dispatch using the existing concrete engine implementations.

Placement and port ownership remain job-wide. An engine owns only expansion of
its assigned endpoints and construction of commands for those processes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from srtctl.core.topology import Endpoint, NodePortAllocator, Process, allocate_endpoints
from srtctl.ports import PROXY_PORTS, SIDECAR_GRPC_PORTS

if TYPE_CHECKING:
    from srtctl.backends import BackendConfig
    from srtctl.core.schema import SrtConfig


def role_name(mode: str) -> str:
    role = "agg" if mode == "aggregated" else mode
    if role not in ("prefill", "decode", "agg"):
        raise ValueError(f"Unknown worker role {mode!r}")
    return role


def allocate_worker_endpoints(config: SrtConfig, nodes: Sequence[str]) -> list[Endpoint]:
    """Allocate once for the job; retain the original engine packer without overrides."""
    topology = config.topology
    allocate = config.backend.allocate_endpoints
    if config.has_role_backends:
        allocate = partial(allocate_endpoints, allow_prefill_decode_colocation=topology.colocated_decode)
    return allocate(
        num_prefill=topology.num_prefill,
        num_decode=topology.num_decode,
        num_agg=topology.num_agg,
        gpus_per_prefill=topology.gpus_per_prefill,
        gpus_per_decode=topology.gpus_per_decode,
        gpus_per_agg=topology.gpus_per_agg,
        gpus_per_node=topology.gpus_per_node,
        available_nodes=nodes,
        spread_workers=config.resources.spread_workers,
    )


def worker_processes(
    config: SrtConfig,
    endpoints: list[Endpoint],
    port_allocator: NodePortAllocator | None = None,
) -> list[Process]:
    """Expand each role with its engine, sharing every job-wide port reservation."""
    allocator = port_allocator or NodePortAllocator(bases={SIDECAR_GRPC_PORTS.name: config.dynamo.sidecar_port})

    def expand(backend: BackendConfig, selected: list[Endpoint]) -> list[Process]:
        return backend.endpoints_to_processes(
            selected,
            port_allocator=allocator,
            frontend_type=config.frontend.type,
            dynamo_sidecar=config.dynamo.sidecar,
        )

    if not config.has_role_backends:
        processes = expand(config.backend, endpoints)
    else:
        processes = []
        for role in ("prefill", "decode", "agg"):
            selected = [endpoint for endpoint in endpoints if endpoint.mode == role]
            if selected:
                processes.extend(expand(config.backend_for_role(role), selected))
    return with_proxy_ports(config, processes, allocator)


def with_proxy_ports(config: SrtConfig, processes: list[Process], allocator: NodePortAllocator) -> list[Process]:
    """Allocate proxy ports for routable workers in the frontend's proxied modes."""
    from srtctl.frontends import FRONTEND_NONE, get_frontend

    if config.frontend.type == FRONTEND_NONE:
        return processes
    proxied = get_frontend(config.frontend.type).proxied_worker_modes(config)
    return [
        replace(process, proxy_port=allocator.next(PROXY_PORTS, process.node))
        if process.http_port > 0 and process.endpoint_mode in proxied
        else process
        for process in processes
    ]
