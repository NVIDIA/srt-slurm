# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TileRT decode workers; prefill is provided by an independently selected engine."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass as plain_dataclass
from dataclasses import field, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from marshmallow import Schema
from marshmallow_dataclass import dataclass

from srtctl.backends.base import Backend, BoundRolesField, RoleSettings, WorkerMode
from srtctl.ports import DIST_INIT_PORTS, DYN_SYSTEM_PORT_BASE, HTTP_PORTS

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process

DECODE_SERVER_MODULE = "tilert.pd_vllm.decode_server"
SINGLE_NODE_ONLY = "TileRT supports single-node decode workers only"
DECODE_ONLY = "TileRT serves decode workers only"


@plain_dataclass(frozen=True)
class DecodeRendezvous:
    """Where one node of a multi-node TileRT decode worker joins the others."""

    nnodes: int
    node_rank: int
    nproc_per_node: int
    master_addr: str
    master_port: int


def _torchrun_launcher(rdzv: DecodeRendezvous) -> list[str]:
    """torchrun starts one decode-server rank per local GPU and sets RANK / LOCAL_RANK itself.

    Every rank gets the same --http-port. TileRT 0.1.8's dsv41 profile builds the engine with `start_node`, whose
    followers (global rank != 0) return before `decode_server.main` binds the HTTP socket, so only the leader
    listens. A profile without `start_node` builds and binds on every rank; use `node-env` for those.
    """
    return [
        "torchrun",
        "--nnodes",
        str(rdzv.nnodes),
        "--nproc-per-node",
        str(rdzv.nproc_per_node),
        "--node-rank",
        str(rdzv.node_rank),
        "--master-addr",
        rdzv.master_addr,
        "--master-port",
        str(rdzv.master_port),
        "-m",
        DECODE_SERVER_MODULE,
    ]


def _node_env_launcher(rdzv: DecodeRendezvous) -> list[str]:
    """One decode server per node; it re-executes itself under torch.distributed.run with a static rendezvous.

    TileRT 0.1.8's `decode_server.main` reads the node count, this node's rank and the rendezvous address from these
    variables when the profile needs more than one rank per node, and re-executes under `torch.distributed.run
    --rdzv-backend static`; with LOCAL_RANK already set (the torchrun form) it does not. TileRT 0.1.6 has no such
    contract.
    """
    return [
        "env",
        f"TILERT_PD_NNODES={rdzv.nnodes}",
        f"TILERT_PD_NODE_RANK={rdzv.node_rank}",
        f"TILERT_PD_RDZV={rdzv.master_addr}:{rdzv.master_port}",
        "python",
        "-m",
        DECODE_SERVER_MODULE,
    ]


# `engine.multi_node_launch` values and the launcher each one renders for a decode worker spanning nodes.
MULTI_NODE_LAUNCHERS: dict[str, Callable[[DecodeRendezvous], list[str]]] = {
    "node-env": _node_env_launcher,
    "torchrun": _torchrun_launcher,
}


@dataclass(frozen=True)
class TileRTBackend(Backend):
    """Launch TileRT's decode server with recipe-owned model and transport settings."""

    # Engine type discriminator.
    type: Literal["tilert"] = "tilert"
    # Model name the server reports to clients; unset uses the default served name.
    served_model_name: str | None = None
    # How a decode worker that spans several nodes starts on each node; unset keeps decode workers single-node.
    # `node-env` (recommended) runs one `python -m tilert.pd_vllm.decode_server ...` per node under
    # TILERT_PD_NNODES, TILERT_PD_NODE_RANK and TILERT_PD_RDZV, and the server starts its local ranks; `torchrun`
    # starts them with `torchrun --nnodes --nproc-per-node --node-rank --master-addr --master-port -m
    # tilert.pd_vllm.decode_server ...`. Only the worker's first node is advertised to the router; see
    # docs/tilert.md#multi-node-decode.
    multi_node_launch: Literal["node-env", "torchrun"] | None = None
    # The roles this engine runs (`roles.<role>` of the recipe), bound by SrtConfig and
    # never written on `engine:`. The decode role's env and args are read from here.
    roles: Mapping[str, RoleSettings] = field(default_factory=dict, metadata={"marshmallow_field": BoundRolesField()})

    Schema: ClassVar[builtins.type[Schema]] = Schema

    def get_served_model_name(self, default: str) -> str:
        return self.served_model_name or default

    def allocate_endpoints(
        self,
        num_prefill: int,
        num_decode: int,
        num_agg: int,
        gpus_per_prefill: int,
        gpus_per_decode: int,
        gpus_per_agg: int,
        gpus_per_node: int,
        available_nodes: Sequence[str],
        spread_workers: bool = False,
        gpu_indices_by_mode: Mapping[WorkerMode, Sequence[frozenset[int]]] | None = None,
    ) -> list[Endpoint]:
        from srtctl.core.topology import allocate_endpoints

        return allocate_endpoints(
            num_prefill=num_prefill,
            num_decode=num_decode,
            num_agg=num_agg,
            gpus_per_prefill=gpus_per_prefill,
            gpus_per_decode=gpus_per_decode,
            gpus_per_agg=gpus_per_agg,
            gpus_per_node=gpus_per_node,
            available_nodes=available_nodes,
            spread_workers=spread_workers,
            gpu_indices_by_mode=gpu_indices_by_mode,
        )

    @property
    def allows_multi_node_decode(self) -> bool:
        """Whether a decode worker may span nodes (`multi_node_launch` is set)."""
        return self.multi_node_launch is not None

    def endpoints_to_processes(
        self,
        endpoints: list[Endpoint],
        base_sys_port: int = DYN_SYSTEM_PORT_BASE,
        port_allocator: NodePortAllocator | None = None,
        frontend_type: str = "tilert-router",
        dynamo_sidecar: bool = False,
    ) -> list[Process]:
        """One process per node of each decode worker.

        A worker spanning nodes (only with `multi_node_launch`) gets one rendezvous port, allocated on its first
        node and shared by every node, and an HTTP port of their own for the nodes after the first, so a decode
        server that binds on every node does not collide; the router is given the first node only.
        """
        from srtctl.core.topology import endpoints_to_processes, port_allocator_for

        if dynamo_sidecar:
            raise ValueError("TileRT does not support Dynamo sidecars")
        if any(ep.mode != "decode" for ep in endpoints):
            raise ValueError(DECODE_ONLY)
        if not self.allows_multi_node_decode and any(ep.is_multi_node for ep in endpoints):
            raise ValueError(SINGLE_NODE_ONLY)
        allocator = port_allocator_for(port_allocator, base_sys_port)
        processes = endpoints_to_processes(endpoints, port_allocator=allocator)
        master_ports = {ep.index: allocator.next(DIST_INIT_PORTS, ep.nodes[0]) for ep in endpoints if ep.is_multi_node}
        if not master_ports:
            return processes
        return [
            replace(
                process,
                dist_init_port=master_ports[process.endpoint_index],
                http_port=process.http_port or allocator.next(HTTP_PORTS, process.node),
            )
            if process.endpoint_index in master_ports
            else process
            for process in processes
        ]

    def router_worker_address(self, process: Process, frontend_node: str | None = None) -> tuple[str, int] | None:
        """Only a decode worker's first node is routed; the other nodes of a multi-node worker serve its ranks."""
        if not process.is_leader:
            return None
        return super().router_worker_address(process, frontend_node)

    def _launcher(self, process: Process, endpoint_processes: list[Process], runtime: RuntimeContext) -> list[str]:
        """`python -m` the decode server on a single node, or this node's multi-node launcher."""
        nodes = list(dict.fromkeys(p.node for p in endpoint_processes))
        if len(nodes) == 1:
            return ["python", "-m", DECODE_SERVER_MODULE]
        if self.multi_node_launch is None:
            raise ValueError(SINGLE_NODE_ONLY)
        if process.dist_init_port is None:
            raise ValueError(
                f"TileRT decode process {process.node} rank {process.node_rank} has no rendezvous port; "
                "build the topology with TileRTBackend.endpoints_to_processes"
            )
        from srtctl.core.slurm import get_hostname_ip

        leader = next(p for p in endpoint_processes if p.is_leader)
        rendezvous = DecodeRendezvous(
            nnodes=len(nodes),
            node_rank=process.node_rank,
            nproc_per_node=len(process.gpu_indices),
            master_addr=get_hostname_ip(leader.node, runtime.network_interface),
            master_port=process.dist_init_port,
        )
        return MULTI_NODE_LAUNCHERS[self.multi_node_launch](rendezvous)

    def build_worker_command(
        self,
        process: Process,
        endpoint_processes: list[Process],
        runtime: RuntimeContext,
        frontend_type: str = "tilert-router",
        nsys_prefix: list[str] | None = None,
        dump_config_path: Path | None = None,
        profiling: ProfilingConfig | None = None,
    ) -> list[str]:
        if process.endpoint_mode != "decode":
            raise ValueError(DECODE_ONLY)
        if process.nixl_port is None:
            raise ValueError("TileRT decode worker needs an allocated control port")
        config = self.get_config_for_mode("decode")
        managed = {key.lstrip("-").replace("_", "-") for key in config} & {"engine", "http-port", "ctrl-port"}
        if managed:
            raise ValueError(f"TileRT arguments are owned by srtctl: {sorted(managed)}")
        command = [
            *(nsys_prefix or []),
            *self._launcher(process, endpoint_processes, runtime),
            "--engine",
            "tilert",
            "--http-port",
            str(process.http_port),
            "--ctrl-port",
            str(process.nixl_port),
        ]
        for key, value in config.items():
            flag = f"--{key.replace('_', '-')}"
            if value is True:
                command.append(flag)
            elif value is not False and value is not None:
                for item in value if isinstance(value, list) else [value]:
                    command.extend([flag, str(item)])
        return command
