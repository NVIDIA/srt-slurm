# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TokenSpeed inference backend, served through Dynamo (``python3 -m dynamo.tokenspeed``)."""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import field, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from marshmallow import Schema
from marshmallow_dataclass import dataclass

from srtctl.backends.base import Backend, BoundRolesField, RoleSettings, role_args
from srtctl.backends.sglang import _config_to_cli_args
from srtctl.ports import DIST_INIT_PORTS, DYN_SYSTEM_PORT_BASE, TOKENSPEED_PORTS

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process

WorkerMode = Literal["prefill", "decode", "agg"]

# TokenSpeed's leader binds a control-plane cluster from its --dist-init-addr port:
# the port itself, five ports above it, and one scheduler port per attention-DP rank.
# Attention DP never exceeds the endpoint's GPU count.
_RENDEZVOUS_PORTS = 6


@dataclass(frozen=True)
class TokenSpeedBackend(Backend):
    """Launch ``python3 -m dynamo.tokenspeed`` workers behind the Dynamo frontend.

    Example YAML:
        engine: tokenspeed
        frontend:
          type: dynamo
        roles:
          agg:
            args:
              tensor-parallel-size: 1
              max-model-len: 4096
    """

    # Engine type discriminator.
    type: Literal["tokenspeed"] = "tokenspeed"
    # The roles this engine runs (`roles.<role>` of the recipe), bound by SrtConfig and
    # never written on `engine:`. Per-role env and TokenSpeed CLI args are read from here.
    roles: Mapping[str, RoleSettings] = field(default_factory=dict, metadata={"marshmallow_field": BoundRolesField()})

    Schema: ClassVar[builtins.type[Schema]] = Schema

    def get_served_model_name(self, default: str) -> str:
        """A role's ``served-model-name``, else ``default``; srtctl always passes the result."""
        for mode in ("prefill", "agg", "decode"):
            args = role_args(self.roles, mode)
            name = args.get("served-model-name") or args.get("served_model_name")
            if name:
                return name
        return default

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
        )

    def endpoints_to_processes(
        self,
        endpoints: list[Endpoint],
        base_sys_port: int = DYN_SYSTEM_PORT_BASE,
        port_allocator: NodePortAllocator | None = None,
        frontend_type: str = "dynamo",
        dynamo_sidecar: bool = False,
    ) -> list[Process]:
        """Each process gets its own ``--port``; each endpoint a rendezvous block on its leader node."""
        if dynamo_sidecar:
            raise ValueError("TokenSpeed does not support Dynamo sidecars")
        from srtctl.core.topology import endpoints_to_processes, port_allocator_for

        allocator = port_allocator_for(port_allocator, base_sys_port)
        processes = endpoints_to_processes(endpoints, port_allocator=allocator)
        dist_init_ports = {
            (endpoint.mode, endpoint.index): allocator.next(
                DIST_INIT_PORTS, endpoint.nodes[0], size=_RENDEZVOUS_PORTS + endpoint.total_gpus
            )
            for endpoint in endpoints
        }
        return [
            replace(
                process,
                tokenspeed_port=allocator.next(TOKENSPEED_PORTS, process.node),
                dist_init_port=dist_init_ports[(process.endpoint_mode, process.endpoint_index)],
            )
            for process in processes
        ]

    def build_worker_command(
        self,
        process: Process,
        endpoint_processes: list[Process],
        runtime: RuntimeContext,
        frontend_type: str = "dynamo",
        nsys_prefix: list[str] | None = None,
        dump_config_path: Path | None = None,
        profiling: ProfilingConfig | None = None,
    ) -> list[str]:
        from srtctl.core.slurm import get_hostname_ip
        from srtctl.frontends import get_frontend

        if get_frontend(frontend_type).worker_launch != "dynamo":
            raise ValueError("engine: tokenspeed runs dynamo.tokenspeed and requires frontend.type: dynamo")
        if process.tokenspeed_port is None or process.dist_init_port is None:
            raise ValueError("build the topology with TokenSpeedBackend.endpoints_to_processes")

        mode = process.endpoint_mode
        config = self.get_config_for_mode(mode)
        for key in ("model", "model-path", "model_path", "served-model-name", "served_model_name"):
            config.pop(key, None)

        endpoint_nodes = list(dict.fromkeys(p.node for p in endpoint_processes))
        leader_ip = get_hostname_ip(endpoint_nodes[0], runtime.network_interface)
        cmd = [
            *(nsys_prefix or []),
            "python3",
            "-m",
            "dynamo.tokenspeed",
            "--model",
            runtime.worker_model_arg,
            "--served-model-name",
            self.get_served_model_name(runtime.model_path.name),
            # A prefill worker advertises this host for the Mooncake bootstrap.
            "--host",
            get_hostname_ip(process.node, runtime.network_interface),
            "--port",
            str(process.tokenspeed_port),
            "--dist-init-addr",
            f"{leader_ip}:{process.dist_init_port}",
            "--nnodes",
            str(len(endpoint_nodes)),
            "--node-rank",
            str(endpoint_nodes.index(process.node)),
        ]
        if mode != "agg":
            cmd.extend(["--disaggregation-mode", mode])
        if mode == "prefill" and process.bootstrap_port is not None:
            cmd.extend(["--disaggregation-bootstrap-port", str(process.bootstrap_port)])
        cmd.extend(_config_to_cli_args(config))
        return cmd
