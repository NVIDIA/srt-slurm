# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TokenSpeed inference backend.

Behind the Dynamo frontend each worker is ``python3 -m dynamo.tokenspeed``; behind a
static router (SMG) it is TokenSpeed's gRPC engine, ``python3 -m smg_grpc_servicer.tokenspeed``.
"""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import field, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from marshmallow import Schema
from marshmallow_dataclass import dataclass

from srtctl.backends.base import Backend, BoundRolesField, RoleSettings, WorkerMode, role_args
from srtctl.backends.sglang import MooncakeKVStoreConfig, SGLangBackend, _config_to_cli_args
from srtctl.ports import DIST_INIT_PORTS, DYN_SYSTEM_PORT_BASE, TOKENSPEED_PORTS

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process


# TokenSpeed's leader binds a control-plane cluster from its --dist-init-addr port:
# the port itself, five ports above it, and one scheduler port per attention-DP rank.
# Attention DP never exceeds the endpoint's GPU count.
_RENDEZVOUS_PORTS = 6

# The worker module per frontend launch mode. TokenSpeed's engine-only server is its
# gRPC servicer, what `ts serve` runs behind its bundled SMG; it has no HTTP-only server.
_WORKER_MODULE = {"dynamo": "dynamo.tokenspeed", "direct": "smg_grpc_servicer.tokenspeed"}


@dataclass(frozen=True)
class TokenSpeedBackend(Backend):
    """Launch TokenSpeed workers behind the Dynamo frontend or a static router such as SMG.

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
    # Set by a `mooncake-master` service. TokenSpeed's Mooncake Store L3 cache
    # (`kvstore-storage-backend: mooncake`) reads the master from the MOONCAKE_* environment.
    mooncake_kv_store: MooncakeKVStoreConfig | None = None

    Schema: ClassVar[builtins.type[Schema]] = Schema

    # The same MOONCAKE_* variables SGLang's Mooncake store client reads.
    get_mooncake_worker_env = SGLangBackend.get_mooncake_worker_env

    def is_grpc_mode(self, mode: WorkerMode) -> bool:
        """A direct TokenSpeed worker is always the gRPC engine (``smg_grpc_servicer.tokenspeed``)."""
        return True

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
        """Each process gets its own ``--port``; each endpoint a rendezvous block on its leader node.

        A direct worker's leader serves gRPC on that ``--port``, so it is also the
        process's routable port (``http_port``).
        """
        if dynamo_sidecar:
            raise ValueError("TokenSpeed does not support Dynamo sidecars")
        from srtctl.core.topology import endpoints_to_processes, port_allocator_for
        from srtctl.frontends import get_frontend

        direct = get_frontend(frontend_type).worker_launch == "direct"
        allocator = port_allocator_for(port_allocator, base_sys_port)
        processes = endpoints_to_processes(endpoints, port_allocator=allocator)
        dist_init_ports = {
            (endpoint.mode, endpoint.index): allocator.next(
                DIST_INIT_PORTS, endpoint.nodes[0], size=_RENDEZVOUS_PORTS + endpoint.total_gpus
            )
            for endpoint in endpoints
        }
        result = []
        for process in processes:
            port = allocator.next(TOKENSPEED_PORTS, process.node)
            result.append(
                replace(
                    process,
                    tokenspeed_port=port,
                    http_port=port if direct and process.http_port > 0 else process.http_port,
                    dist_init_port=dist_init_ports[(process.endpoint_mode, process.endpoint_index)],
                )
            )
        return result

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
            _WORKER_MODULE[get_frontend(frontend_type).worker_launch],
            "--model",
            runtime.worker_model_arg,
            "--served-model-name",
            self.get_served_model_name(runtime.model_path.name),
            # A prefill worker advertises this host for the Mooncake bootstrap; a direct
            # worker also binds its gRPC server on it.
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
