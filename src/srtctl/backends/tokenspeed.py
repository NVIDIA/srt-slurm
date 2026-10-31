# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TokenSpeed inference backend.

Behind Dynamo each worker is ``python3 -m dynamo.tokenspeed``; behind SMG it is
``python3 -m tokenspeed.cli serve --headless``, dialing the gateway's ZMQ sockets.
"""

from __future__ import annotations

import builtins
import json
from collections.abc import Mapping, Sequence
from dataclasses import field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from marshmallow import Schema
from marshmallow_dataclass import dataclass

from srtctl.backends.base import Backend, BoundRolesField, RoleSettings, WorkerMode, role_args
from srtctl.backends.sglang import MooncakeKVStoreConfig, _config_to_cli_args
from srtctl.ports import (
    DIST_INIT_PORTS,
    DYN_SYSTEM_PORT_BASE,
    TOKENSPEED_METRICS_PORTS,
    TOKENSPEED_PORTS,
    TOKENSPEED_ZMQ_PORTS,
)

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process


# TokenSpeed's leader binds a control-plane cluster from its --dist-init-addr port:
# the port itself, five ports above it, and one scheduler port per attention-DP rank.
# Attention DP never exceeds the endpoint's GPU count.
_RENDEZVOUS_PORTS = 6


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

    def get_config_for_mode(self, mode: WorkerMode) -> dict[str, Any]:
        """Resolve underscore and kebab-case spellings once for all TokenSpeed consumers."""
        return {key.replace("_", "-"): value for key, value in role_args(self.roles, mode).items()}

    def prometheus_enabled(self, mode: WorkerMode) -> bool:
        """Whether the role enables TokenSpeed's HTTP Prometheus reporter."""
        args = self.get_config_for_mode(mode)
        reporters = args.get("metrics-reporters")
        if reporters is None:
            reporters = ["prometheus"]
        return args.get("enable-metrics") is True and "prometheus" in reporters

    def owns_dp_ranks(self, process: Process) -> bool:
        """Each headless process exports the metrics of its local ranks, including followers."""
        return process.tokenspeed_metrics_port is not None

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

    def endpoints_to_processes(
        self,
        endpoints: list[Endpoint],
        base_sys_port: int = DYN_SYSTEM_PORT_BASE,
        port_allocator: NodePortAllocator | None = None,
        frontend_type: str = "dynamo",
        dynamo_sidecar: bool = False,
    ) -> list[Process]:
        """Each process gets its own ``--port``; each endpoint a rendezvous block on its leader node.

        A direct endpoint's routable port names the handshake SMG binds on the
        frontend node. Each worker node has a separate optional metrics listener.
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
        zmq_ports = {
            (endpoint.mode, endpoint.index): allocator.next(TOKENSPEED_ZMQ_PORTS) for endpoint in endpoints if direct
        }
        result = []
        for process in processes:
            port = allocator.next(TOKENSPEED_PORTS, process.node)
            key = (process.endpoint_mode, process.endpoint_index)
            zmq_port = zmq_ports.get(key)
            metrics_port = (
                allocator.next(TOKENSPEED_METRICS_PORTS, process.node)
                if direct and self.prometheus_enabled(process.endpoint_mode)
                else None
            )
            result.append(
                replace(
                    process,
                    tokenspeed_port=port,
                    http_port=zmq_port if zmq_port is not None and process.http_port > 0 else process.http_port,
                    tokenspeed_zmq_port=zmq_port,
                    tokenspeed_metrics_port=metrics_port,
                    dist_init_port=dist_init_ports[key],
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
        frontend = get_frontend(frontend_type)
        direct_workers = frontend.worker_launch == "direct"
        entrypoint = ["tokenspeed.cli", "serve", "--headless"] if direct_workers else ["dynamo.tokenspeed"]
        cmd = [
            *(nsys_prefix or []),
            "python3",
            "-m",
            *entrypoint,
            "--model",
            runtime.worker_model_arg,
            "--served-model-name",
            self.get_served_model_name(runtime.model_path.name),
            # The engine advertises its own address for the P/D bootstrap, not the gateway's.
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
        if direct_workers:
            if process.tokenspeed_zmq_port is None:
                raise ValueError("build the topology with TokenSpeedBackend.endpoints_to_processes")
            cmd.extend(
                [
                    "--data-parallel-address",
                    runtime.head_node_ip,
                    "--data-parallel-rpc-port",
                    str(process.tokenspeed_zmq_port),
                    "--zmq-engine-index",
                    "0",
                ]
            )
            if process.tokenspeed_metrics_port is not None:
                config.setdefault("metrics-host", "0.0.0.0")
                cmd.extend(["--metrics-port", str(process.tokenspeed_metrics_port)])
        # argparse's append action needs one flag per reporter, unlike nargs-based lists.
        reporters = config.get("metrics-reporters")
        if isinstance(reporters, list):
            config.pop("metrics-reporters")
            if not reporters:
                raise ValueError("metrics-reporters must contain at least one reporter")
            for reporter in reporters:
                cmd.extend(["--metrics-reporters", str(reporter)])
        # TokenSpeed parses mapping-valued flags (kv-events-config, Mooncake extra config) as JSON.
        config = {key: json.dumps(value) if isinstance(value, dict) else value for key, value in config.items()}
        cmd.extend(_config_to_cli_args(config))
        return cmd
