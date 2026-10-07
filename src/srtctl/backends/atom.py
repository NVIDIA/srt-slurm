# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ROCm ATOM inference backend."""

from __future__ import annotations

import builtins
import json
from collections.abc import Mapping, Sequence
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from marshmallow import Schema
from marshmallow_dataclass import dataclass

from srtctl.backends.base import WORKER_MODES, Backend, BoundRolesField, RoleSettings, role_args
from srtctl.ports import DYN_SYSTEM_PORT_BASE, LMCACHE_SERVER_PORT

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process


@dataclass(frozen=True)
class AtomBackend(Backend):
    """Launch ``atom.entrypoints.openai_server`` on ROCm workers."""

    # Engine type discriminator.
    type: Literal["atom"] = "atom"
    # The roles this engine runs (`roles.<role>` of the recipe), bound by SrtConfig and
    # never written on `engine:`. Per-role env and native ATOM CLI args are read from here.
    roles: Mapping[str, RoleSettings] = field(default_factory=dict, metadata={"marshmallow_field": BoundRolesField()})
    # KV-transfer connector between prefill and decode workers.
    connector: Literal["mooncake"] = "mooncake"
    # Mooncake transport; unset lets ATOM choose.
    mooncake_protocol: Literal["rdma", "tcp"] | None = None

    Schema: ClassVar[builtins.type[Schema]] = Schema

    def get_served_model_name(self, default: str) -> str:
        """The name ATOM serves: a role's ``served-model-name``, else its literal ``--model``."""
        for mode in WORKER_MODES:
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
        frontend_type: str = "atomesh",
        dynamo_sidecar: bool = False,
    ) -> list[Process]:
        if dynamo_sidecar:
            raise ValueError("ATOM does not support Dynamo sidecars")
        from srtctl.core.topology import endpoints_to_processes

        return endpoints_to_processes(endpoints, base_sys_port=base_sys_port, port_allocator=port_allocator)

    def _kv_transfer_config(
        self, process: Process, worker_ip: str, extra_connectors: list[dict[str, Any]]
    ) -> str | None:
        """Mooncake for P/D workers plus the role's extra connectors; several are wrapped in ``multi``."""
        connectors = []
        for connector in extra_connectors:
            if connector.get("kv_connector") == "lmcache_mp":
                # Dial the node-local lmcache-server service unless the recipe set a port.
                extra = {"lmcache.mp.port": LMCACHE_SERVER_PORT, **connector.get("kv_connector_extra_config", {})}
                connector = {**connector, "kv_connector_extra_config": extra}
            connectors.append(connector)
        if process.endpoint_mode in {"prefill", "decode"}:
            if process.nixl_port is None:
                raise ValueError("ATOM P/D worker is missing its Mooncake handshake port")
            mooncake: dict[str, Any] = {
                "kv_role": "kv_producer" if process.endpoint_mode == "prefill" else "kv_consumer",
                "kv_connector": self.connector,
                "proxy_ip": worker_ip,
                "handshake_port": process.nixl_port,
            }
            if self.mooncake_protocol is not None:
                mooncake["protocol"] = self.mooncake_protocol
            connectors.insert(0, mooncake)
        if not connectors:
            return None
        payload = connectors[0] if len(connectors) == 1 else {"kv_connector": "multi", "connectors": connectors}
        return json.dumps(payload, separators=(",", ":"))

    def build_worker_command(
        self,
        process: Process,
        endpoint_processes: list[Process],
        runtime: RuntimeContext,
        frontend_type: str = "atomesh",
        nsys_prefix: list[str] | None = None,
        dump_config_path: Path | None = None,
        profiling: ProfilingConfig | None = None,
    ) -> list[str]:
        if frontend_type != "atomesh":
            raise ValueError(f"backend.type: atom requires frontend.type: atomesh (got {frontend_type!r})")
        if len({item.node for item in endpoint_processes}) != 1:
            raise ValueError("ATOM currently requires each logical endpoint to fit on one Slurm node")

        from srtctl.core.slurm import get_hostname_ip

        worker_ip = get_hostname_ip(process.node, runtime.network_interface)
        config = self.get_config_for_mode(process.endpoint_mode)
        extra_connectors = config.pop("extra-kv-connectors", [])
        reserved = {"model", "host", "server-port", "tp", "tensor-parallel-size", "kv-transfer-config"}
        overlap = reserved.intersection(_canonical_arg_key(key) for key in config)
        if overlap:
            raise ValueError(f"ATOM config cannot override srtctl-managed argument(s): {sorted(overlap)}")

        command = ["env", f"ATOM_HOST_IP={worker_ip}", *(nsys_prefix or [])]
        command.extend(
            [
                "python3",
                "-m",
                "atom.entrypoints.openai_server",
                "--model",
                runtime.worker_model_arg,
                "--host",
                "0.0.0.0",
                "--server-port",
                str(process.http_port),
                "-tp",
                str(len(process.gpu_indices)),
            ]
        )
        kv_transfer_config = self._kv_transfer_config(process, worker_ip, extra_connectors)
        if kv_transfer_config is not None:
            command.extend(["--kv-transfer-config", kv_transfer_config])
        command.extend(_config_to_cli_args(config))
        return command


def _config_to_cli_args(config: dict[str, Any]) -> list[str]:
    """Preserve ATOM's native CLI spelling, which mixes hyphens and underscores."""
    args: list[str] = []
    for key, value in sorted(config.items()):
        flag = f"--{key}"
        if value is True:
            args.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            args.extend([flag, *(str(item) for item in value)])
        else:
            args.extend([flag, str(value)])
    return args


def _canonical_arg_key(key: str) -> str:
    return key.lstrip("-").replace("_", "-")
