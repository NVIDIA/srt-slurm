# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TileRT decode workers; prefill is provided by an independently selected engine."""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from marshmallow import Schema
from marshmallow_dataclass import dataclass

from srtctl.backends.base import Backend, BoundRolesField, RoleSettings
from srtctl.ports import DYN_SYSTEM_PORT_BASE

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process


@dataclass(frozen=True)
class TileRTBackend(Backend):
    """Launch TileRT's decode server with recipe-owned model and transport settings."""

    # Engine type discriminator.
    type: Literal["tilert"] = "tilert"
    # Model name the server reports to clients; unset uses the default served name.
    served_model_name: str | None = None
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
        frontend_type: str = "tilert-router",
        dynamo_sidecar: bool = False,
    ) -> list[Process]:
        from srtctl.core.topology import endpoints_to_processes

        if dynamo_sidecar:
            raise ValueError("TileRT does not support Dynamo sidecars")
        if any(ep.mode != "decode" or len(ep.nodes) != 1 for ep in endpoints):
            raise ValueError("TileRT supports single-node decode workers only")
        return endpoints_to_processes(endpoints, base_sys_port=base_sys_port, port_allocator=port_allocator)

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
        if process.endpoint_mode != "decode" or len({p.node for p in endpoint_processes}) != 1:
            raise ValueError("TileRT supports single-node decode workers only")
        if process.nixl_port is None:
            raise ValueError("TileRT decode worker needs an allocated control port")
        config = self.get_config_for_mode("decode")
        managed = {key.lstrip("-").replace("_", "-") for key in config} & {"engine", "http-port", "ctrl-port"}
        if managed:
            raise ValueError(f"TileRT arguments are owned by srtctl: {sorted(managed)}")
        command = [
            *(nsys_prefix or []),
            "python",
            "-m",
            "tilert.pd_vllm.decode_server",
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
