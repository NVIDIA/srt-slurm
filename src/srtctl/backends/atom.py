# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ROCm ATOM inference backend."""

from __future__ import annotations

import builtins
import json
from collections.abc import Sequence
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from marshmallow import Schema
from marshmallow_dataclass import dataclass

from srtctl.ports import DYN_SYSTEM_PORT_BASE

if TYPE_CHECKING:
    from srtctl.backends.base import SrunConfig
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process

WorkerMode = Literal["prefill", "decode", "agg"]


@dataclass(frozen=True)
class AtomServerConfig:
    """Native ATOM CLI arguments for each serving role."""

    prefill: dict[str, Any] | None = None
    decode: dict[str, Any] | None = None
    aggregated: dict[str, Any] | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class AtomProtocol:
    """Launch ``atom.entrypoints.openai_server`` on ROCm workers."""

    type: Literal["atom"] = "atom"
    prefill_environment: dict[str, str] = field(default_factory=dict)
    decode_environment: dict[str, str] = field(default_factory=dict)
    aggregated_environment: dict[str, str] = field(default_factory=dict)
    atom_config: AtomServerConfig | None = None
    connector: Literal["mooncake"] = "mooncake"
    mooncake_protocol: Literal["rdma", "tcp"] | None = None

    Schema: ClassVar[builtins.type[Schema]] = Schema

    def get_srun_config(self) -> SrunConfig:
        from srtctl.backends.base import SrunConfig

        return SrunConfig(mpi=None, oversubscribe=False, launch_per_endpoint=False)

    def get_config_for_mode(self, mode: WorkerMode) -> dict[str, Any]:
        if self.atom_config is None:
            return {}
        values = {
            "prefill": self.atom_config.prefill,
            "decode": self.atom_config.decode,
            "agg": self.atom_config.aggregated,
        }
        return dict(values.get(mode) or {})

    def get_environment_for_mode(self, mode: WorkerMode) -> dict[str, str]:
        values = {
            "prefill": self.prefill_environment,
            "decode": self.decode_environment,
            "agg": self.aggregated_environment,
        }
        return dict(values.get(mode) or {})

    def get_process_environment(self, process: Process) -> dict[str, str]:
        return {}

    def get_served_model_name(self, default: str) -> str:
        return default

    @property
    def mooncake_kv_store(self) -> None:
        return None

    @property
    def failover(self) -> None:
        return None

    def get_mooncake_worker_env(self, infra_node_ip: str, local_hostname: str) -> dict[str, str]:
        return {}

    def get_failover_environment(self, process: Process, job_id: str) -> dict[str, str]:
        return {}

    def should_set_visible_devices(self) -> bool:
        return True

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

    def _kv_transfer_config(self, process: Process, worker_ip: str) -> str:
        if process.endpoint_mode not in {"prefill", "decode"}:
            raise ValueError("ATOM KV transfer is only valid for prefill/decode workers")
        if process.nixl_port is None:
            raise ValueError("ATOM P/D worker is missing its Mooncake handshake port")
        payload = {
            "kv_role": "kv_producer" if process.endpoint_mode == "prefill" else "kv_consumer",
            "kv_connector": self.connector,
            "proxy_ip": worker_ip,
            "handshake_port": process.nixl_port,
        }
        if self.mooncake_protocol is not None:
            payload["protocol"] = self.mooncake_protocol
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
        if process.endpoint_mode in {"prefill", "decode"}:
            command.extend(["--kv-transfer-config", self._kv_transfer_config(process, worker_ip)])
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
