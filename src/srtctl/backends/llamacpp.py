# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-node llama.cpp serving with a local GGUF model."""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, ClassVar, Literal

from marshmallow import Schema, ValidationError
from marshmallow_dataclass import dataclass

from srtctl.backends.base import Backend, BoundRolesField, RoleSettings, WorkerMode
from srtctl.backends.sglang import _config_to_cli_args
from srtctl.ports import DYN_SYSTEM_PORT_BASE

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process


@dataclass(frozen=True)
class LlamaCppBackend(Backend):
    """Run one ``llama-server`` against a GGUF file inside ``model.path``.

    ``roles.agg.args`` carries llama.cpp CLI options (for example ``ctx-size``,
    ``n-gpu-layers`` and ``parallel``); ``alias`` also sets the model name used
    by benchmark clients. The official server images place the executable at
    ``/app/llama-server`` and do not put it on PATH.
    """

    # GGUF filename relative to the model.path directory (may include subdirectories).
    model_file: str
    # Engine type discriminator.
    type: Literal["llamacpp"] = "llamacpp"
    # Server executable inside the container; override for a custom image.
    server_binary: str = "/app/llama-server"
    # Enable llama.cpp's /metrics endpoint on the server's HTTP port.
    metrics: bool = True
    # Per-role arguments and environment, bound from the recipe's roles block at load time.
    roles: Mapping[str, RoleSettings] = field(default_factory=dict, metadata={"marshmallow_field": BoundRolesField()})

    required_frontend: ClassVar[str | None] = "llamacpp"
    Schema: ClassVar[builtins.type[Schema]] = Schema

    def __post_init__(self) -> None:
        path = PurePosixPath(self.model_file)
        if not self.model_file.strip() or path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".gguf":
            raise ValidationError("engine.model_file must be a relative .gguf filename inside model.path")
        if not self.server_binary.strip():
            raise ValidationError("engine.server_binary must not be empty")
        for role, settings in self.roles.items():
            managed = {key.replace("_", "-") for key in settings.args} & {"model", "m", "host", "port", "metrics"}
            if managed:
                raise ValidationError(
                    f"roles.{role}.args contains SRT-managed flags: {', '.join(sorted(managed))}; "
                    "use engine.model_file and engine.metrics, and let SRT select the host and port"
                )

    def _model_aliases(self) -> str:
        alias = self.get_config_for_mode("agg").get("alias")
        return str(alias) if alias else PurePosixPath(self.model_file).stem

    def get_served_model_name(self, default: str) -> str:
        return self._model_aliases().split(",", 1)[0]

    def get_metrics_path(self, mode: WorkerMode) -> str | None:
        return super().get_metrics_path(mode) if self.metrics else None

    def launch_settings(self) -> dict[str, str]:
        return {
            "model_file": self.model_file,
            "server_binary": self.server_binary,
            "metrics": str(self.metrics).lower(),
        }

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

        if num_prefill or num_decode or num_agg != 1 or gpus_per_agg > gpus_per_node:
            raise ValueError("llamacpp requires exactly one aggregate worker on one node")
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
        from srtctl.core.topology import endpoints_to_processes

        if dynamo_sidecar or any(endpoint.mode != "agg" or len(endpoint.nodes) != 1 for endpoint in endpoints):
            raise ValueError("llamacpp supports single-node aggregate workers without Dynamo sidecars")
        return endpoints_to_processes(
            endpoints, base_sys_port=base_sys_port, port_allocator=port_allocator, bootstrap_ports=False
        )

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
        from srtctl.frontends import get_frontend

        frontend = get_frontend(frontend_type)
        if frontend.type != self.required_frontend:
            raise ValueError(f"llamacpp requires frontend.type: {self.required_frontend}")
        if process.endpoint_mode != "agg" or not process.is_leader or len(endpoint_processes) != 1:
            raise ValueError("llamacpp requires one aggregate server process on one node")
        config = self.get_config_for_mode(process.endpoint_mode)
        config["alias"] = self._model_aliases()
        port = (
            runtime.frontend_port if frontend.worker_api_port(process.endpoint_mode) == "public" else process.http_port
        )
        command = [
            *(nsys_prefix or []),
            self.server_binary,
            "--model",
            str(PurePosixPath(runtime.worker_model_arg) / self.model_file),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ]
        if self.metrics:
            command.append("--metrics")
        command.extend(_config_to_cli_args(config))
        return command
