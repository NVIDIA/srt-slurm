# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo frontend that dials the engines itself (`frontend.type: dynamo-direct`).

``python3 -m dynamo.frontend --direct --engine-url URL ...`` hosts each listed
engine in the frontend process: it speaks the engine's native gRPC API, keeps
discovery in memory, and routes, orchestrates prefill/decode and migrates as it
does for registered workers. The engines are plain ``sglang.launch_server``
processes with no Dynamo beside them, launched exactly as the engine half of
sidecar mode (see ``SGLangBackend._build_sidecar_command``).

Everything about the frontend process is ``DynamoFrontend``; this type only
replaces the discovery plane with a URL list and reads the workers' own ports.
"""

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from srtctl.core.slurm import get_hostname_ip
from srtctl.frontends.base import register_frontend
from srtctl.frontends.dynamo import DynamoFrontend

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

# The engine family behind --engine-url (``--direct-backend``), by engine type.
_DIRECT_BACKENDS = {"sglang": "sglang"}
_MANAGED_ARGS = ("direct", "direct-backend", "engine-url")


@register_frontend("dynamo-direct")
class DynamoDirectFrontend(DynamoFrontend):
    """Dynamo frontend in ``--direct`` mode in front of native SGLang gRPC engines."""

    type: ClassVar[str] = "dynamo-direct"
    required_backend: ClassVar[str | None] = "sglang"
    # The engine's own server, with no dynamo.<engine> worker and no sidecar.
    worker_launch: ClassVar[Literal["dynamo", "direct"]] = "direct"
    dials_native_grpc: ClassVar[bool] = True
    # Engines are listed, not discovered: no etcd or NATS, only the lifecycle collector.
    uses_discovery_plane: ClassVar[bool] = False

    def validate(self, config: Any) -> None:
        conflicts = sorted(
            {str(key).replace("_", "-") for key in config.frontend.args or {}}.intersection(_MANAGED_ARGS)
        )
        if conflicts:
            raise ValueError(f"frontend.type: dynamo-direct manages frontend.args: {', '.join(conflicts)}")

    def get_managed_frontend_args(
        self, config: Any, backend: Any, backend_processes: list["Process"], runtime: "RuntimeContext"
    ) -> list[str]:
        """``--direct`` and one ``--engine-url`` per node that hosts native gRPC listeners.

        Each engine reports its own role, model and KV publishers, so the list is
        flat. A prefill engine's bootstrap host is the host of its URL, hence the
        worker node's address on the job's network interface.
        """
        args = ["--direct", "--direct-backend", _DIRECT_BACKENDS[backend.type]]
        for process in backend_processes:
            # Allocated only on a node that hosts native gRPC listeners; the frontend discovers the node's others.
            if process.sidecar_grpc_port is not None:
                host = get_hostname_ip(process.node, runtime.network_interface)
                args.extend(["--engine-url", f"grpc://{host}:{process.sidecar_grpc_port}"])
        return args

    def worker_metrics_port(self, process: "Process", runtime: "RuntimeContext") -> int | None:
        """Every node of a native engine serves ``/metrics`` on its own HTTP port."""
        return process.http_port if process.http_port > 0 else None

    def worker_endpoint_port(self, process: "Process", config: Any, runtime: "RuntimeContext") -> int | None:
        """One endpoint per logical worker: the leader's HTTP port."""
        return process.http_port if process.is_leader and process.http_port > 0 else None

    def profiling_control_port(self, process: "Process", config: Any, runtime: "RuntimeContext") -> int | None:
        """The engine's own HTTP server controls its profiler on every node."""
        return process.http_port if process.http_port > 0 else None
