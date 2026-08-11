# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo frontend implementation.

Uses NATS/etcd for communication between frontend and backend workers.
"""

import logging
import threading
from typing import TYPE_CHECKING, Any

from srtctl.core.health import WorkerHealthResult, check_dynamo_health
from srtctl.core.schema import build_otel_env
from srtctl.core.slurm import CONTAINER_REMAP_ROOT_EXPORT, start_srun_process
from srtctl.frontends.base import build_setup_script_preamble, register_frontend
from srtctl.ports import ETCD_CLIENT_PORT, NATS_PORT

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


@register_frontend("dynamo")
class DynamoFrontend:
    """Dynamo frontend implementation.

    Uses dynamo.frontend module with NATS/etcd for worker discovery.
    Health checks via /health endpoint.
    """

    @property
    def type(self) -> str:
        return "dynamo"

    @property
    def health_endpoint(self) -> str:
        return "/health"

    def parse_health(
        self,
        response_json: dict,
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        """Parse dynamo /health endpoint response."""
        return check_dynamo_health(response_json, expected_prefill, expected_decode)

    def get_backend_health_urls(self, backend: Any, backend_processes: list["Process"]) -> list[str]:
        """Dynamo owns backend discovery and exposes readiness through its frontend."""
        del backend, backend_processes
        return []

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        """Convert frontend args dict to CLI arguments."""
        if not args:
            return []
        result = []
        for key, value in args.items():
            if value is True:
                result.append(f"--{key}")
            elif value is not False and value is not None:
                result.extend([f"--{key}", str(value)])
        return result

    def start_frontends(
        self,
        topology: Any,  # FrontendTopology
        runtime: "RuntimeContext",
        config: Any,  # SrtConfig
        backend: Any,  # BackendProtocol
        backend_processes: list["Process"],
        stop_event: "threading.Event | None" = None,  # unused: returns immediately
    ) -> list["ManagedProcess"]:
        """Start dynamo frontends on designated nodes."""
        from srtctl.core.processes import ManagedProcess

        processes: list[ManagedProcess] = []

        for idx, node in enumerate(topology.frontend_nodes):
            logger.info("Starting dynamo frontend %d on %s", idx, node)

            frontend_log = runtime.log_dir / f"{node}_frontend_{idx}.out"
            cmd = ["python3", "-m", "dynamo.frontend", f"--http-port={topology.frontend_port}"]
            cmd.extend(self.get_frontend_args_list(config.frontend.args))

            env_to_set = {
                "ETCD_ENDPOINTS": f"http://{runtime.nodes.infra}:{ETCD_CLIENT_PORT}",
                "NATS_SERVER": f"nats://{runtime.nodes.infra}:{NATS_PORT}",
                "DYN_REQUEST_PLANE": config.dynamo.request_plane,
                "DYN_SKIP_SGLANG_LOG_FORMATTING": "1",
            }
            if config.dynamo.event_plane:
                env_to_set["DYN_EVENT_PLANE"] = config.dynamo.event_plane

            # Add OTEL env vars (before frontend env so OTEL_SERVICE_NAME can be overridden)
            env_to_set.update(build_otel_env(config.observability, "frontend"))

            # Add global recipe environment, including values derived from
            # dynamo.wheel, before frontend-specific overrides.
            env_to_set.update(runtime.environment)

            # Add frontend env from config
            if config.frontend.env:
                env_to_set.update(config.frontend.env)

            # Build bash preamble (setup script + dynamo install)
            bash_preamble = self._build_preamble(config)

            proc = start_srun_process(
                command=cmd,
                nodelist=[node],
                output=str(frontend_log),
                container_image=str(runtime.container_image),
                container_mounts=runtime.container_mounts,
                env_to_set=env_to_set,
                bash_preamble=bash_preamble,
                # Frontend container runs the dynamo install (see _build_preamble), whose
                # cold build needs root inside the container. Remap via enroot env var.
                srun_export_env=CONTAINER_REMAP_ROOT_EXPORT if config.dynamo.install else None,
                # TODO(jthomson): I don't have the faintest clue of
                # why this is needed in later versions of Dynamo, but it is.
                mpi="pmix",
                het_group=runtime.nodes.het_group_for(node),
            )

            processes.append(
                ManagedProcess(
                    name=f"frontend_{idx}",
                    popen=proc,
                    log_file=frontend_log,
                    node=node,
                    critical=True,
                )
            )

        return processes

    def _build_preamble(self, config: Any) -> str | None:
        """Build bash preamble for dynamo frontend processes."""
        parts = []

        # Custom setup script
        setup_preamble = build_setup_script_preamble(getattr(config, "setup_script", None))
        if setup_preamble:
            parts.append(setup_preamble)

        # Dynamo installation (required for dynamo frontend)
        # Skip if dynamo.install is False (container already has dynamo installed)
        if config.dynamo.install:
            parts.append(config.dynamo.get_install_commands())

        if not parts:
            return None

        return " && ".join(parts)
