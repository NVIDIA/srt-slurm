# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo frontend implementation.

Uses NATS/etcd for communication between frontend and backend workers.
"""

import logging
import shlex
import threading
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import yaml

from srtctl.core.health import WorkerHealthResult, check_dynamo_health
from srtctl.core.observability_nsys import wrap_observability_nsys
from srtctl.core.schema import build_otel_env
from srtctl.core.slurm import CONTAINER_REMAP_ROOT_EXPORT, start_srun_process
from srtctl.frontends.base import register_frontend
from srtctl.frontends.dynamic_frontend import DynamicFrontend
from srtctl.services.implicit import discovery_env

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)

ROUTER_POLICY_CONFIG_FILENAME = "router_policy_config.yaml"
ROUTER_POLICY_CONFIG_CONTAINER_PATH = f"/logs/{ROUTER_POLICY_CONFIG_FILENAME}"


@register_frontend("dynamo")
class DynamoFrontend(DynamicFrontend):
    """Dynamo frontend implementation.

    Uses dynamo.frontend module with NATS/etcd for worker discovery.
    Health checks via /health endpoint. The dynamo.* recipe rules (sidecar,
    failover, worker_selection) are dynamo-config validations and stay in the
    schema.
    """

    type: ClassVar[str] = "dynamo"
    # dynamo.<engine> workers register over the request plane; they bind no OpenAI port.
    worker_launch: ClassVar[Literal["dynamo", "direct"]] = "dynamo"

    def worker_metrics_port(self, process: "Process", runtime: "RuntimeContext") -> int | None:
        """Every rank runs the Dynamo system status server (health, metrics) on its system port."""
        del runtime
        return process.sys_port if process.sys_port > 0 else None

    def worker_endpoint_port(self, process: "Process", config: Any, runtime: "RuntimeContext") -> int | None:
        """One endpoint per logical worker: the leader's system port, or the native engine's port behind a sidecar."""
        del runtime
        if not process.is_leader:
            return None
        port = process.http_port if config.dynamo.sidecar else process.sys_port
        return port if port > 0 else None

    def profiling_control_port(self, process: "Process", config: Any, runtime: "RuntimeContext") -> int | None:
        """Iteration-triggered captures are controlled per rank on the system port."""
        del config, runtime
        return process.sys_port if process.sys_port > 0 else None

    def profiling_control_is_leader_only(self, config: Any) -> bool:
        """A Dynamo sidecar exposes one control server per logical endpoint, on its leader."""
        return bool(config.dynamo.sidecar)

    def worker_ready_port(self, process: "Process") -> int:
        """DYN_SYSTEM_PORT: the per-worker axum server reports /health once registered."""
        return process.sys_port

    def parse_health(
        self,
        response_json: dict,
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        """Parse dynamo /health endpoint response."""
        return check_dynamo_health(response_json, expected_prefill, expected_decode)

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
        from srtctl.core.processes import FRONTEND_TERMINATE_TIMEOUT_SECONDS, ManagedProcess

        processes: list[ManagedProcess] = []
        frontend_args = dict(config.frontend.args or {})
        worker_selection = getattr(config.frontend, "worker_selection", None)
        if worker_selection is not None:
            policy_path = runtime.log_dir / ROUTER_POLICY_CONFIG_FILENAME
            policy_path.write_text(yaml.safe_dump({"worker_selection": worker_selection}, sort_keys=False))
            frontend_args["router-policy-config"] = ROUTER_POLICY_CONFIG_CONTAINER_PATH
            logger.info("Dynamo router policy config written to %s", policy_path)

        for idx, node in enumerate(topology.frontend_nodes):
            logger.info("Starting dynamo frontend %d on %s", idx, node)

            frontend_log = runtime.log_dir / f"{node}_frontend_{idx}.out"
            cmd = ["python3", "-m", "dynamo.frontend", f"--http-port={topology.frontend_port}"]
            cmd.extend(self.get_frontend_args_list(frontend_args))

            automatic_nsys = getattr(config, "observability_nsys_enabled", False) is True
            nsys_env: dict[str, str] = {}
            if automatic_nsys:
                cmd, nsys_env = wrap_observability_nsys(
                    cmd,
                    config=config,
                    log_dir=runtime.log_dir,
                    report_name=f"frontend/{node}_frontend_{idx}",
                    frontend=True,
                )
                logger.info("Observability: nsys on frontend %d and all worker ranks", idx)

            env_to_set = {
                **discovery_env(config, runtime),
                "DYN_REQUEST_PLANE": config.dynamo.request_plane,
                "DYN_SKIP_SGLANG_LOG_FORMATTING": "1",
            }
            if config.dynamo.event_plane:
                env_to_set["DYN_EVENT_PLANE"] = config.dynamo.event_plane

            # Add OTEL env vars (before frontend env so OTEL_SERVICE_NAME can be overridden)
            env_to_set.update(build_otel_env(config.observability, "frontend"))
            env_to_set.update(nsys_env)

            # Add global recipe environment, including values derived from
            # dynamo.wheel, before frontend-specific overrides.
            env_to_set.update(runtime.environment)

            # Add frontend env from config
            if config.frontend.env:
                env_to_set.update(config.frontend.env)

            # Build bash preamble (setup script + dynamo install)
            bash_preamble = self._build_preamble(config)

            step_name = f"frontend_{idx}"
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
                step_name=step_name,
            )

            processes.append(
                ManagedProcess(
                    name=step_name,
                    popen=proc,
                    log_file=frontend_log,
                    node=node,
                    critical=True,
                    terminate_timeout=(
                        config.observability.nsys.terminate_timeout
                        if automatic_nsys
                        else FRONTEND_TERMINATE_TIMEOUT_SECONDS
                    ),
                    signal_full=not automatic_nsys,
                    step_name=step_name,
                )
            )

        return processes

    def _build_preamble(self, config: Any) -> str | None:
        """Build bash preamble for dynamo frontend processes."""
        parts = []

        # Custom setup script
        setup_script = getattr(config, "setup_script", None)
        if isinstance(setup_script, str) and setup_script:
            script_name = shlex.quote(setup_script)
            parts.append(
                f"setup_script={script_name} && "
                'script_path="/configs/${setup_script}" && '
                'patch_script_path="/configs/patches/${setup_script}" && '
                'echo "Running setup script: ${script_path} (fallback ${patch_script_path})" && '
                'if [ -f "${script_path}" ]; then bash "${script_path}"; '
                'elif [ -f "${patch_script_path}" ]; then bash "${patch_script_path}"; '
                'else echo "WARNING: ${script_path} or ${patch_script_path} not found"; fi'
            )

        # Dynamo installation (required for dynamo frontend)
        # Skip if dynamo.install is False (container already has dynamo installed)
        if config.dynamo.install:
            parts.append(config.dynamo.get_install_commands())

        if not parts:
            return None

        return " && ".join(parts)
