# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared implementation for routers configured with static worker URLs."""

from __future__ import annotations

import logging
import shlex
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from srtctl.core.health import WorkerHealthResult, check_static_router_health, probe_json_health
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.frontends.base import logical_health_expectations

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process
    from srtctl.services.implicit import EffectiveService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouterWorker:
    """A routable backend HTTP endpoint advertised to a static router."""

    mode: str
    url: str
    bootstrap_port: int | None = None


class StaticRouterFrontend:
    """Base class for routers whose worker topology is supplied on the CLI.

    A subclass sets the class attributes, registers with ``@register_frontend``,
    and overrides only the hooks whose behavior differs.
    """

    type: ClassVar[str]
    required_backend: ClassVar[str | None]
    executable: ClassVar[tuple[str, ...]]
    pd_flag: ClassVar[str]
    process_name: ClassVar[str]
    log_label: ClassVar[str | None] = None
    allow_empty_workers: ClassVar[bool] = False
    # Workers are the engines' own servers, each on its allocated HTTP port.
    worker_launch: ClassVar[Literal["dynamo", "direct"]] = "direct"
    expands_node_local_dp: ClassVar[bool] = False

    @property
    def health_endpoint(self) -> str:
        return "/workers"

    def validate(self, config: Any) -> None:
        """Recipe-level rules beyond the backend pairing; none by default."""

    def worker_api_port(self, mode: str) -> Literal["public", "allocated"]:
        """A routed worker binds its own allocated port; the router owns the public one."""
        return "allocated"

    metrics_path: ClassVar[str] = "/metrics"

    def worker_metrics_port(self, process: Process, runtime: RuntimeContext) -> int | None:
        """A native server's leader rank binds the HTTP server that carries /metrics; followers serve nothing."""
        if process.is_leader and process.http_port > 0:
            return process.http_port
        return None

    def worker_endpoint_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        if process.is_leader and process.http_port > 0:
            return process.http_port
        return None

    def profiling_control_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        return process.http_port if process.http_port > 0 else None

    def profiling_control_is_leader_only(self, config: Any) -> bool:
        return False

    def direct_endpoint_nodes(self, processes: list[Process]) -> list[str]:
        return []

    def worker_ready_port(self, process: Process) -> int:
        return process.sys_port

    def parse_health(
        self,
        response_json: dict,
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        return check_static_router_health(response_json, expected_prefill, expected_decode)

    def probe_ready(self, host: str, port: int, expected_prefill: int, expected_decode: int) -> WorkerHealthResult:
        """One GET of the router's worker registry, parsed against the expected counts."""
        return probe_json_health(host, port, self.health_endpoint, self.parse_health, expected_prefill, expected_decode)

    def health_expectations(self, config: Any, processes: list[Process] | None) -> tuple[int, int, str]:
        """The registry lists one entry per logical worker unless the router expands them."""
        return logical_health_expectations(config)

    def implied_services(self, config: Any) -> list[EffectiveService]:
        """A static router needs no discovery plane."""
        return []

    def frontend_metrics_port(self, frontend_args: dict[str, Any] | None) -> int | None:
        """Metrics share the routing port unless the router runs a separate listener."""
        return None

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        """Convert config values to CLI arguments, preserving repeated values."""
        if not args:
            return []
        result: list[str] = []
        for key, value in args.items():
            flag = f"--{key.replace('_', '-')}"
            if value is True:
                result.append(flag)
            elif value is False or value is None:
                continue
            elif isinstance(value, list):
                for item in value:
                    result.extend([flag, str(item)])
            else:
                result.extend([flag, str(value)])
        return result

    def get_managed_frontend_args(
        self,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
    ) -> list[str]:
        """Return adapter-managed CLI arguments derived from srtctl config."""
        return []

    def worker_scheme(self, backend: Any, mode: str) -> str:
        """Return the protocol used to reach one worker endpoint."""
        return "http"

    def worker_bootstrap_port(self, backend: Any, process: Process) -> int | None:
        """Return the optional P/D bootstrap port advertised for a worker."""
        return process.bootstrap_port

    def resolve_worker_host(self, node: str, network_interface: str | None) -> str:
        """Resolve one worker address; adapters may override this for compatibility."""
        return get_hostname_ip(node, network_interface)

    def start_process(self, **kwargs: Any) -> Any:
        """Launch one router process; split out for adapter-specific tests."""
        return start_srun_process(**kwargs)

    def build_bash_preamble(self, config: Any) -> str | None:
        """Return adapter-specific shell setup to run before the router."""
        return None

    def collect_workers(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[RouterWorker]:
        """Collect every independently routable backend process.

        A positive HTTP port, rather than ``Process.is_leader``, is the source
        of truth. Upstream vLLM per-node DP intentionally exposes one API per
        node-local pool, while cross-node TP/PP followers retain port zero.
        """
        workers: list[RouterWorker] = []
        for process in backend_processes:
            if process.http_port <= 0:
                continue
            scheme = self.worker_scheme(backend, process.endpoint_mode)
            host = self.resolve_worker_host(process.node, network_interface)
            workers.append(
                RouterWorker(
                    mode=process.endpoint_mode,
                    url=f"{scheme}://{host}:{process.http_port}",
                    bootstrap_port=self.worker_bootstrap_port(backend, process),
                )
            )
        return workers

    def get_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[str]:
        """Return extra direct readiness requirements, if any."""
        return []

    def build_router_command(self, workers: list[RouterWorker], host: str, port: int) -> list[str]:
        """Build the router CLI for aggregate or prefill/decode topologies."""
        aggregate = [worker for worker in workers if worker.mode == "agg"]
        prefills = [worker for worker in workers if worker.mode == "prefill"]
        decodes = [worker for worker in workers if worker.mode == "decode"]

        cmd = list(self.executable)
        if prefills or decodes:
            if aggregate:
                raise ValueError("Static router topology cannot mix aggregate and disaggregated workers")
            if not prefills or not decodes:
                raise ValueError("Disaggregated static router topology requires prefill and decode workers")
            cmd.append(self.pd_flag)
            for worker in prefills:
                cmd.extend(["--prefill", worker.url])
                if worker.bootstrap_port is not None:
                    cmd.append(str(worker.bootstrap_port))
            for worker in decodes:
                cmd.extend(["--decode", worker.url])
        else:
            if not aggregate:
                if self.allow_empty_workers:
                    cmd.append("--worker-urls")
                    cmd.extend(["--host", host, "--port", str(port)])
                    return cmd
                raise ValueError("Static router topology has no routable workers")
            cmd.extend(["--worker-urls", *(worker.url for worker in aggregate)])

        cmd.extend(["--host", host, "--port", str(port)])
        return cmd

    def start_frontends(
        self,
        topology: Any,
        runtime: RuntimeContext,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
        stop_event: threading.Event | None = None,
    ) -> list[ManagedProcess]:
        from srtctl.core.processes import FRONTEND_TERMINATE_TIMEOUT_SECONDS, ManagedProcess

        configured_backend = getattr(getattr(config, "backend", None), "type", self.required_backend)
        if configured_backend != self.required_backend:
            raise ValueError(
                f"frontend.type: {self.type} requires backend.type: {self.required_backend} "
                f"(got {configured_backend!r})"
            )

        workers = self.collect_workers(backend, backend_processes, runtime.network_interface)
        processes: list[ManagedProcess] = []
        for idx, node in enumerate(topology.frontend_nodes):
            router_log = runtime.log_dir / f"{node}_{self.log_label or self.type}_{idx}.out"
            cmd = self.build_router_command(workers, "0.0.0.0", topology.frontend_port)
            cmd.extend(self.get_managed_frontend_args(config, backend, backend_processes))
            cmd.extend(self.get_frontend_args_list(config.frontend.args))
            logger.info("Starting %s %d on %s: %s", self.type, idx, node, shlex.join(cmd))

            container_image = getattr(config.frontend, "container_image", None) or str(runtime.container_image)
            router_env = dict(getattr(runtime, "environment", {}))
            router_env.update(config.frontend.env or {})
            step_name = f"{self.process_name}_{idx}"
            proc = self.start_process(
                command=cmd,
                nodelist=[node],
                output=str(router_log),
                container_image=container_image,
                container_mounts=runtime.container_mounts,
                env_to_set=router_env or None,
                bash_preamble=self.build_bash_preamble(config),
                het_group=runtime.nodes.het_group_for(node),
                step_name=step_name,
            )
            processes.append(
                ManagedProcess(
                    name=step_name,
                    popen=proc,
                    log_file=router_log,
                    node=node,
                    critical=True,
                    terminate_timeout=FRONTEND_TERMINATE_TIMEOUT_SECONDS,
                    step_name=step_name,
                )
            )
        return processes
