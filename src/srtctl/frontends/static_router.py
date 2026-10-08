# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared implementation for routers configured with static worker URLs."""

from __future__ import annotations

import logging
import shlex
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import urlsplit

from srtctl.core.health import (
    WorkerHealthResult,
    check_static_router_health,
    probe_json_health,
    wait_for_http_endpoints,
    wait_for_port,
)
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.frontends.base import Frontend, numactl_prefix

if TYPE_CHECKING:
    from srtctl.backends.base import WorkerMode
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouterWorker:
    """A routable backend HTTP endpoint advertised to a static router."""

    mode: str
    url: str
    bootstrap_port: int | None = None


def setup_script_preamble(config: Any) -> str | None:
    """Shell that runs the recipe's ``setup_script`` in a router's own container.

    A router that ships as a package (``pip install``) rather than in the model
    image installs itself this way; the script is looked up in ``/configs`` and
    then ``/configs/patches``, as for the workers.
    """
    setup_script = getattr(config, "setup_script", None)
    if not setup_script:
        return None
    script_name = shlex.quote(setup_script)
    return (
        f"setup_script={script_name} && "
        'script_path="/configs/${setup_script}" && '
        'patch_script_path="/configs/patches/${setup_script}" && '
        'echo "Running setup script: ${script_path} (fallback ${patch_script_path})" && '
        'if [ -f "${script_path}" ]; then bash "${script_path}"; '
        'elif [ -f "${patch_script_path}" ]; then bash "${patch_script_path}"; '
        'else echo "WARNING: ${script_path} or ${patch_script_path} not found"; fi'
    )


class StaticRouterFrontend(Frontend):
    """Base class for routers whose worker topology is supplied on the CLI.

    A subclass sets the class attributes, registers with ``@register_frontend``,
    and overrides only the hooks whose behavior differs.
    """

    @property
    @abstractmethod
    def executable(self) -> tuple[str, ...]:
        """Router executable and any fixed launch arguments."""
        raise NotImplementedError

    @property
    @abstractmethod
    def pd_flag(self) -> str:
        """CLI flag enabling prefill/decode disaggregation."""
        raise NotImplementedError

    @property
    @abstractmethod
    def process_name(self) -> str:
        """Name used for the router's managed process and logs."""
        raise NotImplementedError

    log_label: ClassVar[str | None] = None
    allow_empty_workers: ClassVar[bool] = False
    # Probe every advertised HTTP worker for 200 before launching the router. A router whose
    # static registration expires when model loading outlasts its startup window sets this.
    wait_for_workers_before_start: ClassVar[bool] = False

    @property
    def health_endpoint(self) -> str:
        return "/workers"

    def worker_api_port(self, mode: str) -> Literal["public", "allocated"]:
        """A routed worker binds its own allocated port; the router owns the public one."""
        return "allocated"

    def worker_metrics_port(self, process: Process, runtime: RuntimeContext) -> int | None:
        """A native server's leader rank binds the HTTP server that carries /metrics; followers serve nothing.

        A gRPC-mode leader with an HTTP sidecar serves it there instead.
        """
        if process.grpc_http_port is not None:
            return process.grpc_http_port
        if process.is_leader and process.http_port > 0:
            return process.http_port
        return None

    def worker_endpoint_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        if process.is_leader and process.http_port > 0:
            return process.http_port
        return None

    def profiling_control_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        if process.grpc_http_port is not None:
            return process.grpc_http_port
        return process.http_port if process.http_port > 0 else None

    def worker_ready_port(self, process: Process) -> int:
        return process.sys_port

    def parse_health(
        self,
        response_json: dict,
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        return check_static_router_health(response_json, expected_prefill, expected_decode)

    def probe_ready(
        self, host: str, port: int, expected_prefill: int, expected_decode: int, config: Any
    ) -> WorkerHealthResult:
        """One GET of the router's worker registry, parsed against the expected counts."""
        return probe_json_health(host, port, self.health_endpoint, self.parse_health, expected_prefill, expected_decode)

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

    def worker_scheme(self, backend: Any, mode: WorkerMode) -> str:
        """Return the protocol used to reach one worker endpoint: gRPC when the backend's mode serves it."""
        return "grpc" if backend.is_grpc_mode(mode) else "http"

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

    def wait_for_workers(
        self, workers: list[RouterWorker], config: Any, stop_event: threading.Event | None = None
    ) -> None:
        """Wait for HTTP ``/health`` or gRPC TCP connections; skip when health checks are disabled.

        Raise on cancellation or exhaustion of the recipe's health-check budget.
        """
        if config.health_check is None:
            return
        health_check = config.health_check
        timeout = float(health_check.max_attempts * health_check.interval_seconds)
        health_urls = [
            f"{worker.url.rstrip('/')}/health" for worker in workers if worker.url.startswith(("http://", "https://"))
        ]
        grpc_targets = [urlsplit(worker.url) for worker in workers if worker.url.startswith("grpc://")]
        logger.info("Waiting for %d advertised backend endpoints before starting %s", len(workers), self.type)
        deadline = time.monotonic() + timeout
        ready = all(
            wait_for_port(
                str(target.hostname),
                int(target.port or 0),
                timeout=max(deadline - time.monotonic(), 0.0),
                stop_event=stop_event,
            )
            for target in grpc_targets
        ) and wait_for_http_endpoints(
            health_urls,
            poll_interval=float(health_check.interval_seconds),
            timeout=max(deadline - time.monotonic(), 0.0),
            report_every=60.0,
            stop_event=stop_event,
        )
        if not ready:
            raise RuntimeError(f"Advertised backend endpoints did not become ready before {self.type} startup")

    def discovers_workers(self, backend: Any) -> bool:
        """Whether the router learns its workers by registration instead of from its command line.

        A discovering router gets no ``--prefill``/``--decode`` URLs; the subclass
        adds its discovery flags. Static routers never discover.
        """
        return False

    def build_router_command(self, workers: list[RouterWorker], host: str, port: int, backend: Any) -> list[str]:
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
            if not self.discovers_workers(backend):
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
        if self.required_backend is not None and configured_backend != self.required_backend:
            raise ValueError(
                f"frontend.type: {self.type} requires backend.type: {self.required_backend} "
                f"(got {configured_backend!r})"
            )

        workers = self.collect_workers(backend, backend_processes, runtime.network_interface)
        if self.wait_for_workers_before_start:
            self.wait_for_workers(workers, config, stop_event)

        processes: list[ManagedProcess] = []
        for idx, node in enumerate(topology.frontend_nodes):
            router_log = runtime.log_dir / f"{node}_{self.log_label or self.type}_{idx}.out"
            cmd = self.build_router_command(workers, "0.0.0.0", topology.frontend_port, backend)
            cmd.extend(self.get_managed_frontend_args(config, backend, backend_processes))
            cmd.extend(self.get_frontend_args_list(config.frontend.args))
            cmd = numactl_prefix(config) + cmd
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
                srun_options=runtime.srun_options,
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
