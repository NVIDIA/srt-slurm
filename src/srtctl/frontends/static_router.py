# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared implementation for routers configured with static worker URLs."""

from __future__ import annotations

import logging
import shlex
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from srtctl.core.health import WorkerHealthResult, check_static_router_health
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.frontends.base import build_setup_script_preamble

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouterWorker:
    """A logical backend endpoint exposed to a static router."""

    mode: str
    url: str
    bootstrap_port: int | None = None


class StaticRouterFrontend:
    """Base class for routers whose worker topology is supplied on the CLI."""

    type: ClassVar[str]
    backend_type: ClassVar[str]
    executable: ClassVar[tuple[str, ...]]
    pd_flag: ClassVar[str]
    process_name: ClassVar[str]

    @property
    def health_endpoint(self) -> str:
        return "/workers"

    def parse_health(
        self,
        response_json: dict,
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        return check_static_router_health(response_json, expected_prefill, expected_decode)

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        """Convert config values to CLI arguments, preserving repeated values."""
        if not args:
            return []
        result: list[str] = []
        for key, value in args.items():
            flag = f"--{key}"
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
        backend: Any | None = None,
        backend_processes: list[Process] | None = None,
    ) -> list[str]:
        """Return adapter-managed CLI arguments derived from srtctl config."""
        del backend, backend_processes
        return []

    def get_backend_health_urls(self, backend: Any, backend_processes: list[Process]) -> list[str]:
        """Keep existing static-router readiness semantics unless an adapter opts in."""
        del backend, backend_processes
        return []

    def worker_scheme(self, backend: Any, mode: str) -> str:
        """Return the protocol used to reach a worker endpoint."""
        return "http"

    def get_hostname_ip(self, node: str) -> str:
        """Resolve a worker node to the address advertised to the router."""
        return get_hostname_ip(node)

    def worker_bootstrap_port(self, backend: Any, process: Process) -> int | None:
        """Return the optional P/D bootstrap port advertised for a worker."""
        return process.bootstrap_port

    def start_process(self, **kwargs: Any) -> Any:
        """Launch one router process. Split out for adapter-specific testing."""
        return start_srun_process(**kwargs)

    def collect_workers(self, backend: Any, backend_processes: list[Process]) -> list[RouterWorker]:
        workers: list[RouterWorker] = []
        for process in backend_processes:
            # An allocated HTTP port is the source of truth for whether a backend
            # process is independently routable. Multi-node vLLM DP exposes one
            # node-local pool per URL, while TP follower processes retain port 0.
            if process.http_port <= 0:
                continue
            scheme = self.worker_scheme(backend, process.endpoint_mode)
            workers.append(
                RouterWorker(
                    mode=process.endpoint_mode,
                    url=f"{scheme}://{self.get_hostname_ip(process.node)}:{process.http_port}",
                    bootstrap_port=self.worker_bootstrap_port(backend, process),
                )
            )
        return workers

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
                raise ValueError("Static router topology has no logical workers")
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
        del stop_event  # static routers return immediately after launch
        from srtctl.core.processes import ManagedProcess

        configured_backend = getattr(getattr(config, "backend", None), "type", self.backend_type)
        if configured_backend != self.backend_type:
            raise ValueError(
                f"frontend.type: {self.type} requires backend.type: {self.backend_type} (got {configured_backend!r})"
            )

        workers = self.collect_workers(backend, backend_processes)
        processes: list[ManagedProcess] = []
        for idx, node in enumerate(topology.frontend_nodes):
            router_log = runtime.log_dir / f"{node}_{self.type}_{idx}.out"
            cmd = self.build_router_command(workers, "0.0.0.0", topology.frontend_port)
            cmd.extend(self.get_managed_frontend_args(config, backend, backend_processes))
            cmd.extend(self.get_frontend_args_list(config.frontend.args))
            logger.info("Starting %s %d on %s: %s", self.type, idx, node, shlex.join(cmd))

            container_image = getattr(config.frontend, "container_image", None) or str(runtime.container_image)
            het_group_for = getattr(runtime.nodes, "het_group_for", lambda _node: None)
            router_env = dict(getattr(runtime, "environment", {}))
            router_env.update(config.frontend.env or {})
            proc = self.start_process(
                command=cmd,
                nodelist=[node],
                output=str(router_log),
                container_image=container_image,
                container_mounts=runtime.container_mounts,
                env_to_set=router_env or None,
                bash_preamble=build_setup_script_preamble(getattr(config, "setup_script", None)),
                het_group=het_group_for(node),
            )
            processes.append(
                ManagedProcess(
                    name=f"{self.process_name}_{idx}",
                    popen=proc,
                    log_file=router_log,
                    node=node,
                    critical=True,
                )
            )
        return processes
