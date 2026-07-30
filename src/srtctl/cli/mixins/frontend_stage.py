# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Frontend stage mixin for SweepOrchestrator.

Handles frontend/router and nginx startup.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from srtctl.core.processes import ManagedProcess
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.frontends import get_frontend
from srtctl.ports import FRONTEND_INTERNAL_PORT, FRONTEND_PUBLIC_PORT

if TYPE_CHECKING:
    from srtctl.core.processes import ProcessRegistry
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


@dataclass
class FrontendTopology:
    """Describes where nginx and frontends should run.

    Topology rules:
    - Single node OR multiple_frontends disabled: 1 frontend on head, no nginx
    - 2+ nodes AND multiple_frontends enabled: nginx on head, frontends on other nodes
    - no-nginx multiple frontends: N frontends on nodes round-robin, per-node port offsets
      allow N > len(nodes) without conflicts (first frontend on a node gets 8000,
      second gets 8001, etc.)
    """

    nginx_node: str | None  # Node running nginx, or None if no nginx
    frontend_nodes: list[str]  # Nodes running frontends (parallel to frontend_ports)
    frontend_ports: list[int]  # Per-frontend listen port (parallel to frontend_nodes)
    frontend_port: int  # Canonical port: FRONTEND_INTERNAL_PORT for nginx, PUBLIC otherwise
    public_port: int  # Public-facing port (nginx or direct frontend)

    @property
    def uses_nginx(self) -> bool:
        """Whether this topology uses nginx."""
        return self.nginx_node is not None


class FrontendStageMixin:
    """Mixin for frontend/nginx startup stage.

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
        self.backend: BackendProtocol
        self.backend_processes: list[Process]
    """

    # Type hints for mixin dependencies
    config: "SrtConfig"
    runtime: "RuntimeContext"

    @property
    def backend(self) -> Any:
        """Access the backend config (implements BackendProtocol)."""
        return self.config.backend

    @property
    def backend_processes(self) -> list["Process"]:
        """Compute physical process topology from endpoints (cached)."""
        raise NotImplementedError

    def _compute_frontend_topology(self) -> FrontendTopology:
        """Determine where nginx and frontends should run.

        Topology rules:
        - Single node OR multiple_frontends disabled: 1 frontend on head, no nginx
        - 2+ nodes AND multiple_frontends enabled: nginx on head, frontends on other nodes

        Returns:
            FrontendTopology describing where to run nginx and frontends.
        """
        nodes = self.runtime.nodes.worker
        head = self.runtime.nodes.head
        fe_config = self.config.frontend

        # Multiple frontends WITHOUT nginx: applies when use_nginx=False (explicit opt-out)
        # or when frontend type is trtllm_serve (nginx is incompatible with its protocol).
        # N orchestrators run directly on N worker nodes; srtctl writes
        # /logs/frontend_urls.txt so the benchmark can discover all endpoints.
        no_nginx = not fe_config.use_nginx or fe_config.type == "trtllm_serve"
        if no_nginx and fe_config.enable_multiple_frontends and len(nodes) > 1:
            n_frontends = fe_config.num_additional_frontends + 1
            all_nodes = [head] + [n for n in nodes if n != head]
            # Assign frontends round-robin across available nodes. When n_frontends >
            # len(all_nodes), multiple frontends land on the same node; each gets an
            # incremented port (8000, 8001, …) so there are no conflicts.
            port_offset: dict[str, int] = {}
            assigned_nodes: list[str] = []
            assigned_ports: list[int] = []
            for i in range(n_frontends):
                node = all_nodes[i % len(all_nodes)]
                offset = port_offset.get(node, 0)
                assigned_nodes.append(node)
                assigned_ports.append(FRONTEND_PUBLIC_PORT + offset)
                port_offset[node] = offset + 1
            logger.info(
                "Frontend topology (no nginx): %d frontends → %s",
                n_frontends,
                list(zip(assigned_nodes, assigned_ports)),
            )
            return FrontendTopology(
                nginx_node=None,
                frontend_nodes=assigned_nodes,
                frontend_ports=assigned_ports,
                frontend_port=FRONTEND_PUBLIC_PORT,
                public_port=FRONTEND_PUBLIC_PORT,
            )

        # Single node or multiple frontends disabled: single frontend, no nginx.
        # The orchestrator node honors frontend.orchestrator_placement (default
        # "head" -> unchanged; "first_decode" -> first GEN worker-leader node).
        if len(nodes) == 1 or not fe_config.enable_multiple_frontends:
            placement = getattr(fe_config, "orchestrator_placement", "head")
            if placement == "head":
                orchestrator_node = head
            else:
                from srtctl.core.topology import placed_node

                orchestrator_node = placed_node(
                    self.backend_processes, placement, head, kind="frontend.orchestrator_placement"
                )
            return FrontendTopology(
                nginx_node=None,
                frontend_nodes=[orchestrator_node],
                frontend_ports=[FRONTEND_PUBLIC_PORT],
                frontend_port=FRONTEND_PUBLIC_PORT,
                public_port=FRONTEND_PUBLIC_PORT,
            )

        # Multiple nodes with multiple frontends enabled:
        # nginx on head, frontends on other nodes
        other_nodes = [n for n in nodes if n != head]

        # Limit number of frontends based on config (num_additional_frontends is extra beyond first)
        max_frontends = min(
            fe_config.num_additional_frontends + 1,
            len(other_nodes),
        )
        frontend_nodes = other_nodes[:max_frontends]

        logger.info(
            "Frontend topology: nginx on %s, %d frontends on %s",
            head,
            len(frontend_nodes),
            frontend_nodes,
        )

        return FrontendTopology(
            nginx_node=head,
            frontend_nodes=frontend_nodes,
            frontend_ports=[FRONTEND_INTERNAL_PORT] * len(frontend_nodes),
            frontend_port=FRONTEND_INTERNAL_PORT,
            public_port=FRONTEND_PUBLIC_PORT,
        )

    def _start_nginx(self, topology: FrontendTopology) -> ManagedProcess:
        """Start nginx load balancer on the designated node."""
        assert topology.nginx_node is not None
        logger.info("Starting nginx on %s", topology.nginx_node)

        nginx_log = self.runtime.log_dir / f"{topology.nginx_node}_nginx.out"

        # Generate nginx config from template
        nginx_config = self._generate_nginx_config(topology)
        nginx_config_path = self.runtime.log_dir / "nginx.conf"
        nginx_config_path.write_text(nginx_config)
        logger.debug("Nginx config written to %s", nginx_config_path)

        # Install nginx and run it (daemon off keeps nginx in foreground so srun can manage it)
        # Use container path (/logs) since log_dir is mounted there
        container_config_path = "/logs/nginx.conf"
        # Optional ulimit: use_bash_wrapper=False bypasses default_bash_preamble;
        # some clusters reject raising nofile inside the nginx container.
        fe = self.config.frontend
        inner = (
            f"ulimit -n 1048576 && nginx -c {container_config_path} -g 'daemon off;'"
            if fe.nginx_raise_ulimit
            else f"nginx -c {container_config_path} -g 'daemon off;'"
        )
        cmd = ["bash", "-c", inner]

        proc = start_srun_process(
            command=cmd,
            nodelist=[topology.nginx_node],
            output=str(nginx_log),
            container_image=self.config.frontend.nginx_container,
            container_mounts=self.runtime.container_mounts,
            use_bash_wrapper=False,  # Already wrapped in bash -c
            srun_options={
                "container-remap-root": "",
            },
            het_group=self.runtime.nodes.het_group_for(topology.nginx_node),
        )

        return ManagedProcess(
            name="nginx",
            popen=proc,
            log_file=nginx_log,
            node=topology.nginx_node,
            critical=True,
        )

    def _generate_nginx_config(self, topology: FrontendTopology) -> str:
        """Generate nginx configuration from template."""
        from jinja2 import Environment, FileSystemLoader

        template_dir = Path(__file__).parent.parent.parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template("nginx.conf.j2")

        # Get IPs for frontend nodes
        frontend_hosts = [get_hostname_ip(node) for node in topology.frontend_nodes]

        return template.render(
            frontend_hosts=frontend_hosts,
            backend_port=topology.frontend_port,
            listen_port=topology.public_port,
            nginx_raise_ulimit=self.config.frontend.nginx_raise_ulimit,
            nginx_session_affinity=self.config.frontend.nginx_session_affinity,
            nginx_session_affinity_header=self.config.frontend.nginx_session_affinity_header,
        )

    def start_frontend(
        self, registry: "ProcessRegistry", stop_event: "threading.Event | None" = None
    ) -> list[ManagedProcess]:
        """Start the frontend layer (nginx + frontends if applicable).

        Args:
            registry: Process registry.
            stop_event: Optional event to abort readiness waits a frontend performs
                while starting (e.g. trtllm_serve waiting for workers).

        Returns:
            List of ManagedProcess instances for all frontend processes.
        """
        logger.info("Starting frontend layer")
        topology = self._compute_frontend_topology()
        processes: list[ManagedProcess] = []

        # Start nginx if topology requires it
        if topology.uses_nginx:
            nginx_proc = self._start_nginx(topology)
            processes.append(nginx_proc)

        # Get frontend implementation based on config type
        frontend_impl = get_frontend(self.config.frontend.type)
        frontend_procs = frontend_impl.start_frontends(
            topology=topology,
            runtime=self.runtime,
            config=self.config,
            backend=self.backend,
            backend_processes=self.backend_processes,
            stop_event=stop_event,
        )
        processes.extend(frontend_procs)

        # Write /logs/frontend_urls.txt only for no-nginx multi-frontend deployments
        # so the benchmark can discover all endpoint IPs. Single-frontend and nginx
        # topologies use the old approach (benchmark targets localhost:8000 or nginx IP
        # directly — no file needed).
        if not topology.uses_nginx and len(topology.frontend_nodes) > 1:
            urls = [
                f"http://{get_hostname_ip(n)}:{p}"
                for n, p in zip(topology.frontend_nodes, topology.frontend_ports)
            ]
            url_file = self.runtime.log_dir / "frontend_urls.txt"
            url_file.write_text("\n".join(urls) + "\n")
            logger.info("Frontend URLs written to %s: %s", url_file, urls)

        return processes
