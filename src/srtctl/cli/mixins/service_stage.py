# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service stage mixin for ``SweepOrchestrator``: launches the recipe's ``services:`` list.

Every service kind launches the same way: resolve the nodes its ``placement``
selects, optionally clone and build a ``source`` once, then one ``srun`` per
node with the kind's environment merged around the recipe's ``env``, an
optional TCP readiness gate, and a ``ManagedProcess`` for the shared
``ProcessRegistry`` (which provides crash detection and teardown). The kind
(``srtctl.services.registry.ServiceKind``) never launches anything itself.

``start_services("before_workers")`` runs after the Mooncake master and before
workers; ``start_services("after_frontend")`` runs once workers and the
frontend are healthy. See ``docs/services.md``.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.health import wait_for_port
from srtctl.core.processes import ManagedProcess
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.ports import ETCD_CLIENT_PORT, NATS_PORT
from srtctl.services.registry import ServiceLaunchContext, get_service_kind

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Endpoint
    from srtctl.services.config import ServiceConfig

logger = logging.getLogger(__name__)


def render_placeholders(value: str, replacements: dict[str, str]) -> str:
    """Substitute known ``{placeholder}`` names only, leaving unrelated braces (JSON) untouched."""
    for key, replacement in replacements.items():
        value = value.replace(f"{{{key}}}", replacement)
    return value


def _await_and_cd(work_dir: str) -> str:
    """``cd work_dir`` tolerating a brief lag before the shared mount shows the checkout."""
    quoted = shlex.quote(work_dir)
    return f"for _i in $(seq 1 20); do [ -d {quoted} ] && break; sleep 0.5; done; cd {quoted}"


class ServiceStageMixin:
    """Launch the recipe's ``services:`` entries on the sbatch/SLURM path."""

    config: SrtConfig
    runtime: RuntimeContext
    endpoints: list[Endpoint]

    # -- node resolution ---------------------------------------------------------

    def service_nodes(self, service: ServiceConfig) -> list[str]:
        """Physical nodes a service's ``placement`` selects, in allocation order, deduplicated."""
        where = service.placement.node
        if where == "head":
            return [self.runtime.nodes.head]
        if where == "infra":
            return [self.runtime.nodes.infra]
        if where == "workers":
            return list(self.runtime.nodes.worker)
        seen: dict[str, None] = {}
        for endpoint in self.endpoints:
            if endpoint.mode == where:
                for node in endpoint.nodes:
                    seen.setdefault(node, None)
        order = {node: i for i, node in enumerate(self.runtime.nodes.worker)}
        return sorted(seen, key=lambda n: order.get(n, len(order)))

    def _check_port_collisions(self, services: list[ServiceConfig]) -> None:
        """Two services that both listen on the same readiness port cannot share a node."""
        owners: dict[tuple[str, int], str] = {}
        for service in services:
            if service.readiness is None:
                continue
            for node in self.service_nodes(service):
                key = (node, service.readiness.port)
                other = owners.setdefault(key, service.name)
                if other != service.name:
                    raise ValueError(
                        f"services[{service.name}] and services[{other}] both listen on port "
                        f"{service.readiness.port} on node {node}; give them disjoint placements or ports"
                    )

    # -- source build ------------------------------------------------------------

    def _service_container(self, service: ServiceConfig) -> str:
        kind = get_service_kind(service.type)
        return service.container or kind.container_fallback(self.config) or str(self.runtime.container_image)

    def _container_path(self, host_path: Path) -> str:
        """Host path under ``log_dir`` as seen inside a container (``log_dir`` is mounted at ``/logs``)."""
        return str(Path("/logs") / host_path.relative_to(self.runtime.log_dir))

    def _clone_service_source(self, service: ServiceConfig, node: str) -> Path | None:
        """Clone ``service.source`` once on the bare host of ``node``; returns the work dir (host path)."""
        source = service.source
        if source is None:
            return None
        checkout_root = self.runtime.log_dir / "services" / service.name / "src"
        clone_log = self.runtime.log_dir / f"service_{service.name}.clone.out"
        # HTTP/1.1 and no terminal prompt guard against the intermittent smart-HTTP stalls
        # seen cloning github.com from compute nodes; 600s covers slow checkouts onto /logs.
        git = "GIT_TERMINAL_PROMPT=0 timeout 600s git -c http.version=HTTP/1.1"
        root = shlex.quote(str(checkout_root))
        clone_script = (
            f"set -e; mkdir -p {shlex.quote(str(checkout_root.parent))}; "
            f"if [ ! -d {root} ]; then "
            f"{git} clone --filter=blob:none {shlex.quote(source.git)} {root} && "
            f"{git} -C {root} fetch origin {shlex.quote(source.rev)} && "
            f"{git} -C {root} checkout FETCH_HEAD; "
            "fi"
        )
        logger.info("Cloning service %s source %s@%s on %s", service.name, source.git, source.rev, node)
        proc = start_srun_process(
            command=["bash", "-c", clone_script],
            nodelist=[node],
            output=str(clone_log),
            container_image=None,  # bare host: git and network access are host concerns
            het_group=self.runtime.nodes.het_group_for(node),
        )
        if proc.wait() != 0:
            raise RuntimeError(
                f"services[{service.name}] source clone failed (exit {proc.returncode}); see {clone_log}"
            )
        return checkout_root / source.path if source.path else checkout_root

    def _build_service_source(self, service: ServiceConfig, node: str, work_dir: Path) -> None:
        if not service.build_command:
            return
        build_log = self.runtime.log_dir / f"service_{service.name}.build.out"
        logger.info("Building service %s: %s", service.name, shlex.join(service.build_command))
        proc = start_srun_process(
            command=list(service.build_command),
            nodelist=[node],
            output=str(build_log),
            container_image=self._service_container(service),
            container_mounts=self.runtime.container_mounts,
            srun_options=self.runtime.srun_options,
            het_group=self.runtime.nodes.het_group_for(node),
            bash_preamble=_await_and_cd(self._container_path(work_dir)),
        )
        if proc.wait() != 0:
            raise RuntimeError(
                f"services[{service.name}] build_command failed (exit {proc.returncode}); see {build_log}"
            )

    # -- launch ------------------------------------------------------------------

    def _service_environment(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> dict[str, str]:
        kind = get_service_kind(service.type)
        template = ctx.template_vars()
        env: dict[str, str] = {}
        if service.inherit_discovery_env:
            env["ETCD_ENDPOINTS"] = f"http://{self.runtime.nodes.infra}:{ETCD_CLIENT_PORT}"
            env["NATS_SERVER"] = f"nats://{self.runtime.nodes.infra}:{NATS_PORT}"
        env.update(kind.default_environment(service, ctx))
        env.update({k: render_placeholders(v, template) for k, v in service.env.items()})
        env.update(kind.forced_environment(service, ctx))
        return env

    def _launch_service_instance(
        self, service: ServiceConfig, ctx: ServiceLaunchContext, work_dir: Path | None, instances: int
    ) -> ManagedProcess:
        template = ctx.template_vars()
        command = [render_placeholders(part, template) for part in service.effective_command]
        preamble_parts: list[str] = []
        if work_dir is not None:
            preamble_parts.append(_await_and_cd(self._container_path(work_dir)))
        if service.preamble:
            preamble_parts.append(render_placeholders(service.preamble, template).rstrip())
        suffix = f"_{ctx.node}" if instances > 1 else ""
        log_file = self.runtime.log_dir / f"service_{service.name}{suffix}.out"

        logger.info("Starting service %s (%s) on %s: %s", service.name, service.type, ctx.node, shlex.join(command))
        popen = start_srun_process(
            command=command,
            nodelist=[ctx.node],
            output=str(log_file),
            container_image=self._service_container(service),
            container_mounts=self.runtime.container_mounts,
            env_to_set=self._service_environment(service, ctx),
            bash_preamble="; ".join(preamble_parts) or None,
            cpus_per_task=service.cpus_per_task,
            cpu_bind=service.cpu_bind,
            srun_options={**self.runtime.srun_options, **service.srun_options},
            het_group=self.runtime.nodes.het_group_for(ctx.node),
        )
        return ManagedProcess(
            name=f"service_{service.name}{suffix}",
            popen=popen,
            log_file=log_file,
            node=ctx.node,
            critical=service.effective_critical,
        )

    def start_services(self, start: str) -> list[ManagedProcess]:
        """Launch every service whose (effective) ``start`` matches, in declaration order.

        Readiness gates block per node; a gate that times out terminates every
        process this call started and raises. Returned processes are for the
        caller to register with the shared ``ProcessRegistry``.
        """
        services = [s for s in self.config.services if s.effective_start == start]
        if not services:
            return []
        self._check_port_collisions(list(self.config.services))

        worker_order = {node: i for i, node in enumerate(self.runtime.nodes.worker)}
        started: list[ManagedProcess] = []
        try:
            for service in services:
                nodes = self.service_nodes(service)
                if not nodes:
                    logger.warning(
                        "services[%s]: placement.node=%s selects no nodes in this allocation; skipping",
                        service.name,
                        service.placement.node,
                    )
                    continue
                work_dir = self._clone_service_source(service, nodes[0])
                if work_dir is not None:
                    self._build_service_source(service, nodes[0], work_dir)

                for index, node in enumerate(nodes):
                    ctx = ServiceLaunchContext(
                        runtime=self.runtime,
                        node=node,
                        node_ip=get_hostname_ip(node, self.runtime.network_interface),
                        node_id=worker_order.get(node, index),
                        index=index,
                        role=service.placement.node,
                    )
                    started.append(self._launch_service_instance(service, ctx, work_dir, len(nodes)))
                    readiness = service.readiness
                    if readiness is not None:
                        logger.info(
                            "Waiting for service %s on %s (port %d, timeout %ds)",
                            service.name,
                            node,
                            readiness.port,
                            readiness.timeout_seconds,
                        )
                        if not wait_for_port(node, readiness.port, timeout=readiness.timeout_seconds):
                            raise RuntimeError(
                                f"services[{service.name}] did not open port {readiness.port} on {node} within "
                                f"{readiness.timeout_seconds}s; see {started[-1].log_file}"
                            )
                logger.info("Service %s ready on %d node(s)", service.name, len(nodes))
        except Exception:
            for proc in started:
                proc.terminate()
            raise
        return started
