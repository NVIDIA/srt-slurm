# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service stage mixin for ``SweepOrchestrator``: launches the job's services.

Every service kind launches the same way: resolve the nodes its ``placement``
selects, optionally clone and build a ``source`` once, then one ``srun`` per
node with the kind's command and environment merged around the recipe's, an
optional readiness probe (tcp, http, or log; or the kind's default ports), and a
``ManagedProcess`` for the shared ``ProcessRegistry`` (which provides crash
detection and teardown). The kind (``srtctl.services.registry.ServiceKind``)
never launches anything itself.

The list launched is ``effective_services(config)``: what the recipe declares
plus what it implies (etcd and NATS under the Dynamo frontend, the Mooncake
master for ``backend.mooncake_kv_store``, tachometer's default exporters), with
a declared entry of the same name taking over the implicit one.

Nothing a service launches may outlive the job. Every srun this stage starts,
including the one-shot clone and build steps, is registered with the
``ProcessRegistry`` the moment it exists, so ``registry.cleanup()`` (normal
exit, a failed stage, the SIGTERM handler, the crash monitor) reaches it
without depending on this stage returning. The clone and build steps also
run under a wall-clock timeout so a hung build cannot hold the allocation.
Long-running services are named Slurm steps, so cleanup delivers SIGTERM
through ``scancel --signal`` and they get to flush state before exit.

Phases, in run order: ``start_services("infra")`` (the discovery plane),
``start_services("before_workers")`` (after infra, before any worker),
``start_services("after_frontend")`` (once workers and the frontend are
healthy). See ``docs/services.md``.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.processes import ManagedProcess, ProcessRegistry, terminate_and_reap
from srtctl.core.readiness import ProcessDied, wait_until_ready
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.services.config import ServiceReadinessConfig, TcpProbe
from srtctl.services.implicit import discovery_env, effective_services
from srtctl.services.registry import ServiceLaunchContext, get_service_kind

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Endpoint
    from srtctl.services.config import ServiceConfig

logger = logging.getLogger(__name__)

# The clone script runs three git commands, each under its own `timeout 600s`.
CLONE_TIMEOUT_SECONDS = 3 * 600 + 60
# Long-running services flush state on SIGTERM (etcd its WAL, a scraper its parquet).
SERVICE_TERMINATE_TIMEOUT_SECONDS = 30.0
# The discovery plane itself must not be told where the discovery plane is.
_DISCOVERY_KINDS = frozenset({"etcd", "nats"})


def render_placeholders(value: str, replacements: dict[str, str]) -> str:
    """Substitute known ``{placeholder}`` names only, leaving unrelated braces (JSON) untouched."""
    for key, replacement in replacements.items():
        value = value.replace(f"{{{key}}}", replacement)
    return value


def _await_and_cd(work_dir: str) -> str:
    """``cd work_dir`` tolerating a brief lag before the shared mount shows the checkout."""
    quoted = shlex.quote(work_dir)
    return f"for _i in $(seq 1 20); do [ -d {quoted} ] && break; sleep 0.5; done; cd {quoted}"


def service_step_name(service: ServiceConfig, node: str, instances: int) -> str:
    """Slurm step name (and log stem) for one instance of a service."""
    suffix = f"_{node}" if instances > 1 else ""
    return f"service_{service.name}{suffix}"


class ServiceStageMixin:
    """Launch the job's effective services on the sbatch/SLURM path."""

    config: SrtConfig
    runtime: RuntimeContext
    endpoints: list[Endpoint]

    # -- node resolution ---------------------------------------------------------

    def service_nodes(self, service: ServiceConfig) -> list[str]:
        """Physical nodes a service's ``placement`` selects, in allocation order, deduplicated."""
        where = service.effective_placement
        if where == "head":
            return [self.runtime.nodes.head]
        if where in ("infra", "dedicated"):
            # `dedicated` reserves the infra node (Nodes.from_slurm); both resolve there.
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

    @staticmethod
    def _readiness_ports(service: ServiceConfig) -> tuple[int, ...]:
        """Ports the service is known to listen on: its probe's, or the kind's defaults."""
        if service.readiness is not None:
            port = service.readiness.probe_port
            return (port,) if port is not None else ()
        return get_service_kind(service.type).default_readiness_ports

    def _check_port_collisions(self, services: list[ServiceConfig]) -> None:
        """Two services that both listen on the same readiness port cannot share a node."""
        owners: dict[tuple[str, int], str] = {}
        for service in services:
            for port in self._readiness_ports(service):
                for node in self.service_nodes(service):
                    key = (node, port)
                    other = owners.setdefault(key, service.name)
                    if other != service.name:
                        raise ValueError(
                            f"services[{service.name}] and services[{other}] both listen on port "
                            f"{port} on node {node}; give them disjoint placements or ports"
                        )

    # -- one-shot steps (clone, build) ----------------------------------------------

    @staticmethod
    def _run_step(
        step: ManagedProcess, *, timeout: float, registry: ProcessRegistry | None, what: str, log: Path
    ) -> None:
        """Wait for a one-shot srun, tracked and bounded.

        Registered before waiting so a signal or a crash elsewhere tears it down
        with everything else; killed on timeout so a hung step cannot hold the
        allocation until walltime.
        """
        if registry is not None:
            registry.add_process(step)
        try:
            returncode = step.popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_and_reap(step.popen)
            raise RuntimeError(f"{what} timed out after {int(timeout)}s and was killed; see {log}") from None
        if returncode != 0:
            raise RuntimeError(f"{what} failed (exit {returncode}); see {log}")

    def _service_container(self, service: ServiceConfig) -> str:
        kind = get_service_kind(service.type)
        return service.container or kind.container_fallback(self.config) or str(self.runtime.container_image)

    def _container_path(self, host_path: Path) -> str:
        """Host path under ``log_dir`` as seen inside a container (``log_dir`` is mounted at ``/logs``)."""
        return str(Path("/logs") / host_path.relative_to(self.runtime.log_dir))

    def _clone_service_source(self, service: ServiceConfig, node: str, registry: ProcessRegistry | None) -> Path | None:
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
            f"{git} -C {root} fetch origin {shlex.quote(source.checkout)} && "
            f"{git} -C {root} checkout FETCH_HEAD; "
            "fi"
        )
        logger.info("Cloning service %s source %s@%s on %s", service.name, source.git, source.checkout, node)
        popen = start_srun_process(
            command=["bash", "-c", clone_script],
            nodelist=[node],
            output=str(clone_log),
            container_image=None,  # bare host: git and network access are host concerns
            het_group=self.runtime.nodes.het_group_for(node),
        )
        step = ManagedProcess(
            name=f"service_{service.name}.clone", popen=popen, log_file=clone_log, node=node, critical=False
        )
        self._run_step(
            step,
            timeout=CLONE_TIMEOUT_SECONDS,
            registry=registry,
            what=f"services[{service.name}] source clone",
            log=clone_log,
        )
        return checkout_root / source.path if source.path else checkout_root

    def _build_service_source(
        self, service: ServiceConfig, node: str, work_dir: Path, registry: ProcessRegistry | None
    ) -> None:
        if not service.build_command:
            return
        build_log = self.runtime.log_dir / f"service_{service.name}.build.out"
        logger.info("Building service %s: %s", service.name, shlex.join(service.build_command))
        popen = start_srun_process(
            command=list(service.build_command),
            nodelist=[node],
            output=str(build_log),
            container_image=self._service_container(service),
            container_mounts=self.runtime.container_mounts,
            srun_options=self.runtime.srun_options,
            het_group=self.runtime.nodes.het_group_for(node),
            bash_preamble=_await_and_cd(self._container_path(work_dir)),
        )
        step = ManagedProcess(
            name=f"service_{service.name}.build", popen=popen, log_file=build_log, node=node, critical=False
        )
        self._run_step(
            step,
            timeout=service.build_timeout_seconds,
            registry=registry,
            what=f"services[{service.name}] build_command",
            log=build_log,
        )

    # -- launch ------------------------------------------------------------------

    def _service_environment(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> dict[str, str]:
        kind = get_service_kind(service.type)
        template = ctx.template_vars()
        env: dict[str, str] = {}
        if service.inherit_discovery_env and service.type not in _DISCOVERY_KINDS:
            env.update(discovery_env(self.config, self.runtime))
        env.update(kind.default_environment(service, ctx))
        env.update({k: render_placeholders(v, template) for k, v in service.env.items()})
        env.update(kind.forced_environment(service, ctx))
        return env

    def _launch_service_instance(
        self, service: ServiceConfig, ctx: ServiceLaunchContext, work_dir: Path | None, instances: int
    ) -> ManagedProcess:
        kind = get_service_kind(service.type)
        template = ctx.template_vars()
        command = [render_placeholders(part, template) for part in kind.build_command(service, ctx)]
        preamble_parts: list[str] = []
        if work_dir is not None:
            preamble_parts.append(_await_and_cd(self._container_path(work_dir)))
        kind_preamble = kind.preamble(service, ctx)
        if kind_preamble:
            preamble_parts.append(render_placeholders(kind_preamble, template))
        if service.preamble:
            preamble_parts.append(render_placeholders(service.preamble, template).rstrip())
        step_name = service_step_name(service, ctx.node, instances)
        log_file = self.runtime.log_dir / f"{step_name}.out"

        env = self._service_environment(service, ctx)
        logger.info("Starting service %s (%s) on %s: %s", service.name, service.type, ctx.node, shlex.join(command))
        popen = start_srun_process(
            command=command,
            nodelist=[ctx.node],
            output=str(log_file),
            container_image=self._service_container(service),
            container_mounts=self.runtime.container_mounts,
            # Without the bash wrapper there is no `export`; srun --export carries the env instead.
            env_to_set=env if kind.use_bash_wrapper else None,
            srun_export_env=None if kind.use_bash_wrapper else env,
            bash_preamble=("; ".join(preamble_parts) or None) if kind.use_bash_wrapper else None,
            cpus_per_task=service.cpus_per_task,
            cpu_bind=service.cpu_bind,
            srun_options={**self.runtime.srun_options, **service.srun_options},
            het_group=self.runtime.nodes.het_group_for(ctx.node),
            use_bash_wrapper=kind.use_bash_wrapper,
            step_name=step_name,
        )
        return ManagedProcess(
            name=step_name,
            popen=popen,
            log_file=log_file,
            node=ctx.node,
            critical=service.effective_critical,
            terminate_timeout=SERVICE_TERMINATE_TIMEOUT_SECONDS,
            step_name=step_name,
        )

    @staticmethod
    def _wait_ready(proc: ManagedProcess, service: ServiceConfig, readiness: ServiceReadinessConfig) -> None:
        """Block until the service's readiness probe passes, failing fast if the process dies first."""
        assert proc.node is not None
        logger.info("Waiting for service %s on %s (%s)", service.name, proc.node, readiness.describe())
        try:
            ready = wait_until_ready(
                readiness.probe,
                host=proc.node,
                log_file=proc.log_file,
                timeout=readiness.timeout_seconds,
                interval=readiness.interval_seconds,
                is_alive=lambda: proc.is_running,
            )
        except ProcessDied:
            raise RuntimeError(
                f"services[{service.name}] exited with code {proc.exit_code} on {proc.node} before its readiness "
                f"probe passed ({readiness.describe()}); see {proc.log_file}"
            ) from None
        if not ready:
            raise RuntimeError(
                f"services[{service.name}] on {proc.node} was not ready within {readiness.timeout_seconds}s "
                f"({readiness.describe()}); see {proc.log_file}"
            )

    def _wait_service_ready(self, proc: ManagedProcess, service: ServiceConfig) -> None:
        """The recipe's ``readiness`` probe, or each default port of the kind in turn."""
        if service.readiness is not None:
            self._wait_ready(proc, service, service.readiness)
            return
        kind = get_service_kind(service.type)
        for port in kind.default_readiness_ports:
            probe = ServiceReadinessConfig(tcp=TcpProbe(port=port), timeout_seconds=kind.default_readiness_timeout)
            self._wait_ready(proc, service, probe)

    def start_services(self, start: str, registry: ProcessRegistry | None = None) -> list[ManagedProcess]:
        """Launch every effective service whose ``start`` phase matches: implicit ones first, then declared.

        Each process is added to ``registry`` as soon as its srun exists, so a
        readiness wait interrupted by a signal still leaves nothing untracked.
        A readiness gate that fails terminates every process this call started
        and raises. The started processes are also returned. A service with
        ``external`` set launches nothing; its address is injected instead.
        """
        effective = [entry for entry in effective_services(self.config) if not entry.service.external]
        phase = [entry for entry in effective if entry.service.effective_start == start]
        if not phase:
            return []
        self._check_port_collisions([entry.service for entry in effective])

        worker_order = {node: i for i, node in enumerate(self.runtime.nodes.worker)}
        started: list[ManagedProcess] = []
        try:
            for entry in phase:
                service = entry.service
                nodes = self.service_nodes(service)
                if not nodes:
                    logger.warning(
                        "services[%s]: placement.node=%s selects no nodes in this allocation; skipping",
                        service.name,
                        service.effective_placement,
                    )
                    continue
                if entry.implicit:
                    logger.info("Service %s (%s) implied by %s", service.name, service.type, entry.reason)
                work_dir = self._clone_service_source(service, nodes[0], registry)
                if work_dir is not None:
                    self._build_service_source(service, nodes[0], work_dir, registry)

                for index, node in enumerate(nodes):
                    ctx = ServiceLaunchContext(
                        runtime=self.runtime,
                        node=node,
                        node_ip=get_hostname_ip(node, self.runtime.network_interface),
                        node_id=worker_order.get(node, index),
                        index=index,
                        role=service.effective_placement,
                    )
                    proc = self._launch_service_instance(service, ctx, work_dir, len(nodes))
                    started.append(proc)
                    if registry is not None:
                        registry.add_process(proc)
                    self._wait_service_ready(proc, service)
                logger.info("Service %s ready on %d node(s)", service.name, len(nodes))
        except BaseException:
            # Belt and braces: the registry already tracks these, but terminate
            # here too so a failure inside this stage never depends on the caller.
            for proc in started:
                proc.terminate()
            raise
        return started
