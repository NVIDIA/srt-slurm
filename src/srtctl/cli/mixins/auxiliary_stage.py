# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auxiliary service stage mixin for ``SweepOrchestrator`` (the real multi-node sbatch path).

Generic, user-declared sidecar processes (``config.auxiliary_services``) launched once workers
and the frontend are healthy. Mirrors ``TelemetryStageMixin.start_tachometer()``'s shape: a
single ``srun`` launch on the head node, non-critical (a dead sidecar must never tear down the
benchmark), returning ``list[ManagedProcess]`` for the caller to register with the shared
``ProcessRegistry`` -- which already provides continuous crash detection (``start_process_monitor``)
and teardown (``registry.cleanup()``) for free, the same way it does for Tachometer and the
DCGM/node exporters.

This is the real-SLURM-path counterpart to ``render/direct_stages/auxiliary_stage.py``
(``AuxiliaryServiceStageMixin`` for the ``--bash`` dev-mode ``DirectRunner``). The two are
separate implementations because the dev-mode stage launches local subprocesses directly and
lives in a stdlib-only sibling package (no ``srtctl.core``/control-plane imports allowed), while
this one launches via ``srun`` and needs ``RuntimeContext``/``SrtConfig``. See
``docs/auxiliary-services.md``.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.processes import ManagedProcess
from srtctl.core.slurm import start_srun_process
from srtctl.ports import ETCD_CLIENT_PORT, NATS_PORT

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import AuxiliaryServiceConfig, SrtConfig

logger = logging.getLogger(__name__)


class AuxiliaryServiceStageMixin:
    """Launch user-declared ``auxiliary_services`` sidecars on the real sbatch/SLURM path."""

    config: SrtConfig
    runtime: RuntimeContext

    def _auxiliary_service_container_image(self, service: AuxiliaryServiceConfig) -> str:
        """Resolve the container a service launches in -- its own, or the job's main one."""
        return service.container_image or str(self.runtime.container_image)

    def _build_auxiliary_service_source(self, service: AuxiliaryServiceConfig) -> Path | None:
        """Clone ``service.source`` once on the head node, returning the build/launch dir.

        Runs on the bare host (``container_image=None``, same convention as
        ``_run_host_commands``), not inside the job container: git and network access to
        an arbitrary source host are host-level concerns, and the job container is not
        guaranteed to have git installed. Materialized under ``runtime.log_dir``, which
        is always mounted into every container as ``/logs`` (see ``RuntimeContext``), so
        the subsequent containerized build/launch step can still reach the checkout.
        """
        source = service.source
        if source is None:
            return None
        checkout_root = self.runtime.log_dir / "auxiliary_services" / service.name / "src"
        clone_log = self.runtime.log_dir / f"auxiliary_{service.name}.clone.out"
        clone_script = (
            f"set -e; mkdir -p {shlex.quote(str(checkout_root.parent))}; "
            f"if [ ! -d {shlex.quote(str(checkout_root))} ]; then "
            f"git clone --filter=blob:none {shlex.quote(source.git)} {shlex.quote(str(checkout_root))} && "
            f"git -C {shlex.quote(str(checkout_root))} fetch origin {shlex.quote(source.rev)} && "
            f"git -C {shlex.quote(str(checkout_root))} checkout FETCH_HEAD; "
            "fi"
        )
        logger.info("Cloning auxiliary service %s source: %s@%s", service.name, source.git, source.rev)
        proc = start_srun_process(
            command=["bash", "-c", clone_script],
            nodelist=[self.runtime.nodes.head],
            output=str(clone_log),
            container_image=None,  # bare host: git/network tooling, not the job container
            het_group=self.runtime.nodes.het_group_for(self.runtime.nodes.head),
        )
        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(
                f"auxiliary_services[{service.name}] source clone failed (exit {returncode}); see {clone_log}"
            )
        work_dir = checkout_root
        if source.path:
            work_dir = checkout_root / source.path
        return work_dir

    def _run_auxiliary_service_build(self, service: AuxiliaryServiceConfig, work_dir: Path) -> None:
        """Run ``build_command`` once, inside the service's container, from ``work_dir``."""
        if not service.build_command:
            return
        build_log = self.runtime.log_dir / f"auxiliary_{service.name}.build.out"
        logger.info("Building auxiliary service %s: %s", service.name, shlex.join(service.build_command))
        proc = start_srun_process(
            command=list(service.build_command),
            nodelist=[self.runtime.nodes.head],
            output=str(build_log),
            container_image=self._auxiliary_service_container_image(service),
            container_mounts=self.runtime.container_mounts,
            srun_options=self.runtime.srun_options,
            het_group=self.runtime.nodes.het_group_for(self.runtime.nodes.head),
            bash_preamble=f"cd {shlex.quote(str(work_dir))}",
        )
        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(
                f"auxiliary_services[{service.name}] build_command failed (exit {returncode}); see {build_log}"
            )

    def start_auxiliary_services(self) -> list[ManagedProcess]:
        """Launch every configured auxiliary service, once, on the head node.

        Single-node placement (the head node, same as Tachometer in
        ``TelemetryStageMixin.start_tachometer()``) keeps ``source``/``build_command``
        build-once semantics trivial: there is exactly one instance to build for, so no
        per-node duplication or cross-node build coordination is needed. Call this after
        workers and the frontend are confirmed healthy (see ``do_sweep.py``'s ``run()``).
        Services launch in the order they're declared in ``config.auxiliary_services``.
        """
        services = list(self.config.auxiliary_services)
        if not services:
            return []

        etcd_endpoints = f"http://{self.runtime.nodes.infra}:{ETCD_CLIENT_PORT}"
        nats_server = f"nats://{self.runtime.nodes.infra}:{NATS_PORT}"
        processes: list[ManagedProcess] = []

        for service in services:
            work_dir = self._build_auxiliary_service_source(service)
            if work_dir is not None:
                self._run_auxiliary_service_build(service, work_dir)

            env_to_set: dict[str, str] = {}
            if service.inherit_discovery_env:
                env_to_set["ETCD_ENDPOINTS"] = etcd_endpoints
                env_to_set["NATS_SERVER"] = nats_server
            env_to_set.update(service.env)

            log_file = self.runtime.log_dir / f"auxiliary_{service.name}.out"
            logger.info("Starting auxiliary service %s: %s", service.name, shlex.join(service.command))
            popen = start_srun_process(
                command=list(service.command),
                nodelist=[self.runtime.nodes.head],
                output=str(log_file),
                container_image=self._auxiliary_service_container_image(service),
                container_mounts=self.runtime.container_mounts,
                env_to_set=env_to_set,
                srun_options=self.runtime.srun_options,
                het_group=self.runtime.nodes.het_group_for(self.runtime.nodes.head),
                bash_preamble=(f"cd {shlex.quote(str(work_dir))}" if work_dir is not None else None),
            )
            processes.append(
                ManagedProcess(
                    name=f"auxiliary_{service.name}",
                    popen=popen,
                    log_file=log_file,
                    node=self.runtime.nodes.head,
                    # Best-effort by contract, same as Tachometer and the DCGM/node
                    # exporters: a dead sidecar costs its own log, not the run. Crash
                    # detection is the shared ProcessRegistry's background monitor
                    # thread (start_process_monitor), not a startup-only check.
                    critical=False,
                )
            )

        logger.info("Started %d auxiliary service(s)", len(processes))
        return processes
