# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry stage mixin for SweepOrchestrator."""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from srtctl.core.processes import ManagedProcess
from srtctl.core.slurm import start_srun_process
from srtctl.core.telemetry import generate_telemetry_config

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig, TelemetryExporterConfig
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


class TelemetryStageMixin:
    """Mixin for telemetry startup stage."""

    config: SrtConfig
    runtime: RuntimeContext

    @property
    def backend_processes(self) -> list[Process]:
        """Backend worker processes."""
        raise NotImplementedError

    def _compute_frontend_topology(self) -> Any:
        """Frontend topology helper provided by FrontendStageMixin."""
        raise NotImplementedError

    def _start_exporter_container(
        self,
        *,
        exporter_config: TelemetryExporterConfig,
        name: str,
        nodelist: list[str],
        log_file: Path,
        default_command_template: str,
    ) -> ManagedProcess:
        """Start one exporter container across the requested nodes."""
        if exporter_config.command is None:
            cmd_str = default_command_template.format(port=exporter_config.port)
        elif "{port}" in exporter_config.command:
            cmd_str = exporter_config.command.format(port=exporter_config.port)
        else:
            cmd_str = exporter_config.command

        # Exporter images can be distroless (e.g. prom/node-exporter has no
        # shell), so we run the binary directly via execve instead of the
        # default bash wrapper. Pass nodes=len(nodelist) so SLURM places one
        # task per node (default nodes=1 contradicts a multi-node nodelist).
        proc = start_srun_process(
            command=shlex.split(cmd_str),
            nodes=len(nodelist),
            ntasks=len(nodelist),
            nodelist=nodelist,
            output=str(log_file),
            container_image=exporter_config.container_image,
            container_mounts=self.runtime.container_mounts,
            srun_options=self.runtime.srun_options,
            use_bash_wrapper=False,
        )
        return ManagedProcess(
            name=name,
            popen=proc,
            log_file=log_file,
            node=",".join(nodelist),
        )

    def start_telemetry(self) -> list[ManagedProcess]:
        """Start the configured telemetry provider.

        ``dcgm_exporter`` is always launched when telemetry is enabled. The
        ``node_exporter`` and the scraper (``container_image``) are optional —
        each only runs when configured.
        """
        telemetry = self.config.telemetry
        if not telemetry.enabled:
            logger.info("Telemetry disabled")
            return []
        if telemetry.dcgm_exporter is None:
            raise ValueError("Telemetry is enabled but telemetry.dcgm_exporter is not configured")

        logger.info("Starting telemetry provider: %s", telemetry.provider.value)

        worker_nodes = sorted({process.node for process in self.backend_processes})
        processes: list[ManagedProcess] = []
        processes.append(
            self._start_exporter_container(
                exporter_config=telemetry.dcgm_exporter,
                name="telemetry_dcgm_exporter",
                nodelist=worker_nodes,
                log_file=self.runtime.log_dir / "telemetry_dcgm_exporter.out",
                default_command_template="dcgm-exporter --collect-interval=100 --address :{port}",
            )
        )

        if telemetry.node_exporter is not None:
            processes.append(
                self._start_exporter_container(
                    exporter_config=telemetry.node_exporter,
                    name="telemetry_node_exporter",
                    nodelist=worker_nodes,
                    log_file=self.runtime.log_dir / "telemetry_node_exporter.out",
                    default_command_template=(
                        "/bin/node_exporter --web.listen-address=:{port} "
                        "--collector.disable-defaults --collector.cpu --collector.infiniband --collector.meminfo"
                    ),
                )
            )

        if telemetry.container_image is None:
            logger.info("Telemetry scraper not configured; running exporters only")
            return processes

        topology = self._compute_frontend_topology()
        config_path = self.runtime.log_dir / "telemetry_config.toml"
        config_path.write_text(
            generate_telemetry_config(
                processes=self.backend_processes,
                frontend_topology=topology,
                runtime=self.runtime,
                telemetry=telemetry,
            )
        )

        telemetry_dir = self.runtime.log_dir / telemetry.storage_subdir / self.runtime.job_id
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        local_dir = telemetry_dir / "local"
        local_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            telemetry.binary_path,
            "--config",
            "/telemetry_config.toml",
            "--local-dir",
            f"/logs/{telemetry.storage_subdir}/{self.runtime.job_id}/local",
        ]
        if telemetry.sync_interval_secs > 0:
            cmd.extend(["--sync-interval", str(telemetry.sync_interval_secs)])

        env_to_set: dict[str, str] = {}
        if telemetry.compaction_threads > 0:
            env_to_set["POLARS_MAX_THREADS"] = str(telemetry.compaction_threads)

        scraper_mounts = self.runtime.container_mounts | {
            config_path: Path("/telemetry_config.toml"),
        }
        processes.append(
            ManagedProcess(
                name="telemetry",
                popen=start_srun_process(
                    command=cmd,
                    nodelist=[self.runtime.nodes.head],
                    output=str(self.runtime.log_dir / "telemetry.out"),
                    container_image=telemetry.container_image,
                    container_mounts=scraper_mounts,
                    env_to_set=env_to_set,
                    srun_options=self.runtime.srun_options,
                ),
                log_file=self.runtime.log_dir / "telemetry.out",
                node=self.runtime.nodes.head,
            )
        )
        logger.info("Telemetry scraper started with artifacts under %s", telemetry_dir)
        return processes
