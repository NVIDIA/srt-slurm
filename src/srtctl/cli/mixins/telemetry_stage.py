# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry stage mixin for SweepOrchestrator."""

from __future__ import annotations

import logging
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from srtctl.core.git_state import head_commit
from srtctl.core.power.contract import Reason
from srtctl.core.power.manifest import ExpectedWindow
from srtctl.core.power.session import PowerSessionSettings, PowerTelemetrySession
from srtctl.core.power.topology import build_expected_devices
from srtctl.core.processes import ManagedProcess, ProcessRegistry
from srtctl.core.schema import TelemetryExporterConfig
from srtctl.core.slurm import start_srun_process
from srtctl.core.telemetry import generate_tachometer_config

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)

DCGM_EXPORTER_COMMAND_TEMPLATE = "dcgm-exporter --collect-interval=100 --address :{port}"


def resolve_exporter_command(exporter_config: TelemetryExporterConfig, default_template: str) -> str:
    """The exact command string an exporter is launched with.

    Single source of truth so the manifest records what actually ran rather
    than the default, which would be false provenance under a custom command.
    """
    if exporter_config.command is None:
        return default_template.format(port=exporter_config.port)
    if "{port}" in exporter_config.command:
        return exporter_config.command.format(port=exporter_config.port)
    return exporter_config.command


def read_producer_commit() -> str | None:
    """The srt-slurm commit that produced the artifact, when available."""
    located = head_commit(Path(__file__).resolve())
    if located is None:
        return None
    root, commit = located
    # An installed copy nested inside an unrelated git tree must not stamp that repo's HEAD.
    if not (root / "src" / "srtctl").is_dir():
        return None
    return commit


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
        use_bash_wrapper: bool = True,
        critical: bool = True,
        on_started: Callable[[ManagedProcess], None] | None = None,
    ) -> list[ManagedProcess]:
        """Start one exporter container across the requested nodes.

        Under SLURM heterogeneous jobs the nodelist may span both het
        components (prefill on group 0, decode on group 1). A single srun
        cannot target multiple het components, so we split the launch into
        one srun per group when needed.

        ``on_started`` runs immediately after each group's process is created,
        so a caller can take ownership before the next group is launched and a
        partial launch stays reachable by the outer cleanup path.
        """
        cmd_str = resolve_exporter_command(exporter_config, default_command_template)

        if self.runtime.nodes.het:
            groups: dict[int, list[str]] = {}
            for node in nodelist:
                g = self.runtime.nodes.het_group_for(node)
                if g is None:
                    raise RuntimeError(f"node {node!r} not in any het component")
                groups.setdefault(g, []).append(node)
            chunks = sorted(groups.items())
        else:
            chunks = [(-1, nodelist)]  # sentinel: no --het-group

        managed: list[ManagedProcess] = []
        for group_id, nodes in chunks:
            het_group = group_id if group_id >= 0 else None
            chunk_log = log_file if len(chunks) == 1 else log_file.with_suffix(f".g{group_id}.out")
            proc = start_srun_process(
                command=shlex.split(cmd_str),
                nodes=len(nodes),
                ntasks=len(nodes),
                nodelist=nodes,
                output=str(chunk_log),
                container_image=exporter_config.container_image,
                container_mounts=self.runtime.container_mounts,
                srun_options=self.runtime.srun_options,
                het_group=het_group,
                use_bash_wrapper=use_bash_wrapper,
            )
            chunk_name = name if len(chunks) == 1 else f"{name}_g{group_id}"
            process = ManagedProcess(
                name=chunk_name,
                popen=proc,
                log_file=chunk_log,
                node=",".join(nodes),
                critical=critical,
            )
            if on_started is not None:
                on_started(process)
            managed.append(process)
        return managed

    def start_power_telemetry(self, registry: ProcessRegistry) -> PowerTelemetrySession | None:
        """Start DCGM power telemetry when it is enabled.

        Every provider-originated startup failure becomes session state once the
        session exists, so the orchestrator can still finalize artifacts and
        decide the exit code after the benchmark stage.
        """
        telemetry = self.config.telemetry
        if not telemetry.enabled:
            return None

        exporter_config = telemetry.dcgm_exporter
        if exporter_config is None:  # guaranteed by schema validation
            raise ValueError("telemetry.dcgm_exporter is required when telemetry is enabled")

        worker_nodes = sorted({process.node for process in self.backend_processes})
        power_dir = self.runtime.log_dir / telemetry.storage_subdir
        command = resolve_exporter_command(exporter_config, DCGM_EXPORTER_COMMAND_TEMPLATE)

        session = PowerTelemetrySession(
            settings=PowerSessionSettings(
                power_dir=power_dir,
                log_dir=self.runtime.log_dir,
                job_id=self.runtime.job_id,
                run_name=self.runtime.run_name,
                sample_interval_seconds=telemetry.default_frequency,
                startup_timeout_seconds=telemetry.startup_timeout_seconds,
                request_timeout_seconds=telemetry.request_timeout_seconds,
                collector_join_timeout_seconds=telemetry.resolved_collector_join_timeout_seconds,
                required=telemetry.required,
                exporter_port=exporter_config.port,
                exporter_image=exporter_config.container_image,
                exporter_command=command,
                network_interface=self.runtime.network_interface,
                producer_git_commit=read_producer_commit(),
            ),
            expected_devices=build_expected_devices(self.backend_processes),
            expected_windows=[
                ExpectedWindow(benchmark_type=self.config.benchmark.type, concurrency=concurrency)
                for concurrency in self.config.benchmark.get_concurrency_list()
            ],
            nodes=worker_nodes,
        )
        # NOTE: stored before initialize() so a raise mid-startup still leaves a finalizable session.
        self._power_session = session
        self._power_telemetry_ready = False
        session.initialize()
        logger.info("Starting DCGM power telemetry (artifacts under %s)", power_dir)

        def own(process: ManagedProcess) -> None:
            registry.add_process(process)
            session.add_exporter(process)

        try:
            self._start_exporter_container(
                exporter_config=exporter_config,
                name="telemetry_dcgm_exporter",
                nodelist=worker_nodes,
                log_file=self.runtime.log_dir / "telemetry_dcgm_exporter.out",
                default_command_template=DCGM_EXPORTER_COMMAND_TEMPLATE,
                use_bash_wrapper=False,  # distroless exporter images have no shell
                critical=False,  # an exit is telemetry invalidity, not a sweep-critical failure
                on_started=own,
            )
        except Exception:
            logger.exception("DCGM exporter launch failed")
            session.record_reason(Reason.EXPORTER_LAUNCH_FAILED)
            return session

        if session.start_and_wait_for_readiness():
            self._power_telemetry_ready = True
        else:
            logger.warning("Power collector did not reach readiness within %.1fs", telemetry.startup_timeout_seconds)
        return session

    def power_telemetry_blocks_benchmark(self) -> bool:
        """Whether required-mode telemetry failed startup and must skip the workload.

        Running the formal benchmark without collection would burn the
        allocation producing a result no consumer may use. Best-effort mode
        keeps serving and leaves the gap auditable in the manifest.
        """
        session = getattr(self, "_power_session", None)
        if session is None:
            return False
        # NOTE: fail closed, so only proven readiness clears the gate.
        if getattr(self, "_power_telemetry_ready", False):
            return False
        return self.config.telemetry.required

    def finalize_power_telemetry(self, exit_code: int, *, interrupted: bool = False) -> int:
        """Finalize the power session and fold required-mode invalidity into the exit code.

        Runs before ``ProcessRegistry.cleanup()`` so the writer is closed and
        the manifest is durable while the exporters are still owned. Expected
        measurement invalidity follows required/best-effort policy. An
        unexpected finalizer exception is an operational failure: it is logged,
        forces a nonzero exit, and still allows process cleanup to continue.
        """
        session = getattr(self, "_power_session", None)
        if session is None:
            return exit_code

        reaped = getattr(self, "benchmark_child_reaped", None)
        allows_window_mutation = getattr(self, "benchmark_child_allows_window_mutation", None)
        if reaped is False:
            session.record_reason(Reason.BENCHMARK_CHILD_REAP_TIMEOUT)

        try:
            outcome = session.stop_and_finalize(
                interrupted=interrupted,
                allow_window_mutation=allows_window_mutation is True,
            )
        except Exception:
            logger.exception("Power telemetry finalization failed")
            return 1

        logger.info(
            "Power telemetry: status=%s publication_valid=%s reasons=%s",
            outcome.status,
            outcome.publication_valid,
            ",".join(outcome.reason_codes) or "none",
        )
        if outcome.exit_nonzero and exit_code == 0:
            logger.error("telemetry.required is set and power artifacts are not publishable")
            return 1
        return exit_code

    def start_tachometer(self) -> list[ManagedProcess]:
        """Start optional Tachometer collection from observability."""
        tachometer = self.config.observability.tachometer
        if tachometer.enabled is not True:
            logger.info("Tachometer disabled")
            return []

        logger.info("Starting Tachometer")

        power_telemetry = self.config.telemetry
        dcgm_exporter = power_telemetry.dcgm_exporter if power_telemetry.enabled else tachometer.dcgm_exporter
        topology = self._compute_frontend_topology()
        config_path = self.runtime.log_dir / "tachometer_config.toml"
        config_path.write_text(
            generate_tachometer_config(
                processes=self.backend_processes,
                frontend_topology=topology,
                runtime=self.runtime,
                tachometer=tachometer,
                dcgm_exporter=dcgm_exporter,
                frontend_type=self.config.frontend.type,
            )
        )

        tachometer_dir = self.runtime.log_dir / tachometer.storage_subdir
        tachometer_dir.mkdir(parents=True, exist_ok=True)
        local_dir = tachometer_dir / "local"
        local_dir.mkdir(parents=True, exist_ok=True)

        worker_nodes = sorted({process.node for process in self.backend_processes})
        processes: list[ManagedProcess] = []
        if not power_telemetry.enabled and tachometer.dcgm_exporter is not None:
            processes.extend(
                self._start_exporter_container(
                    exporter_config=tachometer.dcgm_exporter,
                    name="tachometer_dcgm_exporter",
                    nodelist=worker_nodes,
                    log_file=self.runtime.log_dir / "tachometer_dcgm_exporter.out",
                    default_command_template=DCGM_EXPORTER_COMMAND_TEMPLATE,
                )
            )
        if tachometer.node_exporter is not None:
            processes.extend(
                self._start_exporter_container(
                    exporter_config=tachometer.node_exporter,
                    name="tachometer_node_exporter",
                    nodelist=worker_nodes,
                    log_file=self.runtime.log_dir / "tachometer_node_exporter.out",
                    default_command_template=(
                        "/bin/node_exporter --web.listen-address=:{port} "
                        "--collector.disable-defaults --collector.cpu --collector.infiniband --collector.meminfo"
                    ),
                )
            )

        cmd = [
            tachometer.binary_path,
            "--config",
            str(config_path),
            "--local-dir",
            str(local_dir),
        ]
        if tachometer.sync_interval_secs > 0:
            cmd.extend(["--sync-interval", str(tachometer.sync_interval_secs)])

        env_to_set: dict[str, str] = {}
        if tachometer.compaction_threads > 0:
            env_to_set["POLARS_MAX_THREADS"] = str(tachometer.compaction_threads)

        processes.append(
            ManagedProcess(
                name="tachometer",
                popen=start_srun_process(
                    command=cmd,
                    nodelist=[self.runtime.nodes.head],
                    output=str(self.runtime.log_dir / "tachometer.out"),
                    env_to_set=env_to_set,
                    srun_options=self.runtime.srun_options,
                    het_group=self.runtime.nodes.het_group_for(self.runtime.nodes.head),
                ),
                log_file=self.runtime.log_dir / "tachometer.out",
                node=self.runtime.nodes.head,
            )
        )
        logger.info("Tachometer started with artifacts under %s", tachometer_dir)
        return processes
