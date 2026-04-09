# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main orchestration script for benchmark sweeps.

This script is called from within the sbatch job and coordinates:
1. Starting head node infrastructure (NATS, etcd)
2. Starting backend workers (prefill/decode/agg)
3. Starting frontends and nginx
4. Running benchmarks
5. Cleanup
"""

import argparse
import functools
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from srtctl.cli.mixins import BenchmarkStageMixin, FrontendStageMixin, PostProcessStageMixin, WorkerStageMixin
from srtctl.core.config import load_config
from srtctl.core.health import wait_for_port
from srtctl.core.lockfile import write_lockfile
from srtctl.core.processes import (
    ManagedProcess,
    ProcessRegistry,
    setup_signal_handlers,
    start_process_monitor,
)
from srtctl.core.runtime import RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.slurm import get_slurm_job_id, start_srun_process
from srtctl.core.status import JobStage, JobStatus, StatusReporter
from srtctl.core.topology import Endpoint, Process
from srtctl.logging_utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class SweepOrchestrator(WorkerStageMixin, FrontendStageMixin, BenchmarkStageMixin, PostProcessStageMixin):
    """Main orchestrator for benchmark sweeps.

    Usage:
        config = load_config(config_path)  # Returns typed SrtConfig
        runtime = RuntimeContext.from_config(config, job_id)
        orchestrator = SweepOrchestrator(config, runtime)
        exit_code = orchestrator.run()
    """

    config: SrtConfig
    runtime: RuntimeContext

    @property
    def backend(self):
        """Access the backend config (implements BackendProtocol)."""
        return self.config.backend

    @functools.cached_property
    def endpoints(self) -> list[Endpoint]:
        """Compute endpoint allocation topology (cached).

        This is the single source of truth for endpoint assignments.
        """
        r = self.config.resources
        return self.backend.allocate_endpoints(
            num_prefill=r.num_prefill,
            num_decode=r.num_decode,
            num_agg=r.num_agg,
            gpus_per_prefill=r.gpus_per_prefill,
            gpus_per_decode=r.gpus_per_decode,
            gpus_per_agg=r.gpus_per_agg,
            gpus_per_node=r.gpus_per_node,
            available_nodes=self.runtime.nodes.worker,
        )

    @functools.cached_property
    def backend_processes(self) -> list[Process]:
        """Compute physical process topology from endpoints (cached)."""
        return self.backend.endpoints_to_processes(self.endpoints)

    def start_head_infrastructure(self, registry: ProcessRegistry) -> ManagedProcess:
        """Start NATS and etcd on the infra node.

        When etcd_nats_dedicated_node is enabled, services run on a dedicated node.
        Otherwise, they run on the head node (default behavior).
        """
        infra_node = self.runtime.nodes.infra
        logger.info("Starting infrastructure services (NATS, etcd)")
        logger.info("Infra node: %s", infra_node)

        setup_script = Path(__file__).parent / "setup_head.py"
        if not setup_script.exists():
            raise RuntimeError(f"setup_head.py not found at {setup_script}")

        setup_script_container = Path("/tmp/setup_head.py")
        infra_log = self.runtime.log_dir / "infra.out"

        cmd = [
            "python3",
            str(setup_script_container),
            "--name",
            self.config.name,
            "--log-dir",
            str(self.runtime.log_dir),
        ]

        mounts = dict(self.runtime.container_mounts)
        mounts[setup_script] = setup_script_container
        # Mount host /tmp to container /host-tmp for etcd/nats data on local storage
        # This ensures etcd WAL writes go to fast local disk, not network storage
        mounts[Path("/tmp")] = Path("/host-tmp")

        proc = start_srun_process(
            command=cmd,
            nodelist=[infra_node],
            output=str(infra_log),
            container_image=str(self.runtime.container_image),
            container_mounts=mounts,
        )

        managed = ManagedProcess(
            name="infra_services",
            popen=proc,
            log_file=infra_log,
            node=infra_node,
            critical=True,
        )

        # 300s timeout to handle slow container imports on first run
        logger.info("Waiting for NATS (port 4222) on %s...", infra_node)
        if not wait_for_port(infra_node, 4222, timeout=300):
            raise RuntimeError("NATS failed to start")
        logger.info("NATS is ready")

        logger.info("Waiting for etcd (port 2379) on %s...", infra_node)
        if not wait_for_port(infra_node, 2379, timeout=300):
            raise RuntimeError("etcd failed to start")
        logger.info("etcd is ready")

        return managed

    def _print_connection_info(self) -> None:
        """Print srun commands for connecting to nodes."""
        container_args = f"--container-image={self.runtime.container_image}"
        mounts_str = ",".join(f"{src}:{dst}" for src, dst in self.runtime.container_mounts.items())
        if mounts_str:
            container_args += f" --container-mounts={mounts_str}"

        logger.info("")
        logger.info("=" * 60)
        logger.info("Connection Commands")
        logger.info("=" * 60)
        logger.info("Frontend URL: http://%s:8000", self.runtime.nodes.head)
        logger.info("")
        logger.info("To connect to head node (%s):", self.runtime.nodes.head)
        logger.info(
            "  srun %s --jobid %s -w %s --overlap --pty bash",
            container_args,
            self.runtime.job_id,
            self.runtime.nodes.head,
        )

        # Print worker node connection commands
        for node in self.runtime.nodes.worker:
            if node != self.runtime.nodes.head:
                logger.info("")
                logger.info("To connect to worker node (%s):", node)
                logger.info(
                    "  srun %s --jobid %s -w %s --overlap --pty bash",
                    container_args,
                    self.runtime.job_id,
                    node,
                )

        logger.info("=" * 60)
        logger.info("")

    def run(self) -> int:
        """Run the complete sweep."""
        # Create status reporter (fire-and-forget, no-op if not configured)
        reporter = StatusReporter.from_config(self.config.reporting, self.runtime.job_id)
        reporter.report_started(self.config, self.runtime)

        logger.info("Sweep Orchestrator")
        logger.info("Job ID: %s", self.runtime.job_id)
        logger.info("Run name: %s", self.runtime.run_name)
        logger.info("Config: %s", self.config.name)
        logger.info("Infra node: %s", self.runtime.nodes.infra)
        logger.info("Head node: %s", self.runtime.nodes.head)
        logger.info("Worker nodes: %s", ", ".join(self.runtime.nodes.worker))
        if self.config.profiling.enabled:
            logger.info("Profiling: %s", self.config.profiling.type)

        # Write initial lockfile with config + SLURM context (fingerprint added after run)
        write_lockfile(self.runtime.log_dir.parent, self.config)

        registry = ProcessRegistry(job_id=self.runtime.job_id)
        stop_event = threading.Event()
        setup_signal_handlers(stop_event, registry)
        start_process_monitor(stop_event, registry)

        exit_code = 1

        try:
            # Stage 1: Head infrastructure (NATS, etcd)
            reporter.report(JobStatus.STARTING, JobStage.HEAD_INFRASTRUCTURE, "Starting head infrastructure")
            head_proc = self.start_head_infrastructure(registry)
            registry.add_process(head_proc)

            # Stage 2: Workers
            reporter.report(JobStatus.WORKERS, JobStage.WORKERS, "Starting workers")
            worker_procs = self.start_all_workers()
            registry.add_processes(worker_procs)

            # Stage 3: Frontend
            reporter.report(JobStatus.FRONTEND, JobStage.FRONTEND, "Starting frontend")
            frontend_procs = self.start_frontend(registry)
            for proc in frontend_procs:
                registry.add_process(proc)

            self._print_connection_info()

            # Stage 4: Benchmark (status reported AFTER health check passes)
            exit_code = self.run_benchmark(registry, stop_event, reporter)

        except Exception as e:
            logger.exception("Error during sweep: %s", e)
            reporter.report(JobStatus.FAILED, JobStage.CLEANUP, str(e))
            exit_code = 1

        finally:
            logger.info("Cleanup")
            reporter.report_completed(exit_code)
            stop_event.set()
            registry.cleanup()
            if exit_code != 0:
                registry.print_failure_details()
            # Run post-processing (AI analysis if enabled)
            self.run_postprocess(exit_code)

        return exit_code


def main():
    """Main entry point."""
    from dataclasses import replace

    parser = argparse.ArgumentParser(description="Run benchmark sweep")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()

    setup_logging()

    try:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error("Config file not found: %s", config_path)
            sys.exit(1)

        config = load_config(config_path)

        # Check for setup_script override from CLI (passed via env var)
        setup_script_override = os.environ.get("SRTCTL_SETUP_SCRIPT")
        if setup_script_override:
            logger.info("Setup script override: %s", setup_script_override)
            config = replace(config, setup_script=setup_script_override)

        job_id = get_slurm_job_id()
        if not job_id:
            logger.error("Not running in SLURM (SLURM_JOB_ID not set)")
            sys.exit(1)

        # Type narrowing: job_id is str after the check above
        assert job_id is not None
        runtime = RuntimeContext.from_config(config, job_id)
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        exit_code = orchestrator.run()

        sys.exit(exit_code)

    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
