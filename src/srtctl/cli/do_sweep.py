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
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from srtctl.cli.mixins import (
    BenchmarkStageMixin,
    FrontendStageMixin,
    PostProcessStageMixin,
    TelemetryStageMixin,
    WorkerStageMixin,
)
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
class SweepOrchestrator(
    WorkerStageMixin,
    FrontendStageMixin,
    TelemetryStageMixin,
    BenchmarkStageMixin,
    PostProcessStageMixin,
):
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
        if self.config.infra.nats_max_payload_mb is not None:
            cmd += ["--nats-max-payload-mb", str(self.config.infra.nats_max_payload_mb)]

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

    def _get_hf_home(self) -> str | None:
        """Get HF_HOME from backend environment config."""
        for mode in ("prefill", "decode", "agg"):
            env = self.config.backend.get_environment_for_mode(mode)
            if "HF_HOME" in env:
                return env["HF_HOME"]
        return None

    def _get_hf_env(self) -> dict[str, str]:
        """Collect HF-related environment variables from backend config.

        Merges environment from all modes (prefill/decode/agg), keeping
        only HuggingFace-relevant keys (HF_*, HUGGING_FACE_*) so the
        pre-download srun runs with the same auth/endpoint context as workers.
        """
        hf_env: dict[str, str] = {}
        for mode in ("prefill", "decode", "agg"):
            for key, val in self.config.backend.get_environment_for_mode(mode).items():
                if key.startswith(("HF_", "HUGGING_FACE_")):
                    hf_env[key] = val
        return hf_env

    def _clean_stale_hf_locks(self) -> None:
        """Clean stale HuggingFace download lock files from shared cache.

        When multiple workers share a HF cache on a networked filesystem,
        stale .lock files from crashed jobs block all future downloads with
        "Lock acquisition failed". This removes locks older than 30 minutes
        (no legitimate download takes that long).
        """
        hf_home = self._get_hf_home()
        if not hf_home:
            return

        cache_dir = Path(hf_home)
        if not cache_dir.is_dir():
            return

        import time

        threshold = time.time() - 30 * 60  # 30 minutes ago
        removed = 0
        for lock_file in cache_dir.rglob("*.lock"):
            try:
                if lock_file.stat().st_mtime < threshold:
                    lock_file.unlink()
                    removed += 1
            except OSError:
                pass  # Permission denied or already deleted

        if removed > 0:
            logger.info("Cleaned %d stale .lock files from HF cache: %s", removed, hf_home)

    def _ensure_model_cached(self) -> None:
        """Pre-download HuggingFace model on a single node before starting workers.

        srt-slurm launches multiple workers that each independently call
        dynamo's fetch_model(). Without pre-caching, all workers race to
        download the same model on the shared filesystem, causing lock
        contention and "Lock acquisition failed" errors.

        This method runs huggingface-cli download on ONE compute node
        (synchronously, blocking) so the model is fully cached before
        any worker starts. Subsequent worker startups find the model
        in cache and skip downloading entirely - no locks created.
        """
        if not self.runtime.is_hf_model:
            return

        hf_home = self._get_hf_home()
        if not hf_home:
            logger.warning(
                "HF model '%s' specified but HF_HOME is not set in backend environment config. "
                "Workers will use the default HuggingFace cache (~/.cache/huggingface) which may not "
                "be shared across nodes. Set HF_HOME in prefill_environment/decode_environment to use "
                "a shared cache directory (e.g., HF_HOME: /lustre/fsw/.../common/cache).",
                self.runtime.model_path,
            )
            return

        model_id = str(self.runtime.model_path)

        # Check if model is already fully cached using huggingface_hub API.
        # snapshot_download with local_files_only=True succeeds only if every
        # file in the model repo is already present in the local cache.
        # Note: HF_HOME stores models in $HF_HOME/hub/, so we pass cache_dir=$HF_HOME/hub
        # to match the actual storage location used by workers.
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

            snapshot_download(model_id, cache_dir=str(Path(hf_home) / "hub"), local_files_only=True)
            logger.info("Model '%s' already cached at %s, skipping pre-download", model_id, hf_home)
            return
        except ImportError:
            logger.debug("huggingface_hub not installed on host, will use container to check/download")
        except Exception:
            logger.debug("Model '%s' not fully cached, will pre-download", model_id)

        download_node = self.runtime.nodes.worker[0]

        logger.info("Ensuring model '%s' is cached on %s (cache: %s)", model_id, download_node, hf_home)

        # The srun command uses HF_HOME (not --cache-dir) to match the exact
        # cache path workers use ($HF_HOME/hub/models--*/).
        # It first checks with HF_HUB_OFFLINE=1 (fast, no network). Only if
        # that fails does it actually download.
        # Uses 'hf download' (new CLI) with 'huggingface-cli download' as fallback.
        import shlex

        q_hf_home = shlex.quote(hf_home)
        q_model_id = shlex.quote(model_id)
        download_cmd = [
            "bash",
            "-c",
            f"export HF_HOME={q_hf_home}; "
            f"find {q_hf_home} -name '*.lock' -mmin +30 -delete 2>/dev/null; "
            f"DL_CMD='hf download'; "
            f"command -v hf >/dev/null 2>&1 || DL_CMD='huggingface-cli download'; "
            f"if HF_HUB_OFFLINE=1 $DL_CMD {q_model_id} --quiet 2>/dev/null; then "
            f"echo 'Model already cached'; "
            f"else "
            f"echo 'Downloading model...'; "
            f"$DL_CMD {q_model_id} --quiet; "
            f"fi",
        ]

        download_log = self.runtime.log_dir / "model_download.out"

        # Pass all HF-related env vars (HF_TOKEN, HF_ENDPOINT, etc.) so the
        # pre-download runs with the same auth/endpoint context as workers.
        hf_env = self._get_hf_env()

        try:
            proc = start_srun_process(
                command=download_cmd,
                nodelist=[download_node],
                output=str(download_log),
                container_image=str(self.runtime.container_image),
                container_mounts=self.runtime.container_mounts,
                env_to_set=hf_env,
                use_bash_wrapper=False,  # command is already bash -c
            )

            timeout_sec = 60 * 60  # 1 hour; large models can take a while
            try:
                rc = proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Model pre-download timed out after %d seconds, killing (workers will retry at startup). Log: %s",
                    timeout_sec,
                    download_log,
                )
                proc.kill()
                proc.wait()
                return

            if rc != 0:
                logger.warning(
                    "Model pre-download exited with code %d (workers will retry at startup). Log: %s",
                    rc,
                    download_log,
                )
            else:
                logger.info("Model pre-download complete")
        except Exception:
            logger.warning("Model pre-download failed (workers will retry at startup)", exc_info=True)

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

            # Pre-worker: Ensure HF model is cached before starting workers.
            # 1. Clean stale lock files from previous crashed downloads
            # 2. Download model on a single node (blocks until complete)
            # This prevents lock contention when multiple workers start.
            if self.runtime.is_hf_model:
                self._clean_stale_hf_locks()
                self._ensure_model_cached()

            # Stage 2: Workers
            reporter.report(JobStatus.WORKERS, JobStage.WORKERS, "Starting workers")
            worker_procs = self.start_all_workers()
            registry.add_processes(worker_procs)

            # Stage 3: Frontend
            reporter.report(JobStatus.FRONTEND, JobStage.FRONTEND, "Starting frontend")
            frontend_procs = self.start_frontend(registry)
            for proc in frontend_procs:
                registry.add_process(proc)

            telemetry_procs = self.start_telemetry()
            for proc in telemetry_procs:
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
