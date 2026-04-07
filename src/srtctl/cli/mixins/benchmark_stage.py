# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark stage mixin for SweepOrchestrator.

Handles benchmark execution and profiling.
"""

import logging
import shlex
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.health import wait_for_model
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.core.status import JobStage, JobStatus, StatusReporter

if TYPE_CHECKING:
    from srtctl.benchmarks.base import BenchmarkRunner
    from srtctl.core.processes import ProcessRegistry
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Endpoint, Process

logger = logging.getLogger(__name__)


class BenchmarkStageMixin:
    """Mixin for benchmark execution stage.

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
        self.endpoints: list[Endpoint]
        self.backend_processes: list[Process]
    """

    # Type hints for mixin dependencies
    config: "SrtConfig"
    runtime: "RuntimeContext"

    @property
    def endpoints(self) -> list["Endpoint"]:
        """Endpoint allocation topology."""
        raise NotImplementedError

    @property
    def backend_processes(self) -> list["Process"]:
        """Backend worker processes."""
        raise NotImplementedError

    def run_benchmark(
        self, registry: "ProcessRegistry", stop_event: threading.Event, reporter: StatusReporter | None = None
    ) -> int:
        """Run the benchmark."""
        logger.info("Waiting for workers to be ready...")

        r = self.config.resources
        num_workers = r.num_prefill + r.num_decode + r.num_agg

        # Build descriptive worker count string
        worker_desc = f"{r.num_agg} agg" if r.num_agg > 0 else f"{r.num_prefill}P + {r.num_decode}D"

        logger.info("Waiting for server health (expecting %d workers: %s)...", num_workers, worker_desc)

        # For aggregated mode: expect 0 prefill, N decode (backend workers count as decode)
        # For disaggregated mode: expect N prefill, M decode
        if r.num_agg > 0:
            n_prefill = 0
            n_decode = r.num_agg
        else:
            n_prefill = r.num_prefill
            n_decode = r.num_decode

        hc = self.config.health_check
        if not wait_for_model(
            host=self.runtime.nodes.head,
            port=8000,
            n_prefill=n_prefill,
            n_decode=n_decode,
            poll_interval=float(hc.interval_seconds),
            timeout=float(hc.max_attempts * hc.interval_seconds),
            report_every=60.0,
            frontend_type=self.config.frontend.type,
            stop_event=stop_event,
        ):
            logger.error("Server did not become healthy")
            if reporter:
                reporter.report(JobStatus.FAILED, JobStage.BENCHMARK, "Workers failed health check")
            return 1

        logger.info("Server is healthy - starting benchmark")
        if reporter:
            reporter.report(JobStatus.BENCHMARK, JobStage.BENCHMARK, "Running benchmark")

        benchmark_type = self.config.benchmark.type
        if self.config.profiling.enabled:
            logger.info(
                "Profiling enabled (type=%s) with benchmark type '%s'",
                self.config.profiling.type,
                benchmark_type,
            )

        if benchmark_type == "manual":
            logger.info("Benchmark type is 'manual' - server is ready for testing")
            logger.info("Frontend URL: http://%s:8000", self.runtime.nodes.head)
            logger.info("Press Ctrl+C to stop the job")

            while not stop_event.is_set():
                if registry.check_failures():
                    logger.error("Worker failure detected during manual mode")
                    return 1
                time.sleep(5)
            return 0

        # Get the appropriate benchmark runner
        from srtctl.benchmarks import get_runner

        try:
            runner = get_runner(benchmark_type)
        except ValueError as e:
            logger.error("%s", e)
            return 1

        # Validate config
        errors = runner.validate_config(self.config)
        if errors:
            for error in errors:
                logger.error("Config error: %s", error)
            return 1

        logger.info("Running %s benchmark", runner.name)

        # Run the benchmark script
        benchmark_log = self.runtime.log_dir / "benchmark.out"
        exit_code = self._run_benchmark_script(runner, benchmark_log, stop_event)

        if exit_code != 0:
            logger.error("Benchmark failed with exit code %d", exit_code)
        else:
            logger.info("Benchmark completed successfully")

        return exit_code

    def _run_benchmark_script(
        self,
        runner: "BenchmarkRunner",
        log_file: Path,
        stop_event: threading.Event,
    ) -> int:
        """Run the actual benchmark script."""

        cmd = runner.build_command(self.config, self.runtime)
        env_to_set = self._get_benchmark_env(runner)

        logger.info("Script: %s", runner.script_path)
        logger.info("Command: %s", shlex.join(cmd))
        logger.info("Log: %s", log_file)

        proc = start_srun_process(
            command=cmd,
            nodelist=[self.runtime.nodes.head],
            output=str(log_file),
            container_image=str(self.runtime.container_image),
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
        )

        # Wait for benchmark to complete
        while proc.poll() is None:
            if stop_event.is_set():
                logger.info("Stop requested, terminating benchmark")
                proc.terminate()
                return 1
            time.sleep(1)

        return proc.returncode or 0

    def _get_benchmark_profiling_env(self, runner: "BenchmarkRunner") -> dict[str, str]:
        """Get environment variables for the benchmark script."""
        env: dict[str, str] = {}

        p = self.config.profiling
        if not p.enabled:
            return env

        # Inside the container, the host log directory is mounted to /logs. Use the container path so profiling
        # artifacts persist back to the host log directory across nodes.
        profiles_dir_in_container = "/logs/profiles"

        # Profiling type (nsys, torch)
        env["PROFILE_TYPE"] = p.type

        # Phase-specific step configs
        if p.prefill:
            if p.prefill.start_step is not None:
                env["PROFILE_PREFILL_START_STEP"] = str(p.prefill.start_step)
            if p.prefill.stop_step is not None:
                env["PROFILE_PREFILL_STOP_STEP"] = str(p.prefill.stop_step)
        if p.decode:
            if p.decode.start_step is not None:
                env["PROFILE_DECODE_START_STEP"] = str(p.decode.start_step)
            if p.decode.stop_step is not None:
                env["PROFILE_DECODE_STOP_STEP"] = str(p.decode.stop_step)
        if p.aggregated:
            if p.aggregated.start_step is not None:
                env["PROFILE_AGG_START_STEP"] = str(p.aggregated.start_step)
            if p.aggregated.stop_step is not None:
                env["PROFILE_AGG_STOP_STEP"] = str(p.aggregated.stop_step)

        # Torch profiler directory
        if p.is_torch:
            env["SGLANG_TORCH_PROFILER_DIR"] = profiles_dir_in_container

        # Collect worker leader IPs and system server ports by mode
        prefill_ips = []
        decode_ips = []
        agg_ips = []
        prefill_endpoints = []
        decode_endpoints = []
        agg_endpoints = []

        use_sys_port = self.config.frontend.type == "dynamo"
        for process in self.backend_processes:
            if not process.is_leader:
                continue
            leader_ip = get_hostname_ip(process.node, self.runtime.network_interface)
            port = process.sys_port if use_sys_port else process.http_port
            leader_endpoint = f"{leader_ip}:{port}"
            if process.endpoint_mode == "prefill":
                prefill_ips.append(leader_ip)
                prefill_endpoints.append(leader_endpoint)
            elif process.endpoint_mode == "decode":
                decode_ips.append(leader_ip)
                decode_endpoints.append(leader_endpoint)
            elif process.endpoint_mode == "agg":
                agg_ips.append(leader_ip)
                agg_endpoints.append(leader_endpoint)

        if prefill_ips:
            env["PROFILE_PREFILL_IPS"] = ",".join(prefill_ips)
        if decode_ips:
            env["PROFILE_DECODE_IPS"] = ",".join(decode_ips)
        if agg_ips:
            env["PROFILE_AGG_IPS"] = ",".join(agg_ips)
        if prefill_endpoints:
            env["PROFILE_PREFILL_ENDPOINTS"] = ",".join(prefill_endpoints)
        if decode_endpoints:
            env["PROFILE_DECODE_ENDPOINTS"] = ",".join(decode_endpoints)
        if agg_endpoints:
            env["PROFILE_AGG_ENDPOINTS"] = ",".join(agg_endpoints)

        # Set profile output directory and common env vars for benchmarks that support profiling
        if runner.name in ("SA-Bench", "SGLang-Bench"):
            env["PROFILE_OUTPUT_DIR"] = profiles_dir_in_container
            env["BENCH_MODEL_NAME"] = self.config.served_model_name
            env["HEAD_NODE"] = self.runtime.nodes.head
            env["HEAD_PORT"] = str(self.runtime.frontend_port)

        return env

    def _get_aiperf_server_metrics_env(self) -> dict[str, str]:
        """Build server metrics URLs for AIPerf benchmarks.

        Collects metrics endpoints from all backend processes that expose
        a sys_port (vLLM workers with AIPerf metrics enabled).
        """
        urls: list[str] = []
        for process in self.backend_processes:
            if process.sys_port > 0:
                host = get_hostname_ip(process.node, self.runtime.network_interface)
                urls.append(f"http://{host}:{process.sys_port}/metrics")

        if not urls:
            return {}
        return {"AIPERF_SERVER_METRICS_URLS": ",".join(sorted(set(urls)))}

    def _get_benchmark_env(self, runner: "BenchmarkRunner") -> dict[str, str]:
        """Get environment variables for the benchmark script."""
        from srtctl.benchmarks.base import AIPerfBenchmarkRunner

        env = self._get_benchmark_profiling_env(runner)
        env["SRTCTL_FRONTEND_TYPE"] = self.config.frontend.type

        # Add AIPerf metrics URLs for AIPerf-driven benchmarks
        if isinstance(runner, AIPerfBenchmarkRunner):
            env.update(self._get_aiperf_server_metrics_env())

        return env
