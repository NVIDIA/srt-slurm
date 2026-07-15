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

from srtctl.core.fingerprint import format_identity_verification, verify_identity
from srtctl.core.health import wait_for_model
from srtctl.core.lockfile import collect_worker_fingerprints
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.core.status import JobStage, JobStatus, StatusReporter
from srtctl.ports import FRONTEND_PUBLIC_PORT, SGLANG_HTTP_PORT_BASE

if TYPE_CHECKING:
    from srtctl.benchmarks.base import BenchmarkRunner
    from srtctl.core.processes import ProcessRegistry
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Endpoint, Process

logger = logging.getLogger(__name__)


def _vllm_data_parallel_size(config: "SrtConfig", mode: str) -> int:
    """Return vLLM data parallel size for a mode, defaulting to one."""
    backend = config.backend
    if getattr(backend, "type", None) != "vllm":
        return 1

    vllm_config = getattr(backend, "vllm_config", None)
    mode_config = getattr(vllm_config, mode, None) if vllm_config else None
    if not mode_config:
        # Special case: no vllm_config at all defaulting to 1 then
        return 1

    return int(mode_config.get("data-parallel-size") or mode_config.get("data_parallel_size") or 1)


def _get_health_expectations(config: "SrtConfig") -> tuple[int, int, str, int]:
    """Compute expected health counts in the units reported by the frontend.

    Dynamo's /health endpoint reports registered generate instances. For vLLM
    DP workers, that means one entry per DP rank, not one entry per logical
    srt-slurm worker. Other frontends keep using logical worker counts.
    """
    r = config.resources

    if r.num_agg > 0:
        logical_prefill = 0
        logical_decode = r.num_agg
        worker_desc = f"{r.num_agg} agg"
    else:
        logical_prefill = r.num_prefill
        logical_decode = r.num_decode
        worker_desc = f"{r.num_prefill}P + {r.num_decode}D"

    if config.frontend.type == "dynamo" and getattr(config.backend, "type", None) == "vllm":
        if r.num_agg > 0:
            n_prefill = 0
            n_decode = logical_decode * _vllm_data_parallel_size(config, "aggregated")
        else:
            n_prefill = logical_prefill * _vllm_data_parallel_size(config, "prefill")
            n_decode = logical_decode * _vllm_data_parallel_size(config, "decode")

        count_desc = f"{n_prefill}P + {n_decode}D Dynamo generate instances; logical workers: {worker_desc}"
        return n_prefill, n_decode, count_desc, n_prefill + n_decode

    count_desc = worker_desc
    return logical_prefill, logical_decode, count_desc, logical_prefill + logical_decode


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

    def _orchestrator_node(self) -> str:
        """Node the frontend/orchestrator runs on (honors frontend.orchestrator_placement)."""
        placement = getattr(self.config.frontend, "orchestrator_placement", "head")
        if placement == "head":
            return self.runtime.nodes.head
        from srtctl.core.topology import placed_node

        return placed_node(
            self.backend_processes, placement, self.runtime.nodes.head, kind="frontend.orchestrator_placement"
        )

    def _benchmark_node(self) -> str:
        """Node the benchmark client runs on (honors benchmark.client_placement)."""
        placement = getattr(self.config.benchmark, "client_placement", "head")
        if placement == "head":
            return self.runtime.nodes.head
        from srtctl.core.topology import placed_node

        return placed_node(
            self.backend_processes, placement, self.runtime.nodes.head, kind="benchmark.client_placement"
        )

    def run_benchmark(
        self, registry: "ProcessRegistry", stop_event: threading.Event, reporter: StatusReporter | None = None
    ) -> int:
        """Run the benchmark."""
        logger.info("Waiting for workers to be ready...")

        n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(self.config)
        logger.info("Waiting for server health (expecting %d health entries: %s)...", num_workers, count_desc)

        hc = self.config.health_check
        if not wait_for_model(
            host=self._orchestrator_node(),
            port=FRONTEND_PUBLIC_PORT,
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

        # Identity verification: compare recipe identity against runtime fingerprints
        # Store results on self so postprocess can include them in the lockfile
        self._identity_verification = None
        try:
            fingerprints = collect_worker_fingerprints(self.runtime.log_dir)
            has_identity = self.config.identity and (
                (
                    self.config.identity.model
                    and (self.config.identity.model.repo or self.config.identity.model.revision)
                )
                or (self.config.identity.container and self.config.identity.container.image)
                or self.config.identity.frameworks
            )
            if fingerprints and has_identity:
                self._identity_verification = verify_identity(self.config.identity, fingerprints)
                banner = format_identity_verification(self._identity_verification, self.config.identity)
                for line in banner.splitlines():
                    logger.info(line)
        except Exception as e:
            logger.debug("Identity verification skipped: %s", e)

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
            logger.info("Frontend URL: http://%s:%d", self._orchestrator_node(), FRONTEND_PUBLIC_PORT)
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
        from srtctl.analysis.live_metrics import try_start_snapshotter

        cmd = runner.build_command(self.config, self.runtime)
        env_to_set = self._get_benchmark_env(runner)
        env_to_set.update(runner.get_environment(self.config, self.runtime))
        container_image = runner.get_container_image(self.config, self.runtime)
        container_mounts = runner.get_container_mounts(self.config, self.runtime)

        logger.info("Script: %s", runner.script_path)
        logger.info("Command: %s", shlex.join(cmd))
        logger.info("Log: %s", log_file)

        # Optional in-flight batch-metrics snapshotter — no-op unless
        # opted in via reporting.live_metrics in the cluster config.
        snapshotter = try_start_snapshotter(self.runtime.log_dir, stop_event)

        bench_node = self._benchmark_node()
        proc = start_srun_process(
            command=cmd,
            nodelist=[bench_node],
            output=str(log_file),
            container_image=str(container_image),
            container_mounts=container_mounts,
            env_to_set=env_to_set,
            srun_options=self.runtime.srun_options,
            het_group=self.runtime.nodes.het_group_for(bench_node),
        )

        try:
            while proc.poll() is None:
                if stop_event.is_set():
                    logger.info("Stop requested, terminating benchmark")
                    proc.terminate()
                    return 1
                time.sleep(1)
            return proc.returncode or 0
        finally:
            if snapshotter is not None:
                snapshotter.stop()

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
        if runner.name in ("SA-Bench", "SGLang-Bench", "Trace-Replay-Bench"):
            env["PROFILE_OUTPUT_DIR"] = profiles_dir_in_container
            env["BENCH_MODEL_NAME"] = self.config.served_model_name
            env["HEAD_NODE"] = self.runtime.nodes.head
            env["HEAD_PORT"] = str(self.runtime.frontend_port)
            env["PROFILE_WORKER_PORT"] = str(SGLANG_HTTP_PORT_BASE)

        # Let benchmark scripts know the backend type so they can select the right profiling lib
        if self.config.backend_type == "trtllm":
            env["PROFILING_BACKEND"] = "trtllm"

        return env

    def _get_sa_bench_slow_down_env(self) -> dict[str, str]:
        """Build SA-Bench slow_down env from benchmark config and decode worker leaders."""
        b = self.config.benchmark
        if b.slow_down_sleep_time is None or b.slow_down_wait_time is None:
            return {}
        if b.slow_down_sleep_time <= 0 or b.slow_down_wait_time <= 0:
            logger.warning(
                "benchmark slow_down: slow_down_sleep_time and slow_down_wait_time must be positive; skipping"
            )
            return {}
        if self.config.frontend.type != "sglang":
            logger.warning("benchmark.slow_down_* ignored: frontend.type is not sglang")
            return {}

        decode_urls: list[str] = []
        for process in self.backend_processes:
            if not process.is_leader:
                continue
            if process.endpoint_mode != "decode":
                continue
            leader_ip = get_hostname_ip(process.node, self.runtime.network_interface)
            decode_urls.append(f"http://{leader_ip}:{process.http_port}")

        if not decode_urls:
            logger.warning("benchmark slow_down requested but no decode worker leaders found; skipping slow_down env")
            return {}

        return {
            "SA_BENCH_SLOW_DOWN_URLS": ",".join(decode_urls),
            "SA_BENCH_SLOW_DOWN_SLEEP_TIME": str(b.slow_down_sleep_time),
            "SA_BENCH_SLOW_DOWN_WAIT_TIME": str(b.slow_down_wait_time),
        }

    def _get_aiperf_server_metrics_env(self) -> dict[str, str]:
        """Build server metrics URLs for AIPerf benchmarks.

        Collects metrics endpoints from all backend processes that expose
        a sys_port (vLLM workers with AIPerf metrics enabled), plus KVBM
        metrics endpoints if DYN_KVBM_METRICS_PORT is configured.
        """
        urls: list[str] = []
        if self.config.frontend.type == "vllm":
            for process in self.backend_processes:
                if process.endpoint_mode == "agg" and process.is_leader:
                    host = get_hostname_ip(process.node, self.runtime.network_interface)
                    urls.append(f"http://{host}:{FRONTEND_PUBLIC_PORT}/metrics")
            if urls:
                return {"AIPERF_SERVER_METRICS_URLS": ",".join(sorted(set(urls)))}

        for process in self.backend_processes:
            if process.sys_port > 0:
                host = get_hostname_ip(process.node, self.runtime.network_interface)
                urls.append(f"http://{host}:{process.sys_port}/metrics")

        # Add KVBM metrics endpoints for prefill processes with DYN_KVBM_METRICS_PORT
        prefill_env = getattr(self.config.backend, "prefill_environment", {})
        agg_env = getattr(self.config.backend, "aggregated_environment", {})
        kvbm_port = prefill_env.get("DYN_KVBM_METRICS_PORT") or agg_env.get("DYN_KVBM_METRICS_PORT")
        if kvbm_port:
            for process in self.backend_processes:
                if process.endpoint_mode in ("prefill", "agg") and process.is_leader:
                    host = get_hostname_ip(process.node, self.runtime.network_interface)
                    urls.append(f"http://{host}:{kvbm_port}/metrics")

        if not urls:
            return {}
        return {"AIPERF_SERVER_METRICS_URLS": ",".join(sorted(set(urls)))}

    def _get_benchmark_env(self, runner: "BenchmarkRunner") -> dict[str, str]:
        """Get environment variables for the benchmark script."""
        from srtctl.benchmarks.base import AIPerfBenchmarkRunner

        env = self._get_benchmark_profiling_env(runner)
        env["SRTCTL_FRONTEND_TYPE"] = self.config.frontend.type

        # Orchestrator endpoint for the benchmark command. When the client runs on
        # a different node than the orchestrator (e.g. client_placement=last_decode
        # with orchestrator_placement=first_decode), "localhost" is wrong — the
        # command should target http://$SRT_FRONTEND_HOST:$SRT_FRONTEND_PORT.
        env["SRT_FRONTEND_HOST"] = get_hostname_ip(self._orchestrator_node(), self.runtime.network_interface)
        env["SRT_FRONTEND_PORT"] = str(self.runtime.frontend_port)

        # Propagate top-level recipe environment to the bench step. Workers
        # already get this via worker_stage; benches need it too for things
        # like HF_TOKEN that the bench script may consume (e.g. NeMo Skills
        # dataset prep against gated HF datasets).
        for key, value in self.runtime.environment.items():
            env[key] = value

        if runner.name == "SA-Bench":
            env.update(self._get_sa_bench_slow_down_env())

        # Add AIPerf-specific env vars for AIPerf-driven benchmarks only
        if isinstance(runner, AIPerfBenchmarkRunner):
            env.update(self._get_aiperf_server_metrics_env())
            if self.config.benchmark.aiperf_package:
                env["AIPERF_PACKAGE"] = self.config.benchmark.aiperf_package

        return env
