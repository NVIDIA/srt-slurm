# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark stage mixin for SweepOrchestrator.

Handles benchmark execution and profiling.
"""

import json
import logging
import re
import shlex
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

from srtctl.backends.trtllm import TRTLLMProtocol
from srtctl.core.fingerprint import format_identity_verification, verify_identity
from srtctl.core.health import wait_for_model
from srtctl.core.ip_utils import url_host
from srtctl.core.lockfile import collect_worker_fingerprints
from srtctl.core.observability_nsys import benchmark_nsys_env
from srtctl.core.power.contract import (
    MEASUREMENT_WINDOW_DIR_ENV,
    WINDOWS_DIRNAME,
)
from srtctl.core.processes import terminate_and_reap
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.core.status import JobStage, JobStatus, StatusReporter
from srtctl.frontends import FRONTEND_NONE, get_frontend
from srtctl.ports import FRONTEND_PUBLIC_PORT, SGLANG_HTTP_PORT_BASE
from srtctl.runtime_scripts.nsys_window import finish as finish_nsys_windows

_BENCHMARK_TERMINATE_TIMEOUT = 15.0
_BENCHMARK_KILL_TIMEOUT = 10.0
# How often manual mode checks for failures and for terminal services finishing.
MANUAL_POLL_SECONDS = 5.0

if TYPE_CHECKING:
    from srtctl.benchmarks.base import BenchmarkRunner
    from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin
    from srtctl.core.processes import ManagedProcess, ProcessRegistry
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Endpoint, Process
    from srtctl.frontends import FrontendProtocol

logger = logging.getLogger(__name__)


def _get_health_expectations(
    config: "SrtConfig", backend_processes: list["Process"] | None = None
) -> tuple[int, int, str, int]:
    """Expected health counts in the units the frontend reports, a description, and their sum.

    The frontend knows what its readiness endpoint counts (Dynamo generate
    registrations, Router-expanded DP ranks, logical workers); see
    ``FrontendProtocol.health_expectations``.
    """
    frontend = get_frontend(config.frontend.type)
    n_prefill, n_decode, count_desc = frontend.health_expectations(config, backend_processes)
    return n_prefill, n_decode, count_desc, n_prefill + n_decode


SERVER_READY_FILENAME = "server_ready.json"


def write_server_ready_marker(log_dir: Path) -> Path | None:
    """Record that every configured worker passed the health gate.

    An external load generator driving a ``manual`` job has only the frontend to ask,
    and a Dynamo frontend lists the model as soon as its first worker registers — before
    the rest have. This file is the launcher-side signal that srtctl's own gate (all
    prefill and decode workers) has passed, so a client can wait for it instead of
    racing the last worker. Best-effort: a failure to write it is logged, never raised.
    """
    marker = log_dir / SERVER_READY_FILENAME
    try:
        marker.write_text(json.dumps({"schema_version": 1, "ready_at_unix": time.time()}) + "\n")
    except OSError as error:
        logger.warning("could not write %s: %s", marker, error)
        return None
    return marker


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
    benchmark_child_reaped: bool | None = None
    benchmark_child_allows_window_mutation: bool | None = None

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

    @property
    def frontend(self) -> "FrontendProtocol | None":
        """The frontend implementation for ``frontend.type``; ``None`` for a services-only job."""
        if self.config.frontend.type == FRONTEND_NONE:
            return None
        return get_frontend(self.config.frontend.type)

    def _public_api_node(self) -> str:
        """Node hosting the public OpenAI HTTP endpoint clients should probe."""
        frontend = self.frontend
        direct_nodes = frontend.direct_endpoint_nodes(self.backend_processes) if frontend is not None else []
        if len(direct_nodes) == 1:
            return direct_nodes[0]
        return self._orchestrator_node()

    def _benchmark_node(self) -> str:
        """Node the benchmark client runs on (honors benchmark.client_placement).

        ``nodes.bench`` equals ``nodes.head`` unless a dedicated client node was
        carved out (benchmark.client_dedicated_node), in which case it points at
        that reserved node instead.
        """
        placement = getattr(self.config.benchmark, "client_placement", "head")
        if placement == "head":
            return self.runtime.nodes.bench
        from srtctl.core.topology import placed_node

        return placed_node(
            self.backend_processes, placement, self.runtime.nodes.head, kind="benchmark.client_placement"
        )

    def _logical_worker_endpoints(self) -> list[tuple[str, str, int]]:
        """Return ``(mode, IP, port)`` for every routable worker endpoint.

        Positive HTTP ports identify Router-facing node-local vLLM pools;
        follower processes in a cross-node model-parallel replica retain zero.

        Dynamo exposes worker metrics on each leader's system port. Direct
        vLLM exposes aggregate metrics on the public frontend port, while
        other frontends expose them on the worker HTTP port.
        """
        frontend = self.frontend
        if frontend is None:
            return []
        endpoints: list[tuple[str, str, int]] = []
        for process in self.backend_processes:
            port = frontend.worker_endpoint_port(process, self.config, self.runtime)
            if port is None:
                continue
            host = get_hostname_ip(process.node, self.runtime.network_interface)
            endpoints.append((process.endpoint_mode, host, port))
        return endpoints

    def _profiling_worker_endpoints(self) -> list[tuple[str, str, int]]:
        """Return only the process endpoints that control this capture.

        Iteration-triggered Nsight captures for vLLM and SGLang target either
        one selected physical process or all processes in each serving phase.
        Time-based Nsight, Torch, and TRT-LLM retain their existing
        endpoint-wide behavior.
        """
        profiling = self.config.profiling
        if not profiling.is_nsys or profiling.is_nsys_time or self.config.backend_type == "trtllm":
            return self._logical_worker_endpoints()

        frontend = self.frontend
        if frontend is None:
            return []
        leader_only_control = frontend.profiling_control_is_leader_only(self.config)
        endpoints: list[tuple[str, str, int]] = []
        selected_modes: set[str] = set()
        for process in self.backend_processes:
            worker_index = process.endpoint_index
            worker_rank = process.node_rank
            capture_all = profiling.captures_all_processes(process.endpoint_mode)
            if not profiling.selects_process(
                process.endpoint_mode,
                worker_index,
                worker_rank,
            ):
                continue

            if leader_only_control and not process.is_leader:
                # The direct-vLLM server and a Dynamo sidecar expose one
                # control server per logical endpoint, on its leader only.
                # All-process capture still wraps followers, but sends a
                # single request to the leader. A selected follower cannot be
                # controlled independently and must fail explicitly.
                if capture_all:
                    continue
                raise ValueError(
                    "Selected profiling process does not expose its own control endpoint: "
                    f"mode={process.endpoint_mode}, worker_index={worker_index}, "
                    f"worker_rank={worker_rank}"
                )

            port = frontend.profiling_control_port(process, self.config, self.runtime)
            if port is None:
                # Native distributed servers expose one HTTP control endpoint
                # for multiple physical processes. Wrap every process, but send
                # only the routable leader endpoint to the benchmark.
                if capture_all:
                    continue
                raise ValueError(
                    "Selected profiling worker does not expose an HTTP control endpoint: "
                    f"mode={process.endpoint_mode}, worker_index={worker_index}, "
                    f"worker_rank={worker_rank}"
                )

            host = get_hostname_ip(process.node, self.runtime.network_interface)
            endpoints.append((process.endpoint_mode, host, port))
            selected_modes.add(process.endpoint_mode)

        required_modes = {
            mode
            for mode, phase in (
                ("prefill", profiling.prefill),
                ("decode", profiling.decode),
                ("agg", profiling.aggregated),
            )
            if phase is not None
        }
        missing_modes = required_modes - selected_modes
        if missing_modes:
            missing = ", ".join(sorted(missing_modes))
            raise ValueError(f"No physical process matches the profiling selector for: {missing}")

        return list(dict.fromkeys(endpoints))

    def _wait_for_service_ready(self, stop_event: threading.Event) -> bool:
        """Wait for frontend counts and any adapter-specific backend barrier."""
        from srtctl.core import health as health_utils

        if self.config.frontend.type == "none":
            # Services-only job: every service already passed its readiness probe in
            # start_services, and there are no engine workers to count.
            logger.info("frontend.type none: no worker-count health gate; services are ready")
            return True

        n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(self.config, self.backend_processes)
        logger.info("Waiting for server health (expecting %d health entries: %s)...", num_workers, count_desc)

        hc = self.config.health_check
        if not wait_for_model(
            host=self._public_api_node(),
            port=FRONTEND_PUBLIC_PORT,
            n_prefill=n_prefill,
            n_decode=n_decode,
            poll_interval=float(hc.interval_seconds),
            timeout=float(hc.max_attempts * hc.interval_seconds),
            report_every=60.0,
            frontend_type=self.config.frontend.type,
            stop_event=stop_event,
            config=self.config,
        ):
            return False

        frontend = get_frontend(self.config.frontend.type)
        backend_health_urls = frontend.get_backend_health_urls(
            self.config.backend,
            self.backend_processes,
            self.runtime.network_interface,
        )
        if not backend_health_urls:
            return True

        logger.info(
            "Frontend requires direct readiness from %d advertised backend URLs",
            len(backend_health_urls),
        )
        return health_utils.wait_for_http_endpoints(
            backend_health_urls,
            poll_interval=float(hc.interval_seconds),
            timeout=float(hc.max_attempts * hc.interval_seconds),
            report_every=60.0,
            stop_event=stop_event,
        )

    @staticmethod
    def _get_worker_endpoint_env(endpoints: list[tuple[str, str, int]]) -> dict[str, str]:
        """Build mode-specific benchmark environment from logical endpoints."""
        env: dict[str, str] = {}
        prefixes = {"prefill": "PREFILL", "decode": "DECODE", "agg": "AGG"}
        for mode, prefix in prefixes.items():
            mode_endpoints = [(host, port) for endpoint_mode, host, port in endpoints if endpoint_mode == mode]
            if not mode_endpoints:
                continue
            # Keep one IP per logical endpoint, including repeated IPs for
            # co-located workers, so IP and endpoint positions stay aligned.
            env[f"SRT_{prefix}_IPS"] = ",".join(host for host, _ in mode_endpoints)
            env[f"SRT_{prefix}_ENDPOINTS"] = ",".join(f"{host}:{port}" for host, port in mode_endpoints)
        return env

    def _get_service_env(self) -> dict[str, str]:
        """Where every effective service runs, for custom benchmark commands.

        ``SRT_SERVICE_<NAME>_NODES`` / ``_IPS`` (comma-separated, placement order) and
        ``_NODE_COUNT`` for each launched service, ``<NAME>`` being the service name
        upper-cased with non-alphanumerics as ``_``. This is how a script drives a
        service the job brought up: a Ray launcher reads ``SRT_SERVICE_TRAIN_IPS`` for
        the head address. External services (already running elsewhere) are skipped;
        their address is injected by their kind.
        """
        from srtctl.services.implicit import effective_services

        service_nodes = getattr(self, "service_nodes", None)
        if service_nodes is None:
            return {}
        env: dict[str, str] = {}
        for entry in effective_services(self.config):
            service = entry.service
            if service.external:
                continue
            nodes = service_nodes(service)
            if not nodes:
                continue
            key = re.sub(r"[^A-Za-z0-9]", "_", service.name).upper()
            env[f"SRT_SERVICE_{key}_NODES"] = ",".join(nodes)
            env[f"SRT_SERVICE_{key}_IPS"] = ",".join(
                get_hostname_ip(node, self.runtime.network_interface) for node in nodes
            )
            env[f"SRT_SERVICE_{key}_NODE_COUNT"] = str(len(nodes))
        return env

    def run_benchmark(
        self, registry: "ProcessRegistry", stop_event: threading.Event, reporter: StatusReporter | None = None
    ) -> int:
        """Run the benchmark."""
        serve_only = bool(getattr(self, "serve_only", False))
        logger.info("Waiting for workers to be ready...")

        if not self._wait_for_service_ready(stop_event):
            logger.error("Server did not become healthy")
            if reporter:
                stage = JobStage.FRONTEND if serve_only else JobStage.BENCHMARK
                reporter.report(JobStatus.FAILED, stage, "Workers failed health check")
            return 1

        logger.info("Server is healthy")
        write_server_ready_marker(self.runtime.log_dir)

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
        except Exception as e:  # noqa: BLE001
            logger.debug("Identity verification skipped: %s", e)

        benchmark_type = self.config.benchmark.type
        if self.config.profiling.enabled and not serve_only:
            logger.info(
                "Profiling enabled (type=%s) with benchmark type '%s'",
                self.config.profiling.type,
                benchmark_type,
            )

        if serve_only or benchmark_type == "manual":
            # Terminal services (services[].terminal) are the job's run: the job ends when
            # every instance has exited, with the worst exit code. ServiceStageMixin records
            # their processes; getattr because SimpleNamespace runtimes in tests lack the mixin.
            by_service: dict[str, list[ManagedProcess]] = dict(getattr(self, "terminal_processes", {}))
            terminal = [proc for procs in by_service.values() for proc in procs]
            if reporter:
                reporter.report(JobStatus.FRONTEND, JobStage.FRONTEND, "Inference endpoint ready")
            if serve_only:
                logger.info("Serve-only mode - no benchmark will be run")
            elif terminal:
                logger.info("Waiting for terminal service(s) to finish: %s", ", ".join(sorted(by_service)))
            else:
                logger.info("Benchmark type is 'manual' - server is ready for testing")
            if self.config.frontend.type != "none":
                logger.info("Frontend URL: http://%s:%d", self._public_api_node(), FRONTEND_PUBLIC_PORT)
            if not terminal:
                logger.info("Press Ctrl+C to stop the job")

            while not stop_event.is_set():
                if terminal and all(not proc.is_running for proc in terminal):
                    exit_code = max((proc.exit_code or 0) for proc in terminal)
                    for proc in terminal:
                        logger.info("Terminal service step %s exited with code %s", proc.name, proc.exit_code)
                    if exit_code:
                        logger.error("Terminal service(s) failed; job exit code %d", exit_code)
                    else:
                        logger.info("Terminal service(s) finished")
                    return exit_code
                if registry.check_failures():
                    logger.error("Worker failure detected while serving")
                    return 1
                time.sleep(MANUAL_POLL_SECONDS)
            return 0

        logger.info("Starting benchmark")
        if reporter:
            reporter.report(JobStatus.BENCHMARK, JobStage.BENCHMARK, "Running benchmark")

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

        # Tachometer scrapes the load window only, the same window the
        # benchmark client's own AIPERF polling covers. Starting it with the
        # other telemetry (before the health gate) recorded minutes of
        # dead-endpoint noise while workers loaded; stopping it with the
        # registry's hard teardown SIGKILLed the scraper mid-write and
        # stranded the whole capture in the arrow WAL (hecate job 487539).
        # The finally attempts a flush when the benchmark script returns or
        # raises. Signal/monitor cleanup can terminate registered processes
        # earlier using its existing budget. Both hooks live on
        # TelemetryStageMixin (same orchestrator object).
        start_tachometer = getattr(self, "start_tachometer", None)
        tachometer_procs = start_tachometer() if start_tachometer is not None else []
        for proc in tachometer_procs:
            registry.add_process(proc)

        # Run the benchmark script
        benchmark_log = self.runtime.log_dir / "benchmark.out"
        try:
            exit_code = self._run_benchmark_script(runner, benchmark_log, stop_event)
        finally:
            if tachometer_procs:
                cast("TelemetryStageMixin", self).stop_tachometer(tachometer_procs)

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
        env_to_set.update(runner.get_environment(self.config, self.runtime))
        container_image = runner.get_container_image(self.config, self.runtime)
        container_mounts = runner.get_container_mounts(self.config, self.runtime)

        logger.info("Script: %s", runner.script_path)
        logger.info("Command: %s", shlex.join(cmd))
        logger.info("Log: %s", log_file)

        # Host/process telemetry for the benchmark window. The Prometheus
        # families describe what Dynamo publishes; they say nothing about the
        # machine underneath, where host CPU saturation, lock convoys and fd
        # exhaustion live. Follows observability.enabled; best-effort contract.
        #
        # `is True` is deliberate, not a truthiness check: this mixin is
        # routinely driven with a mocked config whose every attribute is
        # truthy, and plain truthiness would silently switch it on there.
        observability = getattr(self.config, "observability", None)
        host_sampler = None
        if getattr(observability, "enabled", False) is True:
            from srtctl.analysis.host_sampler import try_start_host_sampler

            host_sampler = try_start_host_sampler(self.runtime.log_dir, observability, stop_event)

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

        # The signal handler raises SystemExit, so only finally can establish
        # how the local srun client stopped before telemetry finalizes.
        self.benchmark_child_reaped = False
        self.benchmark_child_allows_window_mutation = False
        try:
            while proc.poll() is None:
                if stop_event.is_set():
                    logger.info("Stop requested, terminating benchmark")
                    return 1
                time.sleep(1)
            self.benchmark_child_reaped = True
            self.benchmark_child_allows_window_mutation = True
            exit_code = proc.returncode or 0
            if (
                getattr(self.config, "observability_nsys_enabled", False) is True
                and self.config.observability.nsys.capture_window == "measured_workload"
            ):
                try:
                    finish_nsys_windows(
                        self.runtime.log_dir / "profiles" / ".control",
                        self.config.observability.nsys.report_timeout_secs,
                    )
                except (RuntimeError, TimeoutError, OSError) as exc:
                    logger.error("Observability capture failed: %s", exc)
                    return exit_code or 1
            return exit_code
        finally:
            if proc.poll() is None:
                outcome = terminate_and_reap(
                    proc,
                    terminate_timeout=_BENCHMARK_TERMINATE_TIMEOUT,
                    kill_timeout=_BENCHMARK_KILL_TIMEOUT,
                )
                self.benchmark_child_reaped = outcome.reaped
                # Reaping a force-killed local srun client does not prove that
                # its remote Slurm step can no longer write the window.
                self.benchmark_child_allows_window_mutation = outcome.reaped and not outcome.force_killed
            elif self.benchmark_child_reaped is False:
                proc.wait()
                self.benchmark_child_reaped = True
                self.benchmark_child_allows_window_mutation = True
            if host_sampler is not None:
                host_sampler.stop()

    def _get_benchmark_profiling_env(
        self,
        runner: "BenchmarkRunner",
        profiling_endpoints: list[tuple[str, str, int]] | None = None,
    ) -> dict[str, str]:
        """Get environment variables for the benchmark script."""
        env: dict[str, str] = {}

        p = self.config.profiling
        if not p.enabled:
            return env

        # The benchmark runs inside the container, so point it at the log mount rather than the host path;
        # profiling artifacts then persist back to the host log directory across nodes.
        profiles_dir_in_container = str(self.runtime.container_log_dir / "profiles")

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

        if profiling_endpoints is None:
            profiling_endpoints = self._profiling_worker_endpoints()
        for mode, leader_ip, port in profiling_endpoints:
            leader_endpoint = f"{leader_ip}:{port}"
            if mode == "prefill":
                prefill_ips.append(leader_ip)
                prefill_endpoints.append(leader_endpoint)
            elif mode == "decode":
                decode_ips.append(leader_ip)
                decode_endpoints.append(leader_endpoint)
            elif mode == "agg":
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
        if self.config.frontend.type != "sglang-router":
            logger.warning("benchmark.slow_down_* ignored: frontend.type is not sglang-router")
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

    def _get_measurement_window_env(self) -> dict[str, str]:
        """Point the benchmark child at the power artifact's windows directory.

        ``runtime.log_dir`` is already mounted at ``/logs``, so the container
        path and the host path the collector reads are the same directory.
        """
        telemetry = self.config.telemetry
        if not telemetry.enabled:
            return {}
        windows_dir = self.runtime.container_log_dir / telemetry.storage_subdir / WINDOWS_DIRNAME
        return {MEASUREMENT_WINDOW_DIR_ENV: str(windows_dir)}

    def _get_aiperf_server_metrics_env(
        self,
        logical_endpoints: list[tuple[str, str, int]] | None = None,
        *,
        logical_workers_only: bool = False,
    ) -> dict[str, str]:
        """Build server metrics URLs for AIPerf benchmarks.

        Built-in AIPerf runners retain their existing physical-process metrics
        behavior, which is required by vLLM data-parallel layouts. Custom
        benchmarks use logical worker leaders so distributed SGLang follower
        ranks are not advertised as separate engines.
        """
        urls: list[str] = []
        frontend = self.frontend
        if frontend is None:
            # Services-only job: no workers serve engine metrics.
            return {}
        backend = self.config.backend
        is_trtllm = self.config.backend_type == "trtllm"
        dynamo_trtllm_metrics_disabled = (
            frontend.worker_launch == "dynamo"
            and isinstance(backend, TRTLLMProtocol)
            and not (
                (not self.config.dynamo.sidecar and backend.dynamo_metrics_flags) or backend.publish_events_and_metrics
            )
        )
        metrics_path = frontend.metrics_path
        if logical_workers_only:
            if logical_endpoints is None:
                logical_endpoints = self._logical_worker_endpoints()
            # Sidecars use native worker commands, so publish_metrics does not
            # control their existing logical-worker URL discovery.
            if self.config.dynamo.sidecar or not dynamo_trtllm_metrics_disabled:
                urls = [f"http://{host}:{port}{metrics_path}" for _, host, port in logical_endpoints]
        elif frontend.worker_launch == "direct":
            # Every rank the frontend says serves metrics. trtllm-serve mounts its
            # Prometheus route only when the engine runs with return_perf_metrics
            # (expand_trtllm_serve_defaults sets it on every trtllm_serve recipe;
            # an explicit false opts out), so gate each worker on its own engine
            # config -- publish_events_and_metrics is a dynamo.trtllm flag that
            # never reaches a trtllm-serve worker.
            for process in self.backend_processes:
                port = frontend.worker_metrics_port(process, self.runtime)
                if port is None:
                    continue
                if is_trtllm and not self.config.backend.get_config_for_mode(process.endpoint_mode).get(
                    "return_perf_metrics"
                ):
                    continue
                host = get_hostname_ip(process.node, self.runtime.network_interface)
                urls.append(f"http://{host}:{port}{metrics_path}")
        # Dynamo TRT-LLM engine metrics require either the metrics-only
        # flag (the default) or the legacy combined flag (also enabled by
        # observability). Retain the existing sidecar gate because sidecars
        # do not receive --publish-metrics. An explicit legacy False disables
        # both flags. Runtime-only metrics may still exist with publication disabled,
        # but must not be advertised as an engine-metrics capture.
        elif not dynamo_trtllm_metrics_disabled:
            for process in self.backend_processes:
                port = frontend.worker_metrics_port(process, self.runtime)
                if port is None:
                    continue
                host = get_hostname_ip(process.node, self.runtime.network_interface)
                urls.append(f"http://{host}:{port}{metrics_path}")

        # Add KVBM metrics endpoints for prefill processes with DYN_KVBM_METRICS_PORT
        prefill_env = backend.get_environment_for_mode("prefill")
        agg_env = backend.get_environment_for_mode("agg")
        kvbm_port = prefill_env.get("DYN_KVBM_METRICS_PORT") or agg_env.get("DYN_KVBM_METRICS_PORT")
        if kvbm_port:
            for process in self.backend_processes:
                if process.endpoint_mode in ("prefill", "agg") and process.is_leader:
                    host = get_hostname_ip(process.node, self.runtime.network_interface)
                    urls.append(f"http://{host}:{kvbm_port}/metrics")

        if not urls:
            return {}
        # Custom commands preserve logical topology order; built-in AIPerf
        # runners retain their historical sorted physical-process list.
        urls = list(dict.fromkeys(urls)) if logical_workers_only else sorted(set(urls))

        # Add CPU power exporter endpoints (one per worker node) when configured.
        cpu_power_exporter = getattr(self.config.telemetry, "cpu_power_exporter", None)
        if self.config.telemetry.enabled and cpu_power_exporter is not None:
            worker_nodes = sorted({process.node for process in self.backend_processes})
            for node in worker_nodes:
                host = get_hostname_ip(node, self.runtime.network_interface)
                urls.append(f"http://{url_host(host)}:{cpu_power_exporter.port}/metrics")

        return {"AIPERF_SERVER_METRICS_URLS": ",".join(urls)}

    def _get_benchmark_env(self, runner: "BenchmarkRunner") -> dict[str, str]:
        """Get environment variables for the benchmark script."""
        from srtctl.benchmarks.base import AIPerfBenchmarkRunner

        is_custom = self.config.benchmark.type == "custom"
        logical_endpoints = self._logical_worker_endpoints() if self.config.profiling.enabled or is_custom else None
        profiling_endpoints = self._profiling_worker_endpoints() if self.config.profiling.enabled else None
        env = self._get_benchmark_profiling_env(runner, profiling_endpoints)
        if is_custom:
            assert logical_endpoints is not None
            env.update(self._get_worker_endpoint_env(logical_endpoints))
            env.update(self._get_service_env())
            # getattr: this mixin is also driven by SimpleNamespace runtimes in tests.
            gpus_per_node = getattr(self.runtime, "gpus_per_node", None)
            if gpus_per_node is not None:
                env["SRT_GPUS_PER_NODE"] = str(gpus_per_node)
            worker_nodes = getattr(getattr(self.runtime, "nodes", None), "worker", None)
            if isinstance(worker_nodes, (list, tuple)):
                env["SRT_WORKER_NODES"] = ",".join(worker_nodes)
        env["SRTCTL_FRONTEND_TYPE"] = self.config.frontend.type

        # Orchestrator endpoint for the benchmark command. When the client runs on
        # a different node than the orchestrator (e.g. client_placement=last_decode
        # with orchestrator_placement=first_decode), "localhost" is wrong — the
        # command should target http://$SRT_FRONTEND_HOST:$SRT_FRONTEND_PORT.
        # A services-only job (frontend.type none) has no endpoint to point at.
        if self.config.frontend.type != "none":
            env["SRT_FRONTEND_HOST"] = get_hostname_ip(self._public_api_node(), self.runtime.network_interface)
            env["SRT_FRONTEND_PORT"] = str(self.runtime.frontend_port)

        # Propagate top-level recipe environment to the bench step. Workers
        # already get this via worker_stage; benches need it too for things
        # like HF_TOKEN that the bench script may consume (e.g. NeMo Skills
        # dataset prep against gated HF datasets).
        for key, value in self.runtime.environment.items():
            env[key] = value

        # The windows directory is benchmark-agnostic: whichever benchmark runs
        # may adopt window stamping, so the env is not tied to one runner.
        env.update(self._get_measurement_window_env())
        if getattr(self.config, "observability_nsys_enabled", False) is True:
            env.update(benchmark_nsys_env(self.config))

        if runner.name == "SA-Bench":
            env.update(self._get_sa_bench_slow_down_env())

        # Built-in AIPerf runners retain physical-process metrics for vLLM DP.
        # Custom commands commonly wrap AIPerf but do not inherit from its base
        # class, so give them the logical-worker view needed by SGLang TP.
        # An explicit AIPERF_SERVER_METRICS_URLS in the recipe environment wins:
        # the operator may be pointing the client at a curated endpoint list,
        # and injection used to clobber it here silently.
        if "AIPERF_SERVER_METRICS_URLS" not in env:
            if isinstance(runner, AIPerfBenchmarkRunner):
                env.update(self._get_aiperf_server_metrics_env())
            elif is_custom:
                assert logical_endpoints is not None
                env.update(self._get_aiperf_server_metrics_env(logical_endpoints, logical_workers_only=True))
        if isinstance(runner, AIPerfBenchmarkRunner) and self.config.benchmark.aiperf_package:
            env["AIPERF_PACKAGE"] = self.config.benchmark.aiperf_package

        return env
