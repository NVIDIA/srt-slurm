# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Direct and disaggregated trtllm-serve frontend implementation.

For aggregate jobs, the one backend worker owns the public OpenAI port and no
separate frontend process is needed. For disaggregated jobs, this runs
`trtllm-serve disaggregated` as the router. Unlike Dynamo (which discovers workers
via etcd/NATS), the disaggregated server needs a static config (ser.yaml) listing
the context (prefill) and generation (decode) server URLs.
"""

import logging
import shlex
import threading
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import yaml

from srtctl.core.health import WorkerHealthResult, probe_http_ok, wait_for_health
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.frontends.base import frontend_args_to_cli, logical_health_expectations, register_frontend

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process
    from srtctl.services.implicit import EffectiveService

logger = logging.getLogger(__name__)


@register_frontend("trtllm_serve")
class TRTLLMServeFrontend:
    """Direct aggregate or disaggregated trtllm-serve frontend.

    Aggregate mode launches no extra process because the worker itself binds the
    public port. Disaggregated mode launches `trtllm-serve disaggregated --config
    ser.yaml` on the head node. Health is exposed at /health in both modes.
    """

    required_backend: ClassVar[str | None] = "trtllm"
    worker_launch: ClassVar[Literal["dynamo", "direct"]] = "direct"
    expands_node_local_dp: ClassVar[bool] = False

    @property
    def type(self) -> str:
        return "trtllm_serve"

    def worker_api_port(self, mode: str) -> Literal["public", "allocated"]:
        """The aggregate worker is the endpoint; P/D workers sit behind the disaggregated orchestrator."""
        return "public" if mode == "agg" else "allocated"

    # trtllm-serve (worker and disaggregated orchestrator alike) serves Prometheus
    # text at /prometheus/metrics; GET /metrics on a worker is JSON iteration stats.
    metrics_path: ClassVar[str] = "/prometheus/metrics"

    def worker_metrics_port(self, process: "Process", runtime: "RuntimeContext") -> int | None:
        """P/D leaders serve Prometheus on their OpenAI port; followers bind nothing. Aggregate is out of scope."""
        if process.endpoint_mode == "agg" or process.http_port <= 0:
            return None
        return process.http_port

    def worker_endpoint_port(self, process: "Process", config: Any, runtime: "RuntimeContext") -> int | None:
        if not process.is_leader:
            return None
        port = runtime.frontend_port if self.worker_api_port(process.endpoint_mode) == "public" else process.http_port
        return port if port > 0 else None

    def profiling_control_port(self, process: "Process", config: Any, runtime: "RuntimeContext") -> int | None:
        return self.worker_endpoint_port(process, config, runtime)

    def profiling_control_is_leader_only(self, config: Any) -> bool:
        return False

    def direct_endpoint_nodes(self, processes: list["Process"]) -> list[str]:
        return []

    def worker_ready_port(self, process: "Process") -> int:
        """A trtllm-serve worker reports /health on its own OpenAI port."""
        return process.http_port

    def probe_ready(self, host: str, port: int, expected_prefill: int, expected_decode: int) -> WorkerHealthResult:
        """A 200 from /health is ready: the body may be empty, and every worker was gated before the orchestrator started."""
        return probe_http_ok(host, port, "/health", f"trtllm-serve frontend healthy at http://{host}:{port}/health")

    def health_expectations(self, config: Any, processes: list["Process"] | None) -> tuple[int, int, str]:
        return logical_health_expectations(config)

    def validate(self, config: Any) -> None:
        """One direct aggregate worker or one disaggregated orchestrator; either way one public endpoint."""
        if config.frontend.enable_multiple_frontends:
            raise ValueError(
                "frontend.type: trtllm_serve uses one public endpoint; set frontend.enable_multiple_frontends: false"
            )
        if not config.resources.is_disaggregated and config.resources.num_agg != 1:
            raise ValueError(
                "frontend.type: trtllm_serve aggregate mode requires exactly one "
                "aggregate worker (set resources.agg_workers: 1)"
            )

    def get_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list["Process"],
        network_interface: str | None = None,
    ) -> list[str]:
        return []

    def implied_services(self, config: Any) -> list["EffectiveService"]:
        return []

    def frontend_metrics_port(self, frontend_args: dict[str, Any] | None) -> int | None:
        return None

    @staticmethod
    def _build_ser(config: Any, prefill_urls: list[str], decode_urls: list[str], port: int) -> dict[str, Any]:
        """Build the trtllm-serve disaggregated ser.yaml, merging the optional
        orchestrator-side router / server_config_extra from frontend config."""
        context_servers: dict[str, Any] = {"num_instances": len(prefill_urls), "urls": prefill_urls}
        generation_servers: dict[str, Any] = {"num_instances": len(decode_urls), "urls": decode_urls}
        ser: dict[str, Any] = {
            "context_servers": context_servers,
            "generation_servers": generation_servers,
            "hostname": "0.0.0.0",
            "port": port,
        }
        fe = config.frontend
        if getattr(fe, "ctx_router", None):
            context_servers["router"] = dict(fe.ctx_router)
        if getattr(fe, "gen_router", None):
            generation_servers["router"] = dict(fe.gen_router)
        if getattr(fe, "server_config_extra", None):
            ser.update(dict(fe.server_config_extra))
        return ser

    def start_frontends(
        self,
        topology: Any,  # FrontendTopology
        runtime: "RuntimeContext",
        config: Any,  # SrtConfig
        backend: Any,  # BackendProtocol
        backend_processes: list["Process"],
        stop_event: "threading.Event | None" = None,
    ) -> list["ManagedProcess"]:
        """Use the aggregate worker directly or launch the disaggregated orchestrator."""
        from srtctl.core.processes import FRONTEND_TERMINATE_TIMEOUT_SECONDS, ManagedProcess

        # trtllm-serve disaggregated fronts trtllm workers; it can't route to other backends.
        if config.backend.type != "trtllm":
            raise ValueError(f"frontend.type: trtllm_serve requires backend.type: trtllm (got {config.backend.type!r})")

        # trtllm-serve exposes one public endpoint: either the aggregate worker or
        # a disaggregated orchestrator. The nginx + multi-frontend path is not
        # supported. uses_nginx also catches the two-node case where the topology
        # is nginx + one frontend node (frontend_nodes len == 1).
        if topology.uses_nginx or len(topology.frontend_nodes) != 1:
            raise ValueError(
                "trtllm_serve uses one public endpoint and does not support "
                "the nginx/multi-frontend path; set "
                "frontend.enable_multiple_frontends: false"
            )

        if not config.resources.is_disaggregated:
            agg_leaders = [
                process for process in backend_processes if process.endpoint_mode == "agg" and process.is_leader
            ]
            if len(agg_leaders) != 1:
                raise ValueError(
                    f"trtllm_serve aggregate mode requires exactly one aggregate worker (got {len(agg_leaders)})"
                )
            logger.info(
                "frontend.type=trtllm_serve: no separate frontend process; aggregate trtllm-serve owns port %d",
                topology.public_port,
            )
            return []

        frontend_node = topology.frontend_nodes[0]

        # Collect prefill/decode worker URLs from endpoint leaders.
        prefill_urls: list[str] = []
        decode_urls: list[str] = []
        for process in backend_processes:
            if not process.is_leader:
                continue
            url = f"{get_hostname_ip(process.node)}:{process.http_port}"
            if process.endpoint_mode == "prefill":
                prefill_urls.append(url)
            elif process.endpoint_mode == "decode":
                decode_urls.append(url)
        if not prefill_urls or not decode_urls:
            raise ValueError(
                f"trtllm_serve requires disaggregated prefill and decode workers "
                f"(got {len(prefill_urls)} prefill, {len(decode_urls)} decode)"
            )

        # Wait for each worker's OpenAI endpoint to come up before starting the
        # orchestrator (it does not retry unreachable workers).
        for url in prefill_urls + decode_urls:
            host, port = url.rsplit(":", 1)
            logger.info("Waiting for trtllm-serve worker %s", url)
            if not wait_for_health(
                host,
                int(port),
                max_attempts=config.health_check.max_attempts,
                interval=config.health_check.interval_seconds,
                stop_event=stop_event,
            ):
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError("trtllm-serve worker wait aborted")
                raise RuntimeError(f"trtllm-serve worker {url} did not become healthy")

        # Build ser.yaml (host path in log_dir, mounted to /logs in the container).
        ser = self._build_ser(config, prefill_urls, decode_urls, topology.frontend_port)
        host_ser_path = runtime.log_dir / "ser.yaml"
        host_ser_path.write_text(yaml.safe_dump(ser, sort_keys=False))
        logger.info("Wrote trtllm-serve disagg config:\n%s", host_ser_path.read_text())
        container_ser_path = "/logs/ser.yaml"

        cmd = ["trtllm-serve", "disaggregated", "--config", container_ser_path]
        cmd.extend(frontend_args_to_cli(config.frontend.args))
        logger.info("Orchestrator command: %s", shlex.join(cmd))

        env_to_set: dict[str, str] = {}
        if config.frontend.env:
            env_to_set.update(config.frontend.env)

        # Keep the Dynamo frontend's log naming pattern ({node}_frontend_{i}.out)
        # so downstream tooling that globs *_frontend_*.out (perf dashboard,
        # log collection) treats both frontends identically.
        orch_log = runtime.log_dir / f"{frontend_node}_frontend_0.out"
        step_name = "trtllm_serve_orchestrator"
        proc = start_srun_process(
            command=cmd,
            nodelist=[frontend_node],
            output=str(orch_log),
            container_image=str(runtime.container_image),
            container_mounts=runtime.container_mounts,
            env_to_set=env_to_set if env_to_set else None,
            # trtllm-serve imports tensorrt_llm, which requires an MPI launcher even
            # for the single-rank orchestrator (same reason the dynamo frontend uses it).
            mpi="pmix",
            het_group=runtime.nodes.het_group_for(frontend_node),
            step_name=step_name,
        )

        return [
            ManagedProcess(
                name=step_name,
                popen=proc,
                log_file=orch_log,
                node=frontend_node,
                critical=True,
                terminate_timeout=FRONTEND_TERMINATE_TIMEOUT_SECONDS,
                step_name=step_name,
            )
        ]
