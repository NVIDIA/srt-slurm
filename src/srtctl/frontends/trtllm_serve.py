# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
trtllm-serve disaggregated frontend implementation.

Runs `trtllm-serve disaggregated` as the router. Unlike dynamo (which discovers
workers via etcd/NATS), trtllm-serve needs a static config (ser.yaml) listing the
context (prefill) and generation (decode) server URLs. We build that from the
backend worker leaders, then launch the orchestrator on the head frontend node.
"""

import logging
import shlex
import threading
from typing import TYPE_CHECKING, Any

import yaml

from srtctl.core.health import WorkerHealthResult, check_trtllm_serve_health, wait_for_health
from srtctl.core.slurm import get_hostname_ip, start_srun_process

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


class TRTLLMServeFrontend:
    """trtllm-serve disaggregated frontend.

    Launches `trtllm-serve disaggregated --config ser.yaml` on the head node,
    where ser.yaml lists prefill (context_servers) and decode (generation_servers)
    worker URLs collected from the backend processes. Health via /health.
    """

    @property
    def type(self) -> str:
        return "trtllm_serve"

    @property
    def health_endpoint(self) -> str:
        return "/health"

    def parse_health(
        self,
        response_json: dict,
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        """Parse trtllm-serve /health response (200 => ready)."""
        return check_trtllm_serve_health(response_json, expected_prefill, expected_decode)

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        """Convert frontend args dict to CLI arguments."""
        if not args:
            return []
        result = []
        for key, value in args.items():
            if value is True:
                result.append(f"--{key}")
            elif value is not False and value is not None:
                result.extend([f"--{key}", str(value)])
        return result

    def start_frontends(
        self,
        topology: Any,  # FrontendTopology
        runtime: "RuntimeContext",
        config: Any,  # SrtConfig
        backend: Any,  # BackendProtocol
        backend_processes: list["Process"],
        stop_event: "threading.Event | None" = None,
    ) -> list["ManagedProcess"]:
        """Write ser.yaml from worker leaders and launch the disaggregated orchestrator."""
        from srtctl.core.processes import ManagedProcess

        # trtllm-serve disaggregated fronts trtllm workers; it can't route to other backends.
        if config.backend.type != "trtllm":
            raise ValueError(f"frontend.type: trtllm_serve requires backend.type: trtllm (got {config.backend.type!r})")

        # trtllm-serve disaggregated is a single orchestrator process; the nginx +
        # multi-frontend path is not supported. uses_nginx also catches the 2-node case
        # where the topology is nginx + a single frontend node (frontend_nodes len == 1).
        if topology.uses_nginx or len(topology.frontend_nodes) != 1:
            raise ValueError(
                "trtllm_serve frontend runs a single disaggregated orchestrator and does "
                "not support the nginx/multi-frontend path; set "
                "frontend.enable_multiple_frontends: false"
            )
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
        ser = {
            "context_servers": {"num_instances": len(prefill_urls), "urls": prefill_urls},
            "generation_servers": {"num_instances": len(decode_urls), "urls": decode_urls},
            "hostname": "0.0.0.0",
            "port": topology.frontend_port,
        }
        host_ser_path = runtime.log_dir / "ser.yaml"
        host_ser_path.write_text(yaml.safe_dump(ser, sort_keys=False))
        logger.info("Wrote trtllm-serve disagg config:\n%s", host_ser_path.read_text())
        container_ser_path = "/logs/ser.yaml"

        cmd = ["trtllm-serve", "disaggregated", "--config", container_ser_path]
        cmd.extend(self.get_frontend_args_list(config.frontend.args))
        logger.info("Orchestrator command: %s", shlex.join(cmd))

        env_to_set: dict[str, str] = {}
        if config.frontend.env:
            env_to_set.update(config.frontend.env)

        orch_log = runtime.log_dir / f"{frontend_node}_trtllm_serve_orchestrator.out"
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
        )

        return [
            ManagedProcess(
                name="trtllm_serve_orchestrator",
                popen=proc,
                log_file=orch_log,
                node=frontend_node,
                critical=True,
            )
        ]
