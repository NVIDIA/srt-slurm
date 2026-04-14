# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SGLang router frontend implementation.

Uses sglang_router for direct communication with backend workers.
"""

import logging
import shlex
from typing import TYPE_CHECKING, Any

from srtctl.core.health import WorkerHealthResult, check_sglang_router_health
from srtctl.core.slurm import get_hostname_ip, start_srun_process

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


class SGLangFrontend:
    """SGLang router frontend implementation.

    Uses sglang_router.launch_router for direct worker connections.
    Health checks via /workers endpoint.
    """

    @property
    def type(self) -> str:
        return "sglang"

    @property
    def health_endpoint(self) -> str:
        return "/workers"

    def parse_health(
        self,
        response_json: dict,
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        """Parse sglang /workers endpoint response."""
        return check_sglang_router_health(response_json, expected_prefill, expected_decode)

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
    ) -> list["ManagedProcess"]:
        """Start sglang routers on designated nodes.

        Supports two modes:
        - Aggregated: --worker-urls http://w1:port1 http://w2:port2 ...
        - Disaggregated: --pd-disaggregation --prefill url bootstrap_port --decode url
        """
        from srtctl.backends.sglang import SGLangProtocol
        from srtctl.core.processes import ManagedProcess

        r = config.resources
        is_disaggregated = r.num_prefill > 0 or r.num_decode > 0

        # Collect worker info by mode
        agg_workers: list[tuple[str, int]] = []  # (ip, http_port)
        prefill_leaders: list[tuple[str, int, int | None]] = []  # (ip, http_port, bootstrap_port)
        decode_leaders: list[tuple[str, int]] = []  # (ip, http_port)

        # Determine URL schemes based on gRPC mode
        prefill_scheme = "http://"
        decode_scheme = "http://"
        agg_scheme = "http://"
        if isinstance(backend, SGLangProtocol):
            if backend.is_grpc_mode("prefill"):
                prefill_scheme = "grpc://"
            if backend.is_grpc_mode("decode"):
                decode_scheme = "grpc://"
            if backend.is_grpc_mode("agg"):
                agg_scheme = "grpc://"

        for process in backend_processes:
            if not process.is_leader:
                continue
            leader_ip = get_hostname_ip(process.node)
            if process.endpoint_mode == "agg":
                agg_workers.append((leader_ip, process.http_port))
            elif process.endpoint_mode == "prefill":
                prefill_leaders.append((leader_ip, process.http_port, process.bootstrap_port))
            elif process.endpoint_mode == "decode":
                decode_leaders.append((leader_ip, process.http_port))

        processes: list[ManagedProcess] = []

        for idx, node in enumerate(topology.frontend_nodes):
            logger.info("Starting sglang-router %d on %s", idx, node)

            router_log = runtime.log_dir / f"{node}_router_{idx}.out"

            cmd = ["python", "-m", "sglang_router.launch_router"]

            if is_disaggregated:
                # Disaggregated mode: --pd-disaggregation with --prefill and --decode
                cmd.append("--pd-disaggregation")
                for ip, http_port, bootstrap_port in prefill_leaders:
                    cmd.extend(["--prefill", f"{prefill_scheme}{ip}:{http_port}"])
                    # Add bootstrap port if available
                    if bootstrap_port is not None:
                        cmd.append(str(bootstrap_port))
                for ip, http_port in decode_leaders:
                    cmd.extend(["--decode", f"{decode_scheme}{ip}:{http_port}"])
            else:
                # Aggregated mode: --worker-urls with space-separated URLs
                worker_urls = [f"{agg_scheme}{ip}:{port}" for ip, port in agg_workers]
                cmd.extend(["--worker-urls"] + worker_urls)

            cmd.extend(["--host", "0.0.0.0", "--port", str(topology.frontend_port)])
            cmd.extend(self.get_frontend_args_list(config.frontend.args))

            logger.info("Router command: %s", shlex.join(cmd))

            # Build env vars
            env_to_set: dict[str, str] = {}
            if config.frontend.env:
                env_to_set.update(config.frontend.env)

            proc = start_srun_process(
                command=cmd,
                nodelist=[node],
                output=str(router_log),
                container_image=str(runtime.container_image),
                container_mounts=runtime.container_mounts,
                env_to_set=env_to_set if env_to_set else None,
            )

            processes.append(
                ManagedProcess(
                    name=f"sglang_router_{idx}",
                    popen=proc,
                    log_file=router_log,
                    node=node,
                    critical=True,
                )
            )

        return processes
