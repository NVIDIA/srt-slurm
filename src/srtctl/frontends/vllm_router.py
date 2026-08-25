# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM Router frontend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from srtctl.frontends.base import register_frontend
from srtctl.frontends.static_router import StaticRouterFrontend

if TYPE_CHECKING:
    from srtctl.core.topology import Process


def node_local_data_parallel_size(backend: Any, backend_processes: list[Process]) -> int:
    """Return Router's single node-local DP expansion factor."""
    grouped_processes: dict[tuple[str, int], list[Process]] = {}
    for process in backend_processes:
        if process.http_port > 0:
            grouped_processes.setdefault((process.endpoint_mode, process.endpoint_index), []).append(process)

    local_dp_sizes: set[int] = set()
    for (mode, _endpoint_index), processes in grouped_processes.items():
        global_dp_size = int(backend._get_dp_size(mode) or 1)
        derive_local_dp_size = getattr(type(backend), "_get_local_dp_size", None)
        if derive_local_dp_size is None:
            # Keep lightweight protocol doubles usable while production vLLMProtocol
            # derives this from TP*PP*PCP and the actual local GPU allocation.
            if global_dp_size % len(processes) != 0:
                raise ValueError(
                    f"vLLM Router {mode} data-parallel-size={global_dp_size} cannot be evenly split "
                    f"across {len(processes)} routed servers"
                )
            process_local_dp_sizes = {global_dp_size // len(processes)}
        else:
            process_local_dp_sizes = {
                derive_local_dp_size(backend, mode, len(process.gpu_indices)) for process in processes
            }
        if len(process_local_dp_sizes) != 1:
            raise ValueError(f"vLLM Router {mode} endpoint has non-uniform node-local DP sizes")
        local_dp_size = next(iter(process_local_dp_sizes))
        if global_dp_size != local_dp_size * len(processes):
            raise ValueError(
                f"vLLM Router {mode} data-parallel-size={global_dp_size} does not match "
                f"{len(processes)} routed servers * local DP {local_dp_size}"
            )
        local_dp_sizes.add(local_dp_size)

    if len(local_dp_sizes) > 1:
        raise ValueError("vLLM Router requires the same node-local data-parallel size for every routed backend")
    return next(iter(local_dp_sizes), 1)


@register_frontend("vllm-router")
class VLLMRouterFrontend(StaticRouterFrontend):
    """Route requests to direct vLLM OpenAI-compatible worker endpoints."""

    type: ClassVar[str] = "vllm-router"
    backend_type: ClassVar[str] = "vllm"
    executable: ClassVar[tuple[str, ...]] = ("vllm-router",)
    pd_flag: ClassVar[str] = "--vllm-pd-disaggregation"
    process_name: ClassVar[str] = "vllm_router"

    def get_backend_health_urls(self, backend: Any, backend_processes: list[Process]) -> list[str]:
        """Return the exact logical vLLM endpoints advertised to Router.

        Router expands node-local DP pools internally, but its worker-count view
        can become complete before every advertised HTTP server is accepting
        requests. Polling each logical server closes that readiness race without
        changing the semantics of other frontend adapters.
        """
        return [f"{worker.url.rstrip('/')}/health" for worker in self.collect_workers(backend, backend_processes)]

    def get_managed_frontend_args(
        self,
        config: Any,
        backend: Any | None = None,
        backend_processes: list[Process] | None = None,
    ) -> list[str]:
        """Derive Router DP expansion and worker-readiness arguments."""
        frontend_args = config.frontend.args or {}
        managed_args: list[str] = []

        if backend is not None and backend_processes is not None:
            local_dp_size = node_local_data_parallel_size(backend, backend_processes)
            configured_dp_size = frontend_args.get(
                "intra-node-data-parallel-size",
                frontend_args.get("intra_node_data_parallel_size"),
            )
            if configured_dp_size is not None and int(configured_dp_size) != local_dp_size:
                raise ValueError(
                    "frontend.args.intra-node-data-parallel-size conflicts with the allocated vLLM topology: "
                    f"configured {configured_dp_size}, derived {local_dp_size}"
                )
            if local_dp_size > 1 and configured_dp_size is None:
                managed_args.extend(["--intra-node-data-parallel-size", str(local_dp_size)])

        if "worker-startup-timeout-secs" not in frontend_args:
            health_check = config.health_check
            timeout_seconds = health_check.max_attempts * health_check.interval_seconds
            managed_args.extend(["--worker-startup-timeout-secs", str(timeout_seconds)])
        return managed_args

    def worker_bootstrap_port(self, backend: Any, process: Process) -> int | None:
        """Advertise vLLM's NIXL side-channel port to the P/D router."""
        return process.nixl_port
