# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official vLLM Router frontend adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from srtctl.frontends.static_router import StaticRouterFrontend

if TYPE_CHECKING:
    from srtctl.core.topology import Process


def routed_process_dp_size(backend: Any, process: Process) -> int:
    """Return the number of Router-visible DP ranks behind one base URL.

    Upstream's per-node topology exposes one URL for each node-local hybrid-LB
    pool. A model-parallel replica that spans nodes instead exposes one global
    API leader, so Router must treat that URL as one worker.
    """
    global_dp_size = int(backend._get_dp_size(process.endpoint_mode) or 1)
    if global_dp_size <= 1:
        return 1

    replica_size = backend._get_model_parallel_size(process.endpoint_mode)
    local_gpu_count = len(process.gpu_indices)
    if replica_size > local_gpu_count:
        return 1
    if local_gpu_count % replica_size != 0:
        raise ValueError(
            f"vLLM Router {process.endpoint_mode} local GPU allocation {local_gpu_count} "
            f"is not divisible by TP*PP={replica_size}"
        )
    return local_gpu_count // replica_size


def node_local_data_parallel_size(backend: Any, backend_processes: list[Process]) -> int:
    """Return Router's single DP expansion factor for all advertised URLs."""
    routed_sizes = {routed_process_dp_size(backend, process) for process in backend_processes if process.http_port > 0}
    if len(routed_sizes) > 1:
        sizes = ", ".join(str(size) for size in sorted(routed_sizes))
        raise ValueError(f"vLLM Router requires one uniform node-local DP expansion factor; derived {sizes}")
    return next(iter(routed_sizes), 1)


class VLLMRouterFrontend(StaticRouterFrontend):
    """Route aggregate or P/D traffic to direct vLLM API servers."""

    type: ClassVar[str] = "vllm-router"
    backend_type: ClassVar[str] = "vllm"
    executable: ClassVar[tuple[str, ...]] = ("vllm-router",)
    pd_flag: ClassVar[str] = "--vllm-pd-disaggregation"
    process_name: ClassVar[str] = "vllm_router"

    def get_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[str]:
        """Require every exact advertised base API to be accepting requests.

        Router can expand a partially ready hybrid-DP pool into the expected
        worker count before every base API has bound its port. This second gate
        closes that race before benchmark or eval traffic begins.
        """
        return [
            f"{worker.url.rstrip('/')}/health"
            for worker in self.collect_workers(backend, backend_processes, network_interface)
        ]

    def get_managed_frontend_args(
        self,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
    ) -> list[str]:
        """Derive topology- and health-related Router arguments."""
        frontend_args = config.frontend.args or {}
        normalized_frontend_args = {str(key).replace("_", "-") for key in frontend_args}
        managed_args: list[str] = []

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

        if "worker-startup-timeout-secs" not in normalized_frontend_args:
            health_check = config.health_check
            timeout_seconds = health_check.max_attempts * health_check.interval_seconds
            managed_args.extend(["--worker-startup-timeout-secs", str(timeout_seconds)])
        return managed_args

    def worker_bootstrap_port(self, backend: Any, process: Process) -> int | None:
        """Advertise vLLM's NIXL side-channel port for P/D routing."""
        del backend
        return process.nixl_port
