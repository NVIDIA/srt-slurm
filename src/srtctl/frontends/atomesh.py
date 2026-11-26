# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official ATOMesh router frontend for ATOM and vLLM workers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from srtctl.backends.atom import AtomBackend
from srtctl.backends.vllm import VLLMBackend
from srtctl.frontends.base import register_frontend
from srtctl.frontends.static_router import StaticRouterFrontend
from srtctl.frontends.vllm_common import validate_mooncake_workers

if TYPE_CHECKING:
    from srtctl.backends.base import WorkerMode
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process


@register_frontend("atomesh")
class AtomeshFrontend(StaticRouterFrontend):
    """Route ATOM traffic or vLLM Mooncake / MoRI-IO READ or WRITE prefill/decode traffic."""

    type: ClassVar[str] = "atomesh"
    moriio_routing: ClassVar[Literal["discovery", "static"] | None] = "static"
    supports_mooncake_bootstrap: ClassVar[bool] = True
    executable: ClassVar[tuple[str, ...]] = ("atomesh", "launch")
    pd_flag: ClassVar[str] = "--pd-disaggregation"
    process_name: ClassVar[str] = "atomesh"
    wait_for_workers_before_start: ClassVar[bool] = True

    def validate(self, config: Any) -> None:
        backend = config.backend
        if isinstance(backend, AtomBackend):
            if any(not isinstance(role_backend, AtomBackend) for _, role_backend in config.active_role_backends()):
                raise ValueError("ATOMesh requires the same backend type on every worker role")
            return
        if not isinstance(backend, VLLMBackend):
            raise ValueError("frontend.type: atomesh requires backend.type: atom or vllm")  # noqa: TRY004 - recipe validation
        if config.role_backends:
            raise ValueError("ATOMesh vLLM requires one shared engine configuration")
        if backend.discovers_workers():
            self._validate_moriio_workers(config)
        else:
            if any(backend.bootstrap_connector(mode) is None for mode in ("prefill", "decode")):
                raise ValueError("ATOMesh vLLM requires Mooncake on both prefill and decode")
            validate_mooncake_workers(config)
            if backend.is_external_lb("decode"):
                raise ValueError("ATOMesh Mooncake requires decode internal load balancing")
        if config.frontend.enable_multiple_frontends:
            raise ValueError("ATOMesh vLLM requires frontend.enable_multiple_frontends: false")
        if {"backend", "vllm-kv-connector", "vllm-moriio-peers"} & {
            str(key).replace("_", "-") for key in (config.frontend.args or {})
        }:
            raise ValueError(
                "frontend.args.backend, vllm-kv-connector and vllm-moriio-peers are managed by srtctl for atomesh"
            )

    def _validate_moriio_workers(self, config: Any) -> None:
        backend = config.backend
        if backend._get_parallel_size("prefill", "tensor-parallel-size") != backend._get_parallel_size(
            "decode", "tensor-parallel-size"
        ):
            raise ValueError("ATOMesh MoRI-IO requires the same tensor-parallel-size on prefill and decode")
        modes = {}
        for mode in ("prefill", "decode"):
            row = backend.kv_connector_for_mode(mode)
            if row is None or not row.discovery:
                raise ValueError("ATOMesh MoRI-IO requires MoRI-IO on both prefill and decode")
            try:
                modes[mode] = backend.discovery_read_mode(mode)
            except TypeError as exc:
                raise ValueError(str(exc)) from exc
            if config.topology.worker_restart(mode).enabled:
                raise ValueError("ATOMesh MoRI-IO requires worker restart: never")
        if modes["prefill"] != modes["decode"]:
            raise ValueError("ATOMesh MoRI-IO requires matching read_mode on both roles")
        read_mode = modes["prefill"]
        for mode in ("prefill", "decode"):
            if (read_mode or mode == "prefill") and backend._get_parallel_size(mode, "data-parallel-size") != 1:
                raise ValueError(f"ATOMesh MoRI-IO requires {mode} data-parallel-size: 1")
            if not read_mode:
                gpus = config.topology.gpus_per_prefill if mode == "prefill" else config.topology.gpus_per_decode
                if gpus > config.resources.gpus_per_node or backend._get_local_dp_size(
                    mode, gpus
                ) != backend._get_parallel_size(mode, "data-parallel-size"):
                    raise ValueError("ATOMesh MoRIIO WRITE requires each worker's DP pool to fit on one node")
        if not read_mode and (
            backend._get_parallel_size("decode", "pipeline-parallel-size") != 1 or backend.is_external_lb("decode")
        ):
            raise ValueError("ATOMesh MoRIIO WRITE requires decode PP=1 and internal DP load balancing")
        if config.topology.num_agg:
            raise ValueError("ATOMesh MoRI-IO requires a prefill/decode topology")
        if config.health_check is None:
            raise ValueError("ATOMesh MoRI-IO requires health_check for every static worker")

    def worker_metrics_path(self, backend: Any, mode: WorkerMode) -> str | None:
        """ATOM has no worker metrics endpoint; vLLM serves its native metrics."""
        if isinstance(backend, AtomBackend):
            return None
        return super().worker_metrics_path(backend, mode)

    def worker_bootstrap_port(self, backend: Any, process: Process) -> int | None:
        """ATOM exposes transfer topology through ``/kv_transfer_info``."""
        if (
            isinstance(backend, VLLMBackend)
            and process.endpoint_mode == "prefill"
            and backend.bootstrap_connector("prefill") is not None
        ):
            if process.bootstrap_port is None:
                raise ValueError("ATOMesh vLLM prefill is missing its allocated Mooncake bootstrap port")
            return process.bootstrap_port
        return None

    def get_backend_health_urls(
        self, backend: Any, backend_processes: list[Process], network_interface: str | None = None
    ) -> list[str]:
        if not isinstance(backend, VLLMBackend):
            return []
        return [
            f"{worker.url.rstrip('/')}/health"
            for worker in self.collect_workers(backend, backend_processes, network_interface)
        ]

    def get_managed_frontend_args(
        self,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
        runtime: RuntimeContext | None = None,
    ) -> list[str]:
        normalized = {str(key).replace("_", "-") for key in (config.frontend.args or {})}
        if {"backend", "vllm-kv-connector", "vllm-moriio-peers"} & normalized:
            raise ValueError(
                "frontend.args.backend, vllm-kv-connector and vllm-moriio-peers are managed by srtctl for atomesh"
            )
        args = ["--backend", backend.type]
        if isinstance(backend, VLLMBackend) and backend.discovers_workers():
            read_mode = backend.discovery_read_mode("prefill")
            args.extend(["--vllm-kv-connector", "moriio-read" if read_mode else "moriio-write"])
            if not read_mode:
                if runtime is None:
                    raise ValueError("MoRIIO WRITE peer metadata requires allocated runtime ports")
                peers = []
                for process in backend_processes:
                    if process.http_port <= 0:
                        continue
                    workers = self.collect_workers(backend, [process], runtime.network_interface)
                    if len(workers) != 1:
                        raise ValueError("Expected one static MoRIIO peer per API process")
                    extra = backend._discovery_extra_config(process, runtime, register_with_router=False)
                    peers.append(
                        {
                            "url": workers[0].url,
                            "host": extra["host_ip"],
                            "handshake_port": int(extra["handshake_port"]),
                            "notify_port": int(extra["notify_port"]),
                            "tp_size": backend._get_parallel_size(process.endpoint_mode, "tensor-parallel-size"),
                            "dp_size": backend._get_parallel_size(process.endpoint_mode, "data-parallel-size"),
                        }
                    )
                args.extend(["--vllm-moriio-peers", json.dumps(peers, separators=(",", ":"))])
        return args
