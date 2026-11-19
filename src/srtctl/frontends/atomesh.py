# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official ATOMesh router frontend for ATOM and vLLM workers."""

from __future__ import annotations

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
    """Route ATOM traffic or vLLM Mooncake / MoRI-IO READ prefill/decode traffic."""

    type: ClassVar[str] = "atomesh"
    moriio_routing: ClassVar[Literal["discovery", "static-read"] | None] = "static-read"
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
        if config.frontend.enable_multiple_frontends:
            raise ValueError("ATOMesh vLLM requires frontend.enable_multiple_frontends: false")
        for mode in ("prefill", "decode"):
            if backend._get_parallel_size(mode, "data-parallel-size") != 1:
                raise ValueError(f"ATOMesh vLLM requires {mode} data-parallel-size: 1")
        if backend._get_parallel_size("prefill", "tensor-parallel-size") != backend._get_parallel_size(
            "decode", "tensor-parallel-size"
        ):
            raise ValueError("ATOMesh vLLM requires the same tensor-parallel-size on prefill and decode")
        if {"backend", "vllm-kv-connector"} & {str(key).replace("_", "-") for key in (config.frontend.args or {})}:
            raise ValueError("frontend.args.backend and vllm-kv-connector are managed by srtctl for atomesh")

    def _validate_moriio_workers(self, config: Any) -> None:
        backend = config.backend
        for mode in ("prefill", "decode"):
            row = backend.kv_connector_for_mode(mode)
            if row is None or not row.discovery:
                raise ValueError("ATOMesh MoRI-IO READ requires MoRI-IO on both prefill and decode")
            try:
                read_mode = backend.discovery_read_mode(mode)
            except TypeError as exc:
                raise ValueError(str(exc)) from exc
            if not read_mode:
                raise ValueError("ATOMesh MoRI-IO requires read_mode: true on both roles")
            if config.topology.worker_restart(mode).enabled:
                raise ValueError("ATOMesh MoRI-IO READ requires worker restart: never")
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
        if {"backend", "vllm-kv-connector"} & normalized:
            raise ValueError("frontend.args.backend and vllm-kv-connector are managed by srtctl for atomesh")
        args = ["--backend", backend.type]
        if isinstance(backend, VLLMBackend) and backend.discovers_workers():
            args.extend(["--vllm-kv-connector", "moriio-read"])
        return args
