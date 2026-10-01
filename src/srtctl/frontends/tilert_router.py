# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TileRT's native P/D router in front of compatible prefill and TileRT decode workers."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit

from srtctl.core.health import WorkerHealthResult
from srtctl.frontends.base import register_frontend
from srtctl.frontends.static_router import RouterWorker, StaticRouterFrontend

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

ROUTER_MANAGED_ARGS = frozenset({"vllm-url", "decode", "host", "port"})


@register_frontend("tilert-router")
class TileRTRouterFrontend(StaticRouterFrontend):
    """Launch ``tilert.pd_vllm.pd_router`` once every worker answers ``/health``."""

    type: ClassVar[str] = "tilert-router"
    required_backend: ClassVar[str | None] = None
    model_name_role: ClassVar[str | None] = "prefill"
    executable: ClassVar[tuple[str, ...]] = ("python", "-m", "tilert.pd_vllm.pd_router")
    pd_flag: ClassVar[str] = ""
    process_name: ClassVar[str] = "tilert_router"
    # The router connects to its workers at startup.
    wait_for_workers_before_start: ClassVar[bool] = True

    @property
    def health_endpoint(self) -> str:
        return "/health"

    def validate(self, config: Any) -> None:
        if config.frontend.enable_multiple_frontends:
            raise ValueError("frontend.type: tilert-router requires frontend.enable_multiple_frontends: false")
        _reject_router_args(config.frontend.args)
        if config.backend_for_role("decode").type != "tilert":
            raise ValueError("tilert-router requires a TileRT decode engine")
        resources = config.topology
        if resources.num_agg or resources.num_prefill != 1 or resources.num_decode < 1:
            raise ValueError("TileRT router requires exactly one prefill worker and at least one decode worker")
        if config.backend_for_role("prefill").type != "vllm":
            raise ValueError("tilert-router requires vLLM prefill with TileRTConnector")
        model_name = config.served_model_name
        if config.backend_for_role("decode").get_served_model_name(model_name) != model_name:
            raise ValueError("TileRT served_model_name must match the prefill served-model-name")
        if resources.gpus_per_decode > resources.gpus_per_node:
            raise ValueError("TileRT decode workers must fit on one node")
        if config.profiling.is_torch:
            raise ValueError("TileRT does not support torch profiling")

    def parse_health(self, response_json: dict, expected_prefill: int, expected_decode: int) -> WorkerHealthResult:
        ready = response_json.get("status") == "ok"
        return WorkerHealthResult(
            ready=ready,
            message="TileRT P/D router healthy" if ready else "TileRT P/D router not ready",
            prefill_ready=expected_prefill if ready else 0,
            prefill_expected=expected_prefill,
            decode_ready=expected_decode if ready else 0,
            decode_expected=expected_decode,
        )

    def worker_metrics_port(self, process: Process, runtime: RuntimeContext) -> int | None:
        """Only vLLM prefill serves Prometheus metrics; TileRT decode has no /metrics route."""
        if process.endpoint_mode == "prefill" and process.is_leader and process.http_port > 0:
            return process.http_port
        return None

    def worker_bootstrap_port(self, backend: Any, process: Process) -> int | None:
        return process.nixl_port if process.endpoint_mode == "decode" else None

    def build_bash_preamble(self, config: Any) -> str | None:
        """Apply recipe setup in the router's separate container too."""
        if not config.setup_script:
            return None
        return f"bash {shlex.quote(str(Path('/configs') / config.setup_script))}"

    def get_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[str]:
        """A healthy router does not prove its workers are; require every worker /health as well."""
        return [
            f"{worker.url}/health" for worker in self.collect_workers(backend, backend_processes, network_interface)
        ]

    def build_router_command(self, workers: list[RouterWorker], host: str, port: int, backend: Any) -> list[str]:
        prefills = [worker for worker in workers if worker.mode == "prefill"]
        decodes = [worker for worker in workers if worker.mode == "decode"]
        if len(prefills) != 1 or not decodes or len(prefills) + len(decodes) != len(workers):
            raise ValueError("TileRT router requires exactly one prefill worker and at least one decode worker")

        decode_specs: list[str] = []
        for worker in decodes:
            parsed = urlsplit(worker.url)
            if parsed.hostname is None or parsed.port is None or worker.bootstrap_port is None:
                raise ValueError(f"Incomplete TileRT decode endpoint: {worker}")
            decode_specs.append(f"{parsed.hostname}:{worker.bootstrap_port}:{parsed.port}")
        # --decode is a single nargs="+" option; repeating the flag would keep only the last worker.
        return [
            *self.executable,
            "--vllm-url",
            prefills[0].url,
            "--decode",
            *decode_specs,
            "--host",
            host,
            "--port",
            str(port),
        ]

    def get_managed_frontend_args(self, config: Any, backend: Any, backend_processes: list[Process]) -> list[str]:
        _reject_router_args(config.frontend.args)
        return []


def _reject_router_args(args: dict[str, Any] | None) -> None:
    overlap = {str(key).lstrip("-").replace("_", "-") for key in (args or {})}.intersection(ROUTER_MANAGED_ARGS)
    if overlap:
        raise ValueError(f"frontend.args cannot override srtctl-managed TileRT router argument(s): {sorted(overlap)}")
