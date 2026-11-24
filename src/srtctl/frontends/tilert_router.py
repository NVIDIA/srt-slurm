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
# ``frontend.args.rust-front: true`` runs the router under the release's Rust front supervisor instead of passing a flag:
# the supervisor serves the client port with the packaged front and moves the router to 127.0.0.1 on the next port.
RUST_FRONT_ARG = "rust-front"
RUST_FRONT_SUPERVISOR = ("python", "-m", "tilert.pd_vllm.rs_front_sup")
# The supervisor logs ``rs_front_sup FALLBACK: <reason>; ...`` when it relays the client port to the Python router
# instead (the packaged front is missing, fails its check, does not come up, or exits). With the Rust front requested,
# that line fails the run rather than benchmarking the relay.
RUST_FRONT_FALLBACK_PATTERN = r"\brs_front_sup FALLBACK: "
# TileRT 0.2.0's router sends each prefill straight to one of the prefill workers listed in this variable (in this order);
# earlier releases do not read it.
PREFILL_URLS_ENV = "TILERT_ROUTER_P_URLS"
ROUTER_WORKERS_REQUIRED = "TileRT router requires at least one prefill worker and at least one decode worker"


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
    # Set from frontend.args.rust-front when the frontends start.
    rust_front: bool = False

    @property
    def health_endpoint(self) -> str:
        return "/health"

    def validate(self, config: Any) -> None:
        if config.frontend.enable_multiple_frontends:
            raise ValueError("frontend.type: tilert-router requires frontend.enable_multiple_frontends: false")
        _reject_router_args(config.frontend.args)
        rust_front = _rust_front(config.frontend.args)
        if rust_front is not True and rust_front is not False:
            raise ValueError(f"frontend.args.{RUST_FRONT_ARG} must be true or false")
        if config.backend_for_role("decode").type != "tilert":
            raise ValueError("tilert-router requires a TileRT decode engine")
        resources = config.topology
        if resources.num_agg or resources.num_prefill < 1 or resources.num_decode < 1:
            raise ValueError(ROUTER_WORKERS_REQUIRED)
        if config.backend_for_role("prefill").type != "vllm":
            raise ValueError("tilert-router requires vLLM prefill with TileRTConnector")
        model_name = config.served_model_name
        if config.backend_for_role("decode").get_served_model_name(model_name) != model_name:
            raise ValueError("TileRT served_model_name must match the prefill served-model-name")
        if (
            resources.gpus_per_decode > resources.gpus_per_node
            and not config.backend_for_role("decode").allows_multi_node_decode
        ):
            raise ValueError(
                "TileRT decode workers must fit on one node unless roles.decode.engine.multi_node_launch is set"
            )
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

    def start_frontends(
        self,
        topology: Any,
        runtime: RuntimeContext,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
        stop_event: Any = None,
    ) -> list[Any]:
        self.rust_front = _rust_front(config.frontend.args) is True
        return super().start_frontends(topology, runtime, config, backend, backend_processes, stop_event)

    def fatal_log_patterns(self) -> tuple[str, ...]:
        """With the Rust front requested, the supervisor's fallback to the Python router fails the run."""
        return (RUST_FRONT_FALLBACK_PATTERN,) if self.rust_front else ()

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        """The router's flags; ``rust-front`` selects the launcher (`build_router_command`) and is not one."""
        return super().get_frontend_args_list({k: v for k, v in (args or {}).items() if _arg_name(k) != RUST_FRONT_ARG})

    def build_router_command(self, workers: list[RouterWorker], host: str, port: int, backend: Any) -> list[str]:
        prefills = [worker for worker in workers if worker.mode == "prefill"]
        decodes = [worker for worker in workers if worker.mode == "decode"]
        if not prefills or not decodes or len(prefills) + len(decodes) != len(workers):
            raise ValueError(ROUTER_WORKERS_REQUIRED)

        decode_specs: list[str] = []
        for worker in decodes:
            parsed = urlsplit(worker.url)
            if parsed.hostname is None or parsed.port is None or worker.bootstrap_port is None:
                raise ValueError(f"Incomplete TileRT decode endpoint: {worker}")
            decode_specs.append(f"{parsed.hostname}:{worker.bootstrap_port}:{parsed.port}")
        # --decode is a single nargs="+" option; repeating the flag would keep only the last worker.
        # --vllm-url takes every prefill worker as one comma-separated value (TileRT 0.1.8 pd_router splits it on
        # commas; 0.1.6 accepts one URL); one worker renders its URL alone. PREFILL_URLS_ENV carries the same list.
        prefill_urls = ",".join(worker.url for worker in prefills)
        return [
            "env",
            f"{PREFILL_URLS_ENV}={prefill_urls}",
            *(RUST_FRONT_SUPERVISOR if self.rust_front else ()),
            *self.executable,
            "--vllm-url",
            prefill_urls,
            "--decode",
            *decode_specs,
            "--host",
            host,
            "--port",
            str(port),
        ]

    def get_managed_frontend_args(
        self, config: Any, backend: Any, backend_processes: list[Process], runtime: RuntimeContext | None = None
    ) -> list[str]:
        _reject_router_args(config.frontend.args)
        return []


def _arg_name(key: Any) -> str:
    """The flag a ``frontend.args`` key names, however the recipe spells it (``rust_front``, ``--rust-front``)."""
    return str(key).lstrip("-").replace("_", "-")


def _rust_front(args: dict[str, Any] | None) -> Any:
    """The recipe's ``rust-front`` value under any spelling of the key; ``False`` when unset."""
    return next((value for key, value in (args or {}).items() if _arg_name(key) == RUST_FRONT_ARG), False)


def _reject_router_args(args: dict[str, Any] | None) -> None:
    overlap = {_arg_name(key) for key in (args or {})}.intersection(ROUTER_MANAGED_ARGS)
    if overlap:
        raise ValueError(f"frontend.args cannot override srtctl-managed TileRT router argument(s): {sorted(overlap)}")
