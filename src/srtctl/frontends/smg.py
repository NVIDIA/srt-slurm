# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shepherd Model Gateway router frontend (`frontend.type: smg`).

SMG (https://github.com/smg-project/smg) takes its workers as static URLs on the
command line (``smg launch --worker-urls ...`` or ``--pd-disaggregation --prefill
URL [BOOTSTRAP_PORT] --decode URL``), lists them at ``GET /workers`` with
``stats.{prefill,decode,regular}_count``, and detects each HTTP worker's engine
itself, so the static-router base covers launch, readiness, and worker shape for
every backend.
"""

from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar

from srtctl.backends.tokenspeed import TokenSpeedBackend
from srtctl.frontends.base import register_frontend
from srtctl.frontends.static_router import StaticRouterFrontend, setup_script_preamble
from srtctl.ports import SMG_METRICS_PORT

if TYPE_CHECKING:
    import threading

    from srtctl.backends.base import WorkerMode
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process


@register_frontend("smg")
class SMGFrontend(StaticRouterFrontend):
    """Shepherd Model Gateway in front of any engine's direct HTTP workers."""

    type: ClassVar[str] = "smg"
    executable: ClassVar[tuple[str, ...]] = ("smg", "launch")
    pd_flag: ClassVar[str] = "--pd-disaggregation"
    process_name: ClassVar[str] = "smg"
    # SMG waits at most --worker-startup-timeout-secs (1800 s by default) for a startup
    # worker to register; a large model can load for longer, so every worker is probed first.
    wait_for_workers_before_start: ClassVar[bool] = True

    def validate(self, config: Any) -> None:
        """Reject prefill/decode layouts SMG cannot disaggregate (see docs/smg.md)."""
        if isinstance(config.backend, TokenSpeedBackend):
            if config.frontend.enable_multiple_frontends or config.frontend.placement.location != "head":
                raise ValueError(
                    "TokenSpeed ZMQ requires one SMG frontend on head; set enable_multiple_frontends: false"
                )
            frontend_args = {str(key).replace("_", "-"): value for key, value in (config.frontend.args or {}).items()}
            if frontend_args.get("backend", "tokenspeed") != "tokenspeed":
                raise ValueError("TokenSpeed ZMQ requires frontend.args.backend: tokenspeed")
            managed = {"data-parallel-address", "data-parallel-rpc-port", "zmq-engine-index", "metrics-port"}
            for mode in config.roles:
                conflicts = managed.intersection(config.backend_for_role(mode).get_config_for_mode(mode))
                if conflicts:
                    raise ValueError(f"TokenSpeed ZMQ manages roles.{mode}.args: {', '.join(sorted(conflicts))}")
        if not config.topology.is_disaggregated:
            return
        prefill, decode = config.backend_for_role("prefill"), config.backend_for_role("decode")
        if prefill.type != decode.type:
            raise ValueError("frontend.type: smg prefill/decode needs the same engine in both roles")
        if prefill.type == "trtllm":
            raise ValueError("frontend.type: smg does not support TRT-LLM prefill/decode")
        if prefill.type == "vllm" and not (prefill.is_grpc_mode("prefill") and decode.is_grpc_mode("decode")):
            raise ValueError("frontend.type: smg runs vLLM prefill/decode only over gRPC; set grpc: true in both roles")

    def worker_scheme(self, backend: Any, mode: "WorkerMode") -> str:
        """TokenSpeed dials SMG's TCP handshake; other engines retain their HTTP/gRPC APIs."""
        return "tcp" if isinstance(backend, TokenSpeedBackend) else super().worker_scheme(backend, mode)

    def worker_bootstrap_port(self, backend: Any, process: "Process") -> int | None:
        """ZMQ engines declare their actual P/D bootstrap host and port in the handshake."""
        return None if isinstance(backend, TokenSpeedBackend) else super().worker_bootstrap_port(backend, process)

    def worker_metrics_port(self, process: "Process", runtime: "RuntimeContext") -> int | None:
        """Headless TokenSpeed serves each node's metrics independently of the gateway."""
        if process.tokenspeed_zmq_port is not None:
            return process.tokenspeed_metrics_port
        return super().worker_metrics_port(process, runtime)

    def worker_endpoint_port(self, process: "Process", config: Any, runtime: "RuntimeContext") -> int | None:
        if process.tokenspeed_zmq_port is not None:
            return None
        return super().worker_endpoint_port(process, config, runtime)

    def profiling_control_port(self, process: "Process", config: Any, runtime: "RuntimeContext") -> int | None:
        if process.tokenspeed_zmq_port is not None:
            return None
        return super().profiling_control_port(process, config, runtime)

    def start_frontends(
        self,
        topology: Any,
        runtime: "RuntimeContext",
        config: Any,
        backend: Any,
        backend_processes: "list[Process]",
        stop_event: "threading.Event | None" = None,
    ) -> "list[ManagedProcess]":
        if isinstance(backend, TokenSpeedBackend):
            # Only the router's URL view moves: SMG binds these sockets; workers and
            # metrics retain their actual nodes in the orchestrator's process topology.
            backend_processes = [replace(process, node=runtime.nodes.head) for process in backend_processes]
        return super().start_frontends(topology, runtime, config, backend, backend_processes, stop_event)

    def frontend_metrics_port(self, frontend_args: dict[str, Any] | None) -> int | None:
        """SMG serves Prometheus on its own listener, not on the routing port."""
        return SMG_METRICS_PORT

    def build_bash_preamble(self, config: Any) -> str | None:
        """Run the recipe setup script in SMG's container, e.g. to ``pip install smg``."""
        return setup_script_preamble(config)

    def get_managed_frontend_args(
        self,
        config: Any,
        backend: Any,
        backend_processes: "list[Process]",
    ) -> list[str]:
        """Bind SMG's Prometheus listener on srtctl's port instead of upstream's default."""
        normalized = {str(key).replace("_", "-"): value for key, value in (config.frontend.args or {}).items()}
        if "prometheus-port" in normalized:
            raise ValueError("frontend.args.prometheus-port is managed by srtctl for smg")
        managed = ["--prometheus-port", str(SMG_METRICS_PORT)]
        if isinstance(backend, TokenSpeedBackend):
            model_path = config.model.path
            # The router reads the shared mount, even when worker weights are staged node-locally.
            model = (
                normalized.get("model")
                or config.model.hf_snapshot_path
                or (model_path.removeprefix("hf:") if model_path.startswith("hf:") else "/model")
            )
            if "backend" not in normalized:
                managed.extend(["--backend", "tokenspeed"])
            if "model" not in normalized:
                managed.extend(["--model", model])
            served = config.served_model_name
            if served != model:
                managed.extend(["--model-alias", f"{served}={model}"])
            if "tokenizer-path" not in normalized:
                for mode in config.roles:
                    tokenizer = backend.get_config_for_mode(mode).get("tokenizer")
                    if tokenizer:
                        managed.extend(["--tokenizer-path", tokenizer])
                        break
        return managed
