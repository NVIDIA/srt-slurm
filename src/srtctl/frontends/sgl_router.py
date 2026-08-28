# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental Rust SGL router (`experimental/sgl-router`) frontend."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Any, ClassVar

from srtctl.core.health import WorkerHealthResult
from srtctl.frontends.static_router import StaticRouterFrontend

if TYPE_CHECKING:
    from srtctl.core.topology import Process

# Cargo places the release binary here inside a router source checkout.
CARGO_RELEASE_SUBPATH = "target/release"


def resolve_router_binary(frontend: Any, default: str) -> str:
    """Return the router executable selected by `frontend.binary`/`frontend.source`."""
    binary = getattr(frontend, "binary", None)
    if binary:
        return str(binary)
    source = getattr(frontend, "source", None)
    if source:
        return f"{str(source).rstrip('/')}/{CARGO_RELEASE_SUBPATH}/{default}"
    return default


def router_build_command(frontend: Any) -> str | None:
    """Return the cargo build for `frontend.source`, when one is configured."""
    source = getattr(frontend, "source", None)
    if not source:
        return None
    manifest = f"{str(source).rstrip('/')}/Cargo.toml"
    return f"cargo build --release --manifest-path {shlex.quote(manifest)}"


class SGLRouterFrontend(StaticRouterFrontend):
    """Route aggregate SGLang traffic through the experimental Rust router.

    The router owns tokenization and the cache-aware policy itself: it learns
    each worker's ZMQ KV-event publisher from that worker's ``/server_info``,
    so the only wiring srtctl owns is the static worker list plus the normal
    ``backend.kv_events_config`` on the aggregate workers.
    """

    type: ClassVar[str] = "sgl-router"
    backend_type: ClassVar[str] = "sglang"
    executable: ClassVar[tuple[str, ...]] = ("sgl-router",)
    # Static discovery is aggregate-only; P/D is Kubernetes-selector driven.
    pd_flag: ClassVar[str] = ""
    process_name: ClassVar[str] = "sgl_router"
    log_label: ClassVar[str] = "sgl_router"
    # `/readyz` answers with a status code and no body.
    health_status_only: ClassVar[bool] = True

    def __init__(self) -> None:
        # Resolved from `frontend.binary`/`frontend.source` when frontends start.
        self.resolved_executable: tuple[str, ...] = type(self).executable

    @property
    def health_endpoint(self) -> str:
        return "/readyz"

    def parse_health(
        self,
        response_json: dict,
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        """`/readyz` is a bare status code; a 200 means at least one worker."""
        del response_json
        return WorkerHealthResult(
            ready=True,
            message="sgl-router reports ready",
            prefill_ready=expected_prefill,
            prefill_expected=expected_prefill,
            decode_ready=expected_decode,
            decode_expected=expected_decode,
        )

    def get_backend_health_urls(
        self,
        backend: Any,
        backend_processes: list[Process],
        network_interface: str | None = None,
    ) -> list[str]:
        """Gate on every aggregate worker, since `/readyz` only needs one.

        The router flips ready as soon as its registry is non-empty, so this is
        the barrier that actually means "the configured topology is serving".
        """
        return [
            f"{worker.url.rstrip('/')}/health"
            for worker in self.collect_workers(backend, backend_processes, network_interface)
        ]

    def build_router_command(self, workers: list[Any], host: str, port: int) -> list[str]:
        """Build the static-aggregate router CLI for the experimental router."""
        if any(worker.mode != "agg" for worker in workers):
            raise ValueError("frontend.type: sgl-router routes aggregate workers only")
        cmd = list(self.resolved_executable)
        if workers:
            cmd.extend(["--worker-urls", *(worker.url for worker in workers)])
        cmd.extend(["--host", host, "--port", str(port)])
        return cmd

    def managed_model_args(self, config: Any, model_path: str) -> list[str]:
        """Supply the model identity the router requires, unless overridden.

        ``--model-id`` is mandatory, and ``--tokenizer-path`` must name a
        `tokenizer.json` file or a HuggingFace repo id — a bare model directory
        is rejected by the router's loader.
        """
        configured = {str(key).replace("_", "-") for key in (config.frontend.args or {})}
        managed: list[str] = []
        if "model-id" not in configured:
            managed.extend(["--model-id", config.served_model_name])
        if "tokenizer-path" not in configured:
            tokenizer = model_path if not model_path.startswith("/") else f"{model_path.rstrip('/')}/tokenizer.json"
            managed.extend(["--tokenizer-path", tokenizer])
        return managed

    def get_managed_frontend_args(
        self,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
    ) -> list[str]:
        """Resolve the model identity against the Slurm container mount."""
        del backend, backend_processes
        model_path = str(config.model.path)
        return self.managed_model_args(
            config, model_path.removeprefix("hf:") if model_path.startswith("hf:") else "/model"
        )

    def build_bash_preamble(self, config: Any) -> str | None:
        """Build the router from source when `frontend.source` is set."""
        return router_build_command(config.frontend)

    def start_frontends(self, topology, runtime, config, backend, backend_processes, stop_event=None):
        """Bind the configured executable path, then launch normally."""
        self.resolved_executable = (resolve_router_binary(config.frontend, type(self).executable[0]),)
        return super().start_frontends(topology, runtime, config, backend, backend_processes, stop_event)
