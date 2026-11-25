# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang's Rust router (`frontend.type: sgl-router`).

``sgl-router`` (https://github.com/sgl-project/sglang/tree/main/sgl-router) is the
KV-aware router that ships in the SGLang repository. It replaces the Model Gateway
that ``frontend.type: sglang-router`` launches, and it rejects that gateway's
flags (``--pd-disaggregation``, ``--prefill``, ``--decode``). It takes one flat
``--worker-urls`` list: each engine discloses its prefill/decode role and
bootstrap port in ``/server_info`` (``src/discovery/static_urls.rs``), so the
workers are the same direct ``sglang.launch_server`` processes as for the other
static routers.
"""

from typing import TYPE_CHECKING, Any, ClassVar

from srtctl.frontends.base import register_frontend
from srtctl.frontends.static_router import RouterWorker, StaticRouterFrontend, setup_script_preamble

if TYPE_CHECKING:
    from srtctl.core.health import WorkerHealthResult
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

# sgl-router's WorkerMode (``src/discovery/types.rs``) to the static-router stats key.
_MODE_STATS = {"prefill": "prefill_count", "decode": "decode_count", "plain": "regular_count"}


@register_frontend("sgl-router")
class SGLRouterFrontend(StaticRouterFrontend):
    """SGLang's Rust router in front of direct SGLang HTTP workers."""

    type: ClassVar[str] = "sgl-router"
    required_backend: ClassVar[str | None] = "sglang"
    executable: ClassVar[tuple[str, ...]] = ("sgl-router",)
    # Unused: roles are self-disclosed, and the router rejects the gateway's P/D flags.
    pd_flag: ClassVar[str] = ""
    process_name: ClassVar[str] = "sgl_router"
    log_label: ClassVar[str | None] = "router"
    # The router introspects a static worker three times, 100 ms apart, and then registers it as a
    # plain worker serving no model (``src/workers/introspect.rs``), so every engine is up first.
    wait_for_workers_before_start: ClassVar[bool] = True

    def validate(self, config: Any) -> None:
        """The router reaches engines over HTTP only."""
        for mode in config.roles:
            if config.backend_for_role(mode).is_grpc_mode(mode):
                raise ValueError(f"frontend.type: sgl-router needs HTTP workers; remove grpc-mode from roles.{mode}")

    def build_router_command(self, workers: list[RouterWorker], host: str, port: int, backend: Any) -> list[str]:
        """One flat URL list for aggregate and prefill/decode topologies alike."""
        if not workers:
            raise ValueError("Static router topology has no routable workers")
        return [
            *self.executable,
            "--worker-urls",
            *(worker.url for worker in workers),
            "--host",
            host,
            "--port",
            str(port),
        ]

    def get_managed_frontend_args(
        self,
        config: Any,
        backend: Any,
        backend_processes: "list[Process]",
        runtime: "RuntimeContext | None" = None,
    ) -> list[str]:
        """``--model-id`` (required; must be the engines' served name) and a tokenizer the router can read offline.

        ``--tokenizer-path`` takes a ``tokenizer.json`` file or a HuggingFace repo id;
        without it the router downloads the tokenizer for ``--model-id``.
        """
        normalized = {str(key).replace("_", "-") for key in config.frontend.args or {}}
        managed: list[str] = []
        if "model-id" not in normalized:
            managed.extend(["--model-id", config.served_model_name])
        if not normalized.intersection({"tokenizer-path", "no-tokenizer"}) and runtime is not None:
            model = runtime.shared_model_arg
            managed.extend(["--tokenizer-path", model if runtime.is_hf_model else f"{model}/tokenizer.json"])
        return managed

    def parse_health(self, response_json: Any, expected_prefill: int, expected_decode: int) -> "WorkerHealthResult":
        """``GET /workers`` lists ``WorkerView`` records; count the registered, healthy ones that resolved a model."""
        stats = dict.fromkeys(_MODE_STATS.values(), 0)
        for worker in response_json if isinstance(response_json, list) else []:
            state = worker.get("runtime") or {}
            if state.get("registered") and state.get("healthy") and state.get("resolved_model_ids"):
                stats[_MODE_STATS[state.get("mode", "plain")]] += 1
        return super().parse_health({"stats": stats}, expected_prefill, expected_decode)

    def build_bash_preamble(self, config: Any) -> str | None:
        """Run the recipe setup script in the router's container, e.g. to fetch the binary."""
        return setup_script_preamble(config)
