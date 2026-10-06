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

from typing import TYPE_CHECKING, Any, ClassVar

from srtctl.frontends.base import register_frontend
from srtctl.frontends.static_router import StaticRouterFrontend, setup_script_preamble
from srtctl.ports import SMG_METRICS_PORT

if TYPE_CHECKING:
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
        if not config.topology.is_disaggregated:
            return
        prefill, decode = config.backend_for_role("prefill"), config.backend_for_role("decode")
        if prefill.type != decode.type:
            raise ValueError("frontend.type: smg prefill/decode needs the same engine in both roles")
        if prefill.type == "trtllm":
            raise ValueError("frontend.type: smg does not support TRT-LLM prefill/decode")
        if prefill.type == "vllm" and not (prefill.is_grpc_mode("prefill") and decode.is_grpc_mode("decode")):
            raise ValueError("frontend.type: smg runs vLLM prefill/decode only over gRPC; set grpc: true in both roles")

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
        normalized = {str(key).replace("_", "-") for key in (config.frontend.args or {})}
        if "prometheus-port" in normalized:
            raise ValueError("frontend.args.prometheus-port is managed by srtctl for smg")
        return ["--prometheus-port", str(SMG_METRICS_PORT)]
