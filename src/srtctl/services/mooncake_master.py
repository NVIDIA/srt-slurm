# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``type: mooncake-master``: the Mooncake master that stores and workers register with.

Implied by ``backend.mooncake_kv_store`` (the v1 spelling) and equally the v2 way
to ask for Mooncake: declare the service and srtctl injects ``MOONCAKE_MASTER``,
``MOONCAKE_TE_META_DATA_SERVER``, and ``MOONCAKE_LOCAL_HOSTNAME`` into every
worker (see ``expand_services``, which maps the declared service back onto
``backend.mooncake_kv_store`` so the engine-side validation and env injection
keep working unchanged). Runs on the infra node, before workers, with the
embedded HTTP metadata server and the metrics endpoint on, all three ports gated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.ports import MOONCAKE_HTTP_METADATA_PORT, MOONCAKE_MASTER_PORT, MOONCAKE_METRICS_PORT
from srtctl.services.registry import ServiceKind, ServiceLaunchContext, register_service

if TYPE_CHECKING:
    from srtctl.core.schema import SrtConfig
    from srtctl.services.config import ServiceConfig


def mooncake_master_command(extra_args: list[str] | tuple[str, ...] = ()) -> list[str]:
    """The master command, including recipe-provided version-specific flags."""
    return [
        "mooncake_master",
        f"--port={MOONCAKE_MASTER_PORT}",
        "--enable_http_metadata_server=true",
        f"--http_metadata_server_port={MOONCAKE_HTTP_METADATA_PORT}",
        "--eviction_high_watermark_ratio=0.9",
        "--default_kv_lease_ttl=10000",
        "--rpc_thread_num=16",
        "--enable_metric_reporting=true",
        f"--metrics_port={MOONCAKE_METRICS_PORT}",
        *extra_args,
    ]


@register_service("mooncake-master")
class MooncakeMasterService(ServiceKind):
    """Mooncake master on the infra node; ``args`` are appended to the fixed command."""

    builds_command = True
    default_start = "before_workers"
    default_critical = True
    default_placement = "infra"
    default_readiness_ports = (MOONCAKE_MASTER_PORT, MOONCAKE_HTTP_METADATA_PORT, MOONCAKE_METRICS_PORT)
    supports_dedicated = True
    supports_external = True
    option_keys = ("store_config",)  # vLLM: rendered into MOONCAKE_CONFIG_PATH for the workers

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        if service.command is not None:
            return list(service.effective_command)
        return mooncake_master_command(service.args)

    def container_fallback(self, config: SrtConfig) -> str | None:
        mooncake_cfg = getattr(config.backend, "mooncake_kv_store", None)
        return getattr(mooncake_cfg, "container", None)
