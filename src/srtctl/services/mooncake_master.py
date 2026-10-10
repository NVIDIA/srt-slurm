# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``type: mooncake-master``: the Mooncake master that stores and workers register with.

Declared as a ``services:`` entry, or implied by ``engine.mooncake_kv_store``:
either way srtctl injects ``MOONCAKE_MASTER``, ``MOONCAKE_TE_META_DATA_SERVER``,
and ``MOONCAKE_LOCAL_HOSTNAME`` into every worker (``expand_services`` maps a
declared entry onto the internal ``engine.mooncake_kv_store`` field so the
engine-side validation and env injection read one field). vLLM and TRT-LLM
workers also get ``MOONCAKE_CONFIG_PATH``, a client config srtctl renders from
``options.store_config`` (TRT-LLM: one per role, with ``roles.<role>.mooncake_store_config``).
Runs on the infra node, before workers, with the embedded HTTP metadata server
and the metrics endpoint on, all three ports gated. TRT-LLM's
``trtllm-serve mooncake_master`` wraps this same binary, so it is not needed.
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
    # Worker-side Mooncake client JSON: store_config (vLLM and TRT-LLM), device_names_by_gpu (vLLM).
    option_keys = ("store_config", "device_names_by_gpu")

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        if service.command is not None:
            return list(service.effective_command)
        return mooncake_master_command(service.args)

    def container_fallback(self, config: SrtConfig) -> str | None:
        mooncake_cfg = config.backend.mooncake_kv_store
        return mooncake_cfg.container if mooncake_cfg is not None else None
