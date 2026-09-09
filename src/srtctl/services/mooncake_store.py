# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``type: mooncake-store``: a standalone Mooncake Store process wired to the managed master.

Lets inference workers run embedded Mooncake clients with
``MOONCAKE_GLOBAL_SEGMENT_SIZE=0`` while dedicated per-node stores own the DRAM
segments (decode nodes contribute host memory without an in-process HiCache
pool). Requires ``backend.mooncake_kv_store``, which is what launches the master
the store registers with. Starts before workers and is critical by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from marshmallow import ValidationError

from srtctl.ports import MOONCAKE_HTTP_METADATA_PORT, MOONCAKE_MASTER_PORT
from srtctl.services.registry import ServiceKind, ServiceLaunchContext, register_service

if TYPE_CHECKING:
    from srtctl.core.schema import SrtConfig
    from srtctl.services.config import ServiceConfig


@register_service("mooncake-store")
class MooncakeStoreService(ServiceKind):
    """Standalone Mooncake Store; one instance per placed node, started before workers."""

    default_command = ("python", "-m", "mooncake.mooncake_store_service")
    default_start = "before_workers"
    default_critical = True

    def validate(self, service: ServiceConfig, config: SrtConfig) -> None:
        if getattr(config.backend, "mooncake_kv_store", None) is None:
            raise ValidationError(
                f"services[{service.name}] (type mooncake-store) requires backend.mooncake_kv_store, "
                "which launches the master the store registers with"
            )

    def container_fallback(self, config: SrtConfig) -> str | None:
        mooncake_cfg = getattr(config.backend, "mooncake_kv_store", None)
        return getattr(mooncake_cfg, "container", None)

    def default_environment(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> dict[str, str]:
        # The recipe may pin a specific NIC identity by setting this itself.
        return {"MOONCAKE_LOCAL_HOSTNAME": ctx.node_ip}

    def forced_environment(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> dict[str, str]:
        infra_ip = ctx.runtime.infra_node_ip
        return {
            "MOONCAKE_MASTER": f"{infra_ip}:{MOONCAKE_MASTER_PORT}",
            "MOONCAKE_TE_META_DATA_SERVER": f"http://{infra_ip}:{MOONCAKE_HTTP_METADATA_PORT}/metadata",
        }
