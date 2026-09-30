# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``type: mooncake-master``: the Mooncake master that stores and workers register with.

Implied by ``backend.mooncake_kv_store`` (the v1 spelling) and equally the v2 way
to ask for Mooncake: declare the service and srtctl configures each engine's
workers (see ``expand_services``, which maps the declared service back onto
``backend.mooncake_kv_store``). Runs on the infra node before workers. SGLang
and vLLM enable HTTP metadata; TRT-LLM uses its pool RPC interface and a
shared address file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.backends.trtllm import TRTLLMMooncakeKVStoreConfig, TRTLLMProtocol
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


def scrub_mpi_environment() -> str:
    """The Mooncake binaries import TensorRT-LLM, which must not join an MPI step."""
    prefixes = "PMIX_|PMI_|OMPI_|SLURM_|SLURMD_|MPI_|OPAL_|PRTE_|HYDRA_|I_MPI_"
    return "for var in $(env | grep -oE '^(" + prefixes + ')[A-Za-z0-9_]*=\' | tr -d =); do unset "$var"; done'


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
    option_keys = (
        "store_config",
        "device_names_by_gpu",  # vLLM worker-side JSON
        "eviction_ratio",
        "master_timeout_s",
        "store_role",  # TRT-LLM pool
    )

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        if service.command is not None:
            return list(service.effective_command)
        if (ctx.config is not None and isinstance(ctx.config.backend, TRTLLMProtocol)) or (
            ctx.config is None and "store_role" in service.options
        ):
            cfg = (
                ctx.config.backend.mooncake_kv_store
                if ctx.config is not None and isinstance(ctx.config.backend, TRTLLMProtocol)
                else TRTLLMMooncakeKVStoreConfig(**service.options)
            )
            if cfg is None:
                raise ValueError("TRT-LLM mooncake-master requires engine.mooncake_kv_store")
            return [
                "trtllm-serve",
                "mooncake_master",
                "--rpc_port",
                str(MOONCAKE_MASTER_PORT),
                "--metrics_port",
                str(MOONCAKE_METRICS_PORT),
                "--eviction_ratio",
                str(cfg.eviction_ratio),
                "--address_file",
                "/logs/mooncake_master.addr",
                *service.args,
            ]
        return mooncake_master_command(service.args)

    def readiness(self, service: ServiceConfig, ctx: ServiceLaunchContext):
        if ctx.config is not None and isinstance(ctx.config.backend, TRTLLMProtocol):
            from srtctl.services.config import ServiceReadinessConfig

            return ServiceReadinessConfig(port=MOONCAKE_MASTER_PORT)
        return None

    def srun_options(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> dict[str, str]:
        return {"mpi": "none"} if ctx.config is not None and isinstance(ctx.config.backend, TRTLLMProtocol) else {}

    def preamble(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> str | None:
        if ctx.config is not None and isinstance(ctx.config.backend, TRTLLMProtocol):
            return f"mkdir -p /logs/mooncake/master && {scrub_mpi_environment()}"
        return None

    def default_environment(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> dict[str, str]:
        if ctx.config is not None and isinstance(ctx.config.backend, TRTLLMProtocol):
            cfg = ctx.config.backend.mooncake_kv_store
            return {
                "TRTLLM_MOONCAKE_RUN_DIR": "/logs/mooncake/master",
                "TRTLLM_MOONCAKE_MASTER_TIMEOUT": str(cfg.master_timeout_s if cfg is not None else 60),
            }
        return {}

    def container_fallback(self, config: SrtConfig) -> str | None:
        mooncake_cfg = config.backend.mooncake_kv_store
        return mooncake_cfg.container if mooncake_cfg is not None else None
