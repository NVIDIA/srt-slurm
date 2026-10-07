# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One TensorRT-LLM Mooncake memory donor on each placed decode node."""

from __future__ import annotations

from typing import TYPE_CHECKING

from marshmallow import ValidationError

from srtctl.backends.trtllm import TRTLLMBackend, TRTLLMMooncakeKVStoreConfig
from srtctl.services.mooncake_master import scrub_mpi_environment
from srtctl.services.registry import ServiceKind, ServiceLaunchContext, register_service

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.services.config import ServiceConfig


@register_service("mooncake-donor")
class MooncakeDonorService(ServiceKind):
    builds_command = True
    default_start = "before_workers"
    default_critical = True
    default_placement = "decode"
    option_keys = ("size", "protocol")
    default_srun_options = {"mpi": "none"}

    def validate(self, service: ServiceConfig, config: SrtConfig) -> None:
        if not isinstance(config.backend, TRTLLMBackend) or config.backend.mooncake_kv_store is None:
            raise ValidationError("mooncake-donor requires a TRT-LLM mooncake-master service")
        if not service.options.get("size"):
            raise ValidationError("mooncake-donor requires options.size (for example 640GiB)")
        if service.effective_placement != "decode":
            raise ValidationError("mooncake-donor requires placement.node: decode")
        connector = config.backend.get_config_for_mode("prefill").get("kv_connector_config")
        store = connector.get("mooncake_store") if isinstance(connector, dict) else None
        if isinstance(store, dict) and store.get("protocol", "rdma") != service.options.get("protocol", "rdma"):
            raise ValidationError("mooncake-donor options.protocol must match prefill mooncake_store.protocol")

    def container_fallback(self, config: SrtConfig) -> str | None:
        mooncake_cfg = config.backend.mooncake_kv_store
        return mooncake_cfg.container if mooncake_cfg is not None else None

    def readiness(self, service: ServiceConfig, ctx: ServiceLaunchContext):
        from srtctl.services.config import FileProbe, ServiceReadinessConfig

        return ServiceReadinessConfig(file=FileProbe(path=f"mooncake/donor-{ctx.node}.ready"))

    def prepare(self, service: ServiceConfig, runtime: RuntimeContext) -> None:
        # Remove stale markers before spawning any donor, including when a
        # container starts slowly and has not executed its preamble yet.
        for path in (runtime.log_dir / "mooncake").glob("donor-*.ready"):
            path.unlink(missing_ok=True)

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        if service.command is not None:
            return list(service.effective_command)
        return [
            "trtllm-serve",
            "mooncake_donor",
            "--master_server_address",
            "file:///logs/mooncake_master.addr",
            "--segment_size",
            str(service.options["size"]),
            "--protocol",
            str(service.options.get("protocol", "rdma")),
            "--ready_file",
            f"/logs/mooncake/donor-{ctx.node}.ready",
        ]

    def preamble(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> str:
        return f"mkdir -p /logs/mooncake/donor-{ctx.node} && {scrub_mpi_environment()}"

    def default_environment(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> dict[str, str]:
        # Readiness uses the ready file, independent of the logging level.
        env = {
            "TRTLLM_MOONCAKE_RUN_DIR": f"/logs/mooncake/donor-{ctx.node}",
            "TLLM_LOG_LEVEL": "INFO",
        }
        cfg = ctx.config.backend.mooncake_kv_store if ctx.config is not None else None
        if isinstance(cfg, TRTLLMMooncakeKVStoreConfig):
            env["TRTLLM_MOONCAKE_MASTER_TIMEOUT"] = str(cfg.master_timeout_s)
        return env
