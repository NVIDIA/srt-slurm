# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Mooncake backend configuration."""

from dataclasses import field
from typing import ClassVar

from marshmallow import Schema
from marshmallow_dataclass import dataclass

DEFAULT_MOONCAKE_MASTER_ARGS: tuple[str, ...] = (
    "--eviction_high_watermark_ratio=0.9",
    "--default_kv_lease_ttl=10000",
    "--rpc_thread_num=16",
)


@dataclass(frozen=True)
class MooncakeStandalonePlacementConfig:
    """Role-specific overrides for standalone Mooncake Store services."""

    enabled: bool = True
    env: dict[str, str] = field(default_factory=dict)
    args: tuple[str, ...] = ()

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class MooncakeStandaloneHealthCheckConfig:
    """TCP readiness check for a standalone Mooncake Store service."""

    port: int = 8080
    timeout_seconds: int = 120

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("mooncake standalone health_check.port must be between 1 and 65535")
        if self.timeout_seconds <= 0:
            raise ValueError("mooncake standalone health_check.timeout_seconds must be positive")


@dataclass(frozen=True)
class MooncakeStandaloneStoreConfig:
    """Managed standalone Mooncake Store service configuration.

    ``command`` and ``args`` are passed through to the service process without
    interpreting Mooncake-specific flags. One process is launched per physical
    node selected by ``placements``.
    """

    _VALID_PLACEMENTS: ClassVar[frozenset[str]] = frozenset({"prefill", "decode", "aggregated"})

    enabled: bool = True
    container: str | None = None
    command: tuple[str, ...] = ("python", "-m", "mooncake.mooncake_store_service")
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    placements: dict[str, MooncakeStandalonePlacementConfig] = field(default_factory=dict)
    preamble: str | None = None
    cpus_per_task: int | None = None
    cpu_bind: str | None = None
    srun_options: dict[str, str] = field(default_factory=dict)
    health_check: MooncakeStandaloneHealthCheckConfig | None = field(
        default_factory=MooncakeStandaloneHealthCheckConfig
    )
    critical: bool = True

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if not self.command:
            raise ValueError("mooncake standalone command must not be empty")
        if not self.placements:
            raise ValueError("mooncake standalone placements must select at least one worker role")
        unknown = sorted(set(self.placements) - self._VALID_PLACEMENTS)
        if unknown:
            raise ValueError(
                "invalid mooncake standalone placements: "
                f"{', '.join(unknown)} (expected prefill, decode, or aggregated)"
            )
        if self.cpus_per_task is not None and self.cpus_per_task <= 0:
            raise ValueError("mooncake standalone cpus_per_task must be positive")
