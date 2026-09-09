# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The top-level ``services:`` block: user-declared long-running processes launched next to the job."""

# Import kinds to trigger registration.
from srtctl.services import exporters, generic, infra, mooncake_master, mooncake_store
from srtctl.services.config import (
    SERVICE_PLACEMENTS,
    SERVICE_STARTS,
    HttpProbe,
    LogProbe,
    ServiceConfig,
    ServicePlacementConfig,
    ServiceReadinessConfig,
    ServiceSourceConfig,
    TcpProbe,
)
from srtctl.services.registry import (
    ServiceKind,
    ServiceLaunchContext,
    get_service_kind,
    list_service_types,
    register_service,
)

__all__ = [
    "SERVICE_PLACEMENTS",
    "SERVICE_STARTS",
    "HttpProbe",
    "LogProbe",
    "ServiceConfig",
    "ServiceKind",
    "ServiceLaunchContext",
    "ServicePlacementConfig",
    "ServiceReadinessConfig",
    "ServiceSourceConfig",
    "TcpProbe",
    "exporters",
    "generic",
    "get_service_kind",
    "infra",
    "list_service_types",
    "mooncake_master",
    "mooncake_store",
    "register_service",
]
