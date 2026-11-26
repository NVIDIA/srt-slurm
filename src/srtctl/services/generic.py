# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``type: generic``: launch exactly the command the recipe wrote."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from srtctl.services.registry import ServiceKind, register_service

if TYPE_CHECKING:
    from srtctl.services.config import ServiceConfig


@register_service("generic")
class GenericService(ServiceKind):
    """A user-declared sidecar. No defaults beyond the shared ones: starts after the frontend, non-critical."""

    supports_host_native: ClassVar[bool] = True

    def host_native(self, service: ServiceConfig) -> bool:
        return service.host_native
