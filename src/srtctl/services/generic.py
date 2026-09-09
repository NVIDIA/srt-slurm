# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``type: generic``: launch exactly the command the recipe wrote."""

from __future__ import annotations

from srtctl.services.registry import ServiceKind, register_service


@register_service("generic")
class GenericService(ServiceKind):
    """A user-declared sidecar. No defaults beyond the shared ones: starts after the frontend, non-critical."""
