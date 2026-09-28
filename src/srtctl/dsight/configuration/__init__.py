# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration adapters supply optional metadata; configuration is not a metric."""

from .base import ConfigComparison, ConfigDocument, ConfigEvidence, MetricConfigAdapter, MetricConfiguration
from .tokenspeed import TokenSpeedMetricConfiguration

ADAPTERS: dict[str, MetricConfigAdapter] = {"tokenspeed": TokenSpeedMetricConfiguration()}

__all__ = [
    "ADAPTERS",
    "ConfigComparison",
    "ConfigDocument",
    "ConfigEvidence",
    "MetricConfigAdapter",
    "MetricConfiguration",
]
