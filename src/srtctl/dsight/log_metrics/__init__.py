# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered log metric generators. Adding a dialect does not change the UI."""

from .base import LogMetricDefinition, LogMetricEvent, LogMetricGenerator
from .tokenspeed import DynamoTokenSpeedLogMetrics

GENERATORS: tuple[LogMetricGenerator, ...] = (DynamoTokenSpeedLogMetrics(),)

__all__ = ["GENERATORS", "LogMetricDefinition", "LogMetricEvent", "LogMetricGenerator"]
