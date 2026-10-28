# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered log metric generators. Adding a dialect does not change the UI."""

from .base import LogMetricDefinition, LogMetricEvent, LogMetricGenerator
from .sglang import SGLangLogMetrics
from .tokenspeed import DynamoTokenSpeedLogMetrics
from .vllm import VLLMLogMetrics

GENERATORS: tuple[LogMetricGenerator, ...] = (DynamoTokenSpeedLogMetrics(), SGLangLogMetrics(), VLLMLogMetrics())

__all__ = ["GENERATORS", "LogMetricDefinition", "LogMetricEvent", "LogMetricGenerator"]
