# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage mixins for SweepOrchestrator.

Each mixin handles one stage of the sweep orchestration:
- WorkerStageMixin: Backend worker process startup
- FrontendStageMixin: Frontend/nginx orchestration
- BenchmarkStageMixin: Benchmark execution
- PostProcessStageMixin: Post-benchmark AI analysis
"""

from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin
from srtctl.cli.mixins.frontend_stage import FrontendStageMixin
from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin
from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin
from srtctl.cli.mixins.worker_stage import WorkerStageMixin

__all__ = [
    "WorkerStageMixin",
    "FrontendStageMixin",
    "TelemetryStageMixin",
    "BenchmarkStageMixin",
    "PostProcessStageMixin",
]
