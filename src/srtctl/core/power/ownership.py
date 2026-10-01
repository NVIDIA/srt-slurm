# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who launches the worker-node DCGM exporter: the power session or the implied Tachometer service.

One answer for every consumer (implicit services, Tachometer scrape targets, the sweep
orchestrator), so the exporter is launched exactly once and scraped wherever it runs.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from srtctl.core.schema import SrtConfig


def eval_only_run() -> bool:
    """``EVAL_ONLY=true`` skips the benchmark stage, so no measurement window is ever stamped."""
    return os.environ.get("EVAL_ONLY", "false").lower() == "true"


def power_session_runs(config: SrtConfig) -> bool:
    """Whether the sweep starts the DCGM power session for this run.

    Off when telemetry is off or has no GPU leg. An eval-only run still samples
    best-effort, but a *required* measurement would fail an otherwise successful
    evaluation over the missing windows, so that combination skips the session.
    """
    if not (config.telemetry_enabled and config.telemetry_dcgm_exporter is not None):
        return False
    return not (eval_only_run() and config.telemetry.required)


def power_owns_dcgm_exporter(config: SrtConfig) -> bool:
    """Whether the power session launches (and Tachometer scrapes) the DCGM exporter.

    When it does not, the implied ``dcgm-exporter`` service keeps Tachometer's GPU
    metrics flowing.
    """
    return power_session_runs(config)
