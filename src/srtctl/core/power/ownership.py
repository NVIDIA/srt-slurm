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
    """``EVAL_ONLY=true`` skips the benchmark stage, so there is no measurement window to power-sample."""
    return os.environ.get("EVAL_ONLY", "false").lower() == "true"


def power_owns_dcgm_exporter(config: SrtConfig) -> bool:
    """Whether the power session launches (and Tachometer scrapes) the DCGM exporter.

    False when telemetry is off, has no GPU leg, or the run is eval-only; then the
    implied ``dcgm-exporter`` service keeps Tachometer's GPU metrics flowing.
    """
    return bool(config.telemetry_enabled and config.telemetry_dcgm_exporter is not None and not eval_only_run())
