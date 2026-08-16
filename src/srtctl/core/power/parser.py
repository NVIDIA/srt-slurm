# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict DCGM exporter power parsing.

Only ``DCGM_FI_DEV_POWER_USAGE`` is read. Device identity comes from the ``gpu``
and ``UUID`` labels; the optional ``Hostname`` label is deliberately ignored
because the collector already knows which allocated node it polled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from prometheus_client.parser import text_string_to_metric_families

from srtctl.core.power.contract import POWER_METRIC, Reason, dedupe

_MIG_LABELS = ("GPU_I_ID", "GPU_I_PROFILE")


@dataclass(frozen=True)
class PowerReading:
    """One physical GPU's power draw within a single scrape."""

    gpu_index: int
    gpu_uuid: str
    power_w: float


@dataclass(frozen=True)
class ParsedScrape:
    """Readings that may be persisted, plus why anything was dropped."""

    readings: tuple[PowerReading, ...] = ()
    reason_codes: tuple[str, ...] = ()


def parse_power_scrape(text: str) -> ParsedScrape:
    """Parse one exporter ``/metrics`` body into publishable power readings."""
    reasons: list[str] = []
    try:
        families = list(text_string_to_metric_families(text))
    # prometheus-client releases in our supported >=0.20 range have raised
    # ValueError, KeyError, and IndexError for malformed exposition. Keep this
    # third-party boundary broad while still allowing BaseException control
    # flow (for example KeyboardInterrupt) to propagate.
    except Exception:  # noqa: BLE001
        return ParsedScrape(reason_codes=(Reason.ENDPOINT_PARSE_ERROR,))

    by_index: dict[int, PowerReading] = {}
    duplicated: set[int] = set()
    saw_power_sample = False

    for family in families:
        for sample in family.samples:
            if sample.name != POWER_METRIC:
                continue
            saw_power_sample = True
            labels = sample.labels

            if any(labels.get(label) for label in _MIG_LABELS):
                reasons.append(Reason.MIG_INSTANCE_UNSUPPORTED)
                continue

            gpu_index = _parse_index(labels.get("gpu"))
            if gpu_index is None:
                reasons.append(Reason.GPU_INDEX_MISSING)
                continue

            gpu_uuid = (labels.get("UUID") or "").strip()
            if not gpu_uuid:
                reasons.append(Reason.GPU_UUID_MISSING)
                continue

            value = sample.value
            if not math.isfinite(value) or value < 0:
                reasons.append(Reason.INVALID_POWER_VALUE)
                continue

            if gpu_index in by_index:
                duplicated.add(gpu_index)
                continue
            by_index[gpu_index] = PowerReading(gpu_index=gpu_index, gpu_uuid=gpu_uuid, power_w=value)

    if duplicated:
        reasons.append(Reason.DUPLICATE_POWER_METRIC)
        for gpu_index in duplicated:
            by_index.pop(gpu_index, None)

    if not saw_power_sample:
        reasons.append(Reason.POWER_METRIC_MISSING)

    readings = tuple(by_index[index] for index in sorted(by_index))
    return ParsedScrape(readings=readings, reason_codes=dedupe(reasons))


def _parse_index(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None
