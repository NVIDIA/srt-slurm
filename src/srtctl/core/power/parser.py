# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict DCGM exporter power parsing.

``DCGM_FI_DEV_POWER_USAGE`` is mandatory and decides which GPUs produce a
reading. The utilization fields listed in ``UTILIZATION_METRICS`` are optional
riders: they attach to a GPU's power reading when present and valid, and are
dropped silently otherwise. Device identity comes from the ``gpu`` and
``UUID`` labels; the optional ``Hostname`` label is deliberately ignored
because the collector already knows which allocated node it polled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from prometheus_client.parser import text_string_to_metric_families

from srtctl.core.power.contract import POWER_METRIC, UTILIZATION_METRICS, Reason, dedupe

_MIG_LABELS = ("GPU_I_ID", "GPU_I_PROFILE")
_UTILIZATION_BY_METRIC = {metric.metric: metric for metric in UTILIZATION_METRICS}


@dataclass(frozen=True)
class PowerReading:
    """One physical GPU's power draw within a single scrape, with optional utilization."""

    gpu_index: int
    gpu_uuid: str
    power_w: float
    gpu_util_pct: float | None = None
    sm_active: float | None = None


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

    power_by_index: dict[int, tuple[str, float]] = {}
    duplicated_power: set[int] = set()
    saw_power_sample = False
    # column -> gpu_index -> value; a duplicate poisons that (column, gpu) pair.
    utilization: dict[str, dict[int, float]] = {metric.column: {} for metric in UTILIZATION_METRICS}
    duplicated_utilization: dict[str, set[int]] = {metric.column: set() for metric in UTILIZATION_METRICS}

    for family in families:
        for sample in family.samples:
            if sample.name == POWER_METRIC:
                saw_power_sample = True
                _collect_power(sample.labels, sample.value, power_by_index, duplicated_power, reasons)
                continue
            spec = _UTILIZATION_BY_METRIC.get(sample.name)
            if spec is None:
                continue
            _collect_utilization(
                sample.labels,
                sample.value,
                spec.max_value,
                utilization[spec.column],
                duplicated_utilization[spec.column],
            )

    if duplicated_power:
        reasons.append(Reason.DUPLICATE_POWER_METRIC)
        for gpu_index in duplicated_power:
            power_by_index.pop(gpu_index, None)

    if not saw_power_sample:
        reasons.append(Reason.POWER_METRIC_MISSING)

    readings = []
    for gpu_index in sorted(power_by_index):
        gpu_uuid, power_w = power_by_index[gpu_index]
        extras = {
            column: values[gpu_index]
            for column, values in utilization.items()
            if gpu_index in values and gpu_index not in duplicated_utilization[column]
        }
        readings.append(PowerReading(gpu_index=gpu_index, gpu_uuid=gpu_uuid, power_w=power_w, **extras))
    return ParsedScrape(readings=tuple(readings), reason_codes=dedupe(reasons))


def _collect_power(
    labels: dict[str, str],
    value: float,
    by_index: dict[int, tuple[str, float]],
    duplicated: set[int],
    reasons: list[str],
) -> None:
    if any(labels.get(label) for label in _MIG_LABELS):
        reasons.append(Reason.MIG_INSTANCE_UNSUPPORTED)
        return

    gpu_index = _parse_index(labels.get("gpu"))
    if gpu_index is None:
        reasons.append(Reason.GPU_INDEX_MISSING)
        return

    gpu_uuid = (labels.get("UUID") or "").strip()
    if not gpu_uuid:
        reasons.append(Reason.GPU_UUID_MISSING)
        return

    if not math.isfinite(value) or value < 0:
        reasons.append(Reason.INVALID_POWER_VALUE)
        return

    if gpu_index in by_index:
        duplicated.add(gpu_index)
        return
    by_index[gpu_index] = (gpu_uuid, value)


def _collect_utilization(
    labels: dict[str, str],
    value: float,
    max_value: float,
    by_index: dict[int, float],
    duplicated: set[int],
) -> None:
    """Optional metric: every rejection is silent, so no reason list is threaded through."""
    if any(labels.get(label) for label in _MIG_LABELS):
        return
    gpu_index = _parse_index(labels.get("gpu"))
    if gpu_index is None:
        return
    if not math.isfinite(value) or value < 0 or value > max_value:
        return
    if gpu_index in by_index:
        duplicated.add(gpu_index)
        return
    by_index[gpu_index] = value


def _parse_index(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None
