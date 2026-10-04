# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, mapping-driven exporter power parsing.

The mapping's power metric is mandatory and decides which GPUs produce a
reading. GPU temperature and the mapping's utilization metrics are optional riders: they attach to a
GPU's power reading when present and valid, and are dropped silently otherwise.
Device identity comes from the mapping's index and identity labels; any
exporter hostname label is deliberately ignored because the collector already
knows which allocated node it polled. Samples carrying one of the mapping's
instance labels (MIG instances, partitions) are unsupported and dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from prometheus_client.parser import text_string_to_metric_families

from srtctl.core.power.contract import TEMPERATURE_METRIC, Reason, dedupe, is_valid_temperature_c
from srtctl.core.power.mapping import DCGM_POWER_MAPPING, PowerMetricMapping


@dataclass(frozen=True)
class PowerReading:
    """One physical GPU's power draw within a single scrape, with optional utilization."""

    gpu_index: int
    gpu_uuid: str
    power_w: float
    gpu_util_pct: float | None = None
    sm_active: float | None = None
    temperature_c: float | None = None


@dataclass(frozen=True)
class ParsedScrape:
    """Readings that may be persisted, plus why anything was dropped."""

    readings: tuple[PowerReading, ...] = ()
    reason_codes: tuple[str, ...] = ()


def parse_power_scrape(text: str, mapping: PowerMetricMapping = DCGM_POWER_MAPPING) -> ParsedScrape:
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
    utilization_by_metric = {metric.metric: metric for metric in mapping.utilization_metrics}
    temperatures: dict[tuple[int, str], float] = {}
    duplicated_temperatures: set[tuple[int, str]] = set()
    # column -> gpu_index -> value; a duplicate poisons that (column, gpu) pair.
    utilization: dict[str, dict[int, float]] = {metric.column: {} for metric in mapping.utilization_metrics}
    duplicated_utilization: dict[str, set[int]] = {metric.column: set() for metric in mapping.utilization_metrics}

    for family in families:
        for sample in family.samples:
            if sample.name == mapping.power_metric:
                saw_power_sample = True
                _collect_power(sample.labels, sample.value, power_by_index, duplicated_power, reasons, mapping)
                continue
            if sample.name == TEMPERATURE_METRIC:
                _collect_temperature(sample.labels, sample.value, temperatures, duplicated_temperatures, mapping)
                continue
            spec = utilization_by_metric.get(sample.name)
            if spec is None:
                continue
            _collect_utilization(
                sample.labels,
                sample.value,
                spec.max_value,
                utilization[spec.column],
                duplicated_utilization[spec.column],
                mapping,
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
        key = (gpu_index, gpu_uuid)
        temperature = temperatures.get(key) if key not in duplicated_temperatures else None
        readings.append(
            PowerReading(gpu_index=gpu_index, gpu_uuid=gpu_uuid, power_w=power_w, temperature_c=temperature, **extras)
        )
    return ParsedScrape(readings=tuple(readings), reason_codes=dedupe(reasons))


def _collect_power(
    labels: dict[str, str],
    value: float,
    by_index: dict[int, tuple[str, float]],
    duplicated: set[int],
    reasons: list[str],
    mapping: PowerMetricMapping,
) -> None:
    if any(labels.get(label) for label in mapping.instance_labels):
        reasons.append(Reason.MIG_INSTANCE_UNSUPPORTED)
        return

    gpu_index = _parse_index(labels.get(mapping.gpu_index_label))
    if gpu_index is None:
        reasons.append(Reason.GPU_INDEX_MISSING)
        return

    gpu_uuid = (labels.get(mapping.gpu_identity_label) or "").strip()
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
    mapping: PowerMetricMapping,
) -> None:
    """Optional metric: every rejection is silent, so no reason list is threaded through."""
    if any(labels.get(label) for label in mapping.instance_labels):
        return
    gpu_index = _parse_index(labels.get(mapping.gpu_index_label))
    if gpu_index is None:
        return
    if not math.isfinite(value) or value < 0 or value > max_value:
        return
    if gpu_index in by_index:
        duplicated.add(gpu_index)
        return
    by_index[gpu_index] = value


def _collect_temperature(
    labels: dict[str, str],
    value: float,
    temperatures: dict[tuple[int, str], float],
    duplicated: set[tuple[int, str]],
    mapping: PowerMetricMapping,
) -> None:
    if any(labels.get(label) for label in mapping.instance_labels):
        return
    gpu_index = _parse_index(labels.get(mapping.gpu_index_label))
    gpu_uuid = (labels.get(mapping.gpu_identity_label) or "").strip()
    if gpu_index is None or not gpu_uuid or not is_valid_temperature_c(value):
        return
    key = (gpu_index, gpu_uuid)
    if key in temperatures:
        duplicated.add(key)
    temperatures[key] = value


def _parse_index(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None
