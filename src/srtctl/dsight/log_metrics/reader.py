# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize registered worker log metrics into the same series as Tachometer."""

from __future__ import annotations

import datetime as dt
import math
from typing import TYPE_CHECKING, Any

from ..metrics import finalize_metrics
from ..sources import source_identity
from . import GENERATORS, LogMetricGenerator

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

    from ..importer import Importer


def timestamp_ns(stamp: str, zone: ZoneInfo | None) -> int | None:
    time = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if time.tzinfo is None:
        if zone is None:
            return None
        time = time.replace(tzinfo=zone)
    return int(time.timestamp()) * 10**9 + time.microsecond * 1000


def read_log_metrics(run: Importer, generators: tuple[LogMetricGenerator, ...] = GENERATORS) -> list[dict[str, Any]]:
    """Retain actual sample times and evidence; never materialize assumed samples.

    Settings include the last timestamp group preceding the window. That is the
    evidence for a held configuration, not a synthetic observation at time zero.
    Reference links are exact-scope joins, never cross-file or cross-rank guesses.
    """
    catalog: dict[str, dict[str, Any]] = {}
    definitions = {}
    for generator in generators:
        for definition in generator.definitions:
            if definition.name in definitions or any(m["name"] == definition.name for m in run.metric_catalog):
                raise ValueError(f"Log metric name collision: {definition.name}")
            definitions[definition.name] = definition
            catalog[definition.name] = {
                "name": definition.name,
                "title": definition.title,
                "unit": definition.unit,
                "description": definition.description,
                "component": "Workers",
                "group": "Log-derived metrics",
                "group_order": 90,
                "order": 0,
                "counter": False,
                "value_kind": "stored",
                "source_kind": "worker_log",
                "generator": generator.name,
                "temporal": definition.temporal,
                "quality": "Log evidence; periodic samples are not continuous occupancy. "
                "Configuration carries only within the same file, worker, rank, process and labels.",
                "series_count": 0,
                "samples": 0,
                "endpoints": ["worker_log"],
            }
        for definition in generator.definitions:
            if definition.reference:
                reference = next((d for d in generator.definitions if d.name == definition.reference), None)
                if reference is None or reference.unit != definition.unit:
                    raise ValueError(f"Invalid log metric reference: {definition.name}")
    series_by_key: dict[tuple, dict[str, Any]] = {}
    unaligned = 0
    for path in sorted(run.logs.glob("*_w*.out")):
        source = source_identity(path)
        if source is None:
            continue
        for line_number, line in enumerate(path.open(errors="replace", newline="\n"), 1):
            for generator in generators:
                event = generator.parse_line(line, source)
                if event is None:
                    continue
                stamp = timestamp_ns(event.time, run.iteration_zone)
                if stamp is None:
                    unaligned += 1
                    continue
                time = run.t(stamp)
                if time > run.duration:
                    continue
                sid = run.source(path, "worker_log")
                labels = dict(event.labels)
                labels.update(generator=generator.name, log_source=str(sid))
                scope = (sid, source.worker, event.rank, event.rank_kind, event.process, tuple(sorted(labels.items())))
                for name, value in event.values:
                    definition = definitions[name]
                    if definition.temporal == "sample" and time < 0:
                        continue
                    if value is not None and not math.isfinite(value):
                        raise ValueError(f"{path}:{line_number}: non-finite log metric {name}")
                    key = (name, scope)
                    if key not in series_by_key:
                        series_by_key[key] = {
                            "name": name,
                            "label": definition.title,
                            "unit": definition.unit,
                            "value_kind": "stored",
                            "component": "Workers",
                            "group": "Log-derived metrics",
                            "description": definition.description,
                            "source_kind": "worker_log",
                            "generator": generator.name,
                            "temporal": definition.temporal,
                            "raw_name": name,
                            "endpoint": "worker_log",
                            "worker": source.worker,
                            "host": source.host,
                            "gpu": None,
                            "rank": event.rank,
                            "rank_kind": event.rank_kind,
                            "labels": labels,
                            "worker_process": event.process,
                            "raw_host": source.host,
                            "host_source": "worker log filename",
                            "host_evidence": sid,
                            "time_resolution_s": event.time_resolution_s,
                            "points": [],
                            "source_ids": {sid},
                        }
                    series_by_key[key]["points"].append([time, value, sid, line_number])
    if unaligned:
        run.warnings.append("Log metrics with local timestamps omitted: set --iteration-timezone to align them.")
        run.audit["unaligned_log_metric_records"] += unaligned
    result = list(series_by_key.values())
    for index, ((name, _scope), series) in enumerate(series_by_key.items(), start=len(run.metric_series)):
        series["id"] = index
        if series["temporal"] == "setting":
            before = [p[0] for p in series["points"] if p[0] < 0]
            start = max(before) if before else 0
            series["points"] = [p for p in series["points"] if p[0] >= start]
        definition = definitions[name]
        if definition.reference:
            series["reference"] = {
                "name": definition.reference,
                "label": definition.reference_label or definitions[definition.reference].title,
                "series_id": None,
            }
    # Resolve after IDs have been assigned, independently of log observation order.
    for (name, scope), series in series_by_key.items():
        if definition := definitions[name].reference:
            reference = series_by_key.get((definition, scope))
            series["reference"]["series_id"] = reference["id"] if reference else None
    present = {series["name"] for series in result}
    catalog = {name: item for name, item in catalog.items() if name in present}
    finalize_metrics(run, result, catalog)
    run.audit["log_metric_series"] = len(result)
    run.audit["log_metric_points"] = sum(len(series["points"]) for series in result)
    return result
