# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attach selected configuration fields to existing series without adding samples."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ruamel.yaml.error import YAMLError

from srtctl.core.yaml_utils import load_yaml_text_with_comments

from . import ADAPTERS, ConfigDocument, MetricConfigAdapter

if TYPE_CHECKING:
    from ..importer import Importer


class LogEvidence:
    """Load only requested physical lines, with one scan per worker log."""

    def __init__(self, sources: list[dict[str, Any]], series: list[dict[str, Any]]) -> None:
        self.sources = sources
        self.wanted: dict[int, set[int]] = defaultdict(set)
        self.lines: dict[int, dict[int, str]] = {}
        for item in series:
            if item.get("source_kind") == "worker_log" and item.get("temporal") == "setting":
                evidence = item.get("setting_evidence", [[p[0], p[2], p[3]] for p in item["points"]])
                for _, source, line in evidence:
                    self.wanted[source].add(line)

    def line(self, source_id: int, line: int) -> str | None:
        if source_id not in self.lines:
            found: dict[int, str] = {}
            wanted = self.wanted.get(source_id, set())
            if wanted:
                path = Path(self.sources[source_id]["path"])
                with path.open(errors="replace", newline="\n") as stream:
                    for number, text in enumerate(stream, 1):
                        if number in wanted:
                            found[number] = text
                        if len(found) == len(wanted):
                            break
            self.lines[source_id] = found
        return self.lines[source_id].get(line)


def attach_configuration(run: Importer, adapters: dict[str, MetricConfigAdapter] | None = None) -> None:
    """Missing or unusable configuration never disables runtime evidence.

    Discovery is limited to recipe.yaml in the log directory or its parent. An
    explicit path selects one file; ambiguous discovery imports neither candidate.
    Adapters only annotate existing series and export selected fields with lineage.
    """
    if not run.metric_series:
        return
    candidates = [run.config_path] if run.config_path else [run.logs / "recipe.yaml", run.logs.parent / "recipe.yaml"]
    paths = sorted({path.resolve() for path in candidates if path.is_file()})
    if not paths:
        if run.config_path:
            run.warnings.append(f"Configuration metadata omitted: file does not exist: {run.config_path}")
        return
    if len(paths) != 1:
        run.warnings.append(
            "Configuration metadata omitted: multiple recipe.yaml candidates; select one with --config."
        )
        return
    path = paths[0]
    try:
        source_id = run.source(path, "run_config")
        raw = path.read_bytes()
        data = load_yaml_text_with_comments(raw.decode("utf-8"))
        engine = data.get("engine", {})
        adapter = (
            (ADAPTERS if adapters is None else adapters).get(engine.get("type")) if isinstance(engine, dict) else None
        )
        if adapter is None:
            return
        run.sources[source_id]["sha256"] = hashlib.sha256(raw).hexdigest()
        document = ConfigDocument(data, source_id)
        by_id = {series["id"]: series for series in run.metric_series}
        logs = LogEvidence(run.sources, run.metric_series)
        pending = []
        for series in run.metric_series:
            role = run.workers.get(series["worker"], {}).get("role")
            if role is None:
                continue
            values = adapter.read(document, series, role, by_id, logs)
            annotations = []
            for value in values:
                if isinstance(value.source.value, float) and not math.isfinite(value.source.value):
                    run.warnings.append(f"Configuration field omitted: {value.source.field} is not finite.")
                    continue
                if value.comparison and (
                    value.comparison.unit != series["unit"]
                    or isinstance(value.comparison.value, bool)
                    or not math.isfinite(value.comparison.value)
                    or value.comparison.value <= 0
                    or not math.isfinite(value.comparison.start)
                ):
                    raise ValueError(f"Invalid configuration comparison for {series['name']}")
                annotation = asdict(value)
                annotation["source"]["file"] = path.name
                annotations.append(annotation)
            if annotations:
                pending.append((series, annotations))
        for series, annotations in pending:
            series["configuration"] = annotations
        run.audit["metric_series_with_configuration"] = len(pending)
    except (OSError, YAMLError, TypeError, ValueError) as exc:
        run.warnings.append(f"Configuration metadata omitted: {path.name}: {exc}")
