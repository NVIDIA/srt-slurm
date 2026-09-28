# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional configuration values and their source lineage, independent of metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ruamel.yaml.comments import CommentedMap

Scalar = str | int | float | bool | None


@dataclass(frozen=True)
class ConfigEvidence:
    source_id: int
    field: str
    value: Scalar
    line: int | None = None


@dataclass(frozen=True)
class ConfigComparison:
    """A comparable upper bound; absent when units or scope cannot be established.

    start is the first time the comparison's scope is supported by evidence. The
    configuration value itself is a run setting, not a timestamped metric sample.
    """

    value: int | float
    unit: str
    start: float
    basis: str
    evidence: tuple[ConfigEvidence, ...] = ()


@dataclass(frozen=True)
class MetricConfiguration:
    label: str
    unit: str
    scope: str
    source: ConfigEvidence
    comparison: ConfigComparison | None = None
    note: str = ""


@dataclass(frozen=True)
class ConfigDocument:
    """Read selected fields without exporting the complete recipe."""

    data: CommentedMap
    source_id: int

    def field(self, *path: str) -> ConfigEvidence | None:
        parent = self.data
        for key in path[:-1]:
            parent = parent.get(key) if isinstance(parent, CommentedMap) else None
        if not isinstance(parent, CommentedMap) or path[-1] not in parent:
            return None
        value = parent[path[-1]]
        if value is not None and not isinstance(value, str | int | float | bool):
            return None
        try:
            line = parent.lc.key(path[-1])[0] + 1
        except (KeyError, TypeError):
            line = None  # YAML merge keys may not have a local source line.
        return ConfigEvidence(self.source_id, ".".join(path), value, line)


class ConfigLogEvidence(Protocol):
    def line(self, source_id: int, line: int) -> str | None: ...


class MetricConfigAdapter(Protocol):
    def read(
        self,
        document: ConfigDocument,
        series: dict[str, Any],
        role: str,
        metric_series: dict[int, dict[str, Any]],
        logs: ConfigLogEvidence,
    ) -> tuple[MetricConfiguration, ...]: ...
