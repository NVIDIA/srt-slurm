# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TokenSpeed recipe fields related to independent runtime observations."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from ..log_metrics.tokenspeed import ACTIVE_DECODE, ACTIVE_PAGES, DECODE_LIMIT, POOL_PAGES
from .base import ConfigComparison, ConfigDocument, ConfigEvidence, ConfigLogEvidence, MetricConfiguration


@dataclass(frozen=True)
class FieldBinding:
    argument: str
    label: str
    unit: str
    scope: str
    scheduler_limit: bool = False


_BATCH = FieldBinding("max-num-seqs", "Configured max requests", "requests", "Global across attention DP ranks", True)
_KV = (
    FieldBinding("max-total-tokens", "Configured KV token budget", "tokens", "Worker token budget"),
    FieldBinding("prefix-granularity", "Configured page granularity", "tokens", "Tokens per page"),
)
_BINDINGS = {ACTIVE_DECODE: (_BATCH,), DECODE_LIMIT: (_BATCH,), ACTIVE_PAGES: _KV, POOL_PAGES: _KV}
_DP = re.compile(r"(?:^|[ (])dp_size=(\d+)(?=[ )]|$)")


def _batch_comparison(
    field: ConfigEvidence,
    series: dict[str, Any],
    metric_series: dict[int, dict[str, Any]],
    logs: ConfigLogEvidence,
) -> ConfigComparison | None:
    # A global ceiling equals the scheduler ceiling only for attention DP = 1.
    # No default DP size, division, or topology is inferred from recipe omission.
    value = field.value
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        return None
    limit = series if series["name"] == DECODE_LIMIT else metric_series.get(series.get("reference", {}).get("series_id"))
    if not limit or not limit["points"]:
        return None
    evidence = []
    records = limit.get("setting_evidence", [[p[0], p[2], p[3]] for p in limit["points"]])
    for _time, source_id, line in records:
        raw = logs.line(source_id, line)
        match = _DP.search(raw or "")
        if not match or int(match[1]) != 1:
            return None
        evidence.append(ConfigEvidence(source_id, "dp_size", 1, line))
    return ConfigComparison(
        value,
        "requests",
        min(point[0] for point in limit["points"]),
        "The same scheduler log reports attention dp_size=1 throughout the captured settings; "
        "the global recipe ceiling is directly comparable to this scheduler.",
        tuple(evidence),
    )


class TokenSpeedMetricConfiguration:
    def read(
        self,
        document: ConfigDocument,
        series: dict[str, Any],
        role: str,
        metric_series: dict[int, dict[str, Any]],
        logs: ConfigLogEvidence,
    ) -> tuple[MetricConfiguration, ...]:
        result = []
        for binding in _BINDINGS.get(series["name"], ()):
            # A present role override (including null/invalid) must not fall back.
            roles = document.data.get("roles", {})
            role_config = roles.get(role, {}) if isinstance(roles, dict) else {}
            arguments = role_config.get("args", {}) if isinstance(role_config, dict) else {}
            if not isinstance(arguments, dict):
                continue
            if binding.argument in arguments:
                field = document.field("roles", role, "args", binding.argument)
            else:
                field = document.field("engine", "args", binding.argument)
            if field is None:
                continue
            comparison = _batch_comparison(field, series, metric_series, logs) if binding.scheduler_limit else None
            note = ""
            if binding.scheduler_limit and comparison is None:
                note = "Global recipe setting; a per-scheduler comparison requires a positive numeric value and " \
                    "recorded attention dp_size=1 for every setting in this log scope."
            elif not binding.scheduler_limit:
                note = "Configuration context only. Token budgets and page granularity do not establish the usable KV pool size."
            result.append(MetricConfiguration(binding.label, binding.unit, binding.scope, field, comparison, note))
        return tuple(result)
