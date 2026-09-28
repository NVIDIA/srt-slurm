# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared source identities and Dynamo log encodings, independent of engines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .identities import canonical_role
_WORKER = re.compile(
    r"(?P<host>.+)_(?P<role>prefill|decode|agg|aggregated)_w(?P<index>\d+)"
    r"(?:_e(?P<engine>\d+))?(?:_profile_(?:rank(?P<rank>\d+)|gpu(?P<gpus>[\d-]+)))?"
    r"(?:_window\d+)?\.(?:out|sqlite)$"
)
_FRONTEND = re.compile(r"(.+)_frontend_\d+(?:_window\d+)?\.(?:out|sqlite)$")


@dataclass(frozen=True)
class SourceIdentity:
    worker: str
    host: str
    role: str
    index: int | None = None
    rank: int | None = None
    engine: int | None = None
    gpus: str | None = None


def source_identity(path: Path) -> SourceIdentity | None:
    """Filename identity describes a capture, not ranks hidden inside it."""
    if m := _WORKER.fullmatch(path.name):
        role, index = canonical_role(m["role"]), int(m["index"])
        return SourceIdentity(
            f"{role}-{index}",
            m["host"],
            role,
            index,
            int(m["rank"]) if m["rank"] else None,
            int(m["engine"]) if m["engine"] else None,
            m["gpus"],
        )
    if m := _FRONTEND.fullmatch(path.name):
        return SourceIdentity("frontend", m[1], "frontend")
    return None


def otel_files(logs: Path) -> list[Path]:
    return sorted({p for pattern in ("otel/traces.jsonl", "otel/*/traces.jsonl") for p in logs.glob(pattern)})
