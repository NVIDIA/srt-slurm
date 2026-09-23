# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared source identities and Dynamo log encodings, independent of engines."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ANSI = re.compile(r"\x1b\[[0-9;]*m")
UUID = r"[0-9a-fA-F-]{36}"
_WORKER = re.compile(
    r"(?P<host>.+)_(?P<role>prefill|decode|agg|aggregated)_w(?P<index>\d+)"
    r"(?:_e(?P<engine>\d+))?(?:_profile_(?:rank(?P<rank>\d+)|gpu(?P<gpus>[\d-]+)))?"
    r"(?:_window\d+)?\.(?:out|sqlite)$"
)
_FRONTEND = re.compile(r"(.+)_frontend_\d+(?:_window\d+)?\.(?:out|sqlite)$")


def canonical_role(role: str) -> str:
    return {"aggregated": "agg"}.get(role.lower(), role.lower())


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


def log_fields(line: str) -> dict[str, Any]:
    """Read recorded fields from flat or tracing-subscriber JSON."""
    line = ANSI.sub("", line).strip()
    if not line.startswith("{"):
        return {}
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    fields: dict[str, Any] = {}
    for span in obj.get("spans", []):
        if isinstance(span, dict):
            fields.update(span)
    for key in ("span", "fields"):
        if isinstance(obj.get(key), dict):
            fields.update(obj[key])
    fields.update(obj)
    return fields


def frontend_identity(line: str) -> tuple[str, str] | None:
    fields = log_fields(line)
    if fields:
        client = fields.get("x_request_id")
        server = fields.get("dynamo.request.id") or fields.get("request_id")
    else:
        line = ANSI.sub("", line)
        x = re.search(r'x_request_id="([^"]+)"', line)
        d = re.search(r"(?:dynamo\.request\.id|\brequest_id)=(" + UUID + r")", line)
        client, server = (x[1] if x else None), (d[1] if d else None)
    if isinstance(client, str) and isinstance(server, str) and re.fullmatch(UUID, server):
        return client, server
    return None


def otel_files(logs: Path) -> list[Path]:
    return sorted({p for pattern in ("otel/traces.jsonl", "otel/*/traces.jsonl") for p in logs.glob(pattern)})
