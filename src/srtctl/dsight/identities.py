# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode common Dynamo identities independently of inference-engine formats."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_UUID = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_TEXT_FIELDS = re.compile(
    r'(?<![\w.])(x_request_id|request_id|dynamo\.(?:request\.id|instance\.id|operation\.role|process\.epoch))='
    r'("(?:[^"\\]|\\.)*"|[^\s,}\]]+)'
)


def canonical_role(role: str) -> str:
    return {"aggregated": "agg"}.get(role.lower(), role.lower())


def _log_fields(line: str) -> dict[str, Any]:
    """Read text, flat JSON and tracing-subscriber span/fields encodings."""
    line = _ANSI.sub("", line).strip()
    if not line.startswith("{"):
        try:
            return {key: json.loads(value) if value.startswith('"') else value for key, value in _TEXT_FIELDS.findall(line)}
        except json.JSONDecodeError:
            return {}
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    fields: dict[str, Any] = {}
    spans = obj.get("spans")
    if isinstance(spans, list):
        for span in spans:
            if isinstance(span, dict):
                fields.update(span)
    for key in ("span", "fields"):
        if isinstance(obj.get(key), dict):
            fields.update(obj[key])
    fields.update(obj)
    return fields


def frontend_identity(line: str) -> tuple[str, str] | None:
    fields = _log_fields(line)
    client = fields.get("x_request_id")
    server = fields.get("dynamo.request.id") or fields.get("request_id")
    if isinstance(client, str) and client and isinstance(server, str) and _UUID.fullmatch(server):
        return client, server
    return None


@dataclass(frozen=True)
class WorkerIdentity:
    server_id: str
    host: str
    role: str
    process: str


def worker_identity(line: str) -> WorkerIdentity | None:
    fields = _log_fields(line)
    server, host, role, process = (
        fields.get(key)
        for key in ("dynamo.request.id", "dynamo.instance.id", "dynamo.operation.role", "dynamo.process.epoch")
    )
    if (
        isinstance(server, str) and _UUID.fullmatch(server)
        and isinstance(host, str) and host
        and isinstance(role, str) and role
        and isinstance(process, str) and process
    ):
        return WorkerIdentity(server, host, canonical_role(role), process)
    return None
