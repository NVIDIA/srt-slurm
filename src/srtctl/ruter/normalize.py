# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize saved Dynamo logs without copying or parsing Tachometer Parquet.

The output intentionally stays boring: one JSONL file for the router, one for
workers, and a manifest that records the original Tachometer Parquet path.
That keeps this post-processing step quick, inspectable, and independent of
whatever visualization is added later.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_RFC3339_TIMESTAMP = re.compile(r"(?P<timestamp>20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)")
_SGLANG_TIMESTAMP = re.compile(r"\[(?P<timestamp>20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\]")
_LOGFMT = re.compile(r'(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*(?:"(?P<quoted>[^"]*)"|(?P<bare>[^,\s}]+))')
_SCHEDULER_FIELD = re.compile(r"(?P<key>#[A-Za-z][A-Za-z0-9-]*)\s*:\s*(?P<value>[^,]+)")
_RID = re.compile(r'\brid\s*(?:=|:)\s*(?:"(?P<quoted>[^"]+)"|(?P<bare>[^,\s}]+))')
_ROUTING_WORKER = re.compile(r"\[ROUTING\] Best:\s*(?P<worker>[^\s,]+)")
_ROUTING_OVERLAP = re.compile(r"with\s+(?P<overlap>\d+)\s*/\s*(?P<total>\d+)\s+blocks")
_ROUTING_DYNAMO_REQUEST = re.compile(
    r'\[ROUTING\] Best:.*?\brequest_id\s*=\s*(?:"(?P<quoted>[^"]+)"|(?P<bare>[^,\s}]+))'
)
_ROUTING_FORMULA = re.compile(
    r"Formula for worker_id=(?P<worker>\d+) dp_rank=(?P<dp_rank>\d+) with "
    r"(?P<effective_cached_blocks>[\d.]+) effective cached blocks:\s*"
    r"(?P<cost_blocks>[\d.]+) = prefill_load_scale \* adjusted_prefill_blocks \+ "
    r"decode_blocks \+ active_request_cost_blocks =\s*(?P<prefill_load_scale>[\d.]+) \* "
    r"(?P<adjusted_prefill_blocks>[\d.]+) \+ (?P<decode_blocks>[\d.]+) \+ "
    r"(?P<active_request_cost_blocks>[\d.]+)\s*\(raw_prefill_blocks: "
    r"(?P<raw_prefill_blocks>[\d.]+), overlap_credit_blocks: "
    r"(?P<overlap_credit_blocks>[\d.]+), overlap_credit_decay: "
    r"(?P<overlap_credit_decay>[\d.]+)\)"
)
_WORKER_LOG = re.compile(r"(?:^worker-.*\.log$|.*_(?:prefill|decode|agg)_w\d+\.out$)")
_WORKER_INDEX = re.compile(r"(?:worker-(?:prefill-|decode-|agg-)?|_(?:prefill|decode|agg)_w)(?P<index>\d+)")
_WORKER_ROLE = re.compile(r"(?:worker-|_)(?P<role>prefill|decode|agg)(?:-|_)")


@dataclass(frozen=True)
class NormalizedEvent:
    """One parsed log line with source-specific facts kept in ``fields``."""

    source: str
    kind: str
    timestamp_ns: int | None
    worker_index: int | None
    worker_role: str | None
    request_id: str | None
    fields: dict[str, str]
    raw: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunInputs:
    """Raw benchmark artifacts discovered below a single output directory."""

    root: Path
    router_log: Path | None
    worker_logs: tuple[Path, ...]
    tachometer_parquet: Path | None


@dataclass(frozen=True)
class NormalizationReport:
    """Stable summary printed by the CLI and written to the manifest."""

    output_dir: str
    router_events: int
    worker_events: int
    worker_logs: int
    tachometer_parquet: str | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_output_dir(root: Path) -> Path:
    """Keep generated data under ``logs`` so SLURM's artifact sync includes it."""
    root = root.resolve()
    logs_dir = root / "logs"
    return (logs_dir if logs_dir.is_dir() else root) / ".ruter"


def normalize_run(root: Path, *, output_dir: Path | None = None) -> NormalizationReport:
    """Create or replace a portable normalized bundle for a completed run.

    Raw inputs are never changed. Tachometer's ``final.parquet`` is not loaded
    or copied; the manifest simply records its source location.
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ruter run root does not exist: {root}")
    output_dir = (output_dir or default_output_dir(root)).resolve()
    inputs = discover_inputs(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    router_count = _write_events(
        output_dir / "router-events.jsonl",
        () if inputs.router_log is None else parse_router_file(inputs.router_log),
    )
    worker_count = 0
    with _atomic_text_file(output_dir / "worker-events.jsonl") as handle:
        for ordinal, worker_log in enumerate(inputs.worker_logs):
            worker_index = _worker_log_index(worker_log, fallback=ordinal)
            worker_role = _worker_log_role(worker_log)
            for event in parse_worker_file(worker_log, worker_index=worker_index, worker_role=worker_role):
                json.dump(event.to_dict(), handle, separators=(",", ":"), sort_keys=True)
                handle.write("\n")
                worker_count += 1

    warnings: list[str] = []
    if inputs.router_log is None:
        warnings.append("router.log was not found")
    if not inputs.worker_logs:
        warnings.append("no Dynamo worker logs were found")
    if inputs.tachometer_parquet is None:
        warnings.append("Tachometer final.parquet was not found")

    report = NormalizationReport(
        output_dir=str(output_dir),
        router_events=router_count,
        worker_events=worker_count,
        worker_logs=len(inputs.worker_logs),
        tachometer_parquet=_relative_path(root, inputs.tachometer_parquet),
        warnings=tuple(warnings),
    )
    manifest = {
        "schema": "ruter.bundle.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(root),
        "router_log": _relative_path(root, inputs.router_log),
        "worker_logs": [_relative_path(root, path) for path in inputs.worker_logs],
        "tachometer_parquet": _relative_path(root, inputs.tachometer_parquet),
        "report": report.to_dict(),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return report


def discover_inputs(root: Path) -> RunInputs:
    """Find the known direct-host and SLURM log layouts under ``root``."""
    paths = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: str(path))
    router_logs = [path for path in paths if path.name == "router.log"]
    workers = [path for path in paths if _WORKER_LOG.fullmatch(path.name)]
    parquet = [path for path in paths if path.name == "final.parquet" and "tachometer" in path.parts]
    return RunInputs(
        root=root,
        router_log=_prefer_log(root, router_logs),
        worker_logs=tuple(workers),
        tachometer_parquet=parquet[0] if parquet else None,
    )


def parse_router_file(path: Path) -> Iterator[NormalizedEvent]:
    """Yield Dynamo KV-router events from a saved router log."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            event = parse_router_line(raw.rstrip("\n"))
            if event is not None:
                yield event


def parse_worker_file(path: Path, *, worker_index: int, worker_role: str | None = None) -> Iterator[NormalizedEvent]:
    """Yield Dynamo-hosted SGLang engine events from one worker log."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            event = parse_worker_line(raw.rstrip("\n"), worker_index=worker_index)
            if event is not None:
                yield replace(event, worker_role=worker_role)


def parse_router_line(raw: str) -> NormalizedEvent | None:
    """Parse one Dynamo frontend/KV-router log line."""
    line, fields = _structured_log_fields(raw)
    timestamp_ns = _timestamp_ns(raw)
    formula = _ROUTING_FORMULA.search(line)
    if formula is not None:
        fields.update(
            {
                "worker_id": formula["worker"],
                "dp_rank": formula["dp_rank"],
                "effective_cached_blocks": formula["effective_cached_blocks"],
                "cost_blocks": formula["cost_blocks"],
                "prefill_load_scale": formula["prefill_load_scale"],
                "adjusted_prefill_blocks": formula["adjusted_prefill_blocks"],
                "decode_blocks": formula["decode_blocks"],
                "active_request_cost_blocks": formula["active_request_cost_blocks"],
                "raw_prefill_blocks": formula["raw_prefill_blocks"],
                "overlap_credit_blocks": formula["overlap_credit_blocks"],
                "overlap_credit_decay": formula["overlap_credit_decay"],
            }
        )
        return _event("dynamo_router", "routing_formula", timestamp_ns, None, fields, raw)

    if "Selected worker" in line or "[ROUTING] Best:" in line:
        worker = _ROUTING_WORKER.search(line)
        if worker is not None:
            fields.setdefault("worker_id", worker["worker"])
        overlap = _ROUTING_OVERLAP.search(line)
        if overlap is not None:
            fields.setdefault("overlap_blocks", overlap["overlap"])
            fields.setdefault("total_blocks", overlap["total"])
        routing_request = _ROUTING_DYNAMO_REQUEST.search(line)
        if routing_request is not None:
            value = routing_request["quoted"] or routing_request["bare"]
            fields.setdefault("dynamo_request_id", value)
        kind = "routing_decision" if "[ROUTING] Best:" in line else "routing_candidate"
        if kind == "routing_decision" and fields.get("request_id"):
            fields.setdefault("dynamo_request_id", fields["request_id"])
        return _event("dynamo_router", kind, timestamp_ns, None, fields, raw)

    if fields.get("phase") and fields.get("span_name") == "kv_router.route_request":
        return _event("dynamo_router", "routing_dispatch", timestamp_ns, None, fields, raw)

    if "request received" in line:
        return _event("dynamo_router", "router_admission", timestamp_ns, None, fields, raw)
    return None


def parse_worker_line(raw: str, *, worker_index: int) -> NormalizedEvent | None:
    """Parse one Dynamo-hosted SGLang worker log line."""
    line, fields = _structured_log_fields(raw)
    timestamp_ns = _timestamp_ns(raw)
    if "Prefill batch," in line:
        return _event("dynamo_worker", "worker_prefill_batch", timestamp_ns, worker_index, fields, raw)
    if "Decode batch," in line:
        return _event("dynamo_worker", "worker_decode_batch", timestamp_ns, worker_index, fields, raw)
    rid = _RID.search(line)
    if rid is not None:
        fields["rid"] = rid["quoted"] or rid["bare"]
        return _event("dynamo_worker", "worker_request", timestamp_ns, worker_index, fields, raw)
    if "instance_id" in fields and ("request received" in line or "request completed" in line):
        return _event("dynamo_worker", "worker_request", timestamp_ns, worker_index, fields, raw)
    if "Model registration succeeded" in line or "server ready" in line or "Established keep-alive stream" in line:
        return _event("dynamo_worker", "worker_lifecycle", timestamp_ns, worker_index, fields, raw)
    return None


def _event(
    source: str,
    kind: str,
    timestamp_ns: int | None,
    worker_index: int | None,
    fields: dict[str, str],
    raw: str,
) -> NormalizedEvent:
    return NormalizedEvent(
        source=source,
        kind=kind,
        timestamp_ns=timestamp_ns,
        worker_index=worker_index,
        worker_role=None,
        request_id=fields.get("x_request_id") or fields.get("rid") or fields.get("request_id"),
        fields=fields,
        raw=raw,
    )


def _structured_log_fields(raw: str) -> tuple[str, dict[str, str]]:
    """Return a log message plus scalar JSON fields, when Dynamo emitted JSONL."""
    raw = _ANSI.sub("", raw)
    fields: dict[str, str] = {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)):
                fields[key] = str(value)
        line = payload.get("message")
        if not isinstance(line, str):
            line = raw
    else:
        line = raw
    for match in _LOGFMT.finditer(line):
        fields[match["key"]] = (match["quoted"] or match["bare"]).rstrip("}")
    for match in _SCHEDULER_FIELD.finditer(line):
        fields[match["key"]] = match["value"].strip()
    return line, fields


def _timestamp_ns(line: str) -> int | None:
    rfc3339 = _RFC3339_TIMESTAMP.search(line)
    if rfc3339 is not None:
        timestamp = datetime.fromisoformat(rfc3339["timestamp"].replace("Z", "+00:00"))
    else:
        sglang = _SGLANG_TIMESTAMP.search(line)
        if sglang is None:
            return None
        timestamp = datetime.fromisoformat(sglang["timestamp"]).replace(tzinfo=timezone.utc)
    return int(timestamp.timestamp()) * 1_000_000_000 + timestamp.microsecond * 1_000


def _prefer_log(root: Path, paths: list[Path]) -> Path | None:
    if not paths:
        return None
    direct = root / "logs" / "router.log"
    return direct if direct in paths else paths[0]


def _worker_log_index(path: Path, *, fallback: int) -> int:
    match = _WORKER_INDEX.search(path.name)
    return int(match["index"]) if match is not None else fallback


def _worker_log_role(path: Path) -> str:
    match = _WORKER_ROLE.search(path.name)
    return match["role"] if match is not None else "agg"


def _relative_path(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _write_events(path: Path, events: Iterable[NormalizedEvent]) -> int:
    count = 0
    with _atomic_text_file(path) as handle:
        for event in events:
            json.dump(event.to_dict(), handle, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
            count += 1
    return count


class _atomic_text_file:
    """Write a text output atomically without leaving a partial visible bundle."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary_path = path.with_name(f".{path.name}.tmp")
        self.handle: TextIO | None = None

    def __enter__(self) -> TextIO:
        self.handle = self.temporary_path.open("w", encoding="utf-8")
        return self.handle

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        assert self.handle is not None
        self.handle.close()
        if exc_type is None:
            self.temporary_path.replace(self.path)
        else:
            self.temporary_path.unlink(missing_ok=True)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with _atomic_text_file(path) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
