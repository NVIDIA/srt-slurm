# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize one srt-slurm run. All joins use recorded identities, never proximity.

Time values in the browser are seconds relative to origin_ns. Raw epoch and process
identities remain strings to avoid JavaScript's 53-bit integer limit.
"""

import collections
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .clients import AgentPerfAdapter
from .model import SCHEMA, lifecycle
from .nsys import read_profiles

ANSI = re.compile(r"\x1b\[[0-9;]*m")
UUID = r"[0-9a-fA-F-]{36}"


def epoch_ns(s: str) -> int:
    s = s.replace("Z", "+00:00")
    d = dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        raise ValueError("Timestamp has no UTC offset")
    return int(d.timestamp()) * 10**9 + d.microsecond * 1000


def attrs(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {i["key"]: next(iter(i["value"].values()), None) for i in items}


def metric_value(row: dict[str, Any], name: str) -> float | None:
    return row.get("metrics", {}).get(name, {}).get("value")


class Importer:
    def __init__(
        self,
        logs: Path,
        sqlites: Path | None = None,
        job: str | None = None,
        *,
        client: Path | None = None,
        metrics: Path | None = None,
        phase: str = "profiling",
        iteration_timezone: str | None = None,
        max_profile_events: int = 250_000,
        otel: bool = True,
    ) -> None:
        self.logs = logs / "logs" if (logs / "logs").is_dir() else logs
        if not self.logs.is_dir():
            raise ValueError(f"Run log directory does not exist: {self.logs}")
        self.sqlites, self.job = sqlites, job or self.logs.parent.name
        self.client_path, self.metrics_path = client, metrics
        self.phase = phase
        self.iteration_zone = ZoneInfo(iteration_timezone) if iteration_timezone else None
        self.max_profile_events = max_profile_events
        self.otel = otel
        self.workers: dict[str, dict[str, Any]] = {}
        self.worker_epochs: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
        self.warnings: list[str] = []
        self.iterations: list[dict[str, Any]] = []
        self.sources: list[dict[str, Any]] = []
        self.source_ids: dict[str, int] = {}
        self.audit: collections.Counter[str] = collections.Counter(joined_spans=0, clients_with_lifecycle=0)
        self.origin = 0
        self.profiles_data: list[dict[str, Any]] = []

    def source(self, p: Path, kind: str) -> int:
        key = str(p.resolve())
        if key not in self.source_ids:
            self.source_ids[key] = len(self.sources)
            self.sources.append(
                {
                    "id": len(self.sources),
                    "path": key,
                    "kind": kind,
                    "bytes": p.stat().st_size,
                    "modified_ns": str(p.stat().st_mtime_ns),
                }
            )
        return self.source_ids[key]

    def t(self, ns: int | str) -> float:
        return round((int(ns) - self.origin) / 1e9, 9)

    def clients(self) -> None:
        patterns = (
            "profile_export.jsonl",
            "requests.jsonl",
            "agentperf/requests.jsonl",
            "agentperf/*/requests.jsonl",
            "agentic/*/profile_export.jsonl",
            "agentic/*/aiperf_artifacts/profile_export.jsonl",
            "artifacts/*/profile_export.jsonl",
        )
        files = (
            [self.client_path]
            if self.client_path
            else sorted({p.resolve() for pat in patterns for p in self.logs.glob(pat)})
        )
        if len(files) != 1:
            raise ValueError(
                f"Expected one client export (AIPerf or AgentPerf), found {len(files)}; select it with --client"
            )
        rows = []
        native_adapter = None
        self.source(files[0], "client")
        with files[0].open(encoding="utf-8") as stream:
            for line, content in enumerate(stream, 1):
                if not content.strip():
                    continue
                try:
                    row = json.loads(content)
                    if "metadata" not in row and "start_time" in row:
                        if native_adapter is None:
                            native_adapter = AgentPerfAdapter(files[0], self.source)
                        row = native_adapter.normalize(row, line)
                    metadata = row["metadata"]
                    actual_phase = metadata.get("benchmark_phase")
                    if self.phase != "all" and actual_phase not in (None, self.phase):
                        self.audit["excluded_client_rows"] += 1
                        continue
                    if actual_phase is None:
                        self.audit["clients_without_phase"] += 1
                    for key in ("request_start_ns", "request_end_ns"):
                        if int(metadata[key]) <= 0:
                            raise ValueError(f"Invalid {key}")
                    if int(metadata["request_end_ns"]) < int(metadata["request_start_ns"]):
                        raise ValueError("Request ends before it starts")
                    rows.append((line, row))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{files[0]}:{line}: invalid client record: {exc}") from exc
        if not rows:
            raise ValueError(f"No requests for phase {self.phase!r}; warmup is never substituted")
        self.origin = min(int(r["metadata"]["request_start_ns"]) for _, r in rows)
        self.requests: list[dict[str, Any]] = []
        for line, row in rows:
            m = row["metadata"]
            start, end = self.t(m["request_start_ns"]), self.t(m["request_end_ns"])
            ttft = metric_value(row, "time_to_first_token")
            first = start + ttft / 1000 if isinstance(ttft, int | float) and math.isfinite(ttft) and ttft >= 0 else None
            if first is None:
                ttft = None
            client_id = m.get("x_request_id") or m.get("request_id") or f"client-line-{line}"
            session = m.get("root_correlation_id") or m.get("x_correlation_id") or m.get("session_id") or client_id
            # Keep inconsistencies visible; do not change measured TTFT to fit a bar.
            issues = []
            if first is not None and first > end:
                issues.append("TTFT endpoint exceeds request_end_ns")
                self.audit["client_first_after_end"] += 1
            r = {
                "id": client_id,
                "session": session,
                "agent": m.get("x_correlation_id") or session,
                "parent": m.get("parent_correlation_id"),
                "conversation": m.get("conversation_id"),
                "source_trace": m.get("source_trace_id"),
                "turn": m.get("turn_index"),
                "depth": m.get("agent_depth", 0),
                "start": start,
                "end": end,
                "first": first,
                "ttft_ms": ttft,
                "input_tokens": metric_value(row, "input_sequence_length"),
                "output_tokens": metric_value(row, "output_sequence_length"),
                "cached_tokens": metric_value(row, "usage_prompt_cache_read_tokens"),
                "status": "error"
                if row.get("error") or row.get("errors")
                else "cancelled"
                if m.get("was_cancelled")
                else "completed",
                "evidence": row.get("timing_evidence", [self.source(files[0], "client"), line]),
                "client_kind": row.get("client_kind", "aiperf"),
                "timing_quality": row.get("timing_quality", "client export"),
                "identity_evidence": row.get("identity_evidence"),
                "phase_evidence": row.get("phase_evidence"),
                "original_start_time": row.get("original_start_time"),
                "original_end_time": row.get("original_end_time"),
                "issues": issues,
                "server_ids": [],
                "spans": [],
                "engine": [],
                "raw_start_ns": str(m["request_start_ns"]),
                "raw_end_ns": str(m["request_end_ns"]),
            }
            self.requests.append(r)
        if native_adapter:
            self.warnings.append(
                "AgentPerf sessions group the recorded phase, user and conversation; no agent hierarchy is inferred."
            )
            if any(r["timing_quality"] == "request log (liveness)" for r in self.requests):
                self.warnings.append(
                    "Some AgentPerf timings come only from the liveness request log; phase-end analysis records were unavailable for those requests."
                )
        self.requests.sort(key=lambda r: (r["start"], r["id"]))
        self.by_client = {r["id"]: r for r in self.requests}
        if len(self.by_client) != len(self.requests):
            raise ValueError("Duplicate client request IDs; attempts need distinct identity")
        self.duration = max(r["end"] for r in self.requests)
        if self.duration <= 0:
            raise ValueError("Client export has no positive elapsed time")
        self.audit.update(client_requests=len(rows))
        if self.audit["clients_without_phase"]:
            self.warnings.append(
                "Some client records have no benchmark_phase; included as unclassified, not certified measured-only."
            )

    def frontend_bridge(self) -> None:
        self.by_server = {}
        self.bridge = {}
        for p in sorted(self.logs.glob("*_frontend_*.out")):
            for line, s in enumerate(p.open(errors="replace", newline="\n"), 1):
                if "x_request_id" not in s:
                    continue
                s = ANSI.sub("", s)
                x = re.search(r'x_request_id="([^"]+)"', s)
                d = re.search(r"(?:dynamo\.request\.id|\brequest_id)=(" + UUID + r")", s)
                if not x or not d or x[1] not in self.by_client:
                    continue
                r = self.by_client[x[1]]
                if d[1] not in r["server_ids"]:
                    r["server_ids"].append(d[1])
                    self.bridge[d[1]] = [self.source(p, "frontend_log"), line]
                if d[1] in self.by_server and self.by_server[d[1]]["id"] != r["id"]:
                    raise ValueError("Ambiguous client/server bridge")
                self.by_server[d[1]] = r
        self.audit["clients_with_server_identity"] = sum(bool(r["server_ids"]) for r in self.requests)
        self.audit["multiple_server_attempts"] = sum(len(r["server_ids"]) > 1 for r in self.requests)
        for r in self.requests:
            r["bridge_evidence"] = [self.bridge[x] for x in r["server_ids"]]

    def lifecycle(self) -> None:
        spans: list[dict[str, Any]] = []
        trace_requests = collections.defaultdict(set)
        seen = {}
        for p in sorted(self.logs.glob("otel/*/traces.jsonl")):
            if not p.stat().st_size:
                continue
            sid = self.source(p, "otel")
            for line, s in enumerate(p.open(), 1):
                doc = json.loads(s)
                for resource in doc.get("resourceSpans", []):
                    service = attrs(resource.get("resource", {}).get("attributes", [])).get("service.name", "unknown")
                    for scope in resource.get("scopeSpans", []):
                        for ordinal, span in enumerate(scope.get("spans", [])):
                            a = attrs(span.get("attributes", []))
                            rid = a.get("dynamo.request.id")
                            trace = span["traceId"]
                            if rid in self.by_server:
                                trace_requests[trace].add(self.by_server[rid]["id"])
                            if int(span["endTimeUnixNano"]) < self.origin or int(
                                span["startTimeUnixNano"]
                            ) > self.origin + round(self.duration * 1e9):
                                continue
                            key = (trace, span["spanId"])
                            signature = (
                                span["name"],
                                span["startTimeUnixNano"],
                                span["endTimeUnixNano"],
                                span.get("parentSpanId"),
                            )
                            if key in seen:
                                if seen[key] != signature:
                                    raise ValueError(f"{p}:{line}: conflicting duplicate OTel span {key}")
                                self.audit["duplicate_spans"] += 1
                                continue
                            seen[key] = signature
                            start, end = self.t(span["startTimeUnixNano"]), self.t(span["endTimeUnixNano"])
                            if end < start:
                                self.audit["invalid_spans"] += 1
                                continue
                            role = a.get(
                                "dynamo.operation.role", "frontend" if service == "dynamo-frontend" else "unknown"
                            )
                            host = a.get("dynamo.instance.id", p.parent.name)
                            epoch = a.get("dynamo.process.epoch")
                            if epoch:
                                self.worker_epochs[(host, role)].add(epoch)
                            spans.append(
                                {
                                    "id": span["spanId"],
                                    "trace": trace,
                                    "parent": span.get("parentSpanId"),
                                    "name": span["name"],
                                    "start": start,
                                    "end": end,
                                    "role": role,
                                    "host": host,
                                    "process": epoch,
                                    "request": rid,
                                    "operation": a.get("dynamo.operation.id"),
                                    "service": service,
                                    "evidence": [sid, line, ordinal],
                                    "raw_start_ns": str(span["startTimeUnixNano"]),
                                    "raw_end_ns": str(span["endTimeUnixNano"]),
                                    "routing": {
                                        k: a.get(k)
                                        for k in (
                                            "phase",
                                            "worker_id",
                                            "dp_rank",
                                            "request_id",
                                            "request.attempt",
                                            "migration.is_retry",
                                            "request.outcome",
                                        )
                                    }
                                    if span["name"] == "kv_router.route_request"
                                    else None,
                                    "outcome": a.get("dynamo.request.outcome"),
                                    "session_source": a.get("dynamo.session.source"),
                                }
                            )
        for s in spans:
            candidates = trace_requests.get(s["trace"], set())
            r = self.by_server.get(s["request"])
            if r is None and len(candidates) == 1:
                r = self.by_client[next(iter(candidates))]
            if r is None:
                self.audit["unjoined_spans"] += 1
                continue
            s["join"] = "request_id" if s["request"] in self.by_server else "trace_id"
            r["spans"].append(s)
        for r in self.requests:
            r["spans"].sort(key=lambda s: (s["start"], -s["end"]))
            self.route_context(r)
        self.audit["joined_spans"] = sum(len(r["spans"]) for r in self.requests)

    def route_context(self, request: dict[str, Any]) -> None:
        siblings = collections.defaultdict(list)
        for span in request["spans"]:
            if span["name"] in ("kv_router.select_worker", "kv_router.route_request"):
                siblings[(span["trace"], span["parent"])].append(span)
        for group in siblings.values():
            group.sort(key=lambda s: (s["start"], s["end"], s["id"]))
            for index, span in enumerate(group[:-1]):
                route = group[index + 1]
                if (
                    span["name"] != "kv_router.select_worker"
                    or route["name"] != "kv_router.route_request"
                    or span["end"] > route["start"]
                    or not route.get("routing", {}).get("phase")
                ):
                    continue
                span["routing_context"] = {
                    "phase": route["routing"]["phase"],
                    "route_span_id": route["id"],
                    "route_evidence": route["evidence"],
                    "basis": "Inferred from the immediately following route with the same trace and parent; phase is recorded on that route.",
                }
                self.audit["associated_router_selections"] += 1

    def engine(self) -> None:
        pattern = re.compile(r"Engine ID map: request_id=(\S+) trtllm_client_id=(\S+) disagg_request_id=(\S+)")
        iteration = re.compile(
            r"iter = (\d+).*?global_rank = (\d+).*?rank = (\d+).*?num_scheduled_requests = (\d+).*?kv_cache_util = ([\d.]+).*?host_step_time = ([\d.eE+-]+)ms.*?prev_device_step_time = ([\d.eE+-]+)ms.*?timestamp = ([\d-]+ [\d:]+)"
        )
        id_owners = collections.defaultdict(set)
        for p in sorted(self.logs.glob("*_w*.out")):
            match = re.match(r"(.+)_(prefill|decode|aggregated)_w(\d+)\.out", p.name)
            if not match:
                continue
            host, role, index = match.groups()
            wid = f"{role}-{index}"
            if wid in self.workers and self.workers[wid]["host"] != host:
                raise ValueError(f"Ambiguous worker {wid}: multiple leaders in selected logs")
            epochs = sorted(self.worker_epochs.get((host, role), set()))
            self.workers[wid] = {
                "id": wid,
                "role": role,
                "index": int(index),
                "host": host,
                "process_epochs": epochs,
                "profiles": [],
                "metrics": [],
            }
            with p.open(errors="replace", newline="\n") as stream:
                for line, s in enumerate(stream, 1):
                    if "iter =" in s and (it := iteration.search(s)):
                        time = dt.datetime.fromisoformat(it[8])
                        anchored = (
                            self.t(int(time.replace(tzinfo=self.iteration_zone).timestamp()) * 10**9)
                            if self.iteration_zone
                            else None
                        )
                        if anchored is not None and (anchored + 1 < 0 or anchored > self.duration):
                            continue
                        self.iterations.append(
                            {
                                "worker": wid,
                                "iteration": int(it[1]),
                                "global_rank": int(it[2]),
                                "rank": int(it[3]),
                                "batch_requests": int(it[4]),
                                "kv_cache_util": float(it[5]),
                                "host_step_ms": float(it[6]),
                                "previous_device_step_ms": float(it[7]),
                                "local_time": it[8],
                                "start": anchored,
                                "end": anchored + 1 if anchored is not None else None,
                                "evidence": [self.source(p, "worker_log"), line],
                            }
                        )
                    m = pattern.search(s)
                    if not m or m[1] not in self.by_server:
                        continue
                    r = self.by_server[m[1]]
                    process = {
                        sp["process"]
                        for sp in r["spans"]
                        if sp["host"] == host and sp["role"] == role and sp["process"]
                    }
                    entry = {
                        "worker": wid,
                        "host": host,
                        "role": role,
                        "client_id": m[2],
                        "disagg_id": m[3],
                        "server_id": m[1],
                        "process": next(iter(process)) if len(process) == 1 else None,
                        "evidence": [self.source(p, "worker_log"), line],
                    }
                    stamp = re.search(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)", s)
                    if stamp:
                        entry["observed_at"] = self.t(epoch_ns(stamp[0]))
                    r["engine"].append(entry)
                    id_owners[(wid, entry["process"], m[2])].add(m[1])
        self.audit["ambiguous_engine_ids"] = sum(len(v) > 1 for v in id_owners.values())
        self.audit["clients_with_both_engine_maps"] = sum(
            {e["role"] for e in r["engine"]} >= {"prefill", "decode"} for r in self.requests
        )
        self.audit["engine_map_rows"] = sum(len(r["engine"]) for r in self.requests)
        for r in self.requests:
            for entry in r["engine"]:
                entry["identity_ambiguous"] = (
                    entry["process"] is None
                    or len(id_owners[(entry["worker"], entry["process"], entry["client_id"])]) > 1
                )
            for span in r["spans"]:
                candidates = {
                    e["worker"]
                    for e in r["engine"]
                    if e["host"] == span["host"]
                    and e["role"] == span["role"]
                    and e["process"] is not None
                    and e["process"] == span["process"]
                }
                span["worker"] = next(iter(candidates)) if len(candidates) == 1 else None
                if len(candidates) > 1:
                    self.audit["ambiguous_span_workers"] += 1
            r["workers"] = sorted({e["worker"] for e in r["engine"]})
        self.audit["iteration_rows"] = len(self.iterations)

    def metrics(self) -> None:
        from .metrics import read_metrics

        self.metric_series = read_metrics(self)

    def run(self) -> dict[str, Any]:
        self.clients()
        self.frontend_bridge()
        if self.otel:
            self.lifecycle()
        self.engine()
        self.metrics()
        read_profiles(self)
        sessions = collections.defaultdict(list)
        for request in self.requests:
            sessions[request["session"]].append(request)
            request["lifecycle"] = lifecycle(request)
        self.audit["clients_with_lifecycle"] = sum(r["lifecycle"]["available"] for r in self.requests)
        grouped = [
            {
                "id": sid,
                "start": min(r["start"] for r in rs),
                "end": max(r["end"] for r in rs),
                "requests": [r["id"] for r in rs],
                "source_traces": sorted({r["source_trace"] for r in rs if r["source_trace"]}),
                "agents": sorted({r["agent"] for r in rs}),
            }
            for sid, rs in sessions.items()
        ]
        grouped.sort(key=lambda s: s["start"])
        self.audit["sessions"] = len(grouped)
        self.audit["nsight_reports"] = len(self.profiles_data)
        self.audit["nsight_selected_events"] = sum(len(p["events"]) for p in self.profiles_data)
        limitations = [
            "Milestone blocks measure elapsed time between boundaries, not exclusive backend costs.",
            "Engine per-request queue/compute/KV timing is not inferred from lifecycle spans or iteration logs.",
            "Nsight ranges and iteration logs are shared worker/rank context; overlap is not request ownership.",
            "Client TTFT uses the benchmark metric; frontend SSE readiness is a separate server event.",
            "Cross-host skew is not measured. Recorded UTC anchors are used without correction.",
            "Iteration timestamps have one-second precision; counters and previous-device timers can lag the forward pass. No universal counter shift is applied.",
        ]
        if self.iterations and not self.iteration_zone:
            limitations.append(
                "Iteration timestamps have no timezone: they remain unaligned. Rebuild with --iteration-timezone to align coarse windows."
            )
        if not self.profiles_data:
            self.warnings.append("No Nsight SQLite exports supplied.")
        if not self.metric_series:
            self.warnings.append("No selected raw metric series available.")
        data = {
            "schema": SCHEMA,
            "meta": {
                "job": self.job,
                "origin_ns": str(self.origin),
                "start_utc": dt.datetime.fromtimestamp(self.origin / 1e9, dt.timezone.utc).isoformat(),
                "duration": self.duration,
                "source_root": str(self.logs.resolve()),
                "phase": self.phase,
                "otel_enabled": self.otel,
                "session_key": "root_correlation_id, then client correlation/session/request identity",
                "clock": "Recorded UTC anchors; cross-host skew uncalibrated",
                "iteration_timezone": str(self.iteration_zone) if self.iteration_zone else None,
                "limitations": limitations,
                "warnings": self.warnings,
            },
            "audit": dict(self.audit),
            "sources": self.sources,
            "requests": self.requests,
            "sessions": grouped,
            "workers": list(self.workers.values()),
            "metrics": self.metric_series,
            "profiles": self.profiles_data,
            "iterations": self.iterations,
        }
        for source in self.sources:
            stat = Path(source["path"]).stat()
            if stat.st_size != source["bytes"] or str(stat.st_mtime_ns) != source["modified_ns"]:
                raise ValueError(f"Source changed during import: {source['path']}; use a preserved run")
        return data
