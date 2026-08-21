# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional local server for the static ruter route-decision viewer."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import fmean
from typing import Any
from urllib.parse import parse_qs, urlparse

from srtctl.ruter.normalize import _timestamp_ns, normalize_run

logger = logging.getLogger(__name__)

_ROUTER_PREFILL = "dynamo_frontend_worker_active_prefill_tokens"
_ROUTER_TTFT = "dynamo_frontend_worker_last_time_to_first_token_seconds"
_ROUTER_ITL = "dynamo_frontend_worker_last_inter_token_latency_seconds"
_ENGINE_RUNNING = "sglang:num_running_reqs"
_ENGINE_QUEUED = "sglang:num_queue_reqs"
_ENGINE_KV_USED = "dynamo_component_gpu_cache_usage_percent"
_WORKER_ID_LABEL = re.compile(r'(?:^|[,{])worker_id="(?P<worker_id>[^"]+)"')
_WORKER_ENDPOINT = re.compile(r"worker_agg_(?P<index>\d+)_")


def main(argv: list[str] | None = None) -> None:
    """Launch the local viewer from either a run root or ``logs/.ruter`` bundle."""
    parser = argparse.ArgumentParser(description="Serve the local ruter route-decision viewer")
    parser.add_argument("root", nargs="?", type=Path, default=Path("."), help="srt-slurm run directory or logs/.ruter")
    parser.add_argument("--port", type=int, default=8877, help="Loopback port (default: 8877)")
    parser.add_argument("--refresh", action="store_true", help="Reparse logs before loading the viewer")
    args = parser.parse_args(argv)

    try:
        data = load_view_data(args.root, refresh=args.refresh)
    except ModuleNotFoundError as error:
        if error.name == "pyarrow":
            parser.error("ruter view requires the optional UI dependency: uv sync --extra ruter")
        raise
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    launch(data, args.port)


@dataclass(frozen=True)
class _Candidate:
    timestamp_ns: int | None
    worker_id: str
    values: dict[str, Any]


@dataclass
class ViewData:
    run_root: Path
    bundle_dir: Path
    traces: list[dict[str, Any]]
    aiperf: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    candidates: dict[str, list[_Candidate]]
    snapshots: dict[tuple[str, str], dict[str, float | None]]
    worker_aliases: dict[str, str]
    router_settings: dict[str, Any] | None
    benchmark_start_ns: int | None

    def summary(self) -> dict[str, Any]:
        kv_hits = [row["kv_hit_rate"] for row in self.traces if row["kv_hit_rate"] is not None]
        ttfts = [row["ttft_ms"] for row in self.traces if row["ttft_ms"] is not None]
        return {
            "requestTraces": len(self.traces),
            "aiperfRequests": len(self.aiperf),
            "decisions": len(self.decisions),
            "workers": len(self.worker_aliases),
            "workerAliases": list(self.worker_aliases.values()),
            "avgKvHitRate": fmean(kv_hits) if kv_hits else None,
            "avgTtftMs": fmean(ttfts) if ttfts else None,
            "routerSettings": self.router_settings,
        }

    def timeline(self) -> dict[str, list[dict[str, Any]]]:
        if self.benchmark_start_ns is None:
            return {"traces": [], "aiperf": []}
        decisions = {
            (row["dynamo_request_id"], row["stage"]): row for row in self.decisions if row["dynamo_request_id"]
        }
        traces: list[dict[str, Any]] = []
        for row in self.traces:
            timestamp = row["request_received_ns"] or row["event_time_ns"]
            if timestamp is None:
                continue
            prefill_decision = decisions.get((row["request_id"], "prefill")) or decisions.get(
                (row["request_id"], "aggregate")
            )
            decode_decision = decisions.get((row["request_id"], "decode"))
            prefill_worker_id = row["prefill_worker_id"] or (prefill_decision or {}).get("worker_id")
            decode_worker_id = row["decode_worker_id"] or (decode_decision or {}).get("worker_id")
            traces.append(
                {
                    "benchS": (timestamp - self.benchmark_start_ns) / 1_000_000_000,
                    "requestId": row["x_request_id"] or row["request_id"],
                    "dynamoRequestId": row["request_id"],
                    "inputTokens": row["input_tokens"],
                    "cachedTokens": row["cached_tokens"],
                    "kvHitRate": row["kv_hit_rate"],
                    "ttftMs": row["ttft_ms"],
                    "e2eMs": row["total_time_ms"],
                    "queueDepth": row["queue_depth"],
                    "dpRank": row["prefill_dp_rank"],
                    "prefillWorkerAlias": self.worker_aliases.get(prefill_worker_id) if prefill_worker_id else None,
                    "decodeWorkerAlias": self.worker_aliases.get(decode_worker_id) if decode_worker_id else None,
                    "prefillDecisionId": (prefill_decision or {}).get("decision_id"),
                    "decodeDecisionId": (decode_decision or {}).get("decision_id"),
                    "lowerPrefixSelected": bool((prefill_decision or {}).get("lower_prefix_selected")),
                }
            )
        aiperf = [
            {
                "benchS": (row["credit_issued_ns"] - self.benchmark_start_ns) / 1_000_000_000,
                "requestId": row["request_id"],
                "inputTokens": row["input_tokens"],
                "outputTokens": row["output_tokens"],
                "ttftMs": row["ttft_ms"],
                "e2eMs": row["e2e_ms"],
            }
            for row in self.aiperf
            if row["credit_issued_ns"] is not None
        ]
        return {"traces": sorted(traces, key=lambda row: row["benchS"]), "aiperf": aiperf}

    def decision_rows(self) -> list[dict[str, Any]]:
        if self.benchmark_start_ns is None:
            return []
        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        for decision in self.decisions:
            request_id = decision["dynamo_request_id"] or decision["decision_id"]
            if request_id is None:
                continue
            stage = "decode" if decision["stage"] == "decode" else "prefill"
            grouped.setdefault(request_id, {}).setdefault(stage, decision)

        rows = []
        for request_id, stages in grouped.items():
            prefill = stages.get("prefill")
            decode = stages.get("decode")
            timestamp = (prefill or decode or {}).get("timestamp_ns")
            if timestamp is None:
                continue
            prefill_selected = (
                self._candidate(prefill["decision_id"], prefill["worker_id"]) if prefill is not None else None
            )
            decode_selected = (
                self._candidate(decode["decision_id"], decode["worker_id"]) if decode is not None else None
            )
            rows.append(
                {
                    "benchS": (timestamp - self.benchmark_start_ns) / 1_000_000_000,
                    "requestId": request_id,
                    "prefillDecisionId": prefill["decision_id"] if prefill else None,
                    "decodeDecisionId": decode["decision_id"] if decode else None,
                    "prefillWorkerAlias": self.worker_aliases.get(prefill["worker_id"]) if prefill else None,
                    "decodeWorkerAlias": self.worker_aliases.get(decode["worker_id"]) if decode else None,
                    "overlapBlocks": prefill["overlap_blocks"] if prefill else None,
                    "totalBlocks": prefill["total_blocks"] if prefill else None,
                    "prefillScoreBlocks": prefill_selected.values.get("cost_blocks") if prefill_selected else None,
                    "decodeScoreBlocks": decode_selected.values.get("cost_blocks") if decode_selected else None,
                    "lowerPrefixSelected": bool(prefill and prefill["lower_prefix_selected"]),
                }
            )
        return sorted(rows, key=lambda row: row["benchS"])[:5000]

    def decision(self, decision_id: str | None) -> dict[str, Any]:
        if not decision_id or self.benchmark_start_ns is None:
            return {"found": False}
        decision = next((row for row in self.decisions if row["decision_id"] == decision_id), None)
        if decision is None:
            decision = next(
                (
                    row
                    for row in self.decisions
                    if row["dynamo_request_id"] == decision_id and row["stage"] in {"prefill", "aggregate"}
                ),
                None,
            )
        if decision is None or decision["timestamp_ns"] is None:
            return {"found": False}
        trace = next((row for row in self.traces if row["request_id"] == decision["dynamo_request_id"]), None)
        candidates = []
        for candidate in sorted(
            self.candidates.get(decision["decision_id"], []),
            key=lambda row: (row.values.get("cost_blocks") is None, row.values.get("cost_blocks") or 0, row.worker_id),
        ):
            values = candidate.values
            candidates.append(
                {
                    "workerAlias": self.worker_aliases.get(candidate.worker_id),
                    "selected": candidate.worker_id == decision["worker_id"],
                    "dpRank": values.get("dp_rank"),
                    "costBlocks": values.get("cost_blocks"),
                    "effectiveCachedBlocks": values.get("effective_cached_blocks"),
                    "prefillLoadScale": values.get("prefill_load_scale"),
                    "adjustedPrefillBlocks": values.get("adjusted_prefill_blocks"),
                    "rawPrefillBlocks": values.get("raw_prefill_blocks"),
                    "overlapCreditBlocks": values.get("overlap_credit_blocks"),
                    "overlapCreditDecay": values.get("overlap_credit_decay"),
                    "decodeBlocks": values.get("decode_blocks"),
                    "activeRequestCostBlocks": values.get("active_request_cost_blocks"),
                    **self.snapshots.get((decision["decision_id"], candidate.worker_id), _empty_snapshot()),
                }
            )
        return {
            "found": True,
            "benchS": (decision["timestamp_ns"] - self.benchmark_start_ns) / 1_000_000_000,
            "selectedWorkerAlias": self.worker_aliases.get(decision["worker_id"]),
            "stage": decision["stage"],
            "lowerPrefixSelected": decision["lower_prefix_selected"],
            "requestPath": {
                "prefillWorkerAlias": self.worker_aliases.get(trace["prefill_worker_id"])
                if trace and trace["prefill_worker_id"]
                else None,
                "decodeWorkerAlias": self.worker_aliases.get(trace["decode_worker_id"])
                if trace and trace["decode_worker_id"]
                else None,
            },
            "dpRank": decision["dp_rank"],
            "overlapBlocks": decision["overlap_blocks"],
            "totalBlocks": decision["total_blocks"],
            "kvHitRate": trace["kv_hit_rate"] if trace else None,
            "ttftMs": trace["ttft_ms"] if trace else None,
            "e2eMs": trace["total_time_ms"] if trace else None,
            "inputTokens": trace["input_tokens"] if trace else None,
            "cachedTokens": trace["cached_tokens"] if trace else None,
            "candidates": candidates,
        }

    def _candidate(self, decision_id: str | None, worker_id: str | None) -> _Candidate | None:
        if decision_id is None or worker_id is None:
            return None
        return next((row for row in self.candidates.get(decision_id, []) if row.worker_id == worker_id), None)


def load_view_data(root: Path, *, refresh: bool = False) -> ViewData:
    """Load all UI inputs once, before the browser is served."""
    run_root, bundle_dir = _resolve_paths(root)
    if refresh or not (bundle_dir / "manifest.json").is_file():
        normalize_run(run_root, output_dir=bundle_dir)
    router_events = list(_jsonl(bundle_dir / "router-events.jsonl"))
    worker_events = list(_jsonl(bundle_dir / "worker-events.jsonl"))
    traces = _request_traces(run_root)
    aiperf = _aiperf_requests(run_root)
    decisions, candidates = _routing(router_events, traces)
    workers = {row["worker_id"] for row in decisions if row["worker_id"]}
    workers.update(candidate.worker_id for options in candidates.values() for candidate in options)
    aliases = _worker_aliases(workers, traces, decisions, candidates)
    metrics = _metric_series(run_root)
    snapshots = _snapshots(decisions, candidates, worker_events, metrics)
    start_candidates = [
        *(row["credit_issued_ns"] for row in aiperf if row["credit_issued_ns"] is not None),
        *(row["request_received_ns"] for row in traces if row["request_received_ns"] is not None),
        *(row["timestamp_ns"] for row in decisions if row["timestamp_ns"] is not None),
    ]
    return ViewData(
        run_root=run_root,
        bundle_dir=bundle_dir,
        traces=traces,
        aiperf=aiperf,
        decisions=decisions,
        candidates=candidates,
        snapshots=snapshots,
        worker_aliases=aliases,
        router_settings=_router_settings(run_root),
        benchmark_start_ns=min(start_candidates) if start_candidates else None,
    )


def launch(data: ViewData, port: int) -> None:
    """Serve static UI assets and compact JSON endpoints on loopback only."""
    if not 1 <= port <= 65535:
        raise ValueError("--port must be in 1..65535")
    handler = _handler(data)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    logger.info("ruter view loaded %s", data.run_root)
    print(f"ruter view: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _handler(data: ViewData) -> type[BaseHTTPRequestHandler]:
    assets = Path(__file__).with_name("ui")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path
            if route in {"/", "/index.html"}:
                self._asset(assets / "index.html", "text/html; charset=utf-8")
            elif route == "/app.js":
                self._asset(assets / "app.js", "application/javascript; charset=utf-8")
            elif route == "/assets/certified-beaver.png":
                self._asset(assets / "assets" / "certified-beaver.png", "image/png")
            elif route == "/assets/ruter.mp3":
                self._asset(assets / "assets" / "ruter.mp3", "audio/mpeg")
            elif route == "/api/summary":
                self._json(data.summary())
            elif route == "/api/timeline":
                self._json(data.timeline())
            elif route == "/api/decisions":
                self._json(data.decision_rows())
            elif route == "/api/decision":
                self._json(data.decision(parse_qs(parsed.query).get("id", [None])[0]))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _asset(self, path: Path, content_type: str) -> None:
            self._send(path.read_bytes(), content_type)

        def _json(self, value: Any) -> None:
            self._send(json.dumps(value, separators=(",", ":")).encode(), "application/json")

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            logger.debug("ruter view: " + format, *args)

    return Handler


def _resolve_paths(path: Path) -> tuple[Path, Path]:
    path = path.resolve()
    if path.name == ".ruter" and path.parent.name == "logs":
        return path.parent.parent, path
    bundle_dir = path / "logs" / ".ruter"
    if path.is_dir() and (bundle_dir.is_dir() or (path / "logs").is_dir()):
        return path, bundle_dir
    raise FileNotFoundError(f"expected an srt-slurm run root or logs/.ruter, got {path}")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _request_traces(root: Path) -> list[dict[str, Any]]:
    rows = []
    paths = set(root.rglob("dynamo-request-trace*.jsonl*"))
    paths.update(root.rglob("dynamo-request-trace"))
    for path in sorted(paths):
        for record in _read_jsonl(path):
            event = record.get("event", record)
            if event.get("schema") != "dynamo.request.trace.v1" or event.get("event_type") != "request_end":
                continue
            request = event.get("request") or {}
            rows.append(
                {
                    "request_id": _string(request.get("request_id")),
                    "x_request_id": _string(request.get("x_request_id")),
                    "event_time_ns": _ns(event.get("event_time_unix_ms")),
                    "request_received_ns": _ns(request.get("request_received_ms")),
                    "input_tokens": _integer(request.get("input_tokens")),
                    "output_tokens": _integer(request.get("output_tokens")),
                    "cached_tokens": _integer(request.get("cached_tokens")),
                    "kv_hit_rate": _number(request.get("kv_hit_rate")),
                    "ttft_ms": _number(request.get("ttft_ms")),
                    "total_time_ms": _number(request.get("total_time_ms")),
                    "queue_depth": _integer(request.get("queue_depth")),
                    "prefill_worker_id": _canonical_worker_id(_nested(request, "worker", "prefill_worker_id")),
                    "prefill_dp_rank": _integer(_nested(request, "worker", "prefill_dp_rank")),
                    "decode_worker_id": _canonical_worker_id(_nested(request, "worker", "decode_worker_id")),
                    "decode_dp_rank": _integer(_nested(request, "worker", "decode_dp_rank")),
                }
            )
    return rows


def _aiperf_requests(root: Path) -> list[dict[str, Any]]:
    paths = sorted(path for path in root.rglob("trace.jsonl") if "aiperf" in path.parts)
    rows = []
    for path in paths:
        for record in _read_jsonl(path):
            metadata = record.get("metadata") or {}
            metrics = record.get("metrics") or {}
            rows.append(
                {
                    "request_id": _string(metadata.get("x_request_id")),
                    "credit_issued_ns": _integer(metadata.get("credit_issued_ns")),
                    "input_tokens": _metric_value(metrics, "input_sequence_length"),
                    "output_tokens": _metric_value(metrics, "output_sequence_length"),
                    "ttft_ms": _metric_value(metrics, "time_to_first_token"),
                    "e2e_ms": _metric_value(metrics, "request_latency"),
                }
            )
    return rows


def _routing(
    events: list[dict[str, Any]], traces: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, list[_Candidate]]]:
    """Split each request's sequential Dynamo score batches into route decisions.

    In P/D mode Dynamo writes one formula batch plus ``[ROUTING] Best`` record
    for prefill and then another pair for decode. The log event has no stage
    label, so the completed request trace supplies the authoritative selected
    prefill/decode worker identity.
    """
    decisions: list[dict[str, Any]] = []
    candidates: dict[str, list[_Candidate]] = {}
    pending: dict[str, list[_Candidate]] = defaultdict(list)
    decision_counts: dict[str, int] = defaultdict(int)
    dispatches: dict[str, list[tuple[int | None, str]]] = defaultdict(list)
    for event in events:
        fields = event.get("fields") or {}
        request_id = _string(fields.get("dynamo_request_id") or fields.get("request_id") or event.get("request_id"))
        worker_id = _canonical_worker_id(fields.get("worker_id"))
        if event.get("kind") == "routing_formula" and request_id and worker_id:
            values = {name: _number(fields.get(name)) for name in _formula_fields()}
            values["dp_rank"] = _integer(fields.get("dp_rank"))
            pending[request_id].append(_Candidate(_integer(event.get("timestamp_ns")), worker_id, values))
        elif event.get("kind") == "routing_decision" and request_id and worker_id:
            ordinal = decision_counts[request_id]
            decision_counts[request_id] += 1
            decision_id = f"{request_id}:{ordinal}"
            decisions.append(
                {
                    "decision_id": decision_id,
                    "timestamp_ns": _integer(event.get("timestamp_ns")),
                    "dynamo_request_id": request_id,
                    "worker_id": worker_id,
                    "dp_rank": _integer(fields.get("dp_rank")),
                    "overlap_blocks": _integer(fields.get("overlap_blocks")),
                    "total_blocks": _integer(fields.get("total_blocks")),
                    "stage": "aggregate",
                    "lower_prefix_selected": False,
                }
            )
            candidates[decision_id] = pending.pop(request_id, [])
        elif event.get("kind") == "routing_dispatch" and request_id:
            phase = _string(fields.get("phase"))
            if phase is not None:
                dispatches[request_id].append((_integer(event.get("timestamp_ns")), phase.lower()))
    decisions.sort(key=lambda row: row["timestamp_ns"] or 0)
    traces_by_request = {row["request_id"]: row for row in traces if row["request_id"]}
    for decision in decisions:
        trace = traces_by_request.get(decision["dynamo_request_id"])
        dispatch = next(
            (
                phase
                for timestamp, phase in dispatches.get(decision["dynamo_request_id"], [])
                if timestamp is not None
                and decision["timestamp_ns"] is not None
                and timestamp >= decision["timestamp_ns"]
            ),
            None,
        )
        if dispatch in {"prefill", "decode"}:
            decision["stage"] = dispatch
        elif trace and trace["decode_worker_id"]:
            if decision["worker_id"] == trace["prefill_worker_id"]:
                decision["stage"] = "prefill"
            elif decision["worker_id"] == trace["decode_worker_id"]:
                decision["stage"] = "decode"
            else:
                decision["stage"] = "unknown"
        options = candidates.get(decision["decision_id"], [])
        selected = next((row for row in options if row.worker_id == decision["worker_id"]), None)
        if selected is not None and decision["stage"] != "decode":
            selected_prefix = selected.values.get("effective_cached_blocks")
            decision["lower_prefix_selected"] = selected_prefix is not None and any(
                (row.values.get("effective_cached_blocks") or 0) > selected_prefix for row in options
            )
    return decisions, candidates


def _metric_series(root: Path) -> dict[tuple[str, str], list[tuple[int, float]]]:
    path = _tachometer_parquet(root)
    if path is None:
        return {}
    import pyarrow.parquet as pq

    anchor_ns = _tachometer_anchor_ns(root)
    if anchor_ns is None:
        logger.warning("ruter view: Tachometer timestamp anchor unavailable; worker snapshots will be empty")
        return {}
    wanted = {_ROUTER_PREFILL, _ROUTER_TTFT, _ROUTER_ITL, _ENGINE_RUNNING, _ENGINE_QUEUED, _ENGINE_KV_USED}
    series: dict[tuple[str, str], list[tuple[int, float]]] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=131_072, columns=["scraper_endpoint", "metric_name", "metric_value", "time_since_start"]
    ):
        columns = batch.to_pydict()
        for endpoint, metric_name, value, elapsed in zip(
            columns["scraper_endpoint"],
            columns["metric_name"],
            columns["metric_value"],
            columns["time_since_start"],
            strict=True,
        ):
            if not isinstance(metric_name, str) or not isinstance(endpoint, str) or value is None or elapsed is None:
                continue
            metric = metric_name.split("{", 1)[0]
            if metric not in wanted:
                continue
            if endpoint == "router":
                match = _WORKER_ID_LABEL.search(metric_name)
                if match is None:
                    continue
                subject = f"router:{match['worker_id']}"
            else:
                subject = f"engine:{endpoint}"
            series.setdefault((subject, metric), []).append(
                (anchor_ns + int(float(elapsed) * 1_000_000_000), float(value))
            )
    for points in series.values():
        points.sort()
    return series


def _snapshots(
    decisions: list[dict[str, Any]],
    candidates: dict[str, list[_Candidate]],
    worker_events: list[dict[str, Any]],
    series: dict[tuple[str, str], list[tuple[int, float]]],
) -> dict[tuple[str, str], dict[str, float | None]]:
    worker_endpoints: dict[str, str] = {}
    for event in worker_events:
        fields = event.get("fields") or {}
        worker_id = _canonical_worker_id(fields.get("lease_id") or fields.get("instance_id"))
        worker_index = _integer(event.get("worker_index"))
        worker_role = _string(event.get("worker_role"))
        if worker_id is not None and worker_index is not None and worker_role is not None:
            worker_endpoints.setdefault(worker_id, f"engine:worker_{worker_role}_{worker_index}_0")
    snapshots = {}
    for decision in decisions:
        decision_id = decision["decision_id"]
        timestamp = decision["timestamp_ns"]
        if timestamp is None:
            continue
        options = candidates.get(decision_id, [])
        for candidate in options:
            router_subject = f"router:{candidate.worker_id}"
            engine_subject = worker_endpoints.get(candidate.worker_id)
            prefill = _sample_before(series.get((router_subject, _ROUTER_PREFILL)), timestamp)
            last_ttft = _sample_before(series.get((router_subject, _ROUTER_TTFT)), timestamp)
            last_itl = _sample_before(series.get((router_subject, _ROUTER_ITL)), timestamp)
            running = _sample_before(
                series.get((engine_subject, _ENGINE_RUNNING)) if engine_subject else None, timestamp
            )
            queued = _sample_before(series.get((engine_subject, _ENGINE_QUEUED)) if engine_subject else None, timestamp)
            kv_used = _sample_before(
                series.get((engine_subject, _ENGINE_KV_USED)) if engine_subject else None, timestamp
            )
            snapshots[(decision_id, candidate.worker_id)] = {
                "routerSampleAgeMs": _max_age_ms(timestamp, prefill, last_ttft, last_itl),
                "activePrefillTokens": prefill[1] if prefill else None,
                "lastTtftMs": last_ttft[1] * 1_000 if last_ttft else None,
                "lastItlMs": last_itl[1] * 1_000 if last_itl else None,
                "engineSampleAgeMs": _max_age_ms(timestamp, running, queued, kv_used),
                "runningReqs": running[1] if running else None,
                "queuedReqs": queued[1] if queued else None,
                "gpuCacheUsageFraction": kv_used[1] if kv_used else None,
            }
    return snapshots


def _tachometer_parquet(root: Path) -> Path | None:
    paths = sorted(path for path in root.rglob("final.parquet") if "tachometer" in path.parts)
    return paths[0] if paths else None


def _tachometer_anchor_ns(root: Path) -> int | None:
    preferred = root / "logs" / "tachometer.log"
    paths = [preferred] if preferred.is_file() else sorted(root.rglob("*tachometer*.log"))
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                timestamp = _timestamp_ns(line)
                if timestamp is not None:
                    return timestamp
    return None


def _router_settings(root: Path) -> dict[str, Any] | None:
    path = root / "logs" / "router.log"
    if not path.is_file():
        return None
    names = (
        "router_mode",
        "overlap_score_credit",
        "overlap_score_credit_decay",
        "prefill_load_scale",
        "decode_active_request_weight",
        "router_temperature",
    )
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            try:
                logged = json.loads(raw)
            except json.JSONDecodeError:
                logged = None
            line = logged.get("message") if isinstance(logged, dict) else raw
            if not isinstance(line, str) or "CONFIG_DUMP: " not in line:
                continue
            try:
                config = json.loads(line.split("CONFIG_DUMP: ", 1)[1]).get("config", {})
            except json.JSONDecodeError:
                continue
            return {name: config.get(name) for name in names}
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample_before(points: list[tuple[int, float]] | None, timestamp_ns: int) -> tuple[int, float] | None:
    if not points:
        return None
    index = bisect_right(points, (timestamp_ns, float("inf"))) - 1
    return points[index] if index >= 0 else None


def _max_age_ms(timestamp_ns: int, *points: tuple[int, float] | None) -> float | None:
    ages = [(timestamp_ns - point[0]) / 1_000_000 for point in points if point is not None]
    return max(ages) if ages else None


def _empty_snapshot() -> dict[str, None]:
    return {
        "routerSampleAgeMs": None,
        "activePrefillTokens": None,
        "lastTtftMs": None,
        "lastItlMs": None,
        "engineSampleAgeMs": None,
        "runningReqs": None,
        "queuedReqs": None,
        "gpuCacheUsageFraction": None,
    }


def _formula_fields() -> tuple[str, ...]:
    return (
        "cost_blocks",
        "effective_cached_blocks",
        "prefill_load_scale",
        "adjusted_prefill_blocks",
        "raw_prefill_blocks",
        "overlap_credit_blocks",
        "overlap_credit_decay",
        "decode_blocks",
        "active_request_cost_blocks",
    )


def _metric_value(metrics: dict[str, Any], name: str) -> int | float | None:
    value = (metrics.get(name) or {}).get("value")
    return _number(value)


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _ns(value: Any) -> int | None:
    number = _integer(value)
    return number * 1_000_000 if number is not None else None


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _canonical_worker_id(value: Any) -> str | None:
    worker_id = _string(value)
    if worker_id is None:
        return None
    return worker_id.removeprefix("worker_")


def _worker_aliases(
    workers: set[str],
    traces: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    candidates: dict[str, list[_Candidate]],
) -> dict[str, str]:
    """Give P/D workers short, role-safe aliases while preserving agg aliases."""
    prefill = {row["prefill_worker_id"] for row in traces if row["prefill_worker_id"]}
    decode = {row["decode_worker_id"] for row in traces if row["decode_worker_id"]}
    for decision in decisions:
        candidates_for_decision = candidates.get(decision["decision_id"], [])
        if decision["stage"] == "prefill":
            prefill.update(candidate.worker_id for candidate in candidates_for_decision)
        elif decision["stage"] == "decode":
            decode.update(candidate.worker_id for candidate in candidates_for_decision)
    if not decode:
        return {worker: _alphabetic_alias(index) for index, worker in enumerate(sorted(workers))}

    aliases: dict[str, str] = {}
    for index, worker in enumerate(sorted(prefill)):
        aliases[worker] = f"P-{_alphabetic_alias(index)}"
    for index, worker in enumerate(sorted(decode)):
        aliases[worker] = f"D-{_alphabetic_alias(index)}"
    unknown = sorted(workers - prefill - decode)
    for index, worker in enumerate(unknown):
        aliases[worker] = f"W-{_alphabetic_alias(index)}"
    return aliases


def _alphabetic_alias(index: int) -> str:
    alias = ""
    while True:
        alias = chr(ord("A") + index % 26) + alias
        if index < 26:
            return alias
        index = index // 26 - 1
