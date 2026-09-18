# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit dashboard generation and JSON queries; no workflow hooks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .query import KINDS, TraceDataset


def add_commands(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="dashboard_command", required=True)
    build = commands.add_parser("build", help="Build an offline HTML from existing run artifacts")
    build.add_argument("logs", type=Path, help="Run directory or its logs/ directory")
    build.add_argument(
        "--output", "-o", type=Path, required=True, help="New or previously generated dashboard directory"
    )
    build.add_argument("--client", type=Path, help="Select one AIPerf profile_export.jsonl or AgentPerf requests.jsonl")
    build.add_argument(
        "--metrics", type=Path, help="One raw Tachometer capture leaf or file (default: logs/tachometer/local)"
    )
    build.add_argument(
        "--nsys-sqlite", type=Path, dest="sqlites", help="Existing Nsight SQLite exports; does not enable profiling"
    )
    build.add_argument(
        "--no-otel", action="store_false", dest="otel", help="Skip OTel import and request lifecycle breakdowns"
    )
    build.add_argument("--job", help="Display identifier (default: parent directory of logs)")
    build.add_argument(
        "--phase", default="profiling", help="Client benchmark_phase to include; 'all' includes warmup explicitly"
    )
    build.add_argument(
        "--iteration-timezone", help="IANA timezone of timezone-free iteration logs, e.g. America/Los_Angeles"
    )
    build.add_argument(
        "--max-profile-events",
        type=int,
        default=250_000,
        help="Maximum imported NVTX events per report; truncation is reported",
    )
    query = commands.add_parser("query", help="Query the generated dataset as JSON without a browser")
    query.add_argument("dataset", type=Path, help="Dashboard directory or trace-data.json.gz")
    query.add_argument("kind", choices=KINDS, nargs="?", default="summary")
    query.add_argument("--from", type=float, dest="start")
    query.add_argument("--to", type=float, dest="end")
    query.add_argument("--request", dest="request_id")
    for field in ("session", "agent", "worker", "name", "search"):
        query.add_argument("--" + field)
    query.add_argument("--rank", type=int)
    query.add_argument("--profile", type=int)
    query.add_argument("--min-ttft-ms", type=float, default=0)
    query.add_argument("--offset", type=int, default=0)
    query.add_argument("--limit", type=int, default=100)
    query.add_argument("--points", action="store_true", help="Include up to 1000 raw points per metric series")


def run(args: argparse.Namespace) -> int:
    try:
        if args.dashboard_command == "build":
            from .build import build_dashboard

            if args.max_profile_events <= 0:
                raise ValueError("--max-profile-events must be positive")
            result = build_dashboard(
                args.logs,
                args.output,
                client=args.client,
                metrics=args.metrics,
                sqlites=args.sqlites,
                otel=args.otel,
                job=args.job,
                phase=args.phase,
                iteration_timezone=args.iteration_timezone,
                max_profile_events=args.max_profile_events,
            )
        else:
            options = {
                k: getattr(args, k)
                for k in (
                    "start",
                    "end",
                    "request_id",
                    "session",
                    "agent",
                    "worker",
                    "name",
                    "search",
                    "rank",
                    "profile",
                    "min_ttft_ms",
                    "offset",
                    "limit",
                    "points",
                )
            }
            result = TraceDataset.from_path(args.dataset).query(args.kind, **options)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_commands(parser)
    raise SystemExit(run(parser.parse_args()))
