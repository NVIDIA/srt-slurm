#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate benchmark-rollup.json from AgentX/AIPerf results."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _candidate_files(log_dir: Path) -> list[Path]:
    roots = [
        log_dir / "agentx" / "aiperf_artifacts",
        log_dir / "aiperf_artifacts",
        log_dir / "artifacts",
    ]
    names = [
        "profile_export_aiperf.json",
        "profile_export_aiperf_aggregate.json",
    ]
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            direct = root / name
            if direct.is_file():
                out.append(direct)
            out.extend(root.glob(f"*/{name}"))
            out.extend(root.glob(f"*/*/{name}"))
    return out


def _metric_avg(data: dict[str, Any], name: str) -> float | None:
    metric = data.get(name)
    if not isinstance(metric, dict):
        return None
    value = metric.get("avg")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def main(log_dir: Path) -> None:
    candidates = _candidate_files(log_dir)
    if not candidates:
        print("No AgentX/AIPerf results found", file=sys.stderr)
        return

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(latest.read_text())
    except json.JSONDecodeError as e:
        print(f"Failed to parse {latest}: {e}", file=sys.stderr)
        return

    completed = _metric_avg(data, "completed_request_count")
    request_count = _metric_avg(data, "request_count")
    error_count = _metric_avg(data, "error_request_count") or 0.0
    if completed is None and request_count is not None:
        completed = request_count + error_count

    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    rollup = {
        "benchmark_type": "agentx",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "artifact": str(latest.relative_to(log_dir)) if latest.is_relative_to(log_dir) else str(latest),
        "summary": {
            "scenario": metadata.get("scenario"),
            "submission_valid": metadata.get("submission_valid"),
            "request_count": request_count,
            "completed_request_count": completed,
            "error_request_count": error_count,
            "error_rate": (error_count / completed) if completed else None,
        },
        "data": data,
    }

    output_path = log_dir / "benchmark-rollup.json"
    output_path.write_text(json.dumps(rollup, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/logs"))
