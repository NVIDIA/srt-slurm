#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stamp GPU power measurement windows around an AIPerf run in a ``custom`` benchmark.

srtctl's power telemetry (``telemetry.enabled``) samples GPU watts for the whole job and
publishes them per measured concurrency only when the benchmark writes one window file per
``benchmark.concurrencies`` entry into ``$SRT_MEASUREMENT_WINDOW_DIR`` (see
``docs/power-telemetry.md``). Built-in sa-bench writes them itself; a recipe that runs
``aiperf profile`` directly brackets it with this script:

    python3 /benchmarks/aiperf_power_window.py running   --concurrency 64 --result-dir /logs/agentic/concurrency_64
    aiperf profile ... --concurrency 64 --artifact-dir /logs/agentic/concurrency_64
    python3 /benchmarks/aiperf_power_window.py completed --concurrency 64 --result-dir /logs/agentic/concurrency_64

``running`` opens the window before load starts. ``completed`` reads AIPerf's formal profiling
boundary (``start_time``/``end_time`` in ``profile_export_aiperf.json``, which excludes warmup)
and writes both the result file and the completed window with identical timings, as the
validator requires. The file names follow InferenceX's AgentX power adapter
(``agentic_power_concurrency_<N>.json``), so its consumers read these runs unchanged. A failed
run is never stamped ``completed``: its window stays ``running`` and srtctl marks it
``interrupted``.

Without ``SRT_MEASUREMENT_WINDOW_DIR`` (telemetry disabled) both commands are no-ops. Errors are
reported on stderr and exit 0 so power bookkeeping never fails the benchmark; pass ``--require``
to make them fatal. Standard library only: it runs in the benchmark container.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

WINDOW_DIR_ENV = "SRT_MEASUREMENT_WINDOW_DIR"
WINDOWS_DIRNAME = "windows"
SCHEMA_VERSION = 1
BENCHMARK_TYPE = "custom"
CLOCK_SOURCE = "head_node_unix_clock"
AGGREGATE_NAME = "profile_export_aiperf.json"
RECORDS_NAME = "profile_export.jsonl"


class WindowError(Exception):
    """The window contract cannot be written."""


def _atomic_write_json(path: Path, payload: dict) -> None:
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _paths(concurrency: int, result_dir: Path) -> tuple[Path, Path, str] | None:
    """Return (result file, window file, result path relative to the log root), or None when disabled.

    The window directory is ``<log root>/<storage_subdir>/windows``; srtctl resolves each
    window's ``result_path`` against that log root, so the result must live below it.
    """
    raw = os.environ.get(WINDOW_DIR_ENV, "")
    if not raw:
        return None
    window_dir = Path(raw)
    if window_dir.name != WINDOWS_DIRNAME or not window_dir.is_absolute():
        raise WindowError(f"{WINDOW_DIR_ENV}={raw!r} is not an absolute .../{WINDOWS_DIRNAME} directory")
    window_dir.mkdir(parents=True, exist_ok=True)
    log_root = window_dir.parent.parent.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        relative_dir = result_dir.resolve().relative_to(log_root)
    except ValueError as exc:
        raise WindowError(f"result dir {result_dir} is not below the log root {log_root}") from exc
    stem = f"agentic_power_concurrency_{concurrency}"
    return result_dir / f"{stem}.json", window_dir / f"{stem}.json", (relative_dir / f"{stem}.json").as_posix()


def _window(result_path: str, concurrency: int, status: str, start: float, end: float | None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_type": BENCHMARK_TYPE,
        "result_path": result_path,
        "concurrency": concurrency,
        "benchmark_start_time_unix": start,
        "benchmark_end_time_unix": end,
        "duration": None if end is None else end - start,
        "clock_source": CLOCK_SOURCE,
        "status": status,
        "reason": None,
    }


def _artifact_dir(result_dir: Path) -> Path:
    """AIPerf writes into ``--artifact-dir`` directly, or into one subdirectory of it."""
    if (result_dir / AGGREGATE_NAME).is_file():
        return result_dir
    for child in sorted(p for p in result_dir.iterdir() if p.is_dir()):
        if (child / AGGREGATE_NAME).is_file():
            return child
    raise WindowError(f"no {AGGREGATE_NAME} under {result_dir}")


def _unix(value) -> float:
    """Unix seconds from AIPerf's epoch number or ISO timestamp; a naive one is this host's local time."""
    if isinstance(value, bool):
        raise WindowError(f"invalid profile timestamp {value!r}")
    if isinstance(value, int | float):
        seconds = float(value)
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise WindowError(f"invalid profile timestamp {value!r}") from exc
        seconds = parsed.astimezone(timezone.utc).timestamp()
    else:
        raise WindowError(f"invalid profile timestamp {value!r}")
    if not math.isfinite(seconds):
        raise WindowError(f"invalid profile timestamp {value!r}")
    return seconds


def _metric(record: dict, key: str):
    value = record.get("metrics", {}).get(key)
    return value.get("value") if isinstance(value, dict) else value


def _request_totals(records_path: Path) -> dict[str, int]:
    """Count successful profiling-phase requests and their tokens; warmup and errors are excluded."""
    completed = input_tokens = output_tokens = 0
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            phase = record.get("metadata", {}).get("benchmark_phase")
            if record.get("error") or (phase is not None and phase != "profiling"):
                continue
            completed += 1
            input_tokens += int(_metric(record, "input_sequence_length") or 0)
            output_tokens += int(_metric(record, "output_sequence_length") or 0)
    return {"completed": completed, "total_input_tokens": input_tokens, "total_output_tokens": output_tokens}


def mark_running(concurrency: int, result_dir: Path) -> None:
    paths = _paths(concurrency, result_dir)
    if paths is None:
        return
    _, window_file, result_path = paths
    _atomic_write_json(window_file, _window(result_path, concurrency, "running", time.time(), None))


def mark_completed(concurrency: int, result_dir: Path) -> None:
    paths = _paths(concurrency, result_dir)
    if paths is None:
        return
    result_file, window_file, result_path = paths
    artifacts = _artifact_dir(result_dir)
    aggregate = json.loads((artifacts / AGGREGATE_NAME).read_text(encoding="utf-8"))
    if not isinstance(aggregate, dict):
        raise WindowError(f"{AGGREGATE_NAME} is not a JSON object")
    start, end = _unix(aggregate.get("start_time")), _unix(aggregate.get("end_time"))
    if end <= start:
        raise WindowError(f"profile window end {end} is not after start {start}")
    totals = _request_totals(artifacts / RECORDS_NAME) if (artifacts / RECORDS_NAME).is_file() else {}
    if totals and totals["completed"] <= 0:
        raise WindowError("no successful profiling requests in the window")
    window = _window(result_path, concurrency, "completed", start, end)
    boundary = {key: window[key] for key in ("benchmark_start_time_unix", "benchmark_end_time_unix", "duration")}
    _atomic_write_json(result_file, {"max_concurrency": concurrency, **boundary, **totals})
    _atomic_write_json(window_file, window)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("state", choices=("running", "completed"))
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--result-dir", type=Path, required=True, help="the AIPerf --artifact-dir, below /logs")
    parser.add_argument("--require", action="store_true", help="exit 1 when the window cannot be written")
    args = parser.parse_args(argv)
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    try:
        (mark_running if args.state == "running" else mark_completed)(args.concurrency, args.result_dir.absolute())
    except (WindowError, OSError, ValueError, TypeError) as exc:
        print(f"[aiperf_power_window] {args.state} window not written: {exc}", file=sys.stderr)
        return 1 if args.require else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
