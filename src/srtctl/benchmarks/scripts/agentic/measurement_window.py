#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish an AgentX power-measurement window from AIPerf profile records.

This helper intentionally runs after the pinned InferenceX harness. It does not
change replay behavior: it reads the same successful profiling rows used by the
AgentX aggregate, derives their exact wall-clock extent, and atomically stamps
that extent into both the aggregate result and the formal power-window record.

Standalone by design -- the benchmark directory is mounted into the client
container and this module must not import srtctl.
"""

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
CLOCK_SOURCE = "head_node_unix_clock"
WINDOW_DIR_ENV = "SRT_MEASUREMENT_WINDOW_DIR"
CONTAINER_LOG_DIR = "/logs"


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
    try:
        try:
            handle = os.fdopen(fd, "w", encoding="utf-8", closefd=False)
            with handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(fd)
        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _profile_path(result_dir):
    root = Path(result_dir) / "aiperf_artifacts"
    direct = root / "profile_export.jsonl"
    if direct.is_file():
        return direct
    if root.is_dir():
        for child in sorted(root.iterdir()):
            candidate = child / "profile_export.jsonl"
            if child.is_dir() and candidate.is_file():
                return candidate
    raise ValueError(f"profile_export.jsonl not found below {root}")


def _profile_boundary(profile_path):
    starts_ns = []
    ends_ns = []
    with Path(profile_path).open(encoding="utf-8") as profile:
        for line_number, line in enumerate(profile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {profile_path}:{line_number}") from error
            metadata = record.get("metadata", {})
            phase = metadata.get("benchmark_phase")
            if record.get("error") or (phase is not None and phase != "profiling"):
                continue
            start_ns = metadata.get("request_start_ns")
            end_ns = metadata.get("request_end_ns")
            if (
                isinstance(start_ns, bool)
                or not isinstance(start_ns, int | float)
                or isinstance(end_ns, bool)
                or not isinstance(end_ns, int | float)
                or not math.isfinite(start_ns)
                or not math.isfinite(end_ns)
            ):
                continue
            starts_ns.append(int(start_ns))
            ends_ns.append(int(end_ns))

    if not starts_ns or not ends_ns:
        raise ValueError(f"no successful profiling timestamps in {profile_path}")
    start_ns = min(starts_ns)
    end_ns = max(ends_ns)
    if end_ns <= start_ns:
        raise ValueError(f"non-positive profiling duration in {profile_path}")
    start_unix = start_ns / 1_000_000_000
    end_unix = end_ns / 1_000_000_000
    return start_unix, end_unix, (end_ns - start_ns) / 1_000_000_000


def publish_measurement_window(
    *,
    result_dir,
    result_file,
    concurrency,
    benchmark_type="agentic",
    window_dir=None,
    log_root=CONTAINER_LOG_DIR,
):
    """Stamp and publish one AgentX result/window pair.

    Returns ``None`` when telemetry was not requested. All requested telemetry
    failures are fatal so ``telemetry.required`` can never publish an
    unattributed power artifact.
    """
    window_dir = os.environ.get(WINDOW_DIR_ENV) if window_dir is None else window_dir
    if not window_dir:
        return None
    window_dir = Path(window_dir)
    if not window_dir.is_dir():
        raise ValueError(f"measurement-window directory does not exist: {window_dir}")

    result_file = Path(result_file).resolve()
    log_root = Path(log_root).resolve()
    try:
        result_path = result_file.relative_to(log_root).as_posix()
    except ValueError as error:
        raise ValueError(f"result file must be below {log_root}: {result_file}") from error
    if not result_file.is_file():
        raise ValueError(f"AgentX aggregate result does not exist: {result_file}")

    start_unix, end_unix, duration = _profile_boundary(_profile_path(result_dir))
    with result_file.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"AgentX aggregate result is not a JSON object: {result_file}")
    result.update(
        {
            "benchmark_start_time_unix": start_unix,
            "benchmark_end_time_unix": end_unix,
            "duration": duration,
        }
    )
    _atomic_write_json(result_file, result)

    window_path = window_dir / f"{result_file.stem}.json"
    _atomic_write_json(
        window_path,
        {
            "schema_version": SCHEMA_VERSION,
            "benchmark_type": benchmark_type,
            "result_path": result_path,
            "concurrency": concurrency,
            "benchmark_start_time_unix": start_unix,
            "benchmark_end_time_unix": end_unix,
            "duration": duration,
            "clock_source": CLOCK_SOURCE,
            "status": "completed",
            "reason": None,
        },
    )
    return window_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--benchmark-type", default="agentic", choices=("agentic", "agentx"))
    parser.add_argument("--window-dir")
    parser.add_argument("--log-root", default=CONTAINER_LOG_DIR)
    args = parser.parse_args()
    path = publish_measurement_window(
        result_dir=args.result_dir,
        result_file=args.result_file,
        concurrency=args.concurrency,
        benchmark_type=args.benchmark_type,
        window_dir=args.window_dir,
        log_root=args.log_root,
    )
    if path is not None:
        print(f"Published AgentX power-measurement window: {path}")


if __name__ == "__main__":
    main()
