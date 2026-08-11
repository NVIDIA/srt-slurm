# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned wire format shared by every dcgm-power artifact writer and reader."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, TypeGuard

SCHEMA_VERSION = 1

PRODUCER = "srt-slurm.dcgm-power"
POWER_METRIC = "DCGM_FI_DEV_POWER_USAGE"
POWER_UNIT = "W"
POWER_SCOPE = "gpu_device_board_as_reported_by_dcgm"
CLOCK_SOURCE = "head_node_unix_clock"

MANIFEST_FILENAME = "manifest.json"
SAMPLES_FILENAME = "samples.csv"
WINDOWS_DIRNAME = "windows"

SAMPLES_HEADER = (
    "schema_version",
    "timestamp_unix",
    "scrape_seq",
    "hostname",
    "gpu_index",
    "gpu_uuid",
    "power_w",
)

MAX_SAMPLE_GAP_SECONDS = 3.0
COLLECT_CYCLE_TIMEOUT_GRACE_SECONDS = 1.0

BENCHMARK_TYPE_SA_BENCH = "sa-bench"

CONTAINER_LOG_DIR = "/logs"
MEASUREMENT_WINDOW_DIR_ENV = "SRT_MEASUREMENT_WINDOW_DIR"


class Reason:
    """Stable machine-readable reason codes recorded in artifacts."""

    EXPORTER_STARTUP_TIMEOUT = "exporter_startup_timeout"
    EXPORTER_LAUNCH_FAILED = "exporter_launch_failed"
    EXPORTER_EXITED = "exporter_exited"
    ENDPOINT_TIMEOUT = "endpoint_timeout"
    ENDPOINT_HTTP_ERROR = "endpoint_http_error"
    ENDPOINT_PARSE_ERROR = "endpoint_parse_error"
    ENDPOINT_RESOLUTION_FAILED = "endpoint_resolution_failed"
    POWER_METRIC_MISSING = "power_metric_missing"
    DUPLICATE_POWER_METRIC = "duplicate_power_metric"
    SAMPLES_CSV_MISSING = "samples_csv_missing"
    SAMPLES_CSV_HEADER_MISMATCH = "samples_csv_header_mismatch"
    SAMPLES_CSV_MALFORMED = "samples_csv_malformed"
    DUPLICATE_SAMPLE_ROW = "duplicate_sample_row"
    GPU_INDEX_MISSING = "gpu_index_missing"
    GPU_UUID_MISSING = "gpu_uuid_missing"
    INVALID_POWER_VALUE = "invalid_power_value"
    UNEXPECTED_DEVICE = "unexpected_device"
    EXPECTED_DEVICE_MISSING = "expected_device_missing"
    GPU_UUID_CHANGED = "gpu_uuid_changed"
    MIG_INSTANCE_UNSUPPORTED = "mig_instance_unsupported"
    TIMESTAMP_NON_MONOTONIC = "timestamp_non_monotonic"
    CONFLICTING_WORKER_ROLES = "conflicting_worker_roles"
    CONFLICTING_HET_GROUPS = "conflicting_het_groups"
    COLLECTOR_EXCEPTION = "collector_exception"
    COLLECTOR_INTERRUPTED = "collector_interrupted"
    COLLECTOR_JOIN_TIMEOUT = "collector_join_timeout"
    BENCHMARK_CHILD_REAP_TIMEOUT = "benchmark_child_reap_timeout"
    MEASUREMENT_WINDOW_MISSING = "measurement_window_missing"
    MEASUREMENT_WINDOW_UNEXPECTED = "measurement_window_unexpected"
    MEASUREMENT_WINDOW_DUPLICATE = "measurement_window_duplicate"
    MEASUREMENT_WINDOW_MALFORMED = "measurement_window_malformed"
    MEASUREMENT_WINDOW_ARTIFACT_PATH_INVALID = "measurement_window_artifact_path_invalid"
    MEASUREMENT_WINDOW_INCOMPLETE = "measurement_window_incomplete"
    MEASUREMENT_WINDOW_RESULT_MISSING = "measurement_window_result_missing"
    MEASUREMENT_WINDOW_RESULT_MISMATCH = "measurement_window_result_mismatch"
    MEASUREMENT_WINDOW_RESULT_PATH_INVALID = "measurement_window_result_path_invalid"
    MEASUREMENT_WINDOW_CLOCK_MISMATCH = "measurement_window_clock_mismatch"
    MEASUREMENT_WINDOW_NOT_BRACKETED = "measurement_window_not_bracketed"
    SAMPLE_GAP_EXCEEDED = "sample_gap_exceeded"


FATAL_LIFECYCLE_REASONS = (
    Reason.EXPORTER_EXITED,
    Reason.COLLECTOR_EXCEPTION,
    Reason.COLLECTOR_INTERRUPTED,
    Reason.COLLECTOR_JOIN_TIMEOUT,
    Reason.BENCHMARK_CHILD_REAP_TIMEOUT,
)


OPERATIONAL_FAILURE_REASONS = (
    Reason.BENCHMARK_CHILD_REAP_TIMEOUT,
    Reason.COLLECTOR_JOIN_TIMEOUT,
)

STARTUP_FAILURE_REASONS = (
    Reason.EXPORTER_STARTUP_TIMEOUT,
    Reason.EXPORTER_LAUNCH_FAILED,
    Reason.ENDPOINT_RESOLUTION_FAILED,
)


def is_safe_relative_subpath(value: str) -> bool:
    """Whether ``value`` is a relative POSIX path that stays below its root."""
    if not value or value.startswith(("/", "~")):
        return False
    parts = PurePosixPath(value).parts
    return bool(parts) and not any(part in ("..", "") for part in parts)


def is_finite_number(value: Any) -> TypeGuard[int | float]:
    """Whether ``value`` is a finite real number; bools are not numbers here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def dedupe(values: list[str]) -> tuple[str, ...]:
    """First-seen-order deduplication for reason-code accumulation."""
    return tuple(dict.fromkeys(values))


def atomic_write_json(path: Path, payload: Any) -> None:
    """Replace ``path`` with serialized JSON and leave no partial file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", text=True)
    try:
        try:
            handle = os.fdopen(fd, "w", encoding="utf-8", closefd=False)
            with handle:
                handle.write(json.dumps(payload, indent=2, sort_keys=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(fd)
        os.replace(temp_path, path)
    except BaseException:
        Path(temp_path).unlink(missing_ok=True)
        raise
