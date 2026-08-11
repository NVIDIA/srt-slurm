# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measurement-window coverage audit.

Every regular ``windows/*.json`` file is scanned, so a stale, duplicate,
malformed, or unsafe artifact cannot be ignored. This is a structural audit
only: it never averages power or computes energy.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from srtctl.core.power.contract import (
    CLOCK_SOURCE,
    MAX_SAMPLE_GAP_SECONDS,
    SCHEMA_VERSION,
    WINDOWS_DIRNAME,
    Reason,
    atomic_write_json,
    dedupe,
    is_finite_number,
    is_safe_relative_subpath,
)
from srtctl.core.power.manifest import ArtifactError, ExpectedWindow, WindowValidation
from srtctl.core.power.samples import ObservedDevice
from srtctl.core.power.topology import DeviceKey

logger = logging.getLogger(__name__)

WINDOW_STATUS_RUNNING = "running"
WINDOW_STATUS_COMPLETED = "completed"
WINDOW_STATUS_FAILED = "failed"
WINDOW_STATUS_INTERRUPTED = "interrupted"

_ALLOWED_STATUSES = (WINDOW_STATUS_RUNNING, WINDOW_STATUS_COMPLETED, WINDOW_STATUS_FAILED, WINDOW_STATUS_INTERRUPTED)

_CLOCK_TOLERANCE_SECONDS = 0.5
_CLOCK_TOLERANCE_FRACTION = 0.01


@dataclass(frozen=True)
class _ParsedWindow:
    relative_path: str
    benchmark_type: str
    concurrency: int
    result_path: str
    status: str
    start_unix: float
    end_unix: float | None
    duration: float | None


def validate_expected_windows(
    *,
    power_dir: Path,
    result_root: Path,
    expected_windows: Sequence[ExpectedWindow],
    expected_device_keys: set[DeviceKey],
    observed_devices: Sequence[ObservedDevice],
    artifact_errors: list[ArtifactError],
) -> list[WindowValidation]:
    """Emit exactly one validation row per expected window.

    A missing expected window must never pass vacuously, and every unexpected,
    malformed, duplicate, or unsafe file lands in ``artifact_errors`` so the
    expected key set can be compared against the valid observed key set.
    """
    parsed, duplicates = _scan(power_dir / WINDOWS_DIRNAME, result_root, artifact_errors)
    expected_keys = {window.key for window in expected_windows}

    for key, window in sorted(parsed.items()):
        if key not in expected_keys:
            artifact_errors.append(
                ArtifactError(path=window.relative_path, reason_codes=(Reason.MEASUREMENT_WINDOW_UNEXPECTED,))
            )

    return [
        _validate_one(
            expected=expected,
            window=parsed.get(expected.key),
            duplicated=expected.key in duplicates,
            result_root=result_root,
            expected_device_keys=expected_device_keys,
            observed_devices=observed_devices,
        )
        for expected in expected_windows
    ]


def convert_running_windows(windows_dir: Path, *, reason: str) -> int:
    """Turn every still-``running`` window into ``interrupted``.

    Only safe after the benchmark child was proved reaped: a surviving child
    could otherwise overwrite the file with ``completed``. The orchestrator
    never invents a request-completion boundary, so end and duration stay null.
    """
    if not windows_dir.is_dir():
        return 0
    if not _stays_below(windows_dir.parent, windows_dir.name):
        return 0

    try:
        paths = sorted(windows_dir.glob("*.json"))
    except OSError:
        return 0

    converted = 0
    for path in paths:
        if path.is_symlink() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("status") != WINDOW_STATUS_RUNNING:
            continue
        payload["status"] = WINDOW_STATUS_INTERRUPTED
        payload["benchmark_end_time_unix"] = None
        payload["duration"] = None
        payload["reason"] = reason
        atomic_write_json(path, payload)
        converted += 1
    return converted


def _scan(
    windows_dir: Path,
    result_root: Path,
    artifact_errors: list[ArtifactError],
) -> tuple[dict[tuple[str, int], _ParsedWindow], set[tuple[str, int]]]:
    """Parse every regular window file, recording anything unusable."""
    parsed: dict[tuple[str, int], _ParsedWindow] = {}
    duplicates: set[tuple[str, int]] = set()
    if not windows_dir.is_dir():
        return parsed, duplicates

    # NOTE: rejecting only child symlinks still lets the whole directory be one.
    if not _stays_below(windows_dir.parent, WINDOWS_DIRNAME):
        artifact_errors.append(
            ArtifactError(path=WINDOWS_DIRNAME, reason_codes=(Reason.MEASUREMENT_WINDOW_ARTIFACT_PATH_INVALID,))
        )
        return parsed, duplicates

    try:
        paths = sorted(windows_dir.iterdir())
    except OSError:
        artifact_errors.append(
            ArtifactError(path=WINDOWS_DIRNAME, reason_codes=(Reason.MEASUREMENT_WINDOW_ARTIFACT_PATH_INVALID,))
        )
        return parsed, duplicates

    for path in paths:
        relative = f"{WINDOWS_DIRNAME}/{path.name}"
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            artifact_errors.append(
                ArtifactError(path=relative, reason_codes=(Reason.MEASUREMENT_WINDOW_ARTIFACT_PATH_INVALID,))
            )
            continue

        window, reasons = _parse(path, relative, result_root)
        if window is None:
            artifact_errors.append(ArtifactError(path=relative, reason_codes=reasons))
            continue

        key = (window.benchmark_type, window.concurrency)
        if key in duplicates:
            artifact_errors.append(ArtifactError(path=relative, reason_codes=(Reason.MEASUREMENT_WINDOW_DUPLICATE,)))
            continue

        previous = parsed.pop(key, None)
        if previous is not None:
            duplicates.add(key)
            artifact_errors.append(
                ArtifactError(path=previous.relative_path, reason_codes=(Reason.MEASUREMENT_WINDOW_DUPLICATE,))
            )
            artifact_errors.append(ArtifactError(path=relative, reason_codes=(Reason.MEASUREMENT_WINDOW_DUPLICATE,)))
            continue
        parsed[key] = window

    return parsed, duplicates


def _is_strict_int(value) -> bool:
    """``bool`` is an ``int`` subclass; JSON ``true`` must not key as concurrency 1."""
    return isinstance(value, int) and not isinstance(value, bool)


def _parse(path: Path, relative: str, result_root: Path) -> tuple[_ParsedWindow | None, tuple[str, ...]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None, (Reason.MEASUREMENT_WINDOW_MALFORMED,)
    if not isinstance(payload, dict):
        return None, (Reason.MEASUREMENT_WINDOW_MALFORMED,)

    status = payload.get("status")
    concurrency = payload.get("concurrency")
    benchmark_type = payload.get("benchmark_type")
    start = payload.get("benchmark_start_time_unix")
    if (
        not _is_strict_int(payload.get("schema_version"))
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("clock_source") != CLOCK_SOURCE
        or status not in _ALLOWED_STATUSES
        or not isinstance(benchmark_type, str)
        or not _is_strict_int(concurrency)
        or not is_finite_number(start)
    ):
        return None, (Reason.MEASUREMENT_WINDOW_MALFORMED,)

    result_path = payload.get("result_path")
    if not isinstance(result_path, str) or not is_safe_relative_subpath(result_path):
        return None, (Reason.MEASUREMENT_WINDOW_RESULT_PATH_INVALID,)
    if not _stays_below(result_root, result_path) or Path(result_path).stem != path.stem:
        return None, (Reason.MEASUREMENT_WINDOW_RESULT_PATH_INVALID,)

    end = payload.get("benchmark_end_time_unix")
    duration = payload.get("duration")
    if not _status_invariants_hold(status, end, duration, payload.get("reason")):
        return None, (Reason.MEASUREMENT_WINDOW_MALFORMED,)

    return (
        _ParsedWindow(
            relative_path=relative,
            benchmark_type=benchmark_type,
            concurrency=concurrency,
            result_path=result_path,
            status=status,
            start_unix=float(start),
            end_unix=float(end) if end is not None else None,
            duration=float(duration) if duration is not None else None,
        ),
        (),
    )


def _status_invariants_hold(status: str, end, duration, reason) -> bool:
    """End, duration, and reason nullability are all fixed by the status.

    ``running`` is written before anything can have gone wrong, so it carries no
    reason; ``interrupted`` is only ever produced by the orchestrator, which
    always records why.
    """
    if status == WINDOW_STATUS_RUNNING:
        return end is None and duration is None and reason is None
    if status == WINDOW_STATUS_INTERRUPTED:
        return end is None and duration is None and isinstance(reason, str) and bool(reason)
    if not is_finite_number(end) or not is_finite_number(duration) or duration <= 0:
        return False
    if status == WINDOW_STATUS_COMPLETED:
        return reason is None
    return isinstance(reason, str) and bool(reason)


def _validate_one(
    *,
    expected: ExpectedWindow,
    window: _ParsedWindow | None,
    duplicated: bool,
    result_root: Path,
    expected_device_keys: set[DeviceKey],
    observed_devices: Sequence[ObservedDevice],
) -> WindowValidation:
    if window is None:
        reasons = [Reason.MEASUREMENT_WINDOW_DUPLICATE] if duplicated else [Reason.MEASUREMENT_WINDOW_MISSING]
        return WindowValidation(
            benchmark_type=expected.benchmark_type,
            concurrency=expected.concurrency,
            window_file=None,
            power_coverage_valid=False,
            reason_codes=tuple(reasons),
        )

    reasons: list[str] = []
    if window.status != WINDOW_STATUS_COMPLETED:
        reasons.append(Reason.MEASUREMENT_WINDOW_INCOMPLETE)
    else:
        reasons.extend(_check_result(window, result_root))

    gaps: dict[str, float] = {}
    if not reasons:
        if window.end_unix is None:
            reasons.append(Reason.MEASUREMENT_WINDOW_MALFORMED)
        else:
            gaps, coverage_reasons = _check_coverage(
                window.start_unix, window.end_unix, expected_device_keys, observed_devices
            )
            reasons.extend(coverage_reasons)

    return WindowValidation(
        benchmark_type=expected.benchmark_type,
        concurrency=expected.concurrency,
        window_file=window.relative_path,
        power_coverage_valid=not reasons,
        reason_codes=dedupe(reasons),
        per_device_max_sample_gap_seconds=gaps,
    )


def _check_result(window: _ParsedWindow, result_root: Path) -> list[str]:
    """A completed window must match the result it points at, on both clocks."""
    result_file = result_root / window.result_path
    if not result_file.is_file():
        return [Reason.MEASUREMENT_WINDOW_RESULT_MISSING]
    try:
        result = json.loads(result_file.read_text())
    except (OSError, ValueError):
        return [Reason.MEASUREMENT_WINDOW_RESULT_MISSING]
    if not isinstance(result, dict):
        return [Reason.MEASUREMENT_WINDOW_RESULT_MISMATCH]

    if window.end_unix is None or window.duration is None:
        return [Reason.MEASUREMENT_WINDOW_MALFORMED]
    result_timings = (
        result.get("benchmark_start_time_unix"),
        result.get("benchmark_end_time_unix"),
        result.get("duration"),
    )
    if not all(is_finite_number(value) for value in result_timings) or result_timings != (
        window.start_unix,
        window.end_unix,
        window.duration,
    ):
        return [Reason.MEASUREMENT_WINDOW_RESULT_MISMATCH]

    wall = window.end_unix - window.start_unix
    if wall <= 0:
        return [Reason.MEASUREMENT_WINDOW_CLOCK_MISMATCH]
    tolerance = max(_CLOCK_TOLERANCE_SECONDS, _CLOCK_TOLERANCE_FRACTION * window.duration)
    if abs(wall - window.duration) > tolerance:
        return [Reason.MEASUREMENT_WINDOW_CLOCK_MISMATCH]
    return []


def _check_coverage(
    start: float,
    end: float,
    expected_device_keys: set[DeviceKey],
    observed_devices: Sequence[ObservedDevice],
) -> tuple[dict[str, float], list[str]]:
    """Every expected device must bracket the window with small enough gaps."""
    by_key = {device.key: device for device in observed_devices}
    gaps: dict[str, float] = {}
    reasons: list[str] = []

    for key in sorted(expected_device_keys):
        device = by_key.get(key)
        if device is None:
            reasons.append(Reason.MEASUREMENT_WINDOW_NOT_BRACKETED)
            continue
        # NOTE: a changed UUID cannot be attributed to one GPU, so this window's coverage is unusable.
        if len(device.gpu_uuids) != 1:
            reasons.append(Reason.GPU_UUID_CHANGED)
            continue
        sequence = _bracketing_sequence(device.sample_times, start, end)
        if sequence is None:
            reasons.append(Reason.MEASUREMENT_WINDOW_NOT_BRACKETED)
            continue
        largest = max((later - earlier for earlier, later in itertools.pairwise(sequence)), default=0.0)
        gaps[f"{device.hostname}/{device.gpu_uuids[0]}"] = largest
        if largest > MAX_SAMPLE_GAP_SECONDS:
            reasons.append(Reason.SAMPLE_GAP_EXCEEDED)

    return gaps, reasons


def _bracketing_sequence(times: Sequence[float], start: float, end: float) -> list[float] | None:
    """The last sample at or before start, every in-window sample, the first at or after end."""
    ordered = sorted(times)
    before = [value for value in ordered if value <= start]
    after = [value for value in ordered if value >= end]
    if not before or not after:
        return None
    inside = [value for value in ordered if start < value < end]
    return [before[-1], *inside, after[0]]


def _stays_below(root: Path, relative: str) -> bool:
    try:
        (root / relative).resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True
