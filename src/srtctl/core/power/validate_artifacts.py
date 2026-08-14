# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-validate a retained power artifact package offline.

Disk-derived claims are recomputed from the persisted bytes. Producer identity
and runtime-only reasons are checked for wire and lifecycle consistency, and
the stored publication verdict must agree with recomputation. That way a
reviewer can check a run without access to the live job.

    srtctl-validate-power \
        --power-dir outputs/12345/logs/power --result-root outputs/12345/logs
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from srtctl.core.power.contract import (
    ALL_REASON_CODES,
    CLOCK_SOURCE,
    FATAL_LIFECYCLE_REASONS,
    MANIFEST_FILENAME,
    MAX_SAMPLE_GAP_SECONDS,
    POWER_METRIC,
    POWER_SCOPE,
    POWER_UNIT,
    PRODUCER,
    SAMPLES_FILENAME,
    SCHEMA_VERSION,
    STARTUP_FAILURE_REASONS,
    Reason,
    is_finite_number,
    sha256_file,
)
from srtctl.core.power.manifest import STATUS_COMPLETE, ArtifactError, ExpectedWindow, WindowValidation
from srtctl.core.power.samples import ObservedDevice, SampleRow, derive_observed_devices, read_samples
from srtctl.core.power.topology import (
    WORKER_ROLES,
    DeviceAssignment,
    ExpectedDevice,
    resolve_het_groups,
    resolve_roles,
    validate_devices,
)
from srtctl.core.power.windows import validate_expected_windows

_DISK_DERIVED_REASON_CODES = frozenset(
    {
        Reason.SAMPLES_CSV_MISSING,
        Reason.SAMPLES_CSV_HEADER_MISMATCH,
        Reason.SAMPLES_CSV_MALFORMED,
        Reason.DUPLICATE_SAMPLE_ROW,
        Reason.TIMESTAMP_NON_MONOTONIC,
        Reason.UNEXPECTED_DEVICE,
        Reason.EXPECTED_DEVICE_MISSING,
        Reason.GPU_UUID_CHANGED,
        Reason.CONFLICTING_WORKER_ROLES,
        Reason.CONFLICTING_HET_GROUPS,
        Reason.MEASUREMENT_WINDOW_MISSING,
        Reason.MEASUREMENT_WINDOW_UNEXPECTED,
        Reason.MEASUREMENT_WINDOW_DUPLICATE,
        Reason.MEASUREMENT_WINDOW_MALFORMED,
        Reason.MEASUREMENT_WINDOW_ARTIFACT_PATH_INVALID,
        Reason.MEASUREMENT_WINDOW_INCOMPLETE,
        Reason.MEASUREMENT_WINDOW_RESULT_MISSING,
        Reason.MEASUREMENT_WINDOW_RESULT_MISMATCH,
        Reason.MEASUREMENT_WINDOW_RESULT_PATH_INVALID,
        Reason.MEASUREMENT_WINDOW_CLOCK_MISMATCH,
        Reason.MEASUREMENT_WINDOW_NOT_BRACKETED,
        Reason.SAMPLE_GAP_EXCEEDED,
    }
)

_RUNTIME_ONLY_REASON_CODES = frozenset(
    {
        Reason.EXPORTER_STARTUP_TIMEOUT,
        Reason.EXPORTER_LAUNCH_FAILED,
        Reason.EXPORTER_EXITED,
        Reason.ENDPOINT_TIMEOUT,
        Reason.ENDPOINT_HTTP_ERROR,
        Reason.ENDPOINT_PARSE_ERROR,
        Reason.ENDPOINT_RESOLUTION_FAILED,
        Reason.POWER_METRIC_MISSING,
        Reason.DUPLICATE_POWER_METRIC,
        Reason.GPU_INDEX_MISSING,
        Reason.GPU_UUID_MISSING,
        Reason.INVALID_POWER_VALUE,
        Reason.MIG_INSTANCE_UNSUPPORTED,
        Reason.COLLECTOR_EXCEPTION,
        Reason.COLLECTOR_INTERRUPTED,
        Reason.COLLECTOR_JOIN_TIMEOUT,
        Reason.BENCHMARK_CHILD_REAP_TIMEOUT,
    }
)

_UNRECOVERABLE_STARTUP_REASON_CODES = frozenset(
    {
        Reason.EXPORTER_LAUNCH_FAILED,
        Reason.ENDPOINT_RESOLUTION_FAILED,
    }
)


@dataclass(frozen=True)
class ArtifactReport:
    """Verdict plus the numbers a reviewer wants to see."""

    ok: bool
    failures: tuple[str, ...]
    publication_valid: bool | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"validation_ok: {self.ok}", f"publication_valid: {self.publication_valid}"]
        lines += [f"  {key}: {value}" for key, value in self.summary.items()]
        lines += [f"  FAIL {failure}" for failure in self.failures]
        return "\n".join(lines)


def validate_power_artifacts(
    *,
    power_dir: Path,
    result_root: Path,
    expected_roles: dict[str, int] | None = None,
    require_distinct_het_groups: bool = False,
) -> ArtifactReport:
    """Recompute publication validity from the persisted artifact files."""
    manifest_path = power_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return ArtifactReport(ok=False, failures=(f"manifest missing: {manifest_path}",))
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        return ArtifactReport(ok=False, failures=(f"manifest unreadable: {exc}",))
    if not isinstance(manifest, dict):
        return ArtifactReport(ok=False, failures=(f"manifest malformed: not an object ({type(manifest).__name__})",))

    stored_publication_valid = manifest.get("publication_valid")
    if not isinstance(stored_publication_valid, bool):
        stored_publication_valid = None

    failures: list[str] = _check_wire_contract(manifest)
    failures += _check_samples_digest(manifest, power_dir / SAMPLES_FILENAME)

    try:
        expected_devices = _expected_devices(manifest)
        expected_windows = _expected_windows(manifest)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        failures.append(f"manifest malformed: {exc!r}")
        return ArtifactReport(
            ok=False,
            failures=tuple(failures),
            publication_valid=stored_publication_valid,
        )

    rows, sample_reasons = read_samples(power_dir / SAMPLES_FILENAME)
    failures += [f"{reason} in {SAMPLES_FILENAME}" for reason in sample_reasons]
    failures += _check_sample_lifecycle(manifest, rows)

    observed = derive_observed_devices(rows)
    devices = validate_devices(expected_devices, observed)
    failures += [f"{reason} (device identity/topology)" for reason in devices.reason_codes]

    artifact_errors: list[ArtifactError] = []
    validations = validate_expected_windows(
        power_dir=power_dir,
        result_root=result_root,
        expected_windows=expected_windows,
        expected_device_keys={device.key for device in expected_devices},
        observed_devices=observed,
        artifact_errors=artifact_errors,
    )
    if not expected_windows:
        failures.append("no expected measurement window")
    for validation in validations:
        failures += [
            f"{reason} (window {validation.benchmark_type}/{validation.concurrency})"
            for reason in validation.reason_codes
        ]
    failures += [f"{reason} ({error.path})" for error in artifact_errors for reason in error.reason_codes]

    failures += _check_topology(expected_devices, expected_roles, require_distinct_het_groups)
    failures += _check_stored_evidence(
        manifest=manifest,
        expected_devices=expected_devices,
        expected_windows=expected_windows,
        rows=rows,
        sample_reason_codes=sample_reasons,
        observed=observed,
        device_reason_codes=devices.reason_codes,
        devices_valid=devices.valid,
        validations=validations,
        artifact_errors=artifact_errors,
    )

    gaps = [gap for validation in validations for gap in validation.per_device_max_sample_gap_seconds.values()]
    summary = {
        "producer_git_commit": manifest.get("producer_git_commit"),
        "job_id": manifest.get("job_id"),
        "expected_devices": len(expected_devices),
        "observed_devices": len(observed),
        "stable_uuids": sum(1 for device in observed if len(device.gpu_uuids) == 1),
        "sample_rows": len(rows),
        "windows": len(validations),
        "max_sample_gap_seconds": max(gaps) if gaps else None,
    }
    return ArtifactReport(
        ok=not failures,
        failures=tuple(failures),
        publication_valid=stored_publication_valid,
        summary=summary,
    )


def _check_sample_lifecycle(manifest: dict[str, Any], rows: Sequence[SampleRow]) -> list[str]:
    """Require persisted samples to lie inside the producer lifecycle."""
    if not rows:
        return []
    started = manifest.get("started_at_unix")
    stopped = manifest.get("stopped_at_unix")
    if not (is_finite_number(started) and is_finite_number(stopped)):
        return []

    failures: list[str] = []
    first = min(row.timestamp_unix for row in rows)
    last = max(row.timestamp_unix for row in rows)
    if first < started:
        failures.append(f"samples.csv starts at {first}, before started_at_unix {started}")
    if last > stopped:
        failures.append(f"samples.csv ends at {last}, after stopped_at_unix {stopped}")
    return failures


def _check_samples_digest(manifest: dict[str, Any], samples_path: Path) -> list[str]:
    """Require ``samples.csv`` to match the bytes finalized by the producer."""
    stored = manifest.get("samples_sha256")
    if not (isinstance(stored, str) and re.fullmatch(r"[0-9a-f]{64}", stored)):
        return []
    try:
        actual = sha256_file(samples_path)
    except OSError as exc:
        return [f"samples_sha256 could not be verified: {exc}"]
    if actual != stored:
        return [f"samples_sha256 mismatch: manifest records {stored}, recomputed {actual}"]
    return []


def _check_wire_contract(manifest: dict[str, Any]) -> list[str]:
    """Reject a manifest whose own v1 wire and lifecycle metadata is invalid.

    The stored ``publication_valid`` verdict is type-checked here and compared
    with recomputation in :func:`_check_stored_evidence`.
    """
    failures: list[str] = []

    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
        failures.append(f"schema_version is {schema_version!r}, expected {SCHEMA_VERSION!r}")

    for key, expected in (
        ("producer", PRODUCER),
        ("source_metric", POWER_METRIC),
        ("unit", POWER_UNIT),
        ("power_scope", POWER_SCOPE),
        ("timestamp_source", CLOCK_SOURCE),
    ):
        if manifest.get(key) != expected:
            failures.append(f"{key} is {manifest.get(key)!r}, expected {expected!r}")

    for key in ("producer_version", "job_id", "run_name"):
        value = manifest.get(key)
        if not (isinstance(value, str) and value):
            failures.append(f"{key} is not a non-empty string")

    commit = manifest.get("producer_git_commit")
    if commit is not None and not (isinstance(commit, str) and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit)):
        failures.append("producer_git_commit is neither null nor a full lowercase Git object ID")

    status = manifest.get("status")
    if status != STATUS_COMPLETE:
        failures.append(f"status is {status!r}, expected {STATUS_COMPLETE!r}")

    started = manifest.get("started_at_unix")
    stopped = manifest.get("stopped_at_unix")
    if not is_finite_number(started):
        failures.append("started_at_unix is not a finite number")
    if not is_finite_number(stopped):
        failures.append("stopped_at_unix is not finite in a terminal manifest")
    elif is_finite_number(started) and stopped < started:
        failures.append("stopped_at_unix precedes started_at_unix")
    for key in ("sample_interval_seconds", "request_timeout_seconds"):
        value = manifest.get(key)
        if not (is_finite_number(value) and value > 0):
            failures.append(f"{key} is not finite and positive")
    sample_interval = manifest.get("sample_interval_seconds")
    if is_finite_number(sample_interval) and sample_interval > MAX_SAMPLE_GAP_SECONDS:
        failures.append(f"sample_interval_seconds exceeds the {MAX_SAMPLE_GAP_SECONDS}s coverage limit")

    max_scrape_duration = manifest.get("max_scrape_duration_seconds")
    if max_scrape_duration is not None and not (is_finite_number(max_scrape_duration) and max_scrape_duration >= 0):
        failures.append("max_scrape_duration_seconds is neither null nor a finite non-negative number")
    for key in ("scrape_count", "sample_row_count"):
        value = manifest.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            failures.append(f"{key} is not a non-negative integer")

    samples_digest = manifest.get("samples_sha256")
    if not (isinstance(samples_digest, str) and re.fullmatch(r"[0-9a-f]{64}", samples_digest)):
        failures.append("samples_sha256 is not a lowercase SHA-256 digest")

    required = manifest.get("required")
    if not isinstance(required, bool):
        failures.append(f"required is {required!r}, expected a boolean")
    publication_valid = manifest.get("publication_valid")
    if not isinstance(publication_valid, bool):
        failures.append(f"publication_valid is {publication_valid!r}, expected a boolean")

    # NOTE: item types are checked before the set op, or a non-hashable entry raises.
    reasons = manifest.get("reason_codes")
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        failures.append("reason_codes is not a list of strings")
    else:
        reason_strings = [str(reason) for reason in reasons]
        unknown = sorted(set(reason_strings) - ALL_REASON_CODES)
        if unknown:
            failures.append(f"reason_codes contains unknown v1 values: {', '.join(unknown)}")
        if len(reason_strings) != len(set(reason_strings)):
            failures.append("reason_codes contains duplicates")
        # NOTE: startup reasons block `complete` only in required mode, mirroring _terminal_status.
        blocking = set(FATAL_LIFECYCLE_REASONS)
        if required is True:
            blocking |= set(STARTUP_FAILURE_REASONS)
        recorded = set(reason_strings)
        incompatible = sorted(recorded & blocking)
        if incompatible:
            failures.append(f"complete manifest carries lifecycle-failure reasons: {', '.join(incompatible)}")

    exporter = manifest.get("dcgm_exporter")
    if not isinstance(exporter, dict):
        failures.append("dcgm_exporter is not an object")
    else:
        image = exporter.get("container_image_resolved")
        if not (isinstance(image, str) and image):
            failures.append("dcgm_exporter.container_image_resolved is empty")
        digest = exporter.get("container_image_sha256")
        if digest is not None and not (isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)):
            failures.append("dcgm_exporter.container_image_sha256 is neither null nor a lowercase SHA-256 digest")
        port = exporter.get("port")
        if not isinstance(port, int) or isinstance(port, bool) or not 0 < port < 65536:
            failures.append("dcgm_exporter.port is not a valid port number")
        command = exporter.get("command")
        if not (isinstance(command, str) and command):
            failures.append("dcgm_exporter.command is empty")

    return failures


def _check_stored_evidence(
    *,
    manifest: dict[str, Any],
    expected_devices: Sequence[ExpectedDevice],
    expected_windows: Sequence[ExpectedWindow],
    rows: Sequence[SampleRow],
    sample_reason_codes: Sequence[str],
    observed: Sequence[ObservedDevice],
    device_reason_codes: Sequence[str],
    devices_valid: bool,
    validations: Sequence[WindowValidation],
    artifact_errors: Sequence[ArtifactError],
) -> list[str]:
    """Reject a manifest that contradicts the evidence on disk.

    Disk-derived claims and the stored verdict must agree with recomputation.
    Runtime-only reasons are retained evidence whose source events are no
    longer available offline, so their enum and lifecycle consistency are
    checked by :func:`_check_wire_contract`.
    """
    failures: list[str] = []

    stored_rows = manifest.get("sample_row_count")
    if stored_rows != len(rows):
        failures.append(f"sample_row_count is {stored_rows!r}, disk has {len(rows)} rows")

    # NOTE: trailing empty cycles may exceed max(scrape_seq)+1, but can never fall below it.
    stored_scrapes = manifest.get("scrape_count")
    if rows:
        least = max(row.scrape_seq for row in rows) + 1
        if not isinstance(stored_scrapes, int) or isinstance(stored_scrapes, bool) or stored_scrapes < least:
            failures.append(f"scrape_count is {stored_scrapes!r}, disk needs at least {least}")

    if not observed:
        failures.append("observed_devices is empty")
    if not _same_json_evidence(manifest.get("observed_devices"), [device.to_dict() for device in observed]):
        failures.append("observed_devices does not match the devices derived from samples.csv")
    if not _same_json_evidence(
        manifest.get("window_validations"), [validation.to_dict() for validation in validations]
    ):
        failures.append("window_validations does not match the recomputed window audit")
    if not _same_json_evidence(manifest.get("artifact_errors"), [error.to_dict() for error in artifact_errors]):
        failures.append("artifact_errors does not match the recomputed artifact scan")

    recomputed_disk_reasons = {
        *sample_reason_codes,
        *device_reason_codes,
        *(reason for validation in validations for reason in validation.reason_codes),
        *(reason for error in artifact_errors for reason in error.reason_codes),
    }
    stored_reasons = manifest.get("reason_codes")
    stored_reason_strings = (
        [str(reason) for reason in stored_reasons]
        if isinstance(stored_reasons, list) and all(isinstance(reason, str) for reason in stored_reasons)
        else None
    )
    if stored_reason_strings is not None:
        stored_disk_reasons = set(stored_reason_strings) & _DISK_DERIVED_REASON_CODES
        if stored_disk_reasons != recomputed_disk_reasons:
            stored_only = sorted(stored_disk_reasons - recomputed_disk_reasons)
            recomputed_only = sorted(recomputed_disk_reasons - stored_disk_reasons)
            failures.append(
                f"disk-derived reason_codes mismatch: stored-only={stored_only}, recomputed-only={recomputed_only}"
            )

    recorded = set(stored_reason_strings or ())
    lifecycle_complete = not (recorded & set(FATAL_LIFECYCLE_REASONS)) and not (
        manifest.get("required") is True and recorded & set(STARTUP_FAILURE_REASONS)
    )
    # A readiness timeout can recover in best-effort mode because collection
    # keeps running. A failed launch never starts collection, while resolution
    # failure permanently omits an expected node; neither can produce the
    # complete on-disk evidence this validator is recomputing.
    startup_recovered = not (recorded & _UNRECOVERABLE_STARTUP_REASON_CODES)
    windows_valid = bool(expected_windows) and all(validation.power_coverage_valid for validation in validations)
    recomputed_publication_valid = (
        lifecycle_complete
        and startup_recovered
        and devices_valid
        and windows_valid
        and not sample_reason_codes
        and not artifact_errors
    )
    stored_publication_valid = manifest.get("publication_valid")
    if isinstance(stored_publication_valid, bool):
        if stored_publication_valid != recomputed_publication_valid:
            failures.append(
                f"publication_valid is {stored_publication_valid}, recomputed {recomputed_publication_valid}"
            )
        elif not stored_publication_valid:
            failures.append("stored publication_valid is false")

    for label, keys in (
        ("expected_devices", [device.key for device in expected_devices]),
        ("expected_windows", [window.key for window in expected_windows]),
    ):
        if len(set(keys)) != len(keys):
            failures.append(f"{label} contains duplicate keys")

    return failures


def _same_json_evidence(stored: Any, recomputed: Any) -> bool:
    """Compare JSON evidence without letting booleans impersonate numbers."""
    stored_is_number = isinstance(stored, (int, float)) and not isinstance(stored, bool)
    recomputed_is_number = isinstance(recomputed, (int, float)) and not isinstance(recomputed, bool)
    if stored_is_number or recomputed_is_number:
        return is_finite_number(stored) and is_finite_number(recomputed) and stored == recomputed
    if type(stored) is not type(recomputed):
        return False
    if isinstance(stored, list):
        return len(stored) == len(recomputed) and all(
            _same_json_evidence(stored_value, recomputed_value)
            for stored_value, recomputed_value in zip(stored, recomputed, strict=True)
        )
    if isinstance(stored, dict):
        return stored.keys() == recomputed.keys() and all(
            _same_json_evidence(stored[key], recomputed[key]) for key in stored
        )
    return stored == recomputed


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a non-empty string: {value!r}")
    return value


def _role(value: Any) -> str:
    if value not in WORKER_ROLES:
        raise ValueError(f"worker_role is not one of {WORKER_ROLES}: {value!r}")
    return value


def _whole(value: Any, label: str, *, minimum: int) -> int:
    """A JSON integer. ``bool`` is excluded because ``int(True)`` silently passes."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} is not an integer >= {minimum}: {value!r}")
    return value


def _expected_windows(manifest: dict[str, Any]) -> list[ExpectedWindow]:
    entries = manifest.get("expected_windows") or []
    if not isinstance(entries, list):
        raise TypeError(f"expected_windows is not a list: {entries!r}")
    return [
        ExpectedWindow(
            benchmark_type=_text(entry["benchmark_type"], "benchmark_type"),
            concurrency=_whole(entry["concurrency"], "concurrency", minimum=1),
        )
        for entry in entries
    ]


def _expected_devices(manifest: dict[str, Any]) -> list[ExpectedDevice]:
    devices: list[ExpectedDevice] = []
    entries = manifest.get("expected_devices") or []
    if not isinstance(entries, list):
        raise TypeError(f"expected_devices is not a list: {entries!r}")
    for entry in entries:
        raw_assignments = entry.get("assignments") or []
        if not isinstance(raw_assignments, list) or not raw_assignments:
            raise ValueError(f"assignments is not a non-empty list: {raw_assignments!r}")
        assignments = tuple(
            DeviceAssignment(
                worker_role=_role(assignment["worker_role"]),
                worker_index=_whole(assignment["worker_index"], "worker_index", minimum=0),
                worker_process=_whole(assignment["worker_process"], "worker_process", minimum=0),
                het_group=(
                    None
                    if assignment.get("het_group") is None
                    else _whole(assignment["het_group"], "het_group", minimum=0)
                ),
            )
            for assignment in raw_assignments
        )
        devices.append(
            ExpectedDevice(
                hostname=_text(entry["hostname"], "hostname"),
                gpu_index=_whole(entry["gpu_index"], "gpu_index", minimum=0),
                assignments=assignments,
            )
        )
    return devices


def _check_topology(
    expected_devices: Sequence[ExpectedDevice],
    expected_roles: dict[str, int] | None,
    require_distinct_het_groups: bool,
) -> list[str]:
    """Optional canary-shape assertions on top of the generic publication rules.

    Deliberately exact: positive-count roles must match the observed role set,
    while a zero count asserts that a known role is absent. Each present role
    must occupy exactly one heterogeneous group with no group shared between
    roles.
    """
    failures: list[str] = []
    roles, role_conflicts = resolve_roles(expected_devices)
    if role_conflicts:
        return [f"{reason} (topology)" for reason in role_conflicts]

    counts: dict[str, int] = {}
    for role in roles.values():
        counts[role] = counts.get(role, 0) + 1

    if expected_roles is not None:
        unknown = sorted(set(expected_roles) - set(WORKER_ROLES))
        if unknown:
            failures.append(f"unknown expected roles: {unknown}; allowed roles are {list(WORKER_ROLES)}")
        invalid_counts = sorted(
            role
            for role, count in expected_roles.items()
            if not isinstance(count, int) or isinstance(count, bool) or count < 0
        )
        if invalid_counts:
            failures.append(f"expected role counts must be non-negative integers: {invalid_counts}")

        valid_expected = {
            role: count
            for role, count in expected_roles.items()
            if role in WORKER_ROLES and isinstance(count, int) and not isinstance(count, bool) and count >= 0
        }
        expected_present_roles = {role for role, count in valid_expected.items() if count > 0}
        if set(counts) != expected_present_roles:
            failures.append(f"expected roles {sorted(expected_present_roles)}, found {sorted(counts)}")
        for role, count in sorted(valid_expected.items()):
            if counts.get(role, 0) != count:
                failures.append(f"expected {count} {role} GPUs, found {counts.get(role, 0)}")

    if require_distinct_het_groups:
        groups, group_conflicts = resolve_het_groups(expected_devices)
        if group_conflicts:
            return failures + [f"{reason} (topology)" for reason in group_conflicts]

        per_role: dict[str, set[int | None]] = {}
        for device in expected_devices:
            per_role.setdefault(roles[device.key], set()).add(groups[device.hostname])

        assigned: list[int] = []
        for role, role_groups in sorted(per_role.items()):
            if len(role_groups) != 1:
                failures.append(f"role {role} spans het groups {sorted(role_groups, key=str)}, expected exactly one")
                continue
            group = next(iter(role_groups))
            if not isinstance(group, int) or isinstance(group, bool) or group < 0:
                failures.append(f"role {role} has het group {group!r}, expected a non-negative integer")
                continue
            assigned.append(group)
        if len(set(assigned)) != len(assigned):
            failures.append(f"roles share a het group: { {role: sorted(g, key=str) for role, g in per_role.items()} }")
    return failures
