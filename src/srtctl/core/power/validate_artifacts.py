# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-validate a retained power artifact package offline.

Everything is recomputed from the persisted bytes: the manifest supplies the
expected topology and producer identity, never the verdict. That way a reviewer
can check a run without access to the live job.

    python -m srtctl.core.power.validate_artifacts \
        --power-dir outputs/12345/logs/power --result-root outputs/12345/logs
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from srtctl.core.power.contract import (
    CLOCK_SOURCE,
    FATAL_LIFECYCLE_REASONS,
    MANIFEST_FILENAME,
    POWER_METRIC,
    POWER_SCOPE,
    POWER_UNIT,
    PRODUCER,
    SAMPLES_FILENAME,
    SCHEMA_VERSION,
    STARTUP_FAILURE_REASONS,
    is_finite_number,
)
from srtctl.core.power.manifest import STATUS_COMPLETE, ArtifactError, ExpectedWindow
from srtctl.core.power.samples import derive_observed_devices, read_samples
from srtctl.core.power.topology import (
    WORKER_ROLES,
    DeviceAssignment,
    ExpectedDevice,
    resolve_het_groups,
    resolve_roles,
    validate_devices,
)
from srtctl.core.power.windows import validate_expected_windows


@dataclass(frozen=True)
class ArtifactReport:
    """Verdict plus the numbers a reviewer wants to see."""

    ok: bool
    failures: tuple[str, ...]
    summary: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"publication_valid: {self.ok}"]
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

    failures: list[str] = _check_wire_contract(manifest)

    try:
        expected_devices = _expected_devices(manifest)
        expected_windows = _expected_windows(manifest)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return ArtifactReport(ok=False, failures=(f"manifest malformed: {exc!r}",))

    rows, sample_reasons = read_samples(power_dir / SAMPLES_FILENAME)
    failures += [f"{reason} in {SAMPLES_FILENAME}" for reason in sample_reasons]

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
        manifest, expected_devices, expected_windows, rows, observed, validations, artifact_errors
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
    return ArtifactReport(ok=not failures, failures=tuple(failures), summary=summary)


def _check_wire_contract(manifest: dict[str, Any]) -> list[str]:
    """Reject a manifest whose own v1 wire and lifecycle metadata is invalid.

    The stored ``publication_valid`` verdict is deliberately ignored — that is
    what this tool recomputes — but the surrounding metadata must be internally
    consistent, or the recomputation would be describing a different contract.
    """
    failures: list[str] = []

    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("producer", PRODUCER),
        ("source_metric", POWER_METRIC),
        ("unit", POWER_UNIT),
        ("power_scope", POWER_SCOPE),
        ("timestamp_source", CLOCK_SOURCE),
    ):
        if manifest.get(key) != expected:
            failures.append(f"{key} is {manifest.get(key)!r}, expected {expected!r}")

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
    for key in ("scrape_count", "sample_row_count"):
        value = manifest.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            failures.append(f"{key} is not a non-negative integer")

    required = manifest.get("required")
    if not isinstance(required, bool):
        failures.append(f"required is {required!r}, expected a boolean")

    # NOTE: item types are checked before the set op, or a non-hashable entry raises.
    reasons = manifest.get("reason_codes")
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        failures.append("reason_codes is not a list of strings")
    else:
        # NOTE: startup reasons block `complete` only in required mode, mirroring _terminal_status.
        blocking = set(FATAL_LIFECYCLE_REASONS)
        if required is True:
            blocking |= set(STARTUP_FAILURE_REASONS)
        recorded: set[str] = {str(reason) for reason in reasons}
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
        if digest is not None and not (isinstance(digest, str) and digest):
            failures.append("dcgm_exporter.container_image_sha256 is neither null nor a digest string")
        port = exporter.get("port")
        if not isinstance(port, int) or isinstance(port, bool) or not 0 < port < 65536:
            failures.append("dcgm_exporter.port is not a valid port number")
        command = exporter.get("command")
        if not (isinstance(command, str) and command):
            failures.append("dcgm_exporter.command is empty")

    return failures


def _check_stored_evidence(
    manifest: dict[str, Any],
    expected_devices: Sequence[ExpectedDevice],
    expected_windows: Sequence[ExpectedWindow],
    rows: Sequence[Any],
    observed: Sequence[Any],
    validations: Sequence[Any],
    artifact_errors: Sequence[ArtifactError],
) -> list[str]:
    """Reject a manifest that contradicts the evidence on disk.

    Only ``publication_valid`` is ignored, because that is the verdict being
    recomputed. Every other derived field is a claim about the artifact that
    must still hold, or the package describes a different run than it contains.
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
    if manifest.get("observed_devices") != [device.to_dict() for device in observed]:
        failures.append("observed_devices does not match the devices derived from samples.csv")
    if manifest.get("window_validations") != [validation.to_dict() for validation in validations]:
        failures.append("window_validations does not match the recomputed window audit")
    if manifest.get("artifact_errors") != [error.to_dict() for error in artifact_errors]:
        failures.append("artifact_errors does not match the recomputed artifact scan")

    for label, keys in (
        ("expected_devices", [device.key for device in expected_devices]),
        ("expected_windows", [window.key for window in expected_windows]),
    ):
        if len(set(keys)) != len(keys):
            failures.append(f"{label} contains duplicate keys")

    return failures


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

    Deliberately exact: the role set must match, not merely contain, the
    requested counts, and each role must occupy exactly one heterogeneous group
    with no group shared between roles.
    """
    failures: list[str] = []
    roles, role_conflicts = resolve_roles(expected_devices)
    if role_conflicts:
        return [f"{reason} (topology)" for reason in role_conflicts]

    counts: dict[str, int] = {}
    for role in roles.values():
        counts[role] = counts.get(role, 0) + 1

    if expected_roles is not None:
        if set(counts) != set(expected_roles):
            failures.append(f"expected roles {sorted(expected_roles)}, found {sorted(counts)}")
        for role, count in sorted(expected_roles.items()):
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
