# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expected GPU topology derived from srt-slurm backend-process placement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from srtctl.core.power.contract import Reason, dedupe
from srtctl.core.power.samples import ObservedDevice
from srtctl.core.topology import Process

DeviceKey = tuple[str, int]

WORKER_ROLES = ("prefill", "decode", "agg")


@dataclass(frozen=True)
class DeviceAssignment:
    """One backend process's claim on a physical GPU."""

    worker_role: str
    worker_index: int
    worker_process: int
    het_group: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_role": self.worker_role,
            "worker_index": self.worker_index,
            "worker_process": self.worker_process,
            "het_group": self.het_group,
        }


@dataclass(frozen=True)
class ExpectedDevice:
    """A physical GPU the run is expected to sample."""

    hostname: str
    gpu_index: int
    assignments: tuple[DeviceAssignment, ...]

    def __post_init__(self) -> None:
        if not self.assignments:
            raise ValueError("expected device requires at least one assignment")

    @property
    def key(self) -> DeviceKey:
        return (self.hostname, self.gpu_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "gpu_index": self.gpu_index,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
        }


def build_expected_devices(processes: Sequence[Process]) -> list[ExpectedDevice]:
    """Map every allocated backend process onto the GPUs it occupies."""
    grouped: dict[DeviceKey, list[DeviceAssignment]] = {}
    for process in processes:
        assignment = DeviceAssignment(
            worker_role=process.endpoint_mode,
            worker_index=process.endpoint_index,
            worker_process=process.node_rank,
            het_group=process.het_group,
        )
        for gpu_index in sorted(process.gpu_indices):
            grouped.setdefault((process.node, gpu_index), []).append(assignment)

    return [
        ExpectedDevice(hostname=key[0], gpu_index=key[1], assignments=tuple(assignments))
        for key, assignments in sorted(grouped.items())
    ]


def resolve_roles(devices: Sequence[ExpectedDevice]) -> tuple[dict[DeviceKey, str], tuple[str, ...]]:
    """Resolve one semantic role per device, or report the conflict."""
    roles: dict[DeviceKey, str] = {}
    for device in devices:
        distinct = {assignment.worker_role for assignment in device.assignments}
        if len(distinct) != 1:
            return {}, (Reason.CONFLICTING_WORKER_ROLES,)
        roles[device.key] = distinct.pop()
    return roles, ()


def resolve_het_groups(devices: Sequence[ExpectedDevice]) -> tuple[dict[str, int | None], tuple[str, ...]]:
    """Resolve one heterogeneous Slurm group per node, or report the conflict."""
    groups: dict[str, int | None] = {}
    for device in devices:
        distinct = {assignment.het_group for assignment in device.assignments}
        if len(distinct) != 1:
            return {}, (Reason.CONFLICTING_HET_GROUPS,)
        group = distinct.pop()
        if device.hostname in groups and groups[device.hostname] != group:
            return {}, (Reason.CONFLICTING_HET_GROUPS,)
        groups[device.hostname] = group
    return groups, ()


@dataclass(frozen=True)
class DeviceValidation:
    """Whether device identity and topology permit publication."""

    valid: bool
    reason_codes: tuple[str, ...]


def validate_devices(
    expected: Sequence[ExpectedDevice],
    observed: Sequence[ObservedDevice],
) -> DeviceValidation:
    """Require a non-empty expected set that exactly matches stable observations."""
    reasons: list[str] = []

    expected_keys = {device.key for device in expected}
    observed_keys = {device.key for device in observed}

    if not expected_keys or expected_keys - observed_keys:
        reasons.append(Reason.EXPECTED_DEVICE_MISSING)
    if observed_keys - expected_keys:
        reasons.append(Reason.UNEXPECTED_DEVICE)
    # NOTE: a UUID must map 1:1 to a device key, or one physical GPU is counted twice.
    if any(len(device.gpu_uuids) != 1 for device in observed):
        reasons.append(Reason.GPU_UUID_CHANGED)
    else:
        uuids = [device.gpu_uuids[0] for device in observed]
        if len(set(uuids)) != len(uuids):
            reasons.append(Reason.GPU_UUID_CHANGED)

    _, role_conflicts = resolve_roles(expected)
    _, group_conflicts = resolve_het_groups(expected)
    reasons.extend(role_conflicts)
    reasons.extend(group_conflicts)

    return DeviceValidation(valid=not reasons, reason_codes=dedupe(reasons))
