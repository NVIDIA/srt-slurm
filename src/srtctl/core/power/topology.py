# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expected GPU topology derived from srt-slurm backend-process placement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from srtctl.core.power.contract import Reason, dedupe
from srtctl.core.power.samples import ObservedDevice
from srtctl.core.topology import Process

if TYPE_CHECKING:
    from srtctl.core.power.identity import WorkerIdentityProof

DeviceKey = tuple[str, int]

WORKER_ROLES = ("prefill", "decode", "agg")


@dataclass(frozen=True)
class DeviceAssignment:
    """One backend process's claim on a physical GPU."""

    worker_role: str
    worker_index: int
    worker_process: int
    het_group: int | None

    @classmethod
    def from_process(cls, process: Process) -> DeviceAssignment:
        return cls(
            worker_role=process.endpoint_mode,
            worker_index=process.endpoint_index,
            worker_process=process.node_rank,
            het_group=process.het_group,
        )

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
        assignment = DeviceAssignment.from_process(process)
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
    worker_identity: WorkerIdentityProof | None = None,
) -> DeviceValidation:
    """Require every expected device with a stable identity.

    A pool node's GPUs are never planned, so any of them is ``unexpected_device``.
    An exclusive worker node's exporter also reports the GPUs no worker uses;
    those are excused only when ``worker_identity`` shows every worker holding
    exactly the UUIDs the exporter reported at its planned indices, because the
    plan's CUDA numbering and the exporter's numbering are joined on index alone.
    """
    reasons: list[str] = []

    expected_keys = {device.key for device in expected}
    observed_keys = {device.key for device in observed}
    worker_hosts = {hostname for hostname, _ in expected_keys}
    extra_keys = observed_keys - expected_keys
    idle_worker_gpus = {key for key in extra_keys if key[0] in worker_hosts}
    identity_reasons = () if worker_identity is None else _worker_identity_reasons(worker_identity, expected, observed)
    workers_proven = worker_identity is not None and not identity_reasons

    if not expected_keys or expected_keys - observed_keys:
        reasons.append(Reason.EXPECTED_DEVICE_MISSING)
    if extra_keys - idle_worker_gpus or (idle_worker_gpus and not workers_proven):
        reasons.append(Reason.UNEXPECTED_DEVICE)
    # Unusable evidence only matters when idle GPUs needed excusing; a mismatch
    # shows the published rows would be mis-attributed, so it always blocks.
    if idle_worker_gpus or Reason.WORKER_GPU_UUID_MISMATCH in identity_reasons:
        reasons.extend(identity_reasons)
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


def _worker_identity_reasons(
    proof: WorkerIdentityProof,
    expected: Sequence[ExpectedDevice],
    observed: Sequence[ObservedDevice],
) -> tuple[str, ...]:
    """Empty when every planned worker was probed and holds the UUIDs sampled at its planned devices."""
    if proof.error is not None or not proof.workers:
        return (Reason.WORKER_IDENTITY_INVALID,)
    planned: dict[tuple[str, DeviceAssignment], set[DeviceKey]] = {}
    for device in expected:
        for assignment in device.assignments:
            planned.setdefault((device.hostname, assignment), set()).add(device.key)
    if {(worker.hostname, worker.assignment) for worker in proof.workers} != set(planned):
        return (Reason.WORKER_IDENTITY_INVALID,)
    sampled_uuid = {device.key: device.gpu_uuids[0] for device in observed if len(device.gpu_uuids) == 1}
    for worker in proof.workers:
        keys = planned[(worker.hostname, worker.assignment)]
        # A missing or unstable planned device is already reported by the device checks.
        if keys <= sampled_uuid.keys() and {sampled_uuid[key] for key in keys} != set(worker.gpu_uuids):
            return (Reason.WORKER_GPU_UUID_MISMATCH,)
    return ()
