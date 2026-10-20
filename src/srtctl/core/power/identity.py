# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which physical GPUs (by UUID) each worker step holds.

The plan numbers a worker's GPUs by ``CUDA_VISIBLE_DEVICES`` while the exporter
labels samples by its own device index, and nothing guarantees the two orders
agree on a node the job only partly uses. Once the server is healthy, one
bare-host probe per worker node asks Slurm which PIDs belong to each worker
step and the vendor tool which GPU UUIDs those PIDs hold. ``validate_devices``
excuses the idle GPUs an exclusive node's exporter also reports only when this
record matches the UUIDs the exporter saw at the planned indices.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from typing import Any

from srtctl.core.power.topology import DeviceAssignment

MPS_SERVER_PROCESS = "nvidia-cuda-mps-server"
LISTPIDS_HEADER = ["PID", "JOBID", "STEPID", "LOCALID", "GLOBALID"]


@dataclass(frozen=True)
class WorkerGpuIdentity:
    """The GPUs one worker step held when the probe ran."""

    hostname: str
    assignment: DeviceAssignment
    step_name: str
    step_id: str
    gpu_uuids: tuple[str, ...]


@dataclass(frozen=True)
class WorkerIdentityProof:
    """The probe's outcome: every worker's GPUs, or why they could not be established."""

    captured_at_unix: float
    workers: tuple[WorkerGpuIdentity, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_at_unix": self.captured_at_unix,
            "workers": [asdict(worker) for worker in self.workers],
            "error": self.error,
        }


NVIDIA_PROBE_SCRIPT = """import json, subprocess, sys

steps = {}
for step_id in sys.argv[1:]:
    listed = subprocess.run(["scontrol", "listpids", step_id], capture_output=True, text=True, timeout=10, check=True)
    steps[step_id] = listed.stdout
apps = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name", "--format=csv,noheader"],
    capture_output=True, text=True, timeout=10, check=True,
)
print(json.dumps({"steps": steps, "compute_apps": apps.stdout}))
"""

# Keyed by the exporter config's ``kind``: the probe must report the same identity
# the exporter writes into ``gpu_uuid``, which only DCGM's NVML UUID is known to be.
WORKER_PROBE_SCRIPTS: dict[str, str] = {"dcgm": NVIDIA_PROBE_SCRIPT}


def parse_node_probe(raw_output: str) -> dict[str, tuple[str, ...]]:
    """``{step id: UUIDs its processes hold}`` from one node's probe output.

    The JSON is the probe's last line, so stderr noise or an appended earlier
    run above it is ignored. A step whose processes hold no GPU cannot be
    attributed, and neither can any step on a node running MPS, where clients
    hide behind the MPS server's PID. The listpids JOBID column is not compared
    with the step id: for a heterogeneous job ``squeue`` names a step
    ``<leader>+<offset>.<step>`` while ``scontrol`` prints the component's job id.
    """
    lines = [line for line in raw_output.splitlines() if line.strip()]
    payload = json.loads(lines[-1] if lines else "")
    if not isinstance(payload, dict) or set(payload) != {"steps", "compute_apps"}:
        raise ValueError("invalid worker probe output")
    steps, apps = payload["steps"], payload["compute_apps"]
    if not isinstance(steps, dict) or not steps or not isinstance(apps, str):
        raise TypeError("missing worker probe evidence")

    uuids_by_pid: dict[int, set[str]] = {}
    for row in csv.reader(io.StringIO(apps), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 3 or not row[0].isdigit() or int(row[0]) <= 0 or not is_gpu_uuid(row[1]):
            raise ValueError("invalid compute-app identity")
        if MPS_SERVER_PROCESS in row[2]:
            raise ValueError("MPS worker attribution is unsupported")
        uuids_by_pid.setdefault(int(row[0]), set()).add(row[1])

    attributed = {}
    for step_id, listed in steps.items():
        if not isinstance(step_id, str) or not isinstance(listed, str):
            raise TypeError("invalid step PID output")
        rows = listed.splitlines()
        if not rows or rows[0].split() != LISTPIDS_HEADER:
            raise ValueError("missing scontrol listpids header")
        uuids: set[str] = set()
        for row in rows[1:]:
            columns = row.split()
            if len(columns) != 5 or not columns[0].isdigit() or int(columns[0]) <= 0:
                raise ValueError("invalid scontrol listpids row")
            uuids |= uuids_by_pid.get(int(columns[0]), set())
        if not uuids:
            raise ValueError(f"worker step {step_id} has no attributable compute-app UUIDs")
        attributed[step_id] = tuple(sorted(uuids))
    return attributed


def is_gpu_uuid(value: Any) -> bool:
    """A full-GPU ``GPU-…`` UUID; MIG instances are not sampled per device."""
    return (
        isinstance(value, str) and value.startswith("GPU-") and len(value) > 4 and not any(c.isspace() for c in value)
    )
