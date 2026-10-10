# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restrict the worker's UCX device filter to the GPU's NUMA node."""

import os
import sys
from pathlib import Path


def local_devices(sysfs: Path, node: str) -> tuple[list[str], list[str]]:
    rdma: list[str] = []
    ethernet: list[str] = []
    for kind, devices in (("infiniband", rdma), ("net", ethernet)):
        for device in sorted((sysfs / "class" / kind).glob("*")):
            try:
                if (device / "device" / "numa_node").read_text().strip() != node:
                    continue
                if kind == "net":
                    if (device / "type").read_text().strip() == "1":
                        devices.append(device.name)
                else:
                    devices.extend(f"{device.name}:{port.name}" for port in sorted((device / "ports").glob("*")))
            except OSError:
                continue
    return rdma, ethernet


def restrict(value: str, candidates: list[str]) -> str:
    if not value or value == "all":
        return ",".join(candidates)
    exclude = value.startswith("^")
    value = value.removeprefix("^")
    entries = value.removeprefix("=").split(",")
    selected = []
    for candidate in candidates:
        name, _, port = candidate.partition(":")
        matches = []
        for entry in entries:
            fields = entry.split(":")
            device_match = name == fields[0]
            if device_match and (len(fields) == 1 or not fields[1] or fields[1] == port):
                matches.append(fields)
        if exclude:
            if not matches:
                selected.append(candidate)
        elif matches:
            selected.append(candidate)
    return ",".join(dict.fromkeys(selected))


def resolve(sysfs: Path, node: str, env: dict[str, str]) -> dict[str, str]:
    rdma, ethernet = local_devices(sysfs, node)
    ucx_ethernet = ethernet.copy()
    ucx_filter = env.get("UCX_NET_DEVICES", "")
    if ucx_filter and not ucx_filter.startswith("^"):
        # Explicit TCP interfaces can serve every rank, even when the interface
        # is remote to its GPU or has no NUMA affinity (e.g. a container veth).
        for name in ucx_filter.removeprefix("=").split(","):
            if ":" in name or name in ucx_ethernet:
                continue
            try:
                if (sysfs / "class" / "net" / name / "type").read_text().strip() == "1":
                    ucx_ethernet.append(name)
            except OSError:
                continue
    candidates = rdma + ucx_ethernet
    if not candidates and "UCX_NET_DEVICES" not in env:
        return {}
    value = restrict(ucx_filter, candidates)
    if not value:
        raise ValueError(f"UCX_NET_DEVICES has no permitted devices on NUMA node {node}")
    return {"UCX_NET_DEVICES": value}


if __name__ == "__main__":
    try:
        for key, value in resolve(Path("/sys"), sys.argv[1], dict(os.environ)).items():
            print(f"{key}\t{value}")
    except ValueError as error:
        print(f"numa_cpu_bind.sh: cannot bind network devices: {error}", file=sys.stderr)
        sys.exit(2)
