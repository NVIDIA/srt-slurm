# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restrict worker network device filters to the GPU's NUMA node."""

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


def restrict(value: str, candidates: list[str], *, nccl: bool = False) -> str:
    if not value or value == "all":
        return ("=" if nccl and candidates else "") + ",".join(candidates)
    exclude = value.startswith("^")
    value = value.removeprefix("^")
    exact = value.startswith("=") or not nccl
    entries = value.removeprefix("=").split(",")
    selected = []
    for candidate in candidates:
        name, _, port = candidate.partition(":")
        matches = []
        for entry in entries:
            fields = entry.split(":")
            device_match = name == fields[0] if exact else name.startswith(fields[0])
            if device_match and (len(fields) == 1 or not fields[1] or fields[1] == port):
                matches.append(fields)
        if exclude:
            if not matches:
                selected.append(candidate)
        elif matches:
            fields = next((fields for fields in matches if fields[0] == name), matches[0])
            # Retain explicit NCCL rail/plane identities, using the resolved port.
            selected.append(":".join([name, port, *fields[2:]]) if nccl else candidate)
    # A single leading '=' applies exact matching to the whole NCCL list.
    return ("=" if nccl and selected else "") + ",".join(dict.fromkeys(selected))


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
    result: dict[str, str] = {}
    for key, candidates in (
        ("UCX_NET_DEVICES", rdma + ucx_ethernet),
        ("NCCL_IB_HCA", rdma),
    ):
        # Ethernet-only hosts can run UCX TCP; do not invent RDMA settings there.
        if key == "NCCL_IB_HCA" and not rdma and key not in env:
            continue
        if not candidates and key not in env:
            continue
        value = restrict(env.get(key, ""), candidates, nccl=key == "NCCL_IB_HCA")
        if not value:
            raise ValueError(f"{key} has no permitted devices on NUMA node {node}")
        result[key] = value
    return result


if __name__ == "__main__":
    try:
        for key, value in resolve(Path("/sys"), sys.argv[1], dict(os.environ)).items():
            print(f"{key}\t{value}")
    except ValueError as error:
        print(f"numa_cpu_bind.sh: cannot bind network devices: {error}", file=sys.stderr)
        sys.exit(2)
