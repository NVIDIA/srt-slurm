# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find assigned ports that a node's own outbound sockets can take first.

A port inside net.ipv4.ip_local_port_range, and not in ip_local_reserved_ports,
can be the source port of any outbound connection on the node, so a listener
binding it later fails with EADDRINUSE.
"""

import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import fields

from srtctl.core.topology import Process

# One line per node: "<node> <low> <high> <reserved>"; reserved is the kernel's
# comma-separated list (ports and ranges), "-" when empty.
PROBE_SCRIPT = (
    "read lo hi < /proc/sys/net/ipv4/ip_local_port_range; r=$(cat /proc/sys/net/ipv4/ip_local_reserved_ports); "
    'echo "${SLURMD_NODENAME:-$(hostname -s)} $lo $hi ${r:--}"'
)


def probe_nodes(nodes: Sequence[str], job_id: str, het_group: int | None = None) -> str:
    """The probe's lines from one overlapping srun step on the bare hosts; empty when it cannot run."""
    command = ["srun", "--quiet", "--overlap", "--jobid", job_id, f"--nodes={len(nodes)}", f"--ntasks={len(nodes)}"]
    command += [f"--nodelist={','.join(nodes)}", *([f"--het-group={het_group}"] if het_group is not None else [])]
    try:
        probe = [*command, "bash", "-c", PROBE_SCRIPT]
        return subprocess.run(probe, capture_output=True, text=True, timeout=60, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def process_ports(process: Process) -> list[tuple[str, int]]:
    """``(owner, port)`` for every listener port assigned to a worker process."""
    owner = f"{process.endpoint_mode}_{process.endpoint_index}"
    return [
        (f"{owner} {f.name}", value)
        for f in fields(process)
        if f.name.endswith("_port") and isinstance(value := getattr(process, f.name), int) and value > 0
    ]


def _reserved(spec: str) -> set[int]:
    ports: set[int] = set()
    for part in spec.split(","):
        low, _, high = part.partition("-")
        if low.isdigit():
            ports.update(range(int(low), int(high or low) + 1))
    return ports


def ephemeral_conflicts(probe_output: str, assigned: Iterable[tuple[str, str, int]]) -> list[str]:
    """One message per ``(node, owner, port)`` inside that node's unreserved ephemeral range."""
    ranges = {}
    for line in probe_output.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[1].isdigit() and parts[2].isdigit():
            ranges[parts[0]] = (int(parts[1]), int(parts[2]), _reserved(parts[3]))
    messages = []
    for node, owner, port in assigned:
        if node in ranges:
            low, high, reserved = ranges[node]
            if low <= port <= high and port not in reserved:
                messages.append(f"{node}: port {port} ({owner}) is inside ip_local_port_range {low}-{high}")
    return messages
