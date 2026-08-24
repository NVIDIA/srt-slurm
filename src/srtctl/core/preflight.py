# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail a run before it starts when the hardware is not fit to serve it.

The ``before`` hwinfo snapshot already contains everything needed to predict the
common startup failures, but nobody reads three thousand lines of it until a job
has burned twenty minutes and died with a message that names no culprit:

- a worker dies with ``Free memory on device ... is less than desired GPU memory
  utilization`` because a previous job is still holding memory on the card;
- an MNNVL-backed all2all backend hangs or crashes because a node in the fabric
  domain is UNAVAILABLE — including a node outside this allocation, since the
  domain spans every node in ``nodes_config.cfg``;
- a GPU never joined the fabric clique, so its local links look healthy while
  remote peers are unreachable.

This module parses the snapshot and turns those into one readable failure at
startup. It only reports what the snapshot actually recorded: a command that was
missing or timed out yields no finding rather than a false alarm.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

GIB = 1024**3

# nvidia-smi prints "12345 MiB" / "1.5 GiB" style values in --format=csv output.
_MEMORY_VALUE = re.compile(r"^\s*(?P<value>[0-9.]+)\s*(?P<unit>[KMG]i?B)?\s*$", re.IGNORECASE)
_UNIT_SCALE = {"kib": 1024, "mib": 1024**2, "gib": 1024**3, "b": 1}

# "Node #11  - 10.66.6.19  - UNAVAILABLE  - Version:  - Hostname: theia0282..."
_IMEX_NODE = re.compile(
    r"^Node\s+#(?P<index>\d+)\s+[-*]\s*(?P<ip>[0-9a-fA-F.:]+)\s*\*?\s*-\s*"
    r"(?P<state>[A-Z_]+)\b.*?(?:Hostname:\s*(?P<host>\S+))?\s*$"
)

_SECTION = re.compile(r"^# -+ (?P<name>.+?) -+$")
_COMMAND = re.compile(r"^\$ (?P<command>.+)$")
_NODE_HEADER = re.compile(r"^===== (?P<node>\S+) \| ")


@dataclass(frozen=True)
class Finding:
    """One reason the run should not start."""

    node: str
    detail: str

    def __str__(self) -> str:
        return f"{self.node}: {self.detail}"


def check_snapshot(
    snapshot_path: Path,
    *,
    gpu_memory_utilization: float | None = None,
) -> list[Finding]:
    """Inspect a merged hwinfo snapshot and return everything that looks fatal.

    ``gpu_memory_utilization`` is the fraction of each card the engine will
    request. When given, a card with less free memory than
    ``utilization x total`` is reported — the same arithmetic, and the same
    verdict, the worker would reach minutes later.
    """
    try:
        text = snapshot_path.read_text(errors="replace")
    except OSError as error:
        logger.warning("Preflight skipped, cannot read %s: %s", snapshot_path, error)
        return []

    # Domain-level facts are reported by every node that can see them, so the
    # same broken peer would otherwise appear once per observer.
    findings: dict[Finding, None] = {}
    for node, blocks in _parse(text).items():
        for finding in (
            *_check_compute_apps(node, blocks),
            *_check_free_memory(node, blocks, gpu_memory_utilization),
            *_check_imex_domain(blocks),
            *_check_fabric(node, blocks),
        ):
            findings.setdefault(finding)
    return list(findings)


def _parse(text: str) -> dict[str, dict[str, str]]:
    """Split the snapshot into {node: {command: output}}.

    The snapshot format is stable and line-oriented: a node header, then
    ``# ---------- section ----------`` and ``$ command`` blocks.
    """
    nodes: dict[str, dict[str, str]] = {}
    node = ""
    command = ""
    lines: list[str] = []

    def flush() -> None:
        if node and command:
            nodes.setdefault(node, {})[command] = "\n".join(lines)

    for line in text.splitlines():
        if header := _NODE_HEADER.match(line):
            flush()
            node, command, lines = header.group("node"), "", []
            nodes.setdefault(node, {})
        elif _SECTION.match(line):
            flush()
            command, lines = "", []
        elif found := _COMMAND.match(line):
            flush()
            command, lines = found.group("command"), []
        elif command:
            lines.append(line)
    flush()
    return nodes


def _output_for(blocks: dict[str, str], needle: str) -> str | None:
    """Return the output of the recorded command containing ``needle``."""
    for command, output in blocks.items():
        if needle in command:
            return output
    return None


def _is_usable(output: str | None) -> bool:
    """Whether a block holds real output rather than a failure marker."""
    if not output:
        return False
    stripped = output.strip()
    return bool(stripped) and not stripped.startswith(("[exit ", "[timed out", "[no output"))


def _parse_memory(value: str) -> float | None:
    """Return bytes for an nvidia-smi memory value, or None if unparseable."""
    found = _MEMORY_VALUE.match(value)
    if not found:
        return None
    unit = (found.group("unit") or "MiB").lower().replace("mb", "mib").replace("gb", "gib")
    return float(found.group("value")) * _UNIT_SCALE.get(unit, _UNIT_SCALE["mib"])


def _csv_rows(output: str) -> list[list[str]]:
    """Rows of an nvidia-smi --format=csv block, header dropped."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return [[cell.strip() for cell in line.split(",")] for line in lines[1:]]


def _check_compute_apps(node: str, blocks: dict[str, str]) -> list[Finding]:
    """Any process holding device memory before we start is someone else's."""
    output = _output_for(blocks, "--query-compute-apps")
    if not _is_usable(output):
        return []
    # "No running processes found" and a header-only block both mean clean.
    findings = []
    for row in _csv_rows(output):
        if len(row) < 3:
            continue
        pid, name, used = row[0], row[1], row[2]
        findings.append(
            Finding(node, f"GPU memory held by pid {pid} ({name}), {used} — leftover process")
        )
    return findings


def _check_free_memory(
    node: str, blocks: dict[str, str], gpu_memory_utilization: float | None
) -> list[Finding]:
    """A card with less free memory than the engine will request cannot serve.

    Mirrors the engine's own arithmetic — ``utilization x total`` against free
    memory — so a card that fails here is exactly a card that would fail at
    ``init_device``.
    """
    if gpu_memory_utilization is None:
        return []
    output = _output_for(blocks, "memory.total,memory.used,memory.free")
    if not _is_usable(output):
        return []

    findings = []
    for row in _csv_rows(output):
        if len(row) < 4:
            continue
        index, total, _used, free = row[0], row[1], row[2], row[3]
        free_bytes = _parse_memory(free)
        total_bytes = _parse_memory(total)
        if free_bytes is None or total_bytes is None:
            continue
        required = total_bytes * gpu_memory_utilization
        if free_bytes < required:
            findings.append(
                Finding(
                    node,
                    f"GPU {index} has {free_bytes / GIB:.2f} GiB free of "
                    f"{total_bytes / GIB:.2f} GiB, engine needs {required / GIB:.2f} GiB "
                    f"at gpu_memory_utilization={gpu_memory_utilization:g}",
                )
            )
    return findings


def _check_imex_domain(blocks: dict[str, str]) -> list[Finding]:
    """A node missing from the fabric domain breaks MNNVL for everyone in it.

    Reported even when the node is outside this allocation: the domain is shared,
    and an MNNVL all2all backend addresses it as a whole. The finding is keyed by
    the offending peer rather than the node that observed it, so a domain of
    twelve nodes reports one problem and not twelve.
    """
    output = _output_for(blocks, "nvidia-imex-ctl")
    if not _is_usable(output):
        return []

    findings = []
    for line in output.splitlines():
        found = _IMEX_NODE.match(line.strip())
        if not found or found.group("state") == "READY":
            continue
        host = found.group("host") or found.group("ip")
        findings.append(
            Finding(host, f"IMEX domain node #{found.group('index')} is {found.group('state')}")
        )
    return findings


def _check_fabric(node: str, blocks: dict[str, str]) -> list[Finding]:
    """A GPU that did not join the clique cannot reach remote peers over NVLink."""
    output = _output_for(blocks, "Fabric")
    if not _is_usable(output):
        return []

    findings = []
    gpu = -1
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "Fabric":
            gpu += 1
        elif ":" not in stripped:
            continue
        key, _, value = (part.strip() for part in stripped.partition(":"))
        if key == "State" and value != "Completed":
            findings.append(Finding(node, f"GPU {gpu} fabric State is {value}, expected Completed"))
        elif key == "Status" and value not in ("Success", "N/A"):
            findings.append(Finding(node, f"GPU {gpu} fabric Status is {value}, expected Success"))
        elif key == "Summary" and value not in ("Healthy", "N/A"):
            findings.append(Finding(node, f"GPU {gpu} fabric health is {value}, expected Healthy"))
    return findings


def format_findings(findings: list[Finding], snapshot_path: Path) -> str:
    """One message that says what is wrong, where, and where to look."""
    lines = [
        f"Preflight found {len(findings)} problem(s) that would break this run:",
        *(f"  - {finding}" for finding in findings),
        f"Full snapshot: {snapshot_path}",
        "Set preflight.enabled: false to run anyway.",
    ]
    return "\n".join(lines)
