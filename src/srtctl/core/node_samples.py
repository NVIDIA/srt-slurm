# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-node CSVs and readiness markers written by bare-host collectors.

A host collector (``cpu_power``, ``nic_traffic``) runs once per node and
writes ``<nodes_dir>/<host>.csv`` plus ``<ready_dir>/<host>.ready.json`` or
``<host>.error.json``. The head node reads the markers to learn which nodes
started and merges the CSVs into one file at teardown.
"""

from __future__ import annotations

import contextlib
import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

READY_SUFFIX = ".ready.json"
ERROR_SUFFIX = ".error.json"


@dataclass(frozen=True)
class NodeMarkers:
    """Nodes that published a ready marker, and each node's error-marker message."""

    ready: frozenset[str]
    errors: dict[str, str]


def read_node_markers(ready_dir: Path) -> NodeMarkers:
    ready = frozenset(path.name.removesuffix(READY_SUFFIX) for path in ready_dir.glob(f"*{READY_SUFFIX}"))
    errors: dict[str, str] = {}
    for path in sorted(ready_dir.glob(f"*{ERROR_SUFFIX}")):
        message = ""
        with contextlib.suppress(OSError, json.JSONDecodeError, AttributeError):
            message = str(json.loads(path.read_text()).get("error", ""))
        errors[path.name.removesuffix(ERROR_SUFFIX)] = message
    return NodeMarkers(ready=ready, errors=errors)


def read_node_metadata(nodes_dir: Path) -> dict[str, Any]:
    """Each node's ``<host>.metadata.json``, by host; unreadable files are left out."""
    metadata: dict[str, Any] = {}
    for path in sorted(nodes_dir.glob("*.metadata.json")):
        with contextlib.suppress(OSError, json.JSONDecodeError):
            metadata[path.name.removesuffix(".metadata.json")] = json.loads(path.read_text())
    return metadata


@dataclass(frozen=True)
class MergedNodeCsvs:
    row_count: int
    observed_nodes: tuple[str, ...]
    malformed: bool


def merge_node_csvs(
    nodes_dir: Path,
    header: tuple[str, ...],
    output: Path,
    *,
    sort_key: Callable[[list[str]], Any],
) -> MergedNodeCsvs:
    """Concatenate every ``<nodes_dir>/*.csv`` whose header is ``header`` into ``output``.

    A file with another header, a row of the wrong width, or an unreadable file
    is skipped and reported as ``malformed``; the rows that do parse still merge.
    """
    hostname_column = header.index("hostname")
    rows: list[list[str]] = []
    observed: set[str] = set()
    malformed = False
    for path in sorted(nodes_dir.glob("*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                if tuple(next(reader, ())) != header:
                    malformed = True
                    continue
                for row in reader:
                    if len(row) != len(header):
                        malformed = True
                        continue
                    rows.append(row)
                    observed.add(row[hostname_column])
        except (OSError, csv.Error, UnicodeDecodeError):
            malformed = True
    rows.sort(key=sort_key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return MergedNodeCsvs(row_count=len(rows), observed_nodes=tuple(sorted(observed)), malformed=malformed)
