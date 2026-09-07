# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic readiness probes: is this process ready, by its own definition?

The worker and frontend health checks in :mod:`srtctl.core.health` know what a
Dynamo or SGLang endpoint looks like. A user-declared service does not have that
luxury, so ``services[].readiness`` picks one of three generic probes:

- ``tcp``: a port accepts a connection,
- ``http``: a URL returns an expected status,
- ``log``: the process log matches a regular expression.

:func:`wait_until_ready` runs one probe until it passes, the deadline expires,
or the process dies, checking liveness between attempts so a crashed service
fails at once instead of after the full timeout.
"""

from __future__ import annotations

import logging
import re
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from srtctl.services.config import HttpProbe, LogProbe, TcpProbe

logger = logging.getLogger(__name__)


class ProcessDied(RuntimeError):
    """The probed process exited before its probe passed."""


def probe_tcp(host: str, port: int, *, connect_timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=connect_timeout):
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


def probe_http(host: str, port: int, path: str, status: int, *, request_timeout: float = 5.0) -> bool:
    try:
        return requests.get(f"http://{host}:{port}{path}", timeout=request_timeout).status_code == status
    except requests.exceptions.RequestException:
        return False


def probe_log(log_file: Path | None, pattern: str) -> bool:
    if log_file is None or not log_file.exists():
        return False
    try:
        text = log_file.read_text(errors="replace")
    except OSError:
        return False
    return re.search(pattern, text, flags=re.MULTILINE) is not None


def run_probe(probe: TcpProbe | HttpProbe | LogProbe, *, host: str, log_file: Path | None) -> bool:
    """Run one probe attempt against ``host`` (and ``log_file`` for log probes)."""
    from srtctl.services.config import HttpProbe, LogProbe, TcpProbe

    if isinstance(probe, TcpProbe):
        return probe_tcp(host, probe.port)
    if isinstance(probe, HttpProbe):
        return probe_http(host, probe.port, probe.path, probe.status)
    if isinstance(probe, LogProbe):
        return probe_log(log_file, probe.pattern)
    raise TypeError(f"unknown probe type {type(probe).__name__}")


def wait_until_ready(
    probe: TcpProbe | HttpProbe | LogProbe,
    *,
    host: str,
    log_file: Path | None,
    timeout: float,
    interval: float,
    is_alive: Callable[[], bool],
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Probe until it passes (True), the deadline passes (False), or the process dies (:class:`ProcessDied`).

    Liveness is checked after every failed attempt, so a process that exits
    never waits out the full ``timeout``.
    """
    deadline = clock() + timeout
    while True:
        if run_probe(probe, host=host, log_file=log_file):
            return True
        if not is_alive():
            raise ProcessDied("process exited before its readiness probe passed")
        remaining = deadline - clock()
        if remaining <= 0:
            return False
        sleep(min(interval, remaining))
