# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ports the jobs sharing a cluster's nodes hold, so a new job can allocate around them.

Every job starts its port kinds at the same bases. While each job has its nodes to
itself that is enough; two jobs on one node both bind 6100, 17500, ... and the
second one's bind fails. With ``port_registry_dir`` set in srtslurm.yaml, the
orchestrator reads the ports every live job holds on its nodes before it
allocates (``NodePortAllocator.occupied``), then publishes its own.

The registry is a directory on a filesystem every orchestrator host shares: one
JSON file per job (``job-<id>.json``, ``{node: [port, ...]}``) and a lock file
that serializes reading and publishing. A job's file is live while the
orchestrator that wrote it holds a POSIX record lock on it; the kernel drops the
lock when that process ends, so a killed job leaves nothing that has to be
cleaned up by hand: the next reader finds the file unlocked and removes it.
Record locks (``fcntl.lockf``) rather than ``flock``, because on NFS ``flock``
is emulated and may not be seen by other hosts.
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import time
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

REGISTRY_LOCK_NAME = "registry.lock"
POLL_SECONDS = 1.0


def _job_file(directory: Path, job_id: str) -> Path:
    return directory / f"job-{job_id}.json"


def _try_lock(descriptor: int) -> bool:
    """Take an exclusive record lock without waiting; False when another process holds it."""
    try:
        fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EAGAIN):
            return False
        raise
    return True


def live_reservations(directory: Path, nodes: Sequence[str], job_id: str) -> dict[str, frozenset[int]]:
    """The ports every other live job holds on ``nodes``; removes the files of jobs that ended.

    Call it with the registry lock held (``reserve_ports``), so no job is publishing meanwhile.
    """
    wanted = set(nodes)
    occupied: dict[str, set[int]] = {}
    for path in sorted(directory.glob("job-*.json")):
        if path == _job_file(directory, job_id):
            continue
        try:
            descriptor = os.open(path, os.O_RDWR)
        except FileNotFoundError:
            continue
        try:
            if _try_lock(descriptor):
                # Nobody holds it: the job that wrote it is gone.
                path.unlink(missing_ok=True)
                continue
            with os.fdopen(os.dup(descriptor), encoding="utf-8") as handle:
                held = json.load(handle)
        finally:
            # Closing any descriptor of a file drops this process's locks on it,
            # which is only ever the stale lock taken above.
            os.close(descriptor)
        for node, ports in held.items():
            if node in wanted:
                occupied.setdefault(node, set()).update(ports)
    return {node: frozenset(ports) for node, ports in occupied.items()}


@dataclass
class PortLease:
    """This job's entry in the registry, held until ``release`` or until the process ends."""

    directory: Path
    job_id: str
    _descriptor: int | None = field(default=None, repr=False)

    @property
    def path(self) -> Path:
        return _job_file(self.directory, self.job_id)

    def publish(self, ports_by_node: Mapping[str, frozenset[int]]) -> None:
        """Record the ports this job binds on each node and hold the record for the job's life.

        The file is locked before it is written, so a reader never mistakes it for a
        dead job's file.
        """
        if self._descriptor is None:
            self._descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o664)
            fcntl.lockf(self._descriptor, fcntl.LOCK_EX)
        payload = json.dumps({node: sorted(ports) for node, ports in sorted(ports_by_node.items())})
        os.ftruncate(self._descriptor, 0)
        os.pwrite(self._descriptor, payload.encode(), 0)
        os.fsync(self._descriptor)

    def release(self) -> None:
        """Remove this job's entry; its ports are free for the next job."""
        if self._descriptor is None:
            return
        self.path.unlink(missing_ok=True)
        os.close(self._descriptor)
        self._descriptor = None


@contextmanager
def reserve_ports(
    directory: Path, job_id: str, nodes: Sequence[str], timeout_s: float = 600.0
) -> Generator[tuple[PortLease, dict[str, frozenset[int]]]]:
    """Hold the registry lock and yield this job's lease with the ports other jobs hold on ``nodes``.

    Allocate with the second value as ``NodePortAllocator.occupied`` and call
    ``lease.publish`` before leaving the block: the lock keeps another job from
    reading the registry between the two, which is what lets two jobs that start
    at the same moment on one node end up with different ports.
    """
    directory.mkdir(parents=True, exist_ok=True)
    lock = os.open(directory / REGISTRY_LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o664)
    try:
        deadline = time.monotonic() + timeout_s
        while not _try_lock(lock):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"port registry {directory} stayed locked for {timeout_s:.0f}s")
            time.sleep(POLL_SECONDS)
        occupied = live_reservations(directory, nodes, job_id)
        if occupied:
            logger.info(
                "Port registry: other jobs hold %s",
                ", ".join(f"{node} ({len(ports)} ports)" for node, ports in sorted(occupied.items())),
            )
        yield PortLease(directory, job_id), occupied
    finally:
        os.close(lock)
