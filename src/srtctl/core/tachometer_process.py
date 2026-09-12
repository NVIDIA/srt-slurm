# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deliver Tachometer's shutdown signal to its task, keeping the srun client alive."""

from __future__ import annotations

import logging
import os
import re
import secrets
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from srtctl.core.processes import ManagedProcess

logger = logging.getLogger(__name__)

# Positional arguments preserve the scraper argv without shell interpolation.
# The shell becomes the scraper; no intermediate shell remains to catch signals.
_STEP_WRAPPER = """set -euo pipefail
umask 077
metadata=$1
token=$2
expected_job=$3
shift 3
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_STEP_ID:-} =~ ^[0-9]+$ ]]
[[ $SLURM_JOB_ID == "$expected_job" ]]
set -C
printf '%s %s %s\\n' "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$token" > "$metadata.tmp"
ln -- "$metadata.tmp" "$metadata"
rm -- "$metadata.tmp"
exec "$@"
"""


@dataclass(frozen=True)
class TachometerStep:
    """Private, single-launch handoff from a task in the expected allocation.

    Tachometer runs on the head node (component 0 for heterogeneous jobs).
    A different component job ID is rejected rather than guessed from offsets.
    The run log directory must be shared with that node, as for its config.
    """

    path: Path
    token: str
    job_id: str

    @classmethod
    def create(cls, log_dir: Path, job_id: str) -> TachometerStep:
        if not re.fullmatch(r"[0-9]+", job_id):
            raise ValueError("Tachometer requires a numeric Slurm allocation ID")
        directory = Path(tempfile.mkdtemp(prefix=".tachometer-step-", dir=log_dir))
        return cls(directory / "identity", secrets.token_hex(16), job_id)

    def command(self, scraper_command: list[str]) -> list[str]:
        return [
            "bash",
            "-c",
            _STEP_WRAPPER,
            "tachometer-step",
            str(self.path),
            self.token,
            self.job_id,
            *scraper_command,
        ]

    def read(self) -> str:
        """Return only an owned numeric JOB.STEP from this launch; otherwise fail closed."""
        directory = self.path.parent.lstat()
        if not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.getuid() or directory.st_mode & 0o077:
            raise ValueError("Tachometer step directory is not private and owned")
        fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "r", encoding="ascii") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
                raise ValueError("Tachometer step identity is not a private owned file")
            contents = handle.read(256)
        match = re.fullmatch(r"([0-9]+) ([0-9]+) ([0-9a-f]{32})\n", contents)
        if match is None or match[1] != self.job_id or match[3] != self.token:
            raise ValueError("Tachometer step identity does not match this launch")
        return f"{match[1]}.{match[2]}"


@dataclass
class TachometerProcess(ManagedProcess):
    """Scoped shutdown also applies when ProcessRegistry calls terminate()."""

    step: TachometerStep | None = None
    shutdown_grace_secs: float = 120.0

    def _signal_step(self, step_id: str, signal: str) -> bool:
        try:
            result = subprocess.run(
                ["scancel", f"--signal={signal}", step_id],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.error("Tachometer step %s: scancel %s failed: %s", step_id, signal, exc)
            return False
        if result.returncode != 0:
            logger.error(
                "Tachometer step %s: scancel %s exited %d: %s", step_id, signal, result.returncode, result.stderr
            )
            return False
        logger.info("Sent SIG%s to Tachometer step %s", signal, step_id)
        return True

    def terminate(self, timeout: float | None = None) -> None:
        """Signal the recorded step, wait for compaction, then escalate only that step.

        Never signal the srun client: its SIGTERM handler aborts the remote task
        with SIGKILL. Unverifiable metadata or a failed scancel leaves the client
        alive and reports failure instead of risking another step/allocation.
        """
        returncode = self.popen.poll()
        if returncode is not None:
            if returncode != 0:
                logger.warning("Tachometer launcher had already exited with status %d", returncode)
            return
        try:
            if self.step is None:
                raise ValueError("Tachometer has no step identity")
            step_id = self.step.read()
        except (OSError, ValueError) as exc:
            logger.error("Cannot safely stop Tachometer; leaving srun alive: %s", exc)
            return
        if not self._signal_step(step_id, "TERM"):
            return
        grace = self.shutdown_grace_secs if timeout is None else timeout
        try:
            returncode = self.popen.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            logger.warning("Tachometer step %s exceeded %.0fs after SIGTERM; capture may be partial", step_id, grace)
            if not self._signal_step(step_id, "KILL"):
                return
            try:
                returncode = self.popen.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                logger.error("Tachometer launcher was not reaped after step %s SIGKILL", step_id)
                return
            logger.warning("Tachometer step %s required SIGKILL; launcher exited with status %d", step_id, returncode)
            return
        if returncode != 0:
            logger.warning(
                "Tachometer step %s exited with status %d after SIGTERM; inspect %s", step_id, returncode, self.log_file
            )
        else:
            logger.info("Tachometer step %s exited with status 0 after SIGTERM", step_id)
