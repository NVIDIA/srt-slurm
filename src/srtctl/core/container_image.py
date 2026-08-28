# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare one reusable SquashFS image before a job starts its Slurm steps."""

import fcntl
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_DIGEST_RE = re.compile(r"@sha256:([0-9a-fA-F]{64})$")
_SQUASHFS_MAGICS = {b"hsqs", b"sqsh"}


def prepare_container_image(image: str, cache_root: str, *, job_id: str, node: str) -> Path:
    """Materialize a digest-pinned registry image once and return its shared path."""
    match = _DIGEST_RE.search(image)
    if match is None:
        raise ValueError("container caching requires model.container to end with @sha256:<digest>")

    digest = match.group(1).lower()
    system = platform.system().lower()
    machine = platform.machine().lower()
    machine = {"aarch64": "arm64", "x86_64": "amd64"}.get(machine, machine)
    cache_dir = Path(os.path.expandvars(cache_root)).expanduser() / "v1" / f"{system}-{machine}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_path = cache_dir / f"{digest}.sqsh"

    with image_path.with_suffix(".lock").open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if image_path.exists():
            _validate_squashfs(image_path)
            logger.info("Container cache hit: %s", image_path)
            return image_path

        srun = shutil.which("srun")
        if srun is None:
            raise RuntimeError("container caching requires srun on the orchestration node")

        logger.info("Container cache miss: importing sha256:%s", digest)
        with tempfile.TemporaryDirectory(prefix=f".{digest}.", dir=cache_dir) as temporary_dir:
            temporary_path = Path(temporary_dir) / "image.sqsh"
            result = subprocess.run(
                [
                    srun,
                    "--jobid",
                    job_id,
                    "--overlap",
                    "--nodes",
                    "1",
                    "--ntasks",
                    "1",
                    "--nodelist",
                    node,
                    "--container-image",
                    image,
                    "--no-container-entrypoint",
                    "--no-container-mount-home",
                    f"--container-save={temporary_path}",
                    "/bin/true",
                ],
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Pyxis container import failed with exit code {result.returncode}")
            _validate_squashfs(temporary_path)
            temporary_path.replace(image_path)

        logger.info("Container cached: %s", image_path)
        return image_path


def _validate_squashfs(path: Path) -> None:
    """Reject partial files and symlinks before they are used as containers."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 4:
        raise RuntimeError(f"container cache entry is not a regular SquashFS file: {path}")
    with path.open("rb") as image_file:
        if image_file.read(4) not in _SQUASHFS_MAGICS:
            raise RuntimeError(f"container cache entry is not a SquashFS file: {path}")
