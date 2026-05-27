# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NATS/ETCD bootstrap detection.

The existing ``make setup`` target downloads NATS and ETCD binaries (and
prompts for ``srtslurm.yaml`` on first run). ``srtctl install`` reuses that
target instead of reimplementing its logic — we only detect whether it has
already run and shell out otherwise.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def srtctl_root_from_package() -> Path:
    """Derive the repo root from this file's location.

    Layout: ``<root>/src/srtctl/install/setup.py``.
    """
    return Path(__file__).resolve().parents[3]


def detect_arch() -> str:
    """Return ``ARCH`` value compatible with the Makefile's ``setup`` target."""
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    if machine in {"aarch64", "arm64"}:
        return "aarch64"
    raise RuntimeError(f"Unsupported architecture for srtctl bootstrap: {machine}")


def is_bootstrapped(srtctl_root: Path) -> bool:
    """True iff NATS/ETCD binaries are executable and ``srtslurm.yaml`` exists."""
    configs = srtctl_root / "configs"
    nats = configs / "nats-server"
    etcd = configs / "etcd"
    srtslurm_yaml = srtctl_root / "srtslurm.yaml"
    return (
        nats.is_file()
        and os.access(nats, os.X_OK)
        and etcd.is_file()
        and os.access(etcd, os.X_OK)
        and srtslurm_yaml.is_file()
    )


def run_make_setup(srtctl_root: Path, arch: str) -> None:
    """Invoke ``make setup ARCH=<arch>`` with stdio inherited.

    Inherits stdio so the user can respond to interactive prompts (the
    Makefile prompts for SLURM account/partition/etc. when creating a new
    ``srtslurm.yaml``).
    """
    cmd = ["make", "setup", f"ARCH={arch}"]
    logger.info("Running bootstrap: %s (cwd=%s)", " ".join(cmd), srtctl_root)
    subprocess.run(cmd, cwd=srtctl_root, check=True)


def ensure_bootstrapped(srtctl_root: Path, arch: str | None = None) -> bool:
    """Run ``make setup`` if NATS/ETCD/srtslurm.yaml are missing.

    Returns True if bootstrap was executed, False if everything was already in place.
    """
    if is_bootstrapped(srtctl_root):
        logger.info("Bootstrap already complete (NATS, ETCD, srtslurm.yaml present).")
        return False
    run_make_setup(srtctl_root, arch or detect_arch())
    return True
