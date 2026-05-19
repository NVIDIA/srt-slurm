# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container ``.sqsh`` import via enroot."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def container_filename(image_ref: str) -> str:
    """Convert a Docker image reference to enroot's default .sqsh filename.

    Mirrors enroot's substitution (``/`` and ``:`` → ``+``).
    """
    return image_ref.replace("/", "+").replace(":", "+") + ".sqsh"


def _check_registry_credentials(image_ref: str) -> tuple[bool, str]:
    """Check whether common nvcr.io credential files are present.

    Enroot can source credentials from different places depending on cluster
    configuration, but the two most common user-visible locations are:
      - ~/.config/enroot/.credentials
      - ~/.docker/config.json
    """
    if not image_ref.startswith("nvcr.io/"):
        return (True, "")

    enroot_creds = Path.home() / ".config" / "enroot" / ".credentials"
    docker_creds = Path.home() / ".docker" / "config.json"
    if enroot_creds.exists() or docker_creds.exists():
        return (True, "")

    return (
        False,
        "Missing registry credentials for nvcr.io image pull. "
        "Configure NGC credentials (for example with enroot/docker login) so either "
        f"{enroot_creds} or {docker_creds} exists.",
    )


def import_container(
    image_ref: str,
    dest_path: Path,
    *,
    force: bool = False,
    strict_auth_preflight: bool = False,
) -> Path:
    """Run ``enroot import`` to materialise ``image_ref`` as a ``.sqsh`` at ``dest_path``.

    Skips the download if ``dest_path`` already exists (unless ``force``).
    Raises if ``enroot`` is not on PATH.
    """
    if dest_path.exists() and not force:
        logger.info("Container already present, skipping import: %s", dest_path)
        return dest_path

    if shutil.which("enroot") is None:
        raise RuntimeError(
            "`enroot` not found on PATH. Install enroot or run `srtctl install` "
            "on a node where enroot is available."
        )
    ok, reason = _check_registry_credentials(image_ref)
    if not ok and strict_auth_preflight:
        raise RuntimeError(reason)
    if not ok:
        logger.warning("%s Proceeding with enroot import attempt.", reason)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    docker_url = f"docker://{image_ref}"
    cmd = ["enroot", "import", "-o", str(dest_path), docker_url]
    logger.info("Importing container: %s → %s", docker_url, dest_path)
    subprocess.run(cmd, check=True)
    return dest_path
