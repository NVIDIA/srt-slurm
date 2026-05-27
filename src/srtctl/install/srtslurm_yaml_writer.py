# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register model + container aliases in ``srtslurm.yaml``.

Updates are comment-preserving: we round-trip through ``yaml_utils`` so
existing user comments, formatting, and key order survive the edit.
"""

from __future__ import annotations

import fcntl
import logging
import os
from pathlib import Path
import tempfile

from ruamel.yaml.comments import CommentedMap

from srtctl.core.yaml_utils import dump_yaml_with_comments, load_yaml_with_comments

logger = logging.getLogger(__name__)


def _ensure_section(doc: CommentedMap, key: str) -> CommentedMap:
    """Return doc[key] as a CommentedMap, creating it if missing."""
    if key not in doc or doc[key] is None:
        doc[key] = CommentedMap()
    section = doc[key]
    if not isinstance(section, CommentedMap):
        raise ValueError(f"srtslurm.yaml '{key}' is not a mapping (got {type(section).__name__})")
    return section


def _set_alias(section: CommentedMap, alias: str, value: str, *, label: str) -> tuple[str, str | None]:
    """Set ``section[alias] = value``. Returns (action, previous_value) for logging."""
    previous = section.get(alias)
    if previous == value:
        return ("unchanged", previous)
    if previous is None:
        section[alias] = value
        return ("added", None)
    section[alias] = value
    logger.warning("%s alias %r already pointed at %r; overwriting with %r", label, alias, previous, value)
    return ("updated", previous)


def register_aliases(
    srtslurm_yaml: Path,
    *,
    model_alias: str,
    model_path: Path,
    container_alias: str,
    container_path: Path,
) -> dict[str, str]:
    """Add model_paths + containers entries to ``srtslurm.yaml``.

    Returns a small report dict like ``{"model": "added", "container": "updated"}``
    suitable for printing in the CLI summary.
    """
    if not srtslurm_yaml.exists():
        raise FileNotFoundError(
            f"srtslurm.yaml not found at {srtslurm_yaml}. Run `make setup ARCH=<arch>` first."
        )

    lock_path = srtslurm_yaml.with_name(srtslurm_yaml.name + ".lock")
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        doc = load_yaml_with_comments(srtslurm_yaml)

        model_section = _ensure_section(doc, "model_paths")
        model_action, _ = _set_alias(model_section, model_alias, str(model_path), label="model_paths")

        container_section = _ensure_section(doc, "containers")
        container_action, _ = _set_alias(
            container_section, container_alias, str(container_path), label="containers"
        )

        fd, tmp_path = tempfile.mkstemp(prefix=srtslurm_yaml.name + ".", suffix=".tmp", dir=str(srtslurm_yaml.parent))
        os.close(fd)
        try:
            with open(tmp_path, "w") as f:
                dump_yaml_with_comments(doc, f)
            os.replace(tmp_path, srtslurm_yaml)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return {"model": model_action, "container": container_action}
