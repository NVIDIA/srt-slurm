# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lockfile generation for reproducible benchmark runs.

After a run completes, the lockfile captures the fully-resolved recipe config
plus the aggregated runtime fingerprint (pip freeze, GPU info, etc.) from
per-worker fingerprint files. This is the "exactly what ran" artifact.

All operations are fault-tolerant — lockfile writing never blocks or fails a job.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from srtctl.core.fingerprint import load_fingerprint

if TYPE_CHECKING:
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

# Lockfile format version — bump when the structure changes
_LOCKFILE_VERSION = 1


def aggregate_fingerprints(log_dir: Path) -> dict[str, Any] | None:
    """Aggregate per-worker fingerprint files into a single fingerprint.

    Reads all fingerprint_*.json files from the log directory. Scalar fields
    are taken from the first file. pip_packages are merged (sorted union).

    Returns None if no fingerprint files are found or all fail to load.
    """
    try:
        fp_files = sorted(log_dir.glob("fingerprint_*.json"))
    except Exception as e:
        logger.debug("Failed to glob fingerprint files in %s: %s", log_dir, e)
        return None

    if not fp_files:
        return None

    fingerprints = []
    for fp_file in fp_files:
        fp = load_fingerprint(fp_file)
        if fp is not None:
            fingerprints.append(fp)

    if not fingerprints:
        return None

    # Use first fingerprint as base for scalar fields
    result = {k: v for k, v in fingerprints[0].items() if k != "pip_packages"}

    # Merge pip packages: sorted union across all workers
    all_packages: set[str] = set()
    for fp in fingerprints:
        for pkg in fp.get("pip_packages", []):
            all_packages.add(pkg)
    result["pip_packages"] = sorted(all_packages, key=lambda s: s.lower())

    return result


def build_lockfile(config: SrtConfig, runtime_fingerprint: dict[str, Any] | None) -> dict[str, Any]:
    """Build the lockfile dict from a resolved config and aggregated fingerprint.

    Returns a dict with:
    - _meta: lockfile version and generation timestamp
    - config: the full resolved config as a dict
    - fingerprint: the aggregated runtime fingerprint (or None)
    """
    # Dump config to dict via marshmallow schema
    from srtctl.core.schema import SrtConfig

    config_dict = SrtConfig.Schema().dump(config)

    return {
        "_meta": {
            "version": _LOCKFILE_VERSION,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "config": config_dict,
        "fingerprint": runtime_fingerprint,
    }


def write_lockfile(output_dir: Path, config: SrtConfig, log_dir: Path) -> bool:
    """Aggregate fingerprints and write recipe.lock.yaml to the output directory.

    Returns True on success, False on any failure. Never raises.
    """
    try:
        fingerprint = aggregate_fingerprints(log_dir)
        lockfile_data = build_lockfile(config, fingerprint)

        lockfile_path = output_dir / "recipe.lock.yaml"
        lockfile_path.write_text(yaml.dump(lockfile_data, default_flow_style=False, sort_keys=False))
        logger.info("Wrote lockfile: %s", lockfile_path)
        return True
    except Exception as e:
        logger.warning("Failed to write lockfile: %s", e)
        return False


def load_lockfile_fingerprint(path: Path) -> dict[str, Any] | None:
    """Load a fingerprint from a lockfile, output directory, or raw JSON.

    Accepts:
    - Path to recipe.lock.yaml → reads the 'fingerprint' section
    - Path to an output directory → looks for recipe.lock.yaml inside
    - Path to a fingerprint JSON file → loads directly

    Returns None if the fingerprint cannot be loaded.
    """
    try:
        # If it's a directory, look for lockfile or fingerprint files
        if path.is_dir():
            lockfile = path / "recipe.lock.yaml"
            if lockfile.exists():
                return _load_fingerprint_from_lockfile(lockfile)
            # Fall back to aggregating raw fingerprint files from logs/
            logs_dir = path / "logs"
            if logs_dir.is_dir():
                return aggregate_fingerprints(logs_dir)
            return aggregate_fingerprints(path)

        # If it's a YAML file, try loading as lockfile
        if path.suffix in (".yaml", ".yml"):
            return _load_fingerprint_from_lockfile(path)

        # Otherwise try loading as raw fingerprint JSON
        if path.suffix == ".json":
            return load_fingerprint(path)

        return None
    except Exception as e:
        logger.debug("Failed to load fingerprint from %s: %s", path, e)
        return None


def _load_fingerprint_from_lockfile(path: Path) -> dict[str, Any] | None:
    """Extract the fingerprint section from a lockfile YAML."""
    try:
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            return data.get("fingerprint")
        return None
    except Exception as e:
        logger.debug("Failed to parse lockfile %s: %s", path, e)
        return None
