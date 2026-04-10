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

import contextlib
import getpass
import logging
import os
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

# SLURM environment variables to capture in the lockfile.
# Each entry is (yaml_key, env_var).
_SLURM_ENV_KEYS = [
    ("job_id", "SLURM_JOB_ID"),
    ("job_name", "SLURM_JOB_NAME"),
    ("cluster", "SLURM_CLUSTER_NAME"),
    ("account", "SLURM_JOB_ACCOUNT"),
    ("partition", "SLURM_JOB_PARTITION"),
    ("nodelist", "SLURM_JOB_NODELIST"),
    ("num_nodes", "SLURM_JOB_NUM_NODES"),
    ("gpus_per_node", "SLURM_GPUS_PER_NODE"),
    ("time_limit", "SLURM_TIMELIMIT"),
]


def collect_slurm_context() -> dict[str, Any]:
    """Collect SLURM job context from environment variables.

    Captures job ID, account, partition, nodelist, etc. — everything needed
    to understand where and how the job ran. Returns an empty dict outside SLURM.
    """
    ctx: dict[str, Any] = {}

    for key, env_var in _SLURM_ENV_KEYS:
        val = os.environ.get(env_var)
        if val is not None:
            ctx[key] = val

    # User and working directory (always available)
    with contextlib.suppress(Exception):
        ctx["user"] = getpass.getuser()

    ctx["cwd"] = str(Path.cwd())

    # srtctl installation root (where the tool was invoked from)
    srtctl_root = os.environ.get("SRTCTL_ROOT")
    if srtctl_root:
        ctx["srtctl_root"] = srtctl_root

    return ctx


def collect_worker_fingerprints(log_dir: Path) -> dict[str, Any] | None:
    """Load per-worker fingerprint files into a dict keyed by worker name.

    Returns a dict like:
        {"prefill_w0": {...}, "decode_w0": {...}, "decode_w1": {...}}

    The key is derived from the filename: fingerprint_prefill_w0.json -> "prefill_w0".
    Returns None if no fingerprint files are found or all fail to load.
    """
    try:
        fp_files = sorted(log_dir.glob("fingerprint_*.json"))
    except Exception as e:
        logger.debug("Failed to glob fingerprint files in %s: %s", log_dir, e)
        return None

    if not fp_files:
        return None

    result: dict[str, Any] = {}
    for fp_file in fp_files:
        fp = load_fingerprint(fp_file)
        if fp is not None:
            # fingerprint_prefill_w0.json -> prefill_w0
            worker_key = fp_file.stem.removeprefix("fingerprint_")
            result[worker_key] = fp

    return result if result else None


def build_lockfile(
    config: SrtConfig,
    worker_fingerprints: dict[str, Any] | None = None,
    resolved_log_dir: Path | None = None,
    verification: list[Any] | None = None,
) -> dict[str, Any]:
    """Build the lockfile dict from a resolved config and optional per-worker fingerprints.

    Returns a dict with (in order):
    - _meta: lockfile version, timestamp, SLURM context
    - verification: identity check results (pass/fail for each declared field)
    - config: the full resolved config as a dict
    - fingerprints: per-worker fingerprints keyed by worker name (or None)

    Args:
        config: The SrtConfig to serialize.
        worker_fingerprints: Per-worker fingerprints keyed by worker name.
        resolved_log_dir: The actual resolved log directory path. If provided,
            overrides the template string in config.output.log_dir so the lockfile
            records where logs actually went (not the unresolved template).
        verification: List of IdentityCheckResult from verify_identity().
    """
    from srtctl.core.schema import SrtConfig

    config_dict = SrtConfig.Schema().dump(config)

    # Replace template log_dir with the resolved path
    if resolved_log_dir is not None and "output" in config_dict:
        config_dict["output"]["log_dir"] = str(resolved_log_dir)

    # Build verification summary
    verification_dict = None
    if verification:
        passes = [r for r in verification if r.passed]
        fails = [r for r in verification if not r.passed]
        verification_dict = {
            "result": "all OK" if not fails else f"{len(fails)} FAILED",
            "passed": len(passes),
            "failed": len(fails),
            "checks": [
                {"field": r.field, "status": "OK" if r.passed else "FAIL", "message": r.message} for r in verification
            ],
        }

    return {
        "_meta": {
            "version": _LOCKFILE_VERSION,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "slurm": collect_slurm_context(),
        },
        "verification": verification_dict,
        "config": config_dict,
        "fingerprints": worker_fingerprints,
    }


def write_lockfile(
    output_dir: Path,
    config: SrtConfig,
    log_dir: Path | None = None,
    verification: list[Any] | None = None,
) -> bool:
    """Write recipe.lock.yaml to the output directory.

    Called twice per job:
    1. At job start (log_dir=None) — writes config + SLURM context, fingerprints=null
    2. At job end (log_dir set) — rewrites with per-worker fingerprints + verification

    Returns True on success, False on any failure. Never raises.
    """
    try:
        fingerprints = collect_worker_fingerprints(log_dir) if log_dir else None
        lockfile_data = build_lockfile(config, fingerprints, resolved_log_dir=log_dir, verification=verification)

        lockfile_path = output_dir / "recipe.lock.yaml"
        lockfile_path.write_text(yaml.dump(lockfile_data, default_flow_style=False, sort_keys=False))
        logger.info("Wrote lockfile: %s", lockfile_path)
        return True
    except Exception as e:
        logger.warning("Failed to write lockfile: %s", e)
        return False


def load_lockfile_fingerprints(path: Path) -> dict[str, Any] | None:
    """Load per-worker fingerprints from a lockfile, output directory, or raw JSON.

    Accepts:
    - Path to recipe.lock.yaml → reads the 'fingerprints' section (per-worker dict)
    - Path to an output directory → looks for recipe.lock.yaml or raw fingerprint files
    - Path to a single fingerprint JSON → wraps as {"worker": fingerprint}

    Returns a dict keyed by worker name, e.g.:
        {"prefill_w0": {...}, "decode_w0": {...}}
    Returns None if no fingerprints can be loaded.
    """
    try:
        if path.is_dir():
            lockfile = path / "recipe.lock.yaml"
            if lockfile.exists():
                return _load_fingerprints_from_lockfile(lockfile)
            # Fall back to collecting raw fingerprint files
            logs_dir = path / "logs"
            if logs_dir.is_dir():
                return collect_worker_fingerprints(logs_dir)
            return collect_worker_fingerprints(path)

        if path.suffix in (".yaml", ".yml"):
            return _load_fingerprints_from_lockfile(path)

        if path.suffix == ".json":
            fp = load_fingerprint(path)
            if fp is not None:
                # Single file — derive worker key from filename
                worker_key = path.stem.removeprefix("fingerprint_") or "worker"
                return {worker_key: fp}
            return None

        return None
    except Exception as e:
        logger.debug("Failed to load fingerprints from %s: %s", path, e)
        return None


def _load_fingerprints_from_lockfile(path: Path) -> dict[str, Any] | None:
    """Extract the per-worker fingerprints from a lockfile YAML."""
    try:
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            return None
        # Support both 'fingerprints' (new, per-worker) and 'fingerprint' (old, single)
        fps = data.get("fingerprints")
        if isinstance(fps, dict):
            return fps
        fp = data.get("fingerprint")
        if isinstance(fp, dict):
            return {"worker": fp}
        return None
    except Exception as e:
        logger.debug("Failed to parse lockfile %s: %s", path, e)
        return None
