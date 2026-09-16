# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layout of a run's log directory.

``runtime.log_dir`` (``<output_dir>/<job_id>/logs``, mounted at ``/logs`` in every
container) fans out per-node and per-service output into fixed subdirectories so the
root stays readable::

    logs/
    ├── workers/        <node>_<mode>_w<i>.out, <node>_config.json,
    │                   <node>_frontend_<i>.out, <node>_<router>_<i>.out, <node>_nginx.out
    ├── services/
    │   ├── logs/       service_<name>[_<node>].out, service_<name>.{clone,build}.out
    │   └── <name>/src  git checkouts of source-built services
    ├── telemetry/      telemetry_*.out, tachometer.out, tachometer_config.toml,
    │                   process-exporter.yml
    ├── fingerprints/   fingerprint_<mode>_w<i>.json
    ├── power/, cpu_power/, tachometer/   telemetry data (``*.storage_subdir``, configurable)
    └── sweep_<id>.log, benchmark.out, resource_snapshot.json, ...   per-run singletons

Every writer and reader goes through these constants; nothing else derives the paths.
"""

from __future__ import annotations

from pathlib import Path

CONTAINER_LOG_DIR = Path("/logs")

WORKERS_DIRNAME = "workers"
SERVICES_DIRNAME = "services"
SERVICE_LOGS_DIRNAME = f"{SERVICES_DIRNAME}/logs"
TELEMETRY_DIRNAME = "telemetry"
FINGERPRINTS_DIRNAME = "fingerprints"

LOG_SUBDIRS: tuple[str, ...] = (
    WORKERS_DIRNAME,
    SERVICE_LOGS_DIRNAME,
    TELEMETRY_DIRNAME,
    FINGERPRINTS_DIRNAME,
)


def ensure_log_layout(log_dir: Path) -> None:
    """Create ``log_dir`` and every fixed subdirectory (idempotent).

    Slurm's ``--output`` does not create parent directories, so the layout must exist
    before the first ``srun``.
    """
    for subdir in ("", *LOG_SUBDIRS):
        (log_dir / subdir).mkdir(parents=True, exist_ok=True)


def workers_dir(log_dir: Path) -> Path:
    return Path(log_dir) / WORKERS_DIRNAME


def service_logs_dir(log_dir: Path) -> Path:
    return Path(log_dir) / SERVICE_LOGS_DIRNAME


def telemetry_dir(log_dir: Path) -> Path:
    return Path(log_dir) / TELEMETRY_DIRNAME


def fingerprints_dir(log_dir: Path) -> Path:
    return Path(log_dir) / FINGERPRINTS_DIRNAME


def container_path(subdir: str, name: str) -> str:
    """``/logs/<subdir>/<name>`` as seen inside a container."""
    return str(CONTAINER_LOG_DIR / subdir / name)
