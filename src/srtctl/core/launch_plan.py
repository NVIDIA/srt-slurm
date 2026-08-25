# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist the realized Slurm launch plan for post-run reproduction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SECRET_NAME = re.compile(r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIAL)(?:_|$)")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_secret_name(name: str) -> bool:
    return bool(_SECRET_NAME.search(name.upper()))


def _slug(value: str) -> str:
    normalized = _SAFE_NAME.sub("-", value).strip("-.").lower()
    return normalized[:80] or "step"


def _redacted_command(
    command: list[str],
    *,
    env_to_set: dict[str, str] | None,
    srun_export_env: dict[str, str] | None,
) -> tuple[list[str], list[str]]:
    """Return a replayable command with secret environment values removed."""
    replay = list(command)
    required_secrets: set[str] = set()

    if srun_export_env:
        for idx, arg in enumerate(replay):
            if not arg.startswith("--export=ALL,"):
                continue
            exports = []
            for item in arg.removeprefix("--export=ALL,").split(","):
                name, separator, value = item.partition("=")
                if separator and _is_secret_name(name):
                    exports.append(name)
                    required_secrets.add(name)
                else:
                    exports.append(item)
            replay[idx] = "--export=ALL," + ",".join(exports)

    if env_to_set:
        bash_index = next((idx for idx in range(len(replay) - 1) if replay[idx : idx + 2] == ["bash", "-c"]), None)
        if bash_index is not None:
            bash_command = replay[bash_index + 2]
            for name, value in env_to_set.items():
                if not _is_secret_name(name):
                    continue
                actual = f"export {name}={shlex.quote(value)}"
                replacement = f'export {name}="${{{name}:?Set {name} to replay this step}}"'
                bash_command = bash_command.replace(actual, replacement)
                required_secrets.add(name)
            replay[bash_index + 2] = bash_command

    return replay, sorted(required_secrets)


class LaunchPlanRecorder:
    """Thread-safe recorder for exact, realized ``srun`` invocations."""

    def __init__(self, directory: Path, *, job_id: str | None = None) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.job_id = job_id or os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._write_manifest()

    def record(
        self,
        command: list[str],
        *,
        label: str,
        output: str | None,
        nodelist: list[str] | None,
        het_group: int | None,
        container_image: str | None,
        env_to_set: dict[str, str] | None,
        srun_export_env: dict[str, str] | None,
    ) -> Path:
        replay, required_secrets = _redacted_command(
            command,
            env_to_set=env_to_set,
            srun_export_env=srun_export_env,
        )
        with self._lock:
            sequence = len(self._entries) + 1
            filename = f"{sequence:03d}-{_slug(label)}.sh"
            script_path = self.directory / filename
            rendered = self._render_script(replay, required_secrets=required_secrets)
            script_path.write_text(rendered)
            script_path.chmod(0o750)

            entry = {
                "sequence": sequence,
                "script": filename,
                "recorded_at": _utc_now(),
                "output": output,
                "nodelist": list(nodelist or []),
                "het_group": het_group,
                "container_image": container_image,
                "required_secret_environment": required_secrets,
                "sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            }
            self._entries.append(entry)
            self._write_manifest()
            return script_path

    def _render_script(self, command: list[str], *, required_secrets: list[str]) -> str:
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# Generated from the exact srun argv used by srt-slurm.",
            "# Run inside a compatible active Slurm allocation with the recorded recipe and container available.",
        ]
        if required_secrets:
            lines.append("# Required secret environment (values intentionally omitted): " + ", ".join(required_secrets))
            for name in required_secrets:
                lines.append(f': "${{{name}:?Set {name} to replay this step}}"')
        lines.extend(["", "exec " + shlex.join(command), ""])
        return "\n".join(lines)

    def _write_manifest(self) -> None:
        manifest = {
            "schema_version": 1,
            "job_id": self.job_id,
            "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
            "generated_at": _utc_now(),
            "replay_scope": "Individual realized Slurm steps; orchestration order and lifecycle remain owned by srtctl.",
            "secret_policy": "Secret-like environment values are omitted and named in required_secret_environment.",
            "related_artifacts": [
                "../config.yaml",
                "../sbatch_script.sh",
                "../recipe.lock.yaml",
                "../resource_snapshot.json",
            ],
            "steps": self._entries,
        }
        temporary = self.directory / ".manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.directory / "manifest.json")


_active_recorder: LaunchPlanRecorder | None = None


def configure_launch_plan(directory: Path | None, *, job_id: str | None = None) -> LaunchPlanRecorder | None:
    """Enable or disable process-wide launch recording for this orchestrator."""
    global _active_recorder
    _active_recorder = LaunchPlanRecorder(directory, job_id=job_id) if directory is not None else None
    return _active_recorder


def record_srun_command(
    command: list[str],
    *,
    label: str,
    output: str | None,
    nodelist: list[str] | None,
    het_group: int | None,
    container_image: str | None,
    env_to_set: dict[str, str] | None,
    srun_export_env: dict[str, str] | None,
) -> Path | None:
    """Record a realized command when launch-plan capture is enabled."""
    if _active_recorder is None:
        return None
    return _active_recorder.record(
        command,
        label=label,
        output=output,
        nodelist=nodelist,
        het_group=het_group,
        container_image=container_image,
        env_to_set=env_to_set,
        srun_export_env=srun_export_env,
    )
