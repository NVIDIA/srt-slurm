# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit and manage complete host-mode jobs through passwordless OpenSSH."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from rich.console import Console

from srtctl.core.config import load_cluster_config
from srtctl.core.schema import SSHConfig
from srtctl.runtime_scripts import ssh_job
from srtctl.runtime_scripts.ssh_job import JobState, Request


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The job ID and target stay explicit so a command cannot accidentally address a local Slurm job."""
    commands = parser.add_subparsers(dest="remote_action", required=True)
    commands.add_parser("list", help="List SSH submissions on the configured target")
    status = commands.add_parser("status", help="Read an SSH job's state and output directory")
    status.add_argument("job_id", help="ssh-<id> printed by apply")
    logs = commands.add_parser("logs", help="Read the saved apply log on the SSH target")
    logs.add_argument("job_id", help="ssh-<id> printed by apply")
    logs.add_argument(
        "--follow", action="store_true", help="Follow output until the job ends; Ctrl+C leaves it running"
    )
    stop = commands.add_parser("stop", help="Request graceful shutdown through the remote job supervisor")
    stop.add_argument("job_id", help="ssh-<id> printed by apply")


@dataclass(frozen=True)
class SSHClient:
    """One SSH destination, with no local shell, environment forwarding, or password prompts."""

    config: SSHConfig

    def command(self) -> list[str]:
        source = Path(ssh_job.__file__).read_text(encoding="utf-8")
        remote_command = "exec " + shlex.join([self.config.python, "-u", "-c", source])
        return [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
            "--",
            self.config.host,
            remote_command,
        ]

    def request(self, action: str, job_id: str = "", *, follow: bool = False) -> Request:
        return {
            "action": action,
            "workdir": self.config.workdir,
            "job_id": job_id,
            "uv": self.config.uv,
            "recipe": "",
            "selector": None,
            "cluster_config": {},
            "args": [],
            "follow": follow,
            "source": Path(ssh_job.__file__).read_text(encoding="utf-8") if action == "submit" else "",
        }

    def run(self, request: Request) -> JobState | list[JobState] | None:
        streaming = request["action"] == "logs"
        try:
            result = subprocess.run(
                self.command(),
                input=json.dumps(request),
                text=True,
                capture_output=not streaming,
                timeout=None if streaming else 30,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "SSH request timed out; check 'srtctl remote list' before retrying a submission"
            ) from exc
        if result.returncode:
            detail = result.stderr.strip() if result.stderr else f"exit {result.returncode}"
            raise RuntimeError(f"SSH {request['action']} on {self.config.host}: {detail}")
        if streaming:
            return None
        try:
            return cast(JobState | list[JobState], json.loads(result.stdout))
        except ValueError as exc:
            raise RuntimeError("SSH returned no valid job response; check remote shell startup output") from exc


def _client(cluster: dict[str, Any] | None) -> SSHClient | None:
    settings = (cluster or {}).get("ssh")
    return SSHClient(cast(SSHConfig, SSHConfig.Schema().load(settings))) if settings is not None else None


def _apply_arguments(args: argparse.Namespace) -> list[str]:
    """Forward parsed arguments as words; paths and override values are interpreted only remotely."""
    if args.mock_mode:
        raise ValueError("SSH submission does not support --mock; omit ssh to run a local mock")
    forwarded: list[str] = []
    if args.output_dir is not None:
        forwarded += ["--output", str(args.output_dir)]
    for flag, value in (("--tags", args.tags), ("--setup-script", args.setup_script)):
        if value is not None:
            forwarded += [flag, value]
    for flag, enabled in (
        ("--sweep", args.sweep),
        ("--serve-only", args.serve_only),
        ("--no-preflight", args.no_preflight),
    ):
        if enabled:
            forwarded.append(flag)
    for value in args.set_overrides:
        forwarded += ["--set", value]
    for value in args.unset_overrides:
        forwarded += ["--unset", value]
    return forwarded


def submit_over_ssh(args: argparse.Namespace, config_path: Path, selector: str | None, console: Console) -> int | None:
    """Dispatch before local preflight, model resolution, source pinning, setup checks or output staging."""
    cluster = load_cluster_config()
    client = _client(cluster)
    if client is None:
        return None
    if args.command not in ("apply", "dry-run"):
        raise ValueError(
            f"SSH host jobs support apply and dry-run, not {args.command}; use srtctl remote for job control"
        )
    if not config_path.is_file():
        raise ValueError("SSH apply expects one recipe file (override selectors and --sweep are supported)")
    if args.command == "dry-run":
        console.print("SSH host job plan (no connection or remote files created)", markup=False)
        for label, value in (
            ("Target", client.config.host),
            ("Remote checkout", client.config.workdir),
            ("Execution", "existing host launcher using the remote environment"),
            ("Remote uv", client.config.uv),
            ("Recipe to transfer", str(config_path)),
            ("Selector", selector or "all"),
            ("Overrides", shlex.join(args.set_overrides) or "none"),
            ("Unset fields", shlex.join(args.unset_overrides) or "none"),
            ("Sweep", "yes" if args.sweep else "no"),
            (
                "Output",
                str(args.output_dir)
                if args.output_dir
                else (cluster or {}).get("output_dir") or "remote job directory",
            ),
        ):
            console.print(f"  {label}: {value}", markup=False, highlight=False)
        console.print("Recipe validation, preflight and engine startup run on the target.", markup=False)
        return 0
    request = client.request("submit", "ssh-" + uuid.uuid4().hex)
    request["recipe"] = config_path.read_text(encoding="utf-8")
    request["selector"] = selector
    request["cluster_config"] = {**(cluster or {}), "ssh": None}
    request["args"] = _apply_arguments(args)
    state = cast(JobState, client.run(request))
    record = {**state, "ssh_host": client.config.host}
    if args.json_output:
        print(json.dumps(record), flush=True)
    else:
        job_id = state["job_id"]
        console.print(f"SSH job {job_id} on {client.config.host}: {state['state']}", markup=False)
        console.print(f"  Remote output: {state['output_dir']}", markup=False)
        console.print(f"  Status: srtctl remote status {job_id}", markup=False)
        console.print(f"  Logs:   srtctl remote logs {job_id} --follow", markup=False)
        console.print(f"  Stop:   srtctl remote stop {job_id}", markup=False)
    return 1 if state["state"] == "failed" else 0


def run_remote_command(args: argparse.Namespace) -> int:
    client = _client(load_cluster_config())
    if client is None:
        raise ValueError("srtctl remote needs launcher: host and ssh in srtslurm.yaml")
    request = client.request(args.remote_action, getattr(args, "job_id", ""), follow=getattr(args, "follow", False))
    result = client.run(request)
    if result is not None:
        print(json.dumps(result, indent=2), flush=True)
    return 0
