# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct Docker host runner for the ``srtctl apply --bash`` artifact.

This module deliberately uses only the standard library. The rendered Bash can
therefore invoke it from a source checkout without requiring the control-plane
environment on the GPU host. Docker lifecycle belongs here; serving lifecycle
belongs in :mod:`direct_runner`, inside the selected SGLang image.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TextIO


class DirectHostInterrupted(Exception):
    """Signal delivered while the host runner owns Docker containers."""

    def __init__(self, signal_number: int) -> None:
        self.signal_number = signal_number
        super().__init__(f"received signal {signal_number}")


class DirectHostRunner:
    """Own Docker containers and invoke the in-container direct runner."""

    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.container_plan = self._mapping(plan, "direct_plan")
        self.name = self._string(plan, "name")
        self.source_dir = Path(self._string(plan, "source_dir")).resolve()
        self.output_base = Path(self._string(plan, "output_base")).resolve()
        self.model_path = Path(self._string(plan, "model_path")).resolve()
        self.local_container_image = self._string(plan, "local_container_image")
        self.sglang_source = Path(self._string(plan, "sglang_source")).resolve()
        self.sglang_runtime_key = self._string(plan, "sglang_runtime_key")
        requested_output = os.environ.get("SRTCTL_OUTPUT_DIR")
        self.output_dir = (
            Path(requested_output).resolve()
            if requested_output
            else self.output_base / f"{self.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
        )
        self.log_dir = self.output_dir / "logs"
        self.artifact_dir = self.output_dir / "artifacts"
        self.direct_plan_path = self.output_dir / "direct-plan.json"
        self.direct_host_plan_path = self.output_dir / "direct-host-plan.json"
        self.dynamo_cache_root = self.output_base / ".srtctl-cache" / "dynamo-wheels"
        self.sglang_runtime_dir: Path | None = None
        self.container_name = f"srtctl-direct-{os.getpid()}"
        self.mooncake_container_name: str | None = None
        self.container_label = (
            "io.nvidia.srtctl.direct-output=" + hashlib.sha256(str(self.output_dir).encode("utf-8")).hexdigest()
        )
        self._container_exec: subprocess.Popen[Any] | None = None
        self._mooncake_log_process: subprocess.Popen[Any] | None = None
        self._mooncake_log_file: TextIO | None = None

    @staticmethod
    def _mapping(plan: dict[str, Any], key: str) -> dict[str, Any]:
        value = plan.get(key)
        if not isinstance(value, dict):
            raise TypeError(f"direct host plan requires object field {key!r}")
        return value

    @staticmethod
    def _string(plan: dict[str, Any], key: str) -> str:
        value = plan.get(key)
        if not isinstance(value, str) or not value:
            raise TypeError(f"direct host plan requires non-empty string field {key!r}")
        return value

    def log(self, message: str) -> None:
        print(f"[srtctl:{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)

    def _docker(self, *args: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", *args],
            check=True,
            capture_output=capture_output,
            text=True,
        )

    @staticmethod
    def _mount(path: Path, *, readonly: bool) -> list[str]:
        if not path.exists():
            raise RuntimeError(f"required container mount does not exist: {path}")
        resolved = path.resolve()
        mount = f"type=bind,src={resolved},dst={resolved}"
        if readonly:
            mount += ",readonly"
        return ["--mount", mount]

    def _model_mount_path(self) -> Path:
        # Hugging Face snapshot files contain relative links into the repository
        # blob store, so mount the repository root rather than one snapshot.
        if self.model_path.parent.name == "snapshots" and (self.model_path.parent.parent / "blobs").is_dir():
            return self.model_path.parent.parent
        return self.model_path

    def _prepare_output(self) -> None:
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        host_plan = {**self.plan, "output_dir": str(self.output_dir)}
        self.direct_host_plan_path.write_text(json.dumps(host_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.direct_plan_path.write_text(
            json.dumps(self.container_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _sglang_revision(self) -> str:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={self.sglang_source}",
                "-C",
                str(self.sglang_source),
                "rev-parse",
                "--verify",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _remove_stale_containers(self) -> None:
        result = self._docker("ps", "-aq", "--filter", f"label={self.container_label}", capture_output=True)
        for container in result.stdout.splitlines():
            if container:
                self._docker("rm", "-f", container)

    def _main_container_args(self, trace_path: Path | None) -> list[str]:
        revision = self._sglang_revision()
        self.sglang_runtime_dir = (
            self.output_base / ".srtctl-runtime" / (f"sglang-{revision}-{self.sglang_runtime_key}")
        )
        self.dynamo_cache_root.mkdir(parents=True, exist_ok=True)
        args = [
            "run",
            "--detach",
            "--name",
            self.container_name,
            "--label",
            self.container_label,
            "--network",
            "host",
            "--ipc",
            "host",
            "--gpus",
            "all",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "--entrypoint",
            "bash",
            *self._mount(self.source_dir, readonly=True),
            *self._mount(self.output_base, readonly=False),
            *self._mount(self._model_mount_path(), readonly=True),
            *self._mount(self.sglang_source, readonly=True),
            *self._mount(self.dynamo_cache_root, readonly=False),
        ]
        if not self._is_under(self.output_dir, self.output_base):
            args.extend(self._mount(self.output_dir, readonly=False))
        if trace_path is not None:
            args.extend(self._mount(trace_path, readonly=True))
        args.extend(
            [
                self.local_container_image,
                "-lc",
                'trap "exit 0" TERM INT; while :; do sleep 3600; done',
            ]
        )
        return args

    @staticmethod
    def _is_under(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def _start_main_container(self, trace_path: Path | None) -> None:
        self._docker(*self._main_container_args(trace_path), capture_output=True)

    def _start_mooncake(self) -> None:
        command = self.plan.get("mooncake_master_command")
        if not isinstance(command, list) or not command:
            return
        container = self._string(self.plan, "mooncake_container")
        environment = self.plan.get("mooncake_environment", [])
        if not isinstance(environment, list):
            raise TypeError("direct host plan field 'mooncake_environment' must be a list")
        self.mooncake_container_name = f"{self.container_name}-mooncake"
        args = [
            "run",
            "--detach",
            "--name",
            self.mooncake_container_name,
            "--label",
            self.container_label,
            "--network",
            "host",
            "--ipc",
            "host",
            "--gpus",
            "all",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
        ]
        for item in environment:
            if not isinstance(item, list) or len(item) != 2 or not all(isinstance(value, str) for value in item):
                raise TypeError("direct host plan has invalid mooncake environment entry")
            args.extend(("--env", f"{item[0]}={item[1]}"))
        args.extend((container, *(str(argument) for argument in command)))
        result = self._docker(*args, capture_output=True)
        (self.log_dir / "mooncake-master.container-id").write_text(result.stdout, encoding="utf-8")
        self._mooncake_log_file = (self.log_dir / "mooncake-master.log").open("a", encoding="utf-8")
        self._mooncake_log_process = subprocess.Popen(
            ["docker", "logs", "-f", self.mooncake_container_name],
            stdout=self._mooncake_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _run_container(self, trace_path: Path | None) -> int:
        if self.sglang_runtime_dir is None:
            raise RuntimeError("main container runtime path was not initialized")
        args = [
            "docker",
            "exec",
            "-e",
            "SRTCTL_LOCAL_CONTAINERIZED=1",
            "-e",
            f"SRTCTL_OUTPUT_DIR={self.output_dir}",
            "-e",
            f"OUTPUT_DIR={self.output_dir}",
            "-e",
            f"LOG_DIR={self.log_dir}",
            "-e",
            f"ARTIFACT_DIR={self.artifact_dir}",
            "-e",
            f"SRTCTL_DYNAMO_CACHE_ROOT={self.dynamo_cache_root}",
            "-e",
            f"SRTCTL_SGLANG_RUNTIME_DIR={self.sglang_runtime_dir}",
            "-e",
            "SRTCTL_PYTHON=python3",
        ]
        if trace_path is not None:
            args.extend(("-e", f"RUTER_MOONCAKE_TRACE={trace_path}"))
        args.extend(
            (
                self.container_name,
                "python3",
                str(self.source_dir / "src" / "srtctl" / "render" / "direct_runner.py"),
                "--plan",
                str(self.direct_plan_path),
            )
        )
        self._container_exec = subprocess.Popen(args, start_new_session=True)
        try:
            return self._container_exec.wait()
        finally:
            self._container_exec = None

    def _restore_owner(self, path: Path) -> None:
        if not path.exists():
            return
        owner = f"{os.getuid()}:{os.getgid()}"
        try:
            subprocess.run(
                ["chown", "-R", owner, str(path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                ["sudo", "-n", "chown", "-R", owner, str(path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def _stop_mooncake_log_follower(self) -> None:
        if self._mooncake_log_process is not None:
            self._mooncake_log_process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._mooncake_log_process.wait(timeout=5)
            if self._mooncake_log_process.poll() is None:
                self._mooncake_log_process.kill()
                self._mooncake_log_process.wait()
        if self._mooncake_log_file is not None:
            self._mooncake_log_file.close()

    def _cleanup(self) -> None:
        self._stop_mooncake_log_follower()
        for container in (self.mooncake_container_name, self.container_name):
            if container is not None:
                with contextlib.suppress(OSError, subprocess.CalledProcessError):
                    self._docker("rm", "-f", container)
        for path, label in ((self.output_dir, "output"), (self.sglang_runtime_dir, "SGLang runtime")):
            if path is None:
                continue
            try:
                self._restore_owner(path)
            except (OSError, subprocess.CalledProcessError) as error:
                self.log(f"WARNING: failed to restore {label} ownership: {path}: {error}")

    def _on_signal(self, signal_number: int, _frame: Any) -> None:
        if self._container_exec is not None and self._container_exec.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                self._container_exec.send_signal(signal_number)
        raise DirectHostInterrupted(signal_number)

    def run(self) -> int:
        previous_handlers = {
            signal.SIGINT: signal.signal(signal.SIGINT, self._on_signal),
            signal.SIGTERM: signal.signal(signal.SIGTERM, self._on_signal),
        }
        try:
            if shutil.which("docker") is None:
                raise RuntimeError("SRTCTL_LOCAL_CONTAINER_IMAGE requires docker")
            self._prepare_output()
            trace_value = os.environ.get("RUTER_MOONCAKE_TRACE")
            trace_path = Path(trace_value).resolve() if trace_value else None
            self._remove_stale_containers()
            self._start_mooncake()
            self._start_main_container(trace_path)
            return self._run_container(trace_path)
        except DirectHostInterrupted as error:
            self.log(f"Interrupted by signal {error.signal_number}")
            return 128 + error.signal_number
        except (OSError, RuntimeError, TypeError, subprocess.CalledProcessError) as error:
            self.log(f"ERROR: {error}")
            return 1
        finally:
            self._cleanup()
            for signal_number, previous in previous_handlers.items():
                signal.signal(signal_number, previous)


def _load_plan(path: str) -> dict[str, Any]:
    raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    plan = json.loads(raw)
    if not isinstance(plan, dict):
        raise TypeError("direct host plan must be a JSON object")
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Rendered direct host plan, or - for stdin")
    args = parser.parse_args(argv)
    try:
        return DirectHostRunner(_load_plan(args.plan)).run()
    except (OSError, RuntimeError, TypeError, json.JSONDecodeError) as error:
        print(f"srtctl: ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
