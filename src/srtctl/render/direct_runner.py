# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct benchmark runner executed inside the serving environment.

The renderer writes a JSON launch plan and the small Bash entrypoint invokes
this module directly. It deliberately uses only the standard library so the
selected SGLang Python can run it before srtctl itself is installed there.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

if __package__:
    from .direct_stages import (
        BenchmarkStageMixin,
        InfrastructureStageMixin,
        PostProcessStageMixin,
        RuntimeSetupStageMixin,
        ServingStageMixin,
        TelemetryStageMixin,
        router_counts,
    )
    from .direct_stages.common import DirectRunInterrupted, ManagedProcess, shell_quote
else:
    # The container invokes this file before the serving venv has srtctl's
    # control-plane dependencies. Load its stdlib-only sibling package instead.
    stages = importlib.import_module("direct_stages")
    common = importlib.import_module("direct_stages.common")
    BenchmarkStageMixin = stages.BenchmarkStageMixin
    InfrastructureStageMixin = stages.InfrastructureStageMixin
    PostProcessStageMixin = stages.PostProcessStageMixin
    RuntimeSetupStageMixin = stages.RuntimeSetupStageMixin
    ServingStageMixin = stages.ServingStageMixin
    TelemetryStageMixin = stages.TelemetryStageMixin
    router_counts = stages.router_counts
    DirectRunInterrupted = common.DirectRunInterrupted
    ManagedProcess = common.ManagedProcess
    shell_quote = common.shell_quote

_router_counts = router_counts


class DirectRunner(
    RuntimeSetupStageMixin,
    InfrastructureStageMixin,
    ServingStageMixin,
    TelemetryStageMixin,
    BenchmarkStageMixin,
    PostProcessStageMixin,
):
    """Compose the direct-only stages and own their process lifecycle."""

    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.output_dir = Path(os.environ["OUTPUT_DIR"]).resolve()
        self.log_dir = Path(os.environ["LOG_DIR"]).resolve()
        self.artifact_dir = Path(os.environ["ARTIFACT_DIR"]).resolve()
        self.source_dir = Path(str(plan["source_dir"])).resolve()
        self.output_base = Path(str(plan["output_base"])).resolve()
        self.sglang_source = Path(str(plan["sglang_source"])).resolve()
        self.ruter_python = self.source_dir / ".venv" / "bin" / "python"
        self.python = os.environ.get("SRTCTL_PYTHON", sys.executable)
        self.processes: list[ManagedProcess] = []
        self.tachometer: ManagedProcess | None = None
        self.tachometer_local_dir: Path | None = None
        self._configure_environment()

    def _configure_environment(self) -> None:
        for key, value in self.plan["global_environment"]:
            os.environ[str(key)] = str(value)
        for key, value in self.plan["benchmark_environment"]:
            os.environ[str(key)] = str(value)
        os.environ["SRT_FRONTEND_URL"] = f"http://127.0.0.1:{self.plan['frontend_port']}"
        os.environ["SRT_FRONTEND_HOST"] = "127.0.0.1"
        os.environ["SRTCTL_RUTER_PYTHON"] = str(self.ruter_python)
        os.environ["AIPERF_ARTIFACT_DIR"] = str(self.artifact_dir / "aiperf")
        os.environ.setdefault("AIPERF_DATASET_MMAP_BASE_PATH", str(self.artifact_dir / "aiperf-mmap"))

    def log(self, message: str) -> None:
        print(f"[srtctl:{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)

    def _die(self, message: str) -> None:
        raise RuntimeError(message)

    def _run_logged(
        self,
        args: list[str],
        *,
        log_name: str,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        path = self.log_dir / log_name
        with path.open("a", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(shell_quote(arg) for arg in args) + "\n")
            handle.flush()
            subprocess.run(args, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    def _launch(
        self, label: str, log_name: str, args: list[str], *, env: dict[str, str] | None = None
    ) -> ManagedProcess:
        log_path = self.log_dir / log_name
        command_path = log_path.with_suffix(log_path.suffix + ".command")
        command_path.write_text(" ".join(shell_quote(arg) for arg in args) + "\n", encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(
                args,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        log_path.with_suffix(log_path.suffix + ".pid").write_text(f"{process.pid}\n", encoding="utf-8")
        managed = ManagedProcess(label=label, process=process, log_path=log_path)
        self.processes.append(managed)
        self.log(f"Started {label}: pid={process.pid} log={log_path}")
        return managed

    def _launch_shell(
        self, label: str, log_name: str, command: str, *, env: dict[str, str] | None = None
    ) -> ManagedProcess:
        python_bin = str(Path(self.python).parent)
        shell_command = f"export PATH={shell_quote(python_bin)}:$PATH; {command}"
        path = self.log_dir / log_name
        path.with_suffix(path.suffix + ".command").write_text(command + "\n", encoding="utf-8")
        return self._launch(label, log_name, ["bash", "-lc", shell_command], env=env)

    def _stop(self, managed: ManagedProcess, timeout_seconds: int = 30) -> None:
        process = managed.process
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                self.log(f"{managed.label} did not stop after {timeout_seconds}s; sending SIGKILL")
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def _assert_services_alive(self) -> None:
        for managed in self.processes:
            if managed.process.poll() is not None:
                self._die(f"owned service {managed.label} exited; inspect {managed.log_path}")
        if self.plan["mooncake_master_command"]:
            self._assert_tcp("127.0.0.1", int(self.plan["mooncake_master_port"]), "mooncake master")

    def _wait_http_ready(self, url: str, label: str) -> None:
        deadline = time.monotonic() + int(self.plan["health_timeout_seconds"])
        interval = int(self.plan["health_interval_seconds"])
        while time.monotonic() < deadline:
            self._assert_services_alive()
            try:
                with urllib.request.urlopen(url, timeout=5):
                    self.log(f"{label} is ready")
                    return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(interval)
        self._die(f"{label} did not become ready: {url}")

    def _wait_tcp_ready(self, host: str, port: int, label: str) -> None:
        deadline = time.monotonic() + int(self.plan["health_timeout_seconds"])
        interval = int(self.plan["health_interval_seconds"])
        while time.monotonic() < deadline:
            try:
                self._assert_tcp(host, port, label)
                self.log(f"{label} is ready")
                return
            except OSError:
                time.sleep(interval)
        self._die(f"{label} did not become ready on {host}:{port}")

    @staticmethod
    def _assert_tcp(host: str, port: int, label: str) -> None:
        with socket.create_connection((host, port), timeout=3):
            return

    def _cleanup(self) -> None:
        if self.tachometer is not None:
            self._stop(self.tachometer)
        try:
            self._compact_tachometer()
        except (OSError, subprocess.CalledProcessError):
            self.log("WARNING: Tachometer compaction failed; inspect " + str(self.log_dir / "tachometer.log"))
        for managed in reversed(self.processes):
            self._stop(managed)
        self._normalize_ruter()

    def _on_signal(self, signal_number: int, _frame: Any) -> None:
        raise DirectRunInterrupted(signal_number)

    def _run_stages(self) -> None:
        self._run_setup_script()
        if self.plan["ruter_enabled"] and not os.access(self.ruter_python, os.X_OK):
            self._die(f"ruter control Python is not executable: {self.ruter_python}")
        self._install_sglang_from_source()
        self._run_logged([self.python, "-c", "import sglang"], log_name="install-sglang.log")
        self._install_dynamo()
        self._start_infrastructure()
        self._start_mooncake()
        self._start_workers_and_router()
        self._smoke_chat()
        self._start_tachometer()
        self._run_benchmark()

    def run(self) -> int:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        Path(os.environ["AIPERF_DATASET_MMAP_BASE_PATH"]).mkdir(parents=True, exist_ok=True)
        self.log(f"Run directory: {self.output_dir}")
        previous_handlers = {
            signal.SIGINT: signal.signal(signal.SIGINT, self._on_signal),
            signal.SIGTERM: signal.signal(signal.SIGTERM, self._on_signal),
        }
        try:
            self._run_stages()
            return 0
        except DirectRunInterrupted as error:
            self.log(f"Interrupted by signal {error.signal_number}")
            return 128 + error.signal_number
        except KeyboardInterrupt:
            self.log("Interrupted")
            return 130
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            self.log(f"ERROR: {error}")
            return 1
        finally:
            self._cleanup()
            for signal_number, previous in previous_handlers.items():
                signal.signal(signal_number, previous)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="Rendered direct execution plan")
    args = parser.parse_args(argv)
    with args.plan.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    return DirectRunner(plan).run()


if __name__ == "__main__":
    raise SystemExit(main())
