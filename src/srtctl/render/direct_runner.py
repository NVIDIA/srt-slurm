# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct benchmark runner executed inside the serving environment.

The renderer writes a JSON launch plan and the small Bash entrypoint invokes
this module directly.  It deliberately uses only the standard library so the
selected SGLang Python can run it before srtctl itself is installed there.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ManagedProcess:
    """A direct-run subprocess and its process-group leader."""

    label: str
    process: subprocess.Popen[Any]
    log_path: Path


class DirectRunInterrupted(Exception):
    """Signal delivered to the supervisor while it owns child process groups."""

    def __init__(self, signal_number: int) -> None:
        self.signal_number = signal_number
        super().__init__(f"received signal {signal_number}")


class DirectRunner:
    """Own one direct benchmark run from install through normalization."""

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
            handle.write("$ " + " ".join(_shell_quote(arg) for arg in args) + "\n")
            handle.flush()
            subprocess.run(args, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    def _launch(
        self, label: str, log_name: str, args: list[str], *, env: dict[str, str] | None = None
    ) -> ManagedProcess:
        log_path = self.log_dir / log_name
        command_path = log_path.with_suffix(log_path.suffix + ".command")
        command_path.write_text(" ".join(_shell_quote(arg) for arg in args) + "\n", encoding="utf-8")
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
        shell_command = f"export PATH={_shell_quote(python_bin)}:$PATH; {command}"
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

    def _run_setup_script(self) -> None:
        name = self.plan.get("setup_script")
        if not name:
            return
        script = self.source_dir / "configs" / str(name)
        patch = self.source_dir / "configs" / "patches" / str(name)
        selected = script if script.is_file() else patch if patch.is_file() else None
        if selected is None:
            self.log(f"WARNING: setup script not found: {script} (or {patch})")
            return
        self.log(f"Running setup script: {selected}")
        self._run_logged(["bash", str(selected)], log_name="setup.log")

    def _install_sglang_from_source(self) -> None:
        source = self.sglang_source
        if not (source / "python" / "sglang").is_dir():
            self._die(f"Invalid SGLang source: {source}")
        revision = _run_capture(
            ["git", "-c", f"safe.directory={source}", "-C", str(source), "rev-parse", "--verify", "HEAD"]
        )
        runtime_dir = Path(os.environ["SRTCTL_SGLANG_RUNTIME_DIR"])
        runtime_root = runtime_dir.parent
        runtime_root.mkdir(parents=True, exist_ok=True)
        lock = runtime_root / f".sglang-{revision}-{self.plan['sglang_runtime_key']}.lock"
        with lock.open("w", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            if not (runtime_dir / ".complete").is_file():
                self.log(f"Installing SGLang {revision} into {runtime_dir / '.venv'}")
                shutil.rmtree(runtime_dir / ".venv", ignore_errors=True)
                shutil.rmtree(runtime_dir / "source", ignore_errors=True)
                (runtime_dir / ".complete").unlink(missing_ok=True)
                runtime_dir.mkdir(parents=True, exist_ok=True)
                venv = runtime_dir / ".venv"
                subprocess.run([self.python, "-m", "venv", "--system-site-packages", str(venv)], check=True)
                source_copy = runtime_dir / "source"
                shutil.copytree(
                    source,
                    source_copy,
                    ignore=shutil.ignore_patterns(".git", "target", ".venv", "__pycache__", ".pytest_cache"),
                )
                rust_toolchain = _rust_toolchain(source_copy / "rust" / "rust-toolchain.toml")
                if rust_toolchain:
                    self.log(f"Installing source-pinned Rust {rust_toolchain}")
                    self._run_logged(
                        ["rustup", "toolchain", "install", rust_toolchain, "--profile", "minimal"],
                        log_name="install-sglang.log",
                    )
                    os.environ["RUSTUP_TOOLCHAIN"] = rust_toolchain
                selected_python = str(venv / "bin" / "python")
                self._run_logged(
                    [selected_python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
                    log_name="install-sglang.log",
                )
                self._run_logged(
                    [selected_python, "-m", "pip", "install", "--quiet", "--editable", str(source_copy / "python")],
                    log_name="install-sglang.log",
                )
                installed = _run_capture([selected_python, "-c", "import sglang; print(sglang.__file__)"])
                expected = str(source_copy / "python" / "sglang" / "__init__.py")
                if installed != expected:
                    self._die(f"SGLang editable install resolved to {installed}, expected {expected}")
                self._ensure_import(selected_python, "nixl", "nixl", "install-sglang.log")
                self._ensure_import(selected_python, "blake3", "blake3", "install-sglang.log")
                (runtime_dir / ".complete").touch()
        self.python = str(runtime_dir / ".venv" / "bin" / "python")
        os.environ["SRTCTL_PYTHON"] = self.python
        self._run_logged([self.python, "-c", "import sglang, nixl, blake3"], log_name="install-sglang.log")

    def _ensure_import(self, python: str, module: str, package: str, log_name: str) -> None:
        if (
            subprocess.run(
                [python, "-c", f"import {module}"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            != 0
        ):
            self._run_logged([python, "-m", "pip", "install", "--quiet", package], log_name=log_name)

    def _prepare_dynamo_source_build(self) -> None:
        if os.environ.get("SRTCTL_LOCAL_CONTAINERIZED") == "1":
            self.log("Installing Dynamo source build prerequisites")
            self._run_logged(["apt-get", "update", "-qq"], log_name="install-dynamo.log")
            environment = dict(os.environ)
            environment["DEBIAN_FRONTEND"] = "noninteractive"
            self._run_logged(
                ["apt-get", "install", "-y", "-qq", "libclang-dev", "protobuf-compiler"],
                log_name="install-dynamo.log",
                env=environment,
            )
        for tool in ("git", "cargo", "protoc"):
            if shutil.which(tool) is None:
                self._die(f"{tool} is required for a Dynamo source build")
        if subprocess.run([self.python, "-m", "pip", "--version"], check=False).returncode != 0:
            self._run_logged([self.python, "-m", "ensurepip", "--upgrade"], log_name="install-dynamo.log")
        self._run_logged(
            [self.python, "-m", "pip", "install", "--quiet", "--upgrade", "maturin"], log_name="install-dynamo.log"
        )

    def _dynamo_source_cache_key(self) -> str:
        base = str(self.plan["dynamo_source_cache_key"])
        flags = ""
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith(("flags", "Features")) and ":" in line:
                    flags = line.split(":", 1)[1]
                    break
        digest = hashlib.sha256(flags.encode("utf-8")).hexdigest()[:12]
        return f"{base}-{os.uname().machine}-{sys.implementation.cache_tag}-{digest}"

    def _install_dynamo_from_source_cache(self) -> None:
        source_hash = str(self.plan["dynamo_source_hash"])
        root = Path(os.environ.get("SRTCTL_DYNAMO_CACHE_ROOT", str(self.source_dir / "configs" / "dynamo-wheels")))
        key = self._dynamo_source_cache_key()
        cache = root / key
        root.mkdir(parents=True, exist_ok=True)
        with (root / f".{key}.lock").open("w", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            if not (cache / ".complete").is_file():
                self._prepare_dynamo_source_build()
                self.log(f"Building Dynamo {source_hash} into {cache}")
                with tempfile.TemporaryDirectory() as raw_build:
                    build = Path(raw_build)
                    repo = build / "dynamo"
                    self._run_logged(
                        ["git", "clone", "--no-checkout", "https://github.com/ai-dynamo/dynamo.git", str(repo)],
                        log_name="install-dynamo.log",
                    )
                    self._run_logged(
                        ["git", "-C", str(repo), "fetch", "--depth", "1", "origin", source_hash],
                        log_name="install-dynamo.log",
                    )
                    self._run_logged(
                        ["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"], log_name="install-dynamo.log"
                    )
                    for command in self.plan["dynamo_cargo_patch_commands"]:
                        self._run_logged(["bash", "-lc", str(command)], log_name="install-dynamo.log", cwd=repo)
                    cache.mkdir(parents=True, exist_ok=True)
                    for stale in [
                        *cache.glob("ai_dynamo_runtime-*.whl"),
                        cache / "dynamo-src.tar.gz",
                        cache / ".complete",
                    ]:
                        stale.unlink(missing_ok=True)
                    environment = dict(os.environ)
                    environment["RUSTFLAGS"] = (
                        f"{environment.get('RUSTFLAGS', '')} -C target-cpu=native --cfg tokio_unstable"
                    )
                    environment["CARGO_TARGET_DIR"] = str(build / "target")
                    self._run_logged(
                        [self.python, "-m", "maturin", "build", "--release", "--out", str(cache)],
                        log_name="install-dynamo.log",
                        cwd=repo / "lib" / "bindings" / "python",
                        env=environment,
                    )
                    self._write_archive(build, "dynamo", cache / "dynamo-src.tar.gz")
                    (cache / ".complete").touch()
        wheel = next(cache.glob("ai_dynamo_runtime-*.whl"), None)
        if wheel is None:
            self._die(f"Dynamo cache is incomplete: {cache}")
        source = self.output_dir / "runtime" / "dynamo-src"
        shutil.rmtree(source, ignore_errors=True)
        source.mkdir(parents=True, exist_ok=True)
        with tarfile.open(cache / "dynamo-src.tar.gz", "r:gz") as archive:
            archive.extractall(source, filter="data")
        self._run_logged(
            [self.python, "-m", "pip", "install", "--quiet", "--force-reinstall", "--no-deps", str(wheel)],
            log_name="install-dynamo.log",
        )
        self._run_logged(
            [self.python, "-m", "pip", "install", "--quiet", "--editable", str(source / "dynamo")],
            log_name="install-dynamo.log",
        )
        self.log(f"Installed Dynamo {source_hash} from {cache}")

    def _install_dynamo_from_top_of_tree(self) -> None:
        self._prepare_dynamo_source_build()
        self.log("Building Dynamo top-of-tree")
        with tempfile.TemporaryDirectory() as raw_build:
            build = Path(raw_build)
            repo = build / "dynamo"
            self._run_logged(
                ["git", "clone", "--depth", "1", "https://github.com/ai-dynamo/dynamo.git", str(repo)],
                log_name="install-dynamo.log",
            )
            environment = dict(os.environ)
            environment["RUSTFLAGS"] = f"{environment.get('RUSTFLAGS', '')} -C target-cpu=native --cfg tokio_unstable"
            environment["CARGO_TARGET_DIR"] = str(build / "target")
            wheel_dir = build / "wheels"
            self._run_logged(
                [self.python, "-m", "maturin", "build", "--release", "--out", str(wheel_dir)],
                log_name="install-dynamo.log",
                cwd=repo / "lib" / "bindings" / "python",
                env=environment,
            )
            wheel = next(wheel_dir.glob("ai_dynamo_runtime-*.whl"), None)
            if wheel is None:
                self._die("Dynamo top-of-tree build produced no runtime wheel")
            source = self.output_dir / "runtime" / "dynamo-src"
            shutil.rmtree(source, ignore_errors=True)
            shutil.copytree(repo, source, ignore=shutil.ignore_patterns(".git", "target", "__pycache__"))
            self._run_logged(
                [self.python, "-m", "pip", "install", "--quiet", "--force-reinstall", "--no-deps", str(wheel)],
                log_name="install-dynamo.log",
            )
            self._run_logged(
                [self.python, "-m", "pip", "install", "--quiet", "--editable", str(source)],
                log_name="install-dynamo.log",
            )
        self.log("Installed Dynamo top-of-tree")

    @staticmethod
    def _write_archive(root: Path, name: str, destination: Path) -> None:
        with tarfile.open(destination, "w:gz") as archive:
            for path in (root / name).rglob("*"):
                relative = path.relative_to(root)
                if ".git" in relative.parts or "target" in relative.parts:
                    continue
                archive.add(path, arcname=str(relative), recursive=False)

    def _install_dynamo(self) -> None:
        if self.plan["dynamo_source_hash"]:
            self._install_dynamo_from_source_cache()
        else:
            self._install_dynamo_from_top_of_tree()
        self._run_logged([self.python, "-c", "import dynamo"], log_name="install-dynamo.log")

    def _start_infrastructure(self) -> None:
        nats = str(self.source_dir / "configs" / "nats-server")
        etcd = str(self.source_dir / "configs" / "etcd")
        if not os.access(nats, os.X_OK):
            self._die(f"NATS binary is not executable: {nats}")
        if not os.access(etcd, os.X_OK):
            self._die(f"etcd binary is not executable: {etcd}")
        (self.output_dir / "nats").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "etcd").mkdir(parents=True, exist_ok=True)
        self._launch(
            "nats",
            "nats.log",
            [nats, "-js", "-a", "127.0.0.1", "-p", str(self.plan["nats_port"]), "-sd", str(self.output_dir / "nats")],
        )
        client_port = str(self.plan["etcd_client_port"])
        peer_port = str(self.plan["etcd_peer_port"])
        self._launch(
            "etcd",
            "etcd.log",
            [
                etcd,
                "--data-dir",
                str(self.output_dir / "etcd"),
                "--listen-client-urls",
                f"http://127.0.0.1:{client_port}",
                "--advertise-client-urls",
                f"http://127.0.0.1:{client_port}",
                "--listen-peer-urls",
                f"http://127.0.0.1:{peer_port}",
                "--initial-advertise-peer-urls",
                f"http://127.0.0.1:{peer_port}",
                "--initial-cluster",
                f"default=http://127.0.0.1:{peer_port}",
            ],
        )
        self._wait_http_ready(f"http://127.0.0.1:{client_port}/health", "etcd")

    def _start_mooncake(self) -> None:
        command = [str(value) for value in self.plan["mooncake_master_command"]]
        if not command:
            return
        self._wait_tcp_ready("127.0.0.1", int(self.plan["mooncake_master_port"]), "mooncake master")
        self._wait_tcp_ready("127.0.0.1", int(self.plan["mooncake_metadata_port"]), "mooncake metadata")
        self._wait_tcp_ready("127.0.0.1", int(self.plan["mooncake_metrics_port"]), "mooncake metrics")

    def _start_workers_and_router(self) -> None:
        for worker in self.plan["worker_processes"]:
            self._launch_shell(
                str(worker["label"]), str(worker["log_name"]), str(worker["command"]), env=dict(os.environ)
            )
        self._launch_shell("router", "router.log", str(self.plan["router_command"]), env=dict(os.environ))
        self._wait_router_ready()

    def _wait_router_ready(self) -> None:
        url = f"http://127.0.0.1:{self.plan['frontend_port']}/health"
        deadline = time.monotonic() + int(self.plan["health_timeout_seconds"])
        interval = int(self.plan["health_interval_seconds"])
        readiness_log = self.log_dir / "readiness.log"
        with readiness_log.open("a", encoding="utf-8") as handle:
            while time.monotonic() < deadline:
                self._assert_services_alive()
                try:
                    with urllib.request.urlopen(url, timeout=5) as response:
                        payload = json.loads(response.read().decode())
                    prefill, decode = _router_counts(payload)
                    if prefill >= int(self.plan["expected_prefill"]) and decode >= int(self.plan["expected_decode"]):
                        handle.write(
                            f"Router ready: prefill={prefill}/{self.plan['expected_prefill']} decode={decode}/{self.plan['expected_decode']}\n"
                        )
                        self.log(
                            f"Router ready: prefill={prefill}/{self.plan['expected_prefill']} decode={decode}/{self.plan['expected_decode']}"
                        )
                        return
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                    pass
                time.sleep(interval)
        self._die(f"Router did not report expected workers before timeout: {url}")

    def _smoke_chat(self) -> None:
        payload = json.dumps(
            {
                "model": self.plan["model_name"],
                "messages": [{"role": "user", "content": "Reply with one word: ready"}],
                "max_tokens": 16,
                "temperature": 0,
            }
        ).encode()
        request = urllib.request.Request(
            f"{os.environ['SRT_FRONTEND_URL']}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read())
        choices = body.get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else None
        if not isinstance(content, str) or not content.strip():
            self._die(f"Smoke chat returned no content: {body}")
        (self.log_dir / "smoke.json").write_text(json.dumps(body), encoding="utf-8")

    def _start_tachometer(self) -> None:
        if not self.plan["tachometer_enabled"]:
            return
        configured = str(self.source_dir / "bin" / "tachometer-scraper")
        if not os.access(configured, os.X_OK):
            self._die(f"Tachometer scraper is not executable: {configured}")
        storage = self.artifact_dir / "tachometer" / "raw" / "scrape"
        local_dir = self.artifact_dir / "tachometer" / "local"
        storage.parent.mkdir(parents=True, exist_ok=True)
        local_dir.mkdir(parents=True, exist_ok=True)
        config = str(self.plan["tachometer_config"]).replace("${TACHOMETER_STORAGE}", str(storage))
        config_path = self.output_dir / "tachometer.toml"
        config_path.write_text(config, encoding="utf-8")
        args = [configured, "--config", str(config_path), "--local-dir", str(local_dir)]
        if int(self.plan["tachometer_sync_interval_secs"]) > 0:
            args.extend(["--sync-interval", str(self.plan["tachometer_sync_interval_secs"])])
        environment = dict(os.environ)
        if int(self.plan["tachometer_compaction_threads"]) > 0:
            environment["POLARS_MAX_THREADS"] = str(self.plan["tachometer_compaction_threads"])
        self.tachometer = self._launch("tachometer", "tachometer.log", args, env=environment)
        self.tachometer_local_dir = local_dir
        time.sleep(2)
        if self.tachometer.process.poll() is not None:
            self._die(f"Tachometer exited at startup; inspect {self.tachometer.log_path}")

    def _compact_tachometer(self) -> None:
        if self.tachometer_local_dir is None:
            return
        if (
            not any(self.tachometer_local_dir.glob("*.parquet"))
            and not (self.tachometer_local_dir / "current.arrow").is_file()
        ):
            return
        environment = dict(os.environ)
        if int(self.plan["tachometer_compaction_threads"]) > 0:
            environment["POLARS_MAX_THREADS"] = str(self.plan["tachometer_compaction_threads"])
        self._run_logged(
            [
                str(self.source_dir / "bin" / "tachometer-scraper"),
                "compact",
                str(self.tachometer_local_dir),
                "--output",
                str(self.artifact_dir / "tachometer" / "final"),
            ],
            log_name="tachometer.log",
            env=environment,
        )

    def _run_benchmark(self) -> None:
        self.log("Starting benchmark")
        benchmark = self._launch_shell(
            "benchmark", "aiperf.log", str(self.plan["benchmark_command"]), env=dict(os.environ)
        )
        while benchmark.process.poll() is None:
            self._assert_services_alive()
            time.sleep(1)
        if benchmark.process.returncode != 0:
            self._die(f"Benchmark exited with status {benchmark.process.returncode}; inspect {benchmark.log_path}")
        self.log("Benchmark completed successfully")

    def _normalize_ruter(self) -> None:
        if not self.plan["ruter_enabled"]:
            return
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(self.source_dir / "src") if not existing else f"{self.source_dir / 'src'}:{existing}"
        )
        try:
            self._run_logged(
                [
                    str(self.ruter_python),
                    "-m",
                    "srtctl.ruter",
                    "init",
                    str(self.output_dir),
                    "--output",
                    str(self.log_dir / ".ruter"),
                ],
                log_name="ruter.log",
                env=environment,
            )
        except (OSError, subprocess.CalledProcessError):
            self.log("WARNING: ruter normalization failed; inspect " + str(self.log_dir / "ruter.log"))

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


def _router_counts(payload: dict[str, Any]) -> tuple[int, int]:
    prefill = decode = 0
    for instance in payload.get("instances", []):
        if instance.get("endpoint") != "generate":
            continue
        if instance.get("component") == "prefill":
            prefill += 1
        elif instance.get("component") in ("decode", "tensorrt_llm", "backend"):
            decode += 1
    return prefill, decode


def _rust_toolchain(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("channel") and "=" in stripped:
            return stripped.split("=", 1)[1].strip().strip('"')
    return None


def _run_capture(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def _shell_quote(value: str) -> str:
    if value and all(character.isalnum() or character in "@%_+=:,./-" for character in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="Rendered direct execution plan")
    args = parser.parse_args(argv)
    with args.plan.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    return DirectRunner(plan).run()


if __name__ == "__main__":
    raise SystemExit(main())
