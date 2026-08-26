# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serving-runtime setup stage for direct execution."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .common import run_capture, rust_toolchain


class RuntimeSetupStageMixin:
    """Install SGLang and Dynamo into the per-run serving environment."""

    plan: dict[str, Any]
    source_dir: Path
    output_dir: Path
    sglang_source: Path
    python: str

    def log(self, message: str) -> None:
        raise NotImplementedError

    def _die(self, message: str) -> None:
        raise NotImplementedError

    def _run_logged(self, args: list[str], **kwargs: Any) -> None:
        raise NotImplementedError

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
        revision = run_capture(
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
                toolchain = rust_toolchain(source_copy / "rust" / "rust-toolchain.toml")
                if toolchain:
                    self.log(f"Installing source-pinned Rust {toolchain}")
                    self._run_logged(
                        ["rustup", "toolchain", "install", toolchain, "--profile", "minimal"],
                        log_name="install-sglang.log",
                    )
                    os.environ["RUSTUP_TOOLCHAIN"] = toolchain
                selected_python = str(venv / "bin" / "python")
                self._run_logged(
                    [selected_python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
                    log_name="install-sglang.log",
                )
                self._run_logged(
                    [selected_python, "-m", "pip", "install", "--quiet", "--editable", str(source_copy / "python")],
                    log_name="install-sglang.log",
                )
                installed = run_capture([selected_python, "-c", "import sglang; print(sglang.__file__)"])
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
