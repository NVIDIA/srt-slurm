# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the EGM installer with isolated fake Python, hashing and locking tools."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parents[1] / "configs/setup_mooncake_egm.sh"


@dataclass(frozen=True)
class SetupEnvironment:
    script: Path
    wheel_dir: Path
    prefix: Path
    call_log: Path
    environment: dict[str, str]

    def run(self, overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", str(self.script), "--mooncake-only"],
            env={**self.environment, **(overrides or {})},
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    def calls(self, action: str) -> list[dict[str, object]]:
        if not self.call_log.exists():
            return []
        return [
            call for line in self.call_log.read_text().splitlines() if (call := json.loads(line))["action"] == action
        ]


@pytest.fixture
def setup_environment(tmp_path: Path) -> SetupEnvironment:
    binaries = tmp_path / "fake-bin"
    binaries.mkdir()
    configs = tmp_path / "fake configs"
    configs.mkdir()
    script = configs / SETUP.name
    shutil.copyfile(SETUP, script)
    # If the service-only path accidentally invokes the Dynamo installer, record
    # it and fail without ever reaching its real package installation commands.
    (configs / "install_dynamo_wheel.sh").write_text(
        'printf \'%s\\n\' \'{"action":"dynamo"}\' >> "$MC_TEST_CALL_LOG"\nexit 91\n'
    )
    (configs / "mooncake_egm_donor.py").write_text("# Execution is intercepted by the fake Python.\n")
    prefix = tmp_path / "fake Python prefix"
    prefix.mkdir()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    call_log = tmp_path / "calls.jsonl"

    # PATH contains only explicit harmless utilities and fakes. Host pip and
    # installed Mooncake/CUDA packages cannot be reached by this harness.
    for name in ("bash", "dirname", "cut", "rm", "touch"):
        resolved = shutil.which(name)
        assert resolved, name
        (binaries / name).symlink_to(resolved)
    stub = binaries / "fake-tool"
    stub.write_text(
        f"#!{sys.executable}\n"
        + r"""
import hashlib
import json
import os
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]
if name == "sha256sum":
    path = Path(args[0])
    print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")
    raise SystemExit(0)
if name == "flock":
    # Tests exercise sequential sentinel behavior, not OS lock concurrency.
    raise SystemExit(0)
if args[:1] == ["-c"]:
    print(os.environ["MC_TEST_PREFIX"])
    raise SystemExit(0)
if args[:3] == ["-m", "pip", "install"]:
    action = "pip"
elif args[-1:] == ["check-install"]:
    action = "check"
else:
    raise SystemExit(f"Unexpected fake Python invocation: {args!r}")
with open(os.environ["MC_TEST_CALL_LOG"], "a") as stream:
    stream.write(json.dumps({"action": action, "args": args}) + "\n")
if action == "check" and os.environ.get("MC_TEST_CHECK_FAIL") == "1":
    print("Mooncake native setup lacks EGM Store support", file=sys.stderr)
    raise SystemExit(7)
"""
    )
    stub.chmod(0o755)
    for name in ("python3", "sha256sum", "flock"):
        (binaries / name).symlink_to(stub)
    environment = {
        **os.environ,
        "PATH": str(binaries),
        "MC_TEST_PREFIX": str(prefix),
        "MC_TEST_CALL_LOG": str(call_log),
        "SRT_MOONCAKE_EGM_WHEEL_DIR": str(wheels),
    }
    environment.pop("MC_USE_NVLINK_IPC", None)
    return SetupEnvironment(script, wheels, prefix, call_log, environment)


def test_service_setup_accepts_a_compatible_installed_build_without_installing_dynamo(
    setup_environment: SetupEnvironment,
) -> None:
    result = setup_environment.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No wheel" in result.stdout
    assert len(setup_environment.calls("check")) == 1
    assert setup_environment.calls("dynamo") == []
    assert setup_environment.calls("pip") == []


def test_missing_egm_build_reports_the_requirement_without_installing_packages(
    setup_environment: SetupEnvironment,
) -> None:
    result = setup_environment.run({"MC_TEST_CHECK_FAIL": "1"})
    assert result.returncode != 0
    assert "native setup lacks EGM Store support" in result.stderr
    assert "Supply an EGM-enabled ARM64 CUDA 13 wheel" in result.stderr
    assert setup_environment.calls("pip") == []
    assert setup_environment.calls("dynamo") == []
    assert list(setup_environment.prefix.glob("*.complete")) == []


@pytest.mark.parametrize("value", ["", "0"])
def test_ipc_mode_is_rejected_by_presence_even_when_set_to_zero(
    setup_environment: SetupEnvironment, value: str
) -> None:
    result = setup_environment.run({"MC_USE_NVLINK_IPC": value})
    assert result.returncode != 0
    assert "unset MC_USE_NVLINK_IPC" in result.stderr
    assert setup_environment.calls("pip") == []
    assert setup_environment.calls("check") == []


def test_identical_wheel_is_installed_once_but_replaced_bytes_trigger_reinstallation(
    setup_environment: SetupEnvironment,
) -> None:
    wheel = setup_environment.wheel_dir / "mooncake_transfer_engine_cuda13-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"synthetic first build")
    first = setup_environment.run()
    assert first.returncode == 0, first.stdout + first.stderr
    repeated = setup_environment.run()
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert len(setup_environment.calls("pip")) == 1
    # Every launch validates the installed module, including a cache hit.
    assert len(setup_environment.calls("check")) == 3

    wheel.write_bytes(b"synthetic rebuilt wheel with the same filename")
    changed = setup_environment.run()
    assert changed.returncode == 0, changed.stdout + changed.stderr
    calls = setup_environment.calls("pip")
    assert len(calls) == 2
    for call in calls:
        args = call["args"]
        assert isinstance(args, list)
        assert "--no-deps" in args
        assert "--force-reinstall" in args
        assert str(wheel) in args
    expected_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert [path.name for path in setup_environment.prefix.glob("*.complete")] == [
        f".srtctl_mooncake_egm.{expected_hash}.complete"
    ]
    assert setup_environment.calls("dynamo") == []


def test_cached_install_is_rechecked_if_another_step_replaces_the_installed_package(
    setup_environment: SetupEnvironment,
) -> None:
    wheel = setup_environment.wheel_dir / "mooncake_transfer_engine_cuda13-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"synthetic compatible wheel")
    installed = setup_environment.run()
    assert installed.returncode == 0, installed.stdout + installed.stderr
    result = setup_environment.run({"MC_TEST_CHECK_FAIL": "1"})
    assert result.returncode != 0
    assert "Supply an EGM-enabled ARM64 CUDA 13 wheel" in result.stderr
    assert len(setup_environment.calls("pip")) == 1
    assert len(setup_environment.calls("check")) == 3
