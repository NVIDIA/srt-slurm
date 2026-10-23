# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check image publishing and failure cleanup without Enroot or package installs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pytest

SETUP = Path(__file__).resolve().parents[1] / "configs/setup_mooncake_egm.sh"


class EnrootCall(TypedDict):
    args: list[str]
    data_path: str | None


@dataclass(frozen=True)
class ImageBuildEnvironment:
    script: Path
    base_image: Path
    output_image: Path
    wheel_dir: Path
    call_log: Path
    environment: dict[str, str]

    def run(self, overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", str(self.script), "--build-image", str(self.base_image), str(self.output_image)],
            env={**self.environment, **(overrides or {})},
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    def calls(self) -> list[EnrootCall]:
        if not self.call_log.exists():
            return []
        return [json.loads(line) for line in self.call_log.read_text().splitlines()]


@pytest.fixture
def image_build_environment(tmp_path: Path) -> ImageBuildEnvironment:
    # Enroot's colon-separated fstab syntax needs escaped spaces as well as
    # shell quoting for these externally supplied paths.
    workspace = tmp_path / "image build workspace"
    workspace.mkdir()
    configs = workspace / "fake configs"
    configs.mkdir()
    script = configs / SETUP.name
    shutil.copyfile(SETUP, script)
    (configs / "mooncake_egm_donor.py").write_text("# Enroot is intercepted by the test.\n")
    # The host image builder must not invoke package installation on the host.
    (configs / "install_dynamo_wheel.sh").write_text("echo unexpected-Dynamo-install >&2\nexit 91\n")
    base_image = workspace / "original image.sqsh"
    base_image.write_bytes(b"original image content")
    output_dir = workspace / "new images"
    output_dir.mkdir()
    wheel_dir = workspace / "Mooncake wheels"
    wheel_dir.mkdir()
    (wheel_dir / "mooncake_transfer_engine_cuda13-0.0.0-py3-none-any.whl").write_bytes(b"test wheel")
    binaries = workspace / "fake bin"
    binaries.mkdir()
    call_log = workspace / "enroot calls.jsonl"
    enroot = binaries / "enroot"
    enroot.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with open(os.environ["MC_BUILD_TEST_LOG"], "a") as stream:
    stream.write(json.dumps({"args": args, "data_path": os.environ.get("ENROOT_DATA_PATH")}) + "\n")
if args[0] == "export":
    output = Path(args[args.index("--output") + 1])
    output.write_bytes(b"partial export" if os.environ.get("MC_BUILD_TEST_FAIL") == "export" else b"new image")
    if os.environ.get("MC_BUILD_TEST_RACE_DESTINATION"):
        Path(os.environ["MC_BUILD_TEST_RACE_DESTINATION"]).write_bytes(b"concurrent image")
if args[0] == os.environ.get("MC_BUILD_TEST_FAIL"):
    raise SystemExit(42)
"""
    )
    enroot.chmod(0o755)
    site_data = workspace / "site container storage"
    site_data.mkdir()
    environment = {
        **os.environ,
        "PATH": f"{binaries}{os.pathsep}{os.defpath}",
        "SRT_MOONCAKE_EGM_WHEEL_DIR": str(wheel_dir),
        "ENROOT_DATA_PATH": str(site_data),
        "MC_BUILD_TEST_LOG": str(call_log),
    }
    environment.pop("MC_USE_NVLINK_IPC", None)
    environment.pop("MC_BUILD_TEST_FAIL", None)
    environment.pop("MC_BUILD_TEST_RACE_DESTINATION", None)
    return ImageBuildEnvironment(script, base_image, output_dir / "EGM image.sqsh", wheel_dir, call_log, environment)


def test_image_build_preserves_source_and_uses_existing_mooncake_installer(
    image_build_environment: ImageBuildEnvironment,
) -> None:
    build = image_build_environment
    result = build.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert build.base_image.read_bytes() == b"original image content"
    assert build.output_image.read_bytes() == b"new image"
    assert "unexpected-Dynamo-install" not in result.stderr
    calls = build.calls()
    arguments = [call["args"] for call in calls]
    assert all(isinstance(args, list) for args in arguments)
    assert [args[0] for args in arguments] == ["create", "start", "export", "remove"]
    create, start, export, remove = arguments
    name = create[create.index("--name") + 1]
    assert create[-1] == str(build.base_image)
    assert "--root" in start
    assert "--rw" in start
    configs_mount = str(build.script.parent).replace(" ", r"\040")
    wheels_mount = str(build.wheel_dir).replace(" ", r"\040")
    assert f"{configs_mount}:/srt-mooncake-configs:none:ro,bind,x-create=dir" in start
    assert f"{wheels_mount}:/srt-mooncake-wheels:none:ro,bind,x-create=dir" in start
    assert "SRT_MOONCAKE_EGM_WHEEL_DIR=/srt-mooncake-wheels" in start
    assert start[-4:] == [name, "bash", "/srt-mooncake-configs/setup_mooncake_egm.sh", "--mooncake-only"]
    exported_path = Path(export[export.index("--output") + 1])
    assert exported_path.is_relative_to(build.output_image.parent)
    assert exported_path != build.output_image
    assert not exported_path.exists()
    assert list(build.output_image.parent.iterdir()) == [build.output_image]
    assert export[-1] == name
    assert remove == ["remove", "--force", name]
    assert all(call["data_path"] == build.environment["ENROOT_DATA_PATH"] for call in calls)


def test_existing_destination_is_never_overwritten(image_build_environment: ImageBuildEnvironment) -> None:
    build = image_build_environment
    build.output_image.write_bytes(b"existing destination")
    result = build.run()
    assert result.returncode != 0
    assert build.output_image.read_bytes() == b"existing destination"
    assert build.calls() == []


@pytest.mark.parametrize("wheel_count", [0, 2])
def test_image_build_requires_exactly_one_wheel_before_enroot(
    image_build_environment: ImageBuildEnvironment, wheel_count: int
) -> None:
    build = image_build_environment
    for wheel in build.wheel_dir.glob("*.whl"):
        wheel.unlink()
    for index in range(wheel_count):
        (build.wheel_dir / f"mooncake_transfer_engine_cuda13-0.0.{index}-py3-none-any.whl").write_bytes(b"wheel")
    result = build.run()
    assert result.returncode != 0
    assert build.calls() == []
    assert not build.output_image.exists()


@pytest.mark.parametrize("failure", ["create", "start", "export"])
def test_failed_image_build_cleans_its_container_without_publishing(
    image_build_environment: ImageBuildEnvironment, failure: str
) -> None:
    build = image_build_environment
    result = build.run({"MC_BUILD_TEST_FAIL": failure})
    assert result.returncode != 0
    assert build.base_image.read_bytes() == b"original image content"
    assert not build.output_image.exists()
    arguments = [call["args"] for call in build.calls()]
    create = arguments[0]
    name = create[create.index("--name") + 1]
    assert arguments[-1] == ["remove", "--force", name]
    assert list(build.output_image.parent.iterdir()) == []


def test_destination_created_during_export_is_never_overwritten(
    image_build_environment: ImageBuildEnvironment,
) -> None:
    build = image_build_environment
    result = build.run({"MC_BUILD_TEST_RACE_DESTINATION": str(build.output_image)})
    assert result.returncode != 0
    assert build.output_image.read_bytes() == b"concurrent image"
    arguments = [call["args"] for call in build.calls()]
    assert arguments[-1][0:2] == ["remove", "--force"]
    assert list(build.output_image.parent.iterdir()) == [build.output_image]
