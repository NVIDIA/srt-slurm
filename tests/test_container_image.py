# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for one-time container image preparation."""

import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.container_image import prepare_container_image
from srtctl.core.runtime import Nodes, RuntimeContext

_DIGEST = "a" * 64
_IMAGE = f"registry.example.com/example/server@sha256:{_DIGEST}"


def _write_squashfs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"hsqs-container-data")


def _importer(calls: list[list[str]], *, returncode: int = 0):
    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(command)
        if returncode == 0:
            save_option = next(arg for arg in command if arg.startswith("--container-save="))
            _write_squashfs(Path(save_option.split("=", 1)[1]))
        return subprocess.CompletedProcess(command, returncode)

    return run


def test_cold_import_is_published_and_reused(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    cache_root = tmp_path / "cache"
    with (
        patch("srtctl.core.container_image.platform.system", return_value="Linux"),
        patch("srtctl.core.container_image.platform.machine", return_value="aarch64"),
        patch("srtctl.core.container_image.shutil.which", return_value="/usr/bin/srun"),
        patch("srtctl.core.container_image.subprocess.run", side_effect=_importer(calls)),
    ):
        cold = prepare_container_image(_IMAGE, str(cache_root), job_id="123", node="node0")
        warm = prepare_container_image(_IMAGE, str(cache_root), job_id="123", node="node0")

    assert cold == warm == cache_root / "v1" / "linux-arm64" / f"{_DIGEST}.sqsh"
    assert cold.read_bytes().startswith(b"hsqs")
    assert len(calls) == 1
    assert calls[0][0] == "/usr/bin/srun"
    assert calls[0][1:3] == ["--jobid", "123"]
    assert calls[0][calls[0].index("--nodelist") + 1] == "node0"
    assert calls[0][calls[0].index("--container-image") + 1] == _IMAGE


def test_concurrent_callers_import_once(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(command)
        time.sleep(0.05)
        save_option = next(arg for arg in command if arg.startswith("--container-save="))
        _write_squashfs(Path(save_option.split("=", 1)[1]))
        return subprocess.CompletedProcess(command, 0)

    results: list[Path] = []
    with (
        patch("srtctl.core.container_image.platform.system", return_value="Linux"),
        patch("srtctl.core.container_image.platform.machine", return_value="x86_64"),
        patch("srtctl.core.container_image.shutil.which", return_value="/usr/bin/srun"),
        patch("srtctl.core.container_image.subprocess.run", side_effect=run),
    ):
        threads = [
            threading.Thread(
                target=lambda: results.append(
                    prepare_container_image(_IMAGE, str(tmp_path), job_id="123", node="node0")
                )
            )
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert len(calls) == 1
    assert len(results) == 2
    assert results[0] == results[1]


def test_failed_import_leaves_no_cache_entry(tmp_path: Path) -> None:
    with (
        patch("srtctl.core.container_image.platform.system", return_value="Linux"),
        patch("srtctl.core.container_image.platform.machine", return_value="aarch64"),
        patch("srtctl.core.container_image.shutil.which", return_value="/usr/bin/srun"),
        patch("srtctl.core.container_image.subprocess.run", side_effect=_importer([], returncode=17)),
        pytest.raises(RuntimeError, match="exit code 17"),
    ):
        prepare_container_image(_IMAGE, str(tmp_path), job_id="123", node="node0")

    assert not list(tmp_path.rglob("*.sqsh"))
    assert not [path for path in tmp_path.rglob("*") if path.is_dir() and path.name.startswith(f".{_DIGEST}")]


def test_missing_srun_is_actionable(tmp_path: Path) -> None:
    with (
        patch("srtctl.core.container_image.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="requires srun"),
    ):
        prepare_container_image(_IMAGE, str(tmp_path), job_id="123", node="node0")


def test_invalid_squashfs_output_is_not_published(tmp_path: Path) -> None:
    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess:
        save_option = next(arg for arg in command if arg.startswith("--container-save="))
        Path(save_option.split("=", 1)[1]).write_bytes(b"not-a-squashfs")
        return subprocess.CompletedProcess(command, 0)

    with (
        patch("srtctl.core.container_image.shutil.which", return_value="/usr/bin/srun"),
        patch("srtctl.core.container_image.subprocess.run", side_effect=run),
        pytest.raises(RuntimeError, match="not a SquashFS"),
    ):
        prepare_container_image(_IMAGE, str(tmp_path), job_id="123", node="node0")

    assert not list(tmp_path.rglob("*.sqsh"))


@pytest.mark.parametrize("image", ["example/server:latest", "example/server@sha256:short"])
def test_mutable_or_invalid_digest_is_rejected(image: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="@sha256"):
        prepare_container_image(image, str(tmp_path), job_id="123", node="node0")


def test_invalid_or_symlinked_cache_entry_is_rejected(tmp_path: Path) -> None:
    entry = tmp_path / "v1" / "linux-arm64" / f"{_DIGEST}.sqsh"
    target = tmp_path / "target.sqsh"
    _write_squashfs(target)
    entry.parent.mkdir(parents=True)
    entry.symlink_to(target)

    with (
        patch("srtctl.core.container_image.platform.system", return_value="Linux"),
        patch("srtctl.core.container_image.platform.machine", return_value="aarch64"),
        pytest.raises(RuntimeError, match="regular SquashFS"),
    ):
        prepare_container_image(_IMAGE, str(tmp_path), job_id="123", node="node0")


def test_orchestrator_replaces_runtime_image() -> None:
    runtime = RuntimeContext(
        job_id="123",
        run_name="test-123",
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node0",)),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=Path("/tmp/logs"),
        model_path=Path("/model"),
        container_image=Path(_IMAGE),
        gpus_per_node=8,
        network_interface=None,
    )
    config = MagicMock()
    config.model.container = _IMAGE
    orchestrator = SweepOrchestrator(config=config, runtime=runtime)

    with (
        patch("srtctl.cli.do_sweep.get_srtslurm_setting", return_value="/shared/cache"),
        patch("srtctl.cli.do_sweep.prepare_container_image", return_value=Path("/shared/cache/image.sqsh")) as prepare,
    ):
        orchestrator._prepare_container_image()

    prepare.assert_called_once_with(_IMAGE, "/shared/cache", job_id="123", node="node0")
    assert runtime.container_image == Path(_IMAGE)
    assert orchestrator.runtime.container_image == Path("/shared/cache/image.sqsh")


@pytest.mark.parametrize(
    ("setting", "image"),
    [
        (None, _IMAGE),
        ("/shared/cache", "/shared/image.sqsh"),
        ("/shared/cache", "registry.example.com/example/server:latest"),
    ],
)
def test_orchestrator_preserves_native_behavior(setting: str | None, image: str) -> None:
    orchestrator = MagicMock(spec=SweepOrchestrator)
    orchestrator.config = MagicMock()
    orchestrator.config.model.container = image
    orchestrator.runtime = MagicMock()
    with (
        patch("srtctl.cli.do_sweep.get_srtslurm_setting", return_value=setting),
        patch("srtctl.cli.do_sweep.prepare_container_image") as prepare,
    ):
        SweepOrchestrator._prepare_container_image(orchestrator)
    prepare.assert_not_called()
