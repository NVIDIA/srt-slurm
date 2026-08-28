# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for safe, one-time container image preparation."""

import fcntl
import os
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.container_image import ContainerPreparationError, PreparedContainer, prepare_container_image
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import ContainerCacheConfig, ContainerCacheMode

_DIGEST = "a" * 64
_IMAGE = f"registry.example.com/example/server@sha256:{_DIGEST}"


def _write_squashfs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"hsqs-container-data")


def _config(path: Path | None, *, mode: ContainerCacheMode = ContainerCacheMode.REQUIRED, timeout: int = 600):
    return ContainerCacheConfig(mode=mode, path=str(path) if path else None, lock_timeout_seconds=timeout)


def _prepare(image: str, config: ContainerCacheConfig, output_dir: Path, **kwargs) -> PreparedContainer:
    return prepare_container_image(image, config, job_id="123", node="node0", output_dir=output_dir, **kwargs)


def _importer(calls: list[list[str]], *, returncode: int = 0, valid: bool = True, probe_returncode: int | None = None):
    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(command)
        if "/bin/test" in command:
            return subprocess.CompletedProcess(command, returncode if probe_returncode is None else probe_returncode)
        if returncode == 0:
            save_option = next(arg for arg in command if arg.startswith("--container-save="))
            Path(save_option.split("=", 1)[1]).write_bytes(b"hsqs-data" if valid else b"invalid")
        return subprocess.CompletedProcess(command, returncode)

    return run


def test_cold_import_is_published_and_reused(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    with (
        patch("srtctl.core.container_image.platform.system", return_value="Linux"),
        patch("srtctl.core.container_image.platform.machine", return_value="aarch64"),
        patch("srtctl.core.container_image.shutil.which", return_value="/usr/bin/srun"),
        patch("srtctl.core.container_image.subprocess.run", side_effect=_importer(calls)),
    ):
        cold = _prepare(_IMAGE, _config(tmp_path / "cache"), tmp_path)
        warm = _prepare(_IMAGE, _config(tmp_path / "cache"), tmp_path)

    expected = tmp_path / "cache" / "v1" / "linux-arm64" / f"{_DIGEST}.sqsh"
    assert cold.effective_image == warm.effective_image == str(expected)
    assert cold.cache_hit is False
    assert warm.cache_hit is True
    assert cold.import_tool == "pyxis"
    assert len(calls) == 1
    assert f"--container-image={_IMAGE}" in calls[0]


def test_concurrent_callers_import_once(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(command)
        time.sleep(0.05)
        save_option = next(arg for arg in command if arg.startswith("--container-save="))
        _write_squashfs(Path(save_option.split("=", 1)[1]))
        return subprocess.CompletedProcess(command, 0)

    results: list[PreparedContainer] = []
    with (
        patch("srtctl.core.container_image.shutil.which", return_value="/usr/bin/srun"),
        patch("srtctl.core.container_image.subprocess.run", side_effect=run),
    ):
        threads = [
            threading.Thread(target=lambda: results.append(_prepare(_IMAGE, _config(tmp_path), tmp_path)))
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert len(calls) == 1
    assert {result.cache_hit for result in results} == {False, True}


@pytest.mark.parametrize(("returncode", "valid", "message"), [(17, True, "exit code 17"), (0, False, "SquashFS")])
def test_failed_import_is_not_published(tmp_path: Path, returncode: int, valid: bool, message: str) -> None:
    with (
        patch("srtctl.core.container_image.shutil.which", return_value="/usr/bin/srun"),
        patch(
            "srtctl.core.container_image.subprocess.run", side_effect=_importer([], returncode=returncode, valid=valid)
        ),
        pytest.raises(ContainerPreparationError, match=message),
    ):
        _prepare(_IMAGE, _config(tmp_path), tmp_path)

    assert not list(tmp_path.rglob("*.sqsh"))
    assert not [path for path in tmp_path.rglob("*") if path.is_dir() and path.name.endswith(".tmp")]


def test_missing_srun_is_actionable(tmp_path: Path) -> None:
    with (
        patch("srtctl.core.container_image.shutil.which", return_value=None),
        pytest.raises(ContainerPreparationError, match="srun is not available"),
    ):
        _prepare(_IMAGE, _config(tmp_path), tmp_path)


def test_auto_falls_back_but_required_rejects_mutable_tag(tmp_path: Path) -> None:
    auto = _prepare("example/server:latest", _config(None, mode=ContainerCacheMode.AUTO), tmp_path)
    assert auto.effective_image == "example/server:latest"
    assert "not pinned" in (auto.fallback_reason or "")

    with pytest.raises(ContainerPreparationError, match="not pinned"):
        _prepare("example/server:latest", _config(None), tmp_path)


def test_auto_falls_back_when_cache_is_unavailable(tmp_path: Path) -> None:
    with patch("srtctl.core.container_image.shutil.which", return_value=None):
        result = _prepare(_IMAGE, _config(tmp_path, mode=ContainerCacheMode.AUTO), tmp_path)
    assert result.effective_image == _IMAGE
    assert "srun" in (result.fallback_reason or "")


def test_native_mode_does_not_touch_cache(tmp_path: Path) -> None:
    result = _prepare(_IMAGE, _config(None, mode=ContainerCacheMode.NATIVE), tmp_path)
    assert result.effective_image == _IMAGE
    assert result.cache_mode == "native"
    assert not list(tmp_path.iterdir())


def test_local_squashfs_is_validated_and_preserved(tmp_path: Path) -> None:
    image = tmp_path / "image.sqsh"
    _write_squashfs(image)
    result = _prepare(str(image), _config(None, mode=ContainerCacheMode.AUTO), tmp_path)
    assert result.effective_image == str(image)
    assert result.cache_hit is None


def test_symlinked_or_unsafe_cache_content_is_rejected(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    entry = cache_root / "v1" / "linux-amd64" / f"{_DIGEST}.sqsh"
    target = tmp_path / "target.sqsh"
    _write_squashfs(target)
    entry.parent.mkdir(parents=True)
    entry.symlink_to(target)

    with (
        patch("srtctl.core.container_image.platform.system", return_value="Linux"),
        patch("srtctl.core.container_image.platform.machine", return_value="x86_64"),
        pytest.raises(ContainerPreparationError, match="cannot validate"),
    ):
        _prepare(_IMAGE, _config(cache_root), tmp_path)


def test_group_writable_cache_directory_is_rejected(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir(mode=0o770)
    cache_root.chmod(0o770)
    with pytest.raises(ContainerPreparationError, match="writable by another user"):
        _prepare(_IMAGE, _config(cache_root), tmp_path)


def test_unsafe_automatic_cache_parent_is_rejected(tmp_path: Path) -> None:
    cache_parent = tmp_path / ".srtctl"
    cache_parent.mkdir(mode=0o770)
    cache_parent.chmod(0o770)
    with pytest.raises(ContainerPreparationError, match="writable by another user"):
        _prepare(_IMAGE, _config(None), tmp_path)


def test_group_writable_cache_entry_is_rejected(tmp_path: Path) -> None:
    # Use patched platform naming to keep the expected layout independent of the host.
    entry = tmp_path / "v1" / "linux-amd64" / f"{_DIGEST}.sqsh"
    _write_squashfs(entry)
    entry.chmod(0o660)
    with (
        patch("srtctl.core.container_image.platform.system", return_value="Linux"),
        patch("srtctl.core.container_image.platform.machine", return_value="x86_64"),
        pytest.raises(ContainerPreparationError, match="safe regular file"),
    ):
        _prepare(_IMAGE, _config(tmp_path), tmp_path)


def test_unsafe_lock_file_is_rejected(tmp_path: Path) -> None:
    lock_path = tmp_path / "v1" / "linux-amd64" / f"{_DIGEST}.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.touch(mode=0o660)
    lock_path.chmod(0o660)
    with (
        patch("srtctl.core.container_image.platform.system", return_value="Linux"),
        patch("srtctl.core.container_image.platform.machine", return_value="x86_64"),
        pytest.raises(ContainerPreparationError, match="lock is unsafe"),
    ):
        _prepare(_IMAGE, _config(tmp_path), tmp_path)


def test_lock_wait_is_bounded(tmp_path: Path) -> None:
    cache_dir = tmp_path / "v1" / "linux-amd64"
    cache_dir.mkdir(parents=True)
    lock_path = cache_dir / f"{_DIGEST}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with (
            patch("srtctl.core.container_image.platform.system", return_value="Linux"),
            patch("srtctl.core.container_image.platform.machine", return_value="x86_64"),
            pytest.raises(ContainerPreparationError, match="timed out"),
        ):
            _prepare(_IMAGE, _config(tmp_path, timeout=1), tmp_path)


def test_automatic_path_is_derived_from_output_directory(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    with (
        patch("srtctl.core.container_image.shutil.which", return_value="/usr/bin/srun"),
        patch("srtctl.core.container_image.subprocess.run", side_effect=_importer(calls)),
    ):
        result = _prepare(_IMAGE, _config(None, mode=ContainerCacheMode.AUTO), tmp_path)
    expected_root = tmp_path / ".srtctl" / "container-cache" / str(os.geteuid())
    assert Path(result.effective_image).is_relative_to(expected_root)
    assert result.cache_path_source == "output_dir"


def test_multi_node_visibility_is_checked_for_each_het_group(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    groups = ((0, ("n0", "n1")), (1, ("n2",)))
    with (
        patch("srtctl.core.container_image.shutil.which", return_value="/usr/bin/srun"),
        patch("srtctl.core.container_image.subprocess.run", side_effect=_importer(calls)),
    ):
        _prepare(_IMAGE, _config(tmp_path), tmp_path, visibility_groups=groups)

    probes = [command for command in calls if "/bin/test" in command]
    assert len(probes) == 2
    assert {option for command in probes for option in command if option.startswith("--het-group=")} == {
        "--het-group=0",
        "--het-group=1",
    }


def test_failed_visibility_check_prevents_required_import(tmp_path: Path) -> None:
    with (
        patch("srtctl.core.container_image.shutil.which", return_value="/usr/bin/srun"),
        patch("srtctl.core.container_image.subprocess.run", side_effect=_importer([], probe_returncode=1)),
        pytest.raises(ContainerPreparationError, match="every allocated node"),
    ):
        _prepare(_IMAGE, _config(tmp_path), tmp_path, visibility_groups=((None, ("n0", "n1")),))


def test_provenance_redacts_registry_credentials() -> None:
    prepared = PreparedContainer(
        declared_image=f"docker://user:secret@registry.example.com/image@sha256:{_DIGEST}",
        resolved_image=f"docker://user:secret@registry.example.com/image@sha256:{_DIGEST}",
        effective_image="/cache/image.sqsh",
        platform="linux/amd64",
        digest=f"sha256:{_DIGEST}",
        cache_mode="auto",
        cache_path_source="explicit",
        cache_hit=True,
    )
    lock_data = prepared.as_lock_data()
    assert "secret" not in lock_data["declared_image"]
    assert "user:<redacted>" in lock_data["declared_image"]


def test_orchestrator_replaces_runtime_image_and_records_provenance() -> None:
    runtime = RuntimeContext(
        job_id="123",
        run_name="test-123",
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node0",)),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=Path("/output/123/logs"),
        model_path=Path("/model"),
        container_image=_IMAGE,
        gpus_per_node=8,
        network_interface=None,
    )
    config = MagicMock()
    config.model.container = _IMAGE
    orchestrator = SweepOrchestrator(config=config, runtime=runtime)
    prepared = PreparedContainer(
        declared_image=_IMAGE,
        resolved_image=_IMAGE,
        effective_image="/cache/image.sqsh",
        platform="linux/amd64",
        digest=f"sha256:{_DIGEST}",
        cache_mode="auto",
        cache_path_source="output_dir",
        cache_hit=True,
    )

    with (
        patch(
            "srtctl.cli.do_sweep.get_container_cache_config", return_value=_config(None, mode=ContainerCacheMode.AUTO)
        ),
        patch("srtctl.cli.do_sweep.prepare_container_image", return_value=prepared) as prepare,
    ):
        orchestrator._prepare_container_image()

    assert prepare.call_args.kwargs["output_dir"] == Path("/output")
    assert orchestrator.runtime.container_image == Path("/cache/image.sqsh")
    assert orchestrator.runtime.prepared_container is prepared
