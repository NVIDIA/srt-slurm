# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safely prepare reusable SquashFS images for registry containers."""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from srtctl.core.schema import ContainerCacheConfig

logger = logging.getLogger(__name__)

_DIGEST_RE = re.compile(r"@sha256:([0-9a-fA-F]{64})$")
_SQUASHFS_MAGICS = {b"hsqs", b"sqsh"}
_URL_USERINFO_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)(?P<userinfo>[^/@\s]+)@")


class ContainerPreparationError(RuntimeError):
    """A container could not be safely prepared for shared reuse."""


@dataclass(frozen=True)
class PreparedContainer:
    """Container identity and the image selected for execution."""

    declared_image: str
    resolved_image: str
    effective_image: str
    platform: str
    digest: str | None
    cache_mode: str
    cache_path_source: str | None
    cache_hit: bool | None
    import_duration_seconds: float | None = None
    import_tool: str | None = None
    fallback_reason: str | None = None

    def as_lock_data(self) -> dict[str, str | bool | float | None]:
        """Return credential-redacted, YAML-safe provenance."""
        data = asdict(self)
        for key in ("declared_image", "resolved_image", "effective_image", "fallback_reason"):
            value = data.get(key)
            if isinstance(value, str):
                data[key] = _redact_credentials(value)
        return data


def prepare_container_image(
    image: str,
    config: ContainerCacheConfig,
    *,
    job_id: str,
    node: str,
    output_dir: Path,
    visibility_groups: Sequence[tuple[int | None, Sequence[str]]] = (),
) -> PreparedContainer:
    """Select native handling or materialize one safe, shared SquashFS image."""
    from srtctl.core.schema import ContainerCacheMode

    declared = os.path.expandvars(image)
    selected_platform = _platform_name()
    if config.mode == ContainerCacheMode.NATIVE:
        return _native_result(declared, selected_platform, config.mode.value)

    if _looks_like_local_path(declared):
        local_path = Path(declared).expanduser().absolute()
        try:
            _validate_squashfs(local_path)
        except ContainerPreparationError as exc:
            if config.mode == ContainerCacheMode.REQUIRED:
                raise
            reason = _redact_credentials(str(exc))
            logger.warning("Container validation failed; using native image handling: %s", reason)
            return _native_result(declared, selected_platform, config.mode.value, fallback_reason=reason)
        return PreparedContainer(
            declared_image=declared,
            resolved_image=str(local_path),
            effective_image=str(local_path),
            platform=selected_platform,
            digest=None,
            cache_mode=config.mode.value,
            cache_path_source=None,
            cache_hit=None,
        )

    match = _DIGEST_RE.search(declared)
    if match is None:
        reason = "container image is not pinned with @sha256:<digest>"
        if config.mode == ContainerCacheMode.REQUIRED:
            raise ContainerPreparationError(reason)
        return _native_result(declared, selected_platform, config.mode.value, fallback_reason=reason)

    path_source = "explicit" if config.path else "output_dir"
    cache_root = (
        Path(os.path.expandvars(config.path)).expanduser()
        if config.path
        else output_dir.absolute() / ".srtctl" / "container-cache" / str(os.geteuid())
    )
    if not cache_root.is_absolute():
        cache_root = cache_root.absolute()
    cache_tree_root = cache_root if config.path else output_dir.absolute() / ".srtctl"
    try:
        return _prepare_registry_image(
            declared,
            match.group(1).lower(),
            cache_root,
            selected_platform,
            config,
            path_source,
            job_id,
            node,
            visibility_groups,
            cache_tree_root,
        )
    except ContainerPreparationError as exc:
        if config.mode == ContainerCacheMode.REQUIRED:
            raise
        reason = _redact_credentials(str(exc))
        logger.warning("Container cache unavailable; using native image handling: %s", reason)
        return _native_result(declared, selected_platform, config.mode.value, fallback_reason=reason)


def _prepare_registry_image(
    image: str,
    digest_hex: str,
    cache_root: Path,
    selected_platform: str,
    config: ContainerCacheConfig,
    path_source: str,
    job_id: str,
    node: str,
    visibility_groups: Sequence[tuple[int | None, Sequence[str]]],
    cache_tree_root: Path,
) -> PreparedContainer:
    _prepare_cache_root(cache_root, cache_tree_root)
    cache_dir = cache_root / "v1" / selected_platform.replace("/", "-")
    _prepare_cache_root(cache_dir, cache_root)
    image_path = cache_dir / f"{digest_hex}.sqsh"
    digest = f"sha256:{digest_hex}"

    with _entry_lock(image_path.with_suffix(".lock"), config.lock_timeout_seconds):
        if image_path.exists() or image_path.is_symlink():
            _validate_squashfs(image_path)
            _verify_shared_visibility(image_path, job_id, visibility_groups)
            logger.info("Container cache hit: %s", image_path)
            return PreparedContainer(
                image, image, str(image_path), selected_platform, digest, config.mode.value, path_source, True
            )

        srun = shutil.which("srun")
        if srun is None:
            raise ContainerPreparationError("srun is not available on the orchestration node")

        logger.info("Container cache miss: importing %s", digest)
        started_at = time.monotonic()
        temporary_dir = Path(tempfile.mkdtemp(prefix=f".{digest_hex}.", suffix=".tmp", dir=cache_dir))
        temporary_path = temporary_dir / "image.sqsh"
        try:
            result = subprocess.run(
                [
                    srun,
                    "--jobid",
                    job_id,
                    "--overlap",
                    "--nodes=1",
                    "--ntasks=1",
                    f"--nodelist={node}",
                    f"--container-image={image}",
                    "--no-container-entrypoint",
                    "--no-container-mount-home",
                    f"--container-save={temporary_path}",
                    "/bin/true",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise ContainerPreparationError(f"Pyxis container import failed with exit code {result.returncode}")
            os.chmod(temporary_path, 0o600, follow_symlinks=False)
            _validate_squashfs(temporary_path)
            _fsync_file(temporary_path)
            os.replace(temporary_path, image_path)
            _fsync_directory(cache_dir)
            _validate_squashfs(image_path)
            _verify_shared_visibility(image_path, job_id, visibility_groups)
        except ContainerPreparationError:
            raise
        except Exception as exc:
            raise ContainerPreparationError(_redact_credentials(str(exc))) from exc
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)

        duration = time.monotonic() - started_at
        logger.info("Prepared container in %.1fs: %s", duration, image_path)
        return PreparedContainer(
            declared_image=image,
            resolved_image=image,
            effective_image=str(image_path),
            platform=selected_platform,
            digest=digest,
            cache_mode=config.mode.value,
            cache_path_source=path_source,
            cache_hit=False,
            import_duration_seconds=duration,
            import_tool="pyxis",
        )


def _prepare_cache_root(cache_root: Path, tree_root: Path) -> None:
    try:
        tree_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        current = tree_root
        _validate_cache_directory(current)
        for part in cache_root.relative_to(tree_root).parts:
            current /= part
            current.mkdir(mode=0o700, exist_ok=True)
            _validate_cache_directory(current)
    except ContainerPreparationError:
        raise
    except OSError as exc:
        raise ContainerPreparationError(f"cannot prepare container cache directory {cache_root}: {exc}") from exc


def _validate_cache_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ContainerPreparationError(f"cannot inspect container cache directory {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ContainerPreparationError(f"container cache path is not a real directory: {path}")
    if info.st_uid not in {os.geteuid(), 0}:
        raise ContainerPreparationError(f"container cache directory has an unexpected owner: {path}")
    if info.st_mode & 0o022:
        raise ContainerPreparationError(f"container cache directory is writable by another user: {path}")


def _validate_squashfs(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid not in {os.geteuid(), 0} or info.st_mode & 0o022:
                raise ContainerPreparationError(f"container image is not a safe regular file: {path}")
            if info.st_size <= 4 or os.read(fd, 4) not in _SQUASHFS_MAGICS:
                raise ContainerPreparationError(f"container image is not a SquashFS file: {path}")
        finally:
            os.close(fd)
    except ContainerPreparationError:
        raise
    except OSError as exc:
        raise ContainerPreparationError(f"cannot validate container image {path}: {exc}") from exc


@contextmanager
def _entry_lock(lock_path: Path, timeout_seconds: int) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ContainerPreparationError(f"cannot open container cache lock {lock_path}: {exc}") from exc

    with os.fdopen(fd, "a+") as lock_file:
        info = os.fstat(lock_file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid not in {os.geteuid(), 0} or info.st_mode & 0o022:
            raise ContainerPreparationError(f"container cache lock is unsafe: {lock_path}")
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ContainerPreparationError(
                        f"timed out after {timeout_seconds}s waiting for container cache lock {lock_path}"
                    ) from exc
                time.sleep(min(0.1, remaining))
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _verify_shared_visibility(
    path: Path,
    job_id: str,
    groups: Sequence[tuple[int | None, Sequence[str]]],
) -> None:
    """Verify a cache path on every allocation node when more than one is used."""
    unique_nodes = {node for _, nodes in groups for node in nodes}
    if len(unique_nodes) <= 1:
        return
    srun = shutil.which("srun")
    if srun is None:
        raise ContainerPreparationError("srun is not available for shared-cache validation")
    for het_group, raw_nodes in groups:
        nodes = tuple(dict.fromkeys(raw_nodes))
        if not nodes:
            continue
        command = [srun, "--jobid", job_id, "--overlap"]
        if het_group is not None:
            command.append(f"--het-group={het_group}")
        command.extend(
            [
                f"--nodes={len(nodes)}",
                f"--ntasks={len(nodes)}",
                "--ntasks-per-node=1",
                f"--nodelist={','.join(nodes)}",
                "/bin/test",
                "-r",
                str(path),
            ]
        )
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ContainerPreparationError(f"cannot validate shared container cache: {exc}") from exc
        if result.returncode != 0:
            raise ContainerPreparationError("container cache is not readable from every allocated node")


def _native_result(
    image: str,
    selected_platform: str,
    cache_mode: str,
    *,
    fallback_reason: str | None = None,
) -> PreparedContainer:
    match = _DIGEST_RE.search(image)
    return PreparedContainer(
        declared_image=image,
        resolved_image=image,
        effective_image=image,
        platform=selected_platform,
        digest=f"sha256:{match.group(1).lower()}" if match else None,
        cache_mode=cache_mode,
        cache_path_source=None,
        cache_hit=None,
        fallback_reason=fallback_reason,
    )


def _looks_like_local_path(image: str) -> bool:
    return image.startswith(("/", "./", "~/"))


def _platform_name() -> str:
    architecture = platform.machine().lower()
    architecture = {"x86_64": "amd64", "aarch64": "arm64"}.get(architecture, architecture)
    return f"{platform.system().lower()}/{architecture}"


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            os.fsync(fd)
    finally:
        os.close(fd)


def _redact_credentials(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        userinfo = match.group("userinfo")
        username = userinfo.split(":", 1)[0] if ":" in userinfo else "<redacted>"
        authority = f"{username}:<redacted>" if username != "<redacted>" else "<redacted>"
        return f"{match.group('scheme')}{authority}@"

    return _URL_USERINFO_RE.sub(replace, text)
