# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one place that answers "which srtctl is this".

The package version comes from the git tag through hatch-vcs (``pyproject.toml``);
this module reads it back from the installed metadata and pairs it with the
commit and the protocol versions srtctl speaks, so ``srtctl --version``, the
lockfile stamp, and the release notes are the same numbers.
"""

from __future__ import annotations

import contextlib
import subprocess
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

FALLBACK_VERSION = "0.0.0+unknown"


def package_version() -> str:
    """Installed ``srtctl`` version, or the hatch-vcs file, or the fallback."""
    with contextlib.suppress(metadata.PackageNotFoundError):
        return metadata.version("srtctl")
    with contextlib.suppress(ImportError):
        from srtctl._version import __version__  # type: ignore[import-not-found]

        return str(__version__)
    return FALLBACK_VERSION


def source_commit() -> str | None:
    """HEAD of the checkout this module was imported from, when it is a git repo."""
    root = Path(__file__).resolve().parents[2]
    with contextlib.suppress(Exception):
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


@dataclass(frozen=True)
class VersionInfo:
    """Everything a release note or a lockfile needs to say about the tool."""

    version: str
    commit: str | None
    recipe_schema_versions: tuple[int, ...]
    lockfile_version: int

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        commit = f" ({self.commit})" if self.commit else ""
        schemas = ", ".join(str(v) for v in self.recipe_schema_versions)
        return f"srtctl {self.version}{commit}; recipe schema {schemas}; lockfile v{self.lockfile_version}"


def version_info() -> VersionInfo:
    from srtctl.core.lockfile import LOCKFILE_VERSION
    from srtctl.core.schema import SUPPORTED_SCHEMA_VERSIONS

    return VersionInfo(
        version=package_version(),
        commit=source_commit(),
        recipe_schema_versions=tuple(SUPPORTED_SCHEMA_VERSIONS),
        lockfile_version=LOCKFILE_VERSION,
    )
