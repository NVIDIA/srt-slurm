# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture git state for source trees mounted into a run."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

GIT_STATE_FILENAME = "git_state.txt"
_GIT_TIMEOUT_S = 10
_MAX_UNTRACKED_BYTES = 200_000
_URL_USERINFO_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)(?P<userinfo>[^/@\s]+)@")


@dataclass(frozen=True)
class GitSnapshotSource:
    """A host path whose enclosing git repository should be captured."""

    label: str
    path: Path


def _expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def _run_git(repo: Path, args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except Exception as e:
        logger.debug("git command failed in %s: %s", repo, e)
        return None
    if check and result.returncode != 0:
        return None
    return result


def _find_git_root(path: Path) -> Path | None:
    candidate = path if path.is_dir() else path.parent
    result = _run_git(candidate, ["rev-parse", "--show-toplevel"], check=True)
    if result is None:
        return None
    root = result.stdout.strip()
    return Path(root).resolve() if root else None


def _split_mount(mount_spec: str) -> tuple[str, str] | None:
    if ":" not in mount_spec:
        return None
    return mount_spec.split(":", 1)


def git_snapshot_sources_from_extra_mounts(config: SrtConfig) -> list[GitSnapshotSource]:
    """Collect git snapshot source paths from explicit extra_mount entries."""
    sources: list[GitSnapshotSource] = []
    if config.extra_mount:
        for mount_spec in config.extra_mount:
            split = _split_mount(mount_spec)
            if split is None:
                sources.append(GitSnapshotSource("extra_mount", _expand_path(mount_spec)))
                continue
            host_path, container_path = split
            sources.append(GitSnapshotSource(f"extra_mount:{container_path}", _expand_path(host_path)))

    return sources


def _git_stdout(repo: Path, args: list[str]) -> str:
    result = _run_git(repo, args)
    if result is None:
        return "<git command failed>\n"
    if result.returncode != 0:
        stderr = _redact_url_credentials(result.stderr.strip())
        return f"<git command failed: {stderr or result.returncode}>\n"
    return _redact_url_credentials(result.stdout) if result.stdout else "<none>\n"


def _redact_url_credentials(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        scheme = match.group("scheme")
        userinfo = match.group("userinfo")
        if ":" in userinfo:
            username, _password = userinfo.split(":", 1)
            return f"{scheme}{username}:<redacted>@"
        if userinfo.lower().startswith(("ghp_", "github_pat_", "glpat-", "x-access-token")):
            return f"{scheme}<redacted>@"
        return match.group(0)

    return _URL_USERINFO_RE.sub(replace, text)


def _untracked_files(repo: Path) -> list[str]:
    result = _run_git(repo, ["ls-files", "--others", "--exclude-standard", "-z"])
    if result is None or result.returncode != 0 or not result.stdout:
        return []
    return [p for p in result.stdout.split("\0") if p]


def _format_untracked_file(repo: Path, rel_path: str) -> list[str]:
    path = repo / rel_path
    try:
        data = path.read_bytes()
    except Exception as e:
        return [f"# unable to read untracked file {rel_path}: {e}\n"]

    header = [
        f"diff --git a/{rel_path} b/{rel_path}\n",
        "new file mode 100644\n",
        "--- /dev/null\n",
        f"+++ b/{rel_path}\n",
    ]
    if len(data) > _MAX_UNTRACKED_BYTES:
        return [*header, f"# omitted: untracked file is {len(data)} bytes\n"]
    if b"\0" in data:
        return [*header, f"# omitted: untracked file appears binary ({len(data)} bytes)\n"]

    text = data.decode("utf-8", errors="replace")
    return [*header, "@@\n", *[f"+{line}\n" for line in text.splitlines()]]


def _format_repo_snapshot(repo: Path, labels: list[str], source_paths: list[Path]) -> list[str]:
    lines = [
        "\n",
        "=" * 80 + "\n",
        f"Repository: {repo}\n",
        f"Labels: {', '.join(sorted(set(labels)))}\n",
        "Source paths:\n",
    ]
    lines.extend(f"  - {p}\n" for p in source_paths)

    for title, args in [
        ("Remote URLs", ["remote", "-v"]),
        ("Branch", ["branch", "--show-current"]),
        ("HEAD", ["rev-parse", "HEAD"]),
        ("Status", ["status", "--short", "--branch"]),
        ("Last 10 commits", ["log", "--decorate", "--oneline", "-n", "10"]),
        ("Staged diff", ["diff", "--cached", "--no-ext-diff"]),
        ("Unstaged diff", ["diff", "--no-ext-diff"]),
    ]:
        lines.extend(["\n", f"## {title}\n", _git_stdout(repo, args)])

    untracked = _untracked_files(repo)
    lines.extend(["\n", "## Untracked files\n"])
    if not untracked:
        lines.append("<none>\n")
    else:
        lines.extend(f"  - {path}\n" for path in untracked)
        lines.append("\n## Untracked file contents\n")
        for rel_path in untracked:
            lines.extend(_format_untracked_file(repo, rel_path))
            lines.append("\n")

    return lines


def write_git_state_snapshot(output_path: Path, sources: Iterable[GitSnapshotSource]) -> bool:
    """Write a best-effort git state snapshot.

    The output includes the last 10 commits, staged diff, unstaged diff,
    and untracked file contents for every unique git repository found
    under the supplied source paths.
    """
    try:
        grouped: dict[Path, tuple[list[str], list[Path]]] = {}
        considered: list[GitSnapshotSource] = []
        for source in sources:
            expanded = GitSnapshotSource(source.label, _expand_path(source.path))
            considered.append(expanded)
            root = _find_git_root(expanded.path)
            if root is None:
                continue
            labels, paths = grouped.setdefault(root, ([], []))
            labels.append(expanded.label)
            paths.append(expanded.path)

        lines = [
            "# srtctl git state snapshot\n",
            f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n",
            "\n",
            "Sources considered:\n",
        ]
        if considered:
            lines.extend(f"  - {source.label}: {source.path}\n" for source in considered)
        else:
            lines.append("  <none>\n")

        if not grouped:
            lines.extend(["\n", "No git repositories found under the considered sources.\n"])
        else:
            for repo, (labels, paths) in sorted(grouped.items(), key=lambda item: str(item[0])):
                lines.extend(_format_repo_snapshot(repo, labels, paths))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(lines))
        logger.info("Wrote git state snapshot: %s", output_path)
        return True
    except Exception as e:
        logger.warning("Failed to write git state snapshot %s: %s", output_path, e)
        return False
