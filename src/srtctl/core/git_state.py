# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture git state for source trees mounted into a run."""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

GIT_STATE_FILENAME = "git_state.txt"
_METADATA_TIMEOUT_S = 10
_CONTENT_TIMEOUT_S = 60
_REPO_TIMEOUT_S = 120
_SLOW_COMMAND_LOG_THRESHOLD_S = 5
_MAX_UNTRACKED_BYTES = 200_000
_URL_USERINFO_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)(?P<userinfo>[^/@\s]+)@")


@dataclass(frozen=True)
class GitSnapshotSource:
    """A host path whose enclosing git repository should be captured."""

    label: str
    path: Path


class GitCommandOutcome(str, Enum):
    """Possible outcomes from a best-effort git command."""

    SUCCESS = "success"
    NONZERO = "nonzero"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class GitCommandResult:
    """Structured result for a git command used in a snapshot."""

    command: tuple[str, ...]
    outcome: GitCommandOutcome
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    timeout_seconds: float | None = None
    elapsed_seconds: float = 0
    error: str = ""


def _expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def _run_git(
    repo: Path,
    args: list[str],
    *,
    timeout_seconds: float = _METADATA_TIMEOUT_S,
    deadline: float | None = None,
) -> GitCommandResult:
    command = ("git", *(_redact_url_credentials(arg) for arg in args))
    started_at = time.monotonic()
    effective_timeout = timeout_seconds
    if deadline is not None:
        remaining = deadline - started_at
        if remaining <= 0:
            return GitCommandResult(command=command, outcome=GitCommandOutcome.BUDGET_EXHAUSTED)
        effective_timeout = min(timeout_seconds, remaining)

    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - started_at
        return GitCommandResult(
            command=command,
            outcome=GitCommandOutcome.TIMEOUT,
            stdout=_redact_url_credentials(_subprocess_text(e.stdout)),
            stderr=_redact_url_credentials(_subprocess_text(e.stderr)),
            timeout_seconds=effective_timeout,
            elapsed_seconds=elapsed,
        )
    except Exception as e:  # noqa: BLE001
        elapsed = time.monotonic() - started_at
        logger.debug("git command could not execute in %s: %s", repo, _redact_url_credentials(str(e)))
        return GitCommandResult(
            command=command,
            outcome=GitCommandOutcome.EXECUTION_ERROR,
            elapsed_seconds=elapsed,
            error=_redact_url_credentials(str(e)),
        )

    elapsed = time.monotonic() - started_at
    if elapsed >= _SLOW_COMMAND_LOG_THRESHOLD_S:
        logger.info("Slow git command in %s completed in %.1fs: %s", repo, elapsed, shlex.join(command))
    return GitCommandResult(
        command=command,
        outcome=GitCommandOutcome.SUCCESS if result.returncode == 0 else GitCommandOutcome.NONZERO,
        stdout=result.stdout,
        stderr=_redact_url_credentials(result.stderr),
        returncode=result.returncode,
        elapsed_seconds=elapsed,
    )


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _find_git_root(path: Path) -> Path | None:
    candidate = path if path.is_dir() else path.parent
    result = _run_git(candidate, ["rev-parse", "--show-toplevel"])
    if result.outcome != GitCommandOutcome.SUCCESS:
        return None
    root = result.stdout.strip()
    return Path(root).resolve() if root else None


def head_commit(path: Path) -> tuple[Path, str] | None:
    """Git root containing ``path`` and its HEAD commit, when available."""
    root = _find_git_root(path)
    if root is None:
        return None
    result = _run_git(root, ["rev-parse", "HEAD"])
    commit = result.stdout.strip() if result.outcome == GitCommandOutcome.SUCCESS else ""
    return (root, commit) if commit else None


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


def _git_stdout(repo: Path, args: list[str], *, timeout_seconds: float, deadline: float) -> str:
    result = _run_git(repo, args, timeout_seconds=timeout_seconds, deadline=deadline)
    if result.outcome != GitCommandOutcome.SUCCESS:
        return _format_git_failure(result)
    return _redact_url_credentials(result.stdout) if result.stdout else "<none>\n"


def _format_git_failure(result: GitCommandResult) -> str:
    command = shlex.join(result.command)
    stderr = _redact_url_credentials(result.stderr.strip())
    error = _redact_url_credentials(result.error.strip())
    if result.outcome == GitCommandOutcome.TIMEOUT:
        timeout = _format_seconds(result.timeout_seconds or 0)
        return f"<git command timed out after {timeout}: {command}>\n"
    if result.outcome == GitCommandOutcome.NONZERO:
        detail = f": {stderr}" if stderr else ""
        return f"<git command failed with exit {result.returncode}: {command}{detail}>\n"
    if result.outcome == GitCommandOutcome.EXECUTION_ERROR:
        detail = f": {error}" if error else ""
        return f"<git command could not execute: {command}{detail}>\n"
    if result.outcome == GitCommandOutcome.BUDGET_EXHAUSTED:
        return f"<git command skipped because repository budget was exhausted: {command}>\n"
    raise ValueError(f"Cannot format successful git command as a failure: {result.command}")


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.0f}s" if float(seconds).is_integer() else f"{seconds:.1f}s"


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


def _untracked_files(repo: Path, *, deadline: float) -> tuple[list[str] | None, str | None]:
    result = _run_git(
        repo,
        ["ls-files", "--others", "--exclude-standard", "-z"],
        timeout_seconds=_CONTENT_TIMEOUT_S,
        deadline=deadline,
    )
    if result.outcome != GitCommandOutcome.SUCCESS:
        return None, _format_git_failure(result)
    return [p for p in result.stdout.split("\0") if p], None


def _format_untracked_file(repo: Path, rel_path: str) -> list[str]:
    path = repo / rel_path
    try:
        data = path.read_bytes()
    except Exception as e:  # noqa: BLE001
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
    deadline = time.monotonic() + _REPO_TIMEOUT_S
    lines = [
        "\n",
        "=" * 80 + "\n",
        f"Repository: {repo}\n",
        f"Labels: {', '.join(sorted(set(labels)))}\n",
        "Source paths:\n",
    ]
    lines.extend(f"  - {p}\n" for p in source_paths)

    for title, args, timeout_seconds in [
        ("Remote URLs", ["remote", "-v"], _METADATA_TIMEOUT_S),
        ("Branch", ["branch", "--show-current"], _METADATA_TIMEOUT_S),
        ("HEAD", ["rev-parse", "HEAD"], _METADATA_TIMEOUT_S),
        (
            "Status",
            ["status", "--short", "--branch", "--untracked-files=no", "--ignore-submodules=all"],
            _CONTENT_TIMEOUT_S,
        ),
        ("Last 10 commits", ["log", "--decorate", "--oneline", "-n", "10"], _CONTENT_TIMEOUT_S),
        ("Staged diff", ["diff", "--cached", "--no-ext-diff"], _CONTENT_TIMEOUT_S),
        ("Unstaged diff", ["diff", "--no-ext-diff"], _CONTENT_TIMEOUT_S),
    ]:
        lines.extend(
            ["\n", f"## {title}\n", _git_stdout(repo, args, timeout_seconds=timeout_seconds, deadline=deadline)]
        )

    untracked, untracked_error = _untracked_files(repo, deadline=deadline)
    lines.extend(["\n", "## Untracked files\n"])
    if untracked_error:
        lines.append(untracked_error)
    elif not untracked:
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
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to write git state snapshot %s: %s", output_path, e)
        return False
