# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One way to say "this code, from git": the ``source:`` shape.

Used by ``services[].source`` (build a sidecar before launching it) and
``dynamo.source`` (which Dynamo to install). A source names a repository and an
immutable ref. Because a ref like ``refs/pull/14000/head`` moves, ``srtctl apply``
resolves every non-SHA ``rev`` to a commit with ``git ls-remote`` and records it
as ``sha`` in the submitted ``config.yaml`` (see :func:`pin_source_revs`), so the
job builds exactly what the lockfile says and build caches key on the commit.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Any, ClassVar

from marshmallow import Schema, ValidationError
from marshmallow_dataclass import dataclass

logger = logging.getLogger(__name__)

# Branch names that move out from under a build. Tags, SHAs, and PR head refs are fine.
MOVING_REFS: frozenset[str] = frozenset({"main", "master", "HEAD"})

# 4 is git's minimum abbreviation; legacy recipes pin short hashes like "abc123".
_SHA_RE = re.compile(r"^[0-9a-f]{4,40}$")


def is_commit_sha(rev: str) -> bool:
    """Whether ``rev`` already names a commit (4 to 40 hex chars, an abbreviated or full SHA)."""
    return bool(_SHA_RE.match(rev.strip()))


def validate_ref(rev: str, *, where: str) -> None:
    """Reject empty or moving refs; ``where`` labels the recipe path in the error."""
    if not rev.strip():
        raise ValidationError(f"{where}.rev must be a non-empty ref (commit SHA, tag, or refs/pull/<n>/head)")
    if rev.strip() in MOVING_REFS:
        raise ValidationError(
            f"{where}.rev must be an immutable ref (commit SHA, tag, or refs/pull/<n>/head), "
            f"not a moving branch name: {rev!r}"
        )


def resolve_rev(git: str, rev: str, *, timeout: float = 60.0) -> str:
    """Resolve ``rev`` in ``git`` to a commit SHA with ``git ls-remote``.

    A SHA is returned as-is. For a tag the peeled commit (``<ref>^{}``) is
    preferred over the tag object so annotated and lightweight tags resolve
    the same way. Raises ``RuntimeError`` when the ref does not exist or the
    remote cannot be reached.
    """
    rev = rev.strip()
    if is_commit_sha(rev):
        return rev
    patterns = [rev, f"refs/tags/{rev}", f"refs/heads/{rev}"] if not rev.startswith("refs/") else [rev]
    cmd = ["git", "-c", "http.version=HTTP/1.1", "ls-remote", git, *patterns, *(f"{p}^{{}}" for p in patterns)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"git ls-remote {git} {rev} failed: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"git ls-remote {git} {rev} failed (exit {result.returncode}): {result.stderr.strip()}")
    found: dict[str, str] = {}
    for line in result.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        if sha and ref:
            found[ref] = sha
    for pattern in patterns:
        for candidate in (f"{pattern}^{{}}", pattern):
            if candidate in found:
                return found[candidate]
    raise RuntimeError(f"ref {rev!r} not found in {git}")


@dataclass(frozen=True)
class SourceConfig:
    """A git repository at an immutable ref.

    Attributes:
        git: Repository URL to clone.
        rev: Immutable ref: a commit SHA, a tag, or ``refs/pull/<n>/head`` for
            an unmerged PR. Branch names are rejected because they move.
        path: Optional subdirectory of the clone to build and run from.
        sha: The commit ``rev`` resolved to. Filled in by ``srtctl apply`` at
            submit time; write it yourself only to pin an exact commit while
            keeping the human-readable ``rev`` beside it.
    """

    git: str
    rev: str
    path: str | None = None
    sha: str | None = None

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if not self.git.strip():
            raise ValidationError("source.git must be a non-empty repository URL")
        validate_ref(self.rev, where="source")
        if self.path is not None and not self.path.strip():
            raise ValidationError("source.path must not be blank when set")
        if self.sha is not None and not is_commit_sha(self.sha):
            raise ValidationError(f"source.sha must be a commit SHA, got {self.sha!r}")

    @property
    def checkout(self) -> str:
        """What to actually check out: the pinned commit when present, else the ref."""
        return self.sha or self.rev.strip()


@dataclass(frozen=True)
class DynamoSourceConfig:
    """Where Dynamo comes from. Exactly one of ``git``, ``pypi``, or ``wheel``.

    Attributes:
        git: Repository URL to build from (default upstream when ``rev`` is set
            without it). Builds ``ai-dynamo-runtime`` with maturin and installs
            ``ai-dynamo`` from the checkout; cached on ``/configs`` by commit.
        rev: Immutable ref in ``git``: commit SHA, tag, or ``refs/pull/<n>/head``.
        sha: The commit ``rev`` resolved to; filled in by ``srtctl apply``.
        patches: Cargo dependency replacements applied tree-wide before the
            build (the legacy ``cargo_patches``).
        pypi: Release version from PyPI (the legacy ``version``).
        wheel: Staged nightly ``ai-dynamo`` version (the legacy ``wheel``).
    """

    DEFAULT_GIT: ClassVar[str] = "https://github.com/ai-dynamo/dynamo.git"

    git: str | None = None
    rev: str | None = None
    sha: str | None = None
    patches: list[str] | None = None
    pypi: str | None = None
    wheel: str | None = None

    Schema: ClassVar[type[Schema]] = Schema

    def __post_init__(self) -> None:
        if self.rev is not None and self.git is None:
            object.__setattr__(self, "git", self.DEFAULT_GIT)
        chosen = [name for name, on in (("git", self.git), ("pypi", self.pypi), ("wheel", self.wheel)) if on]
        if len(chosen) != 1:
            raise ValidationError(
                f"dynamo.source needs exactly one of git (with rev), pypi, or wheel; got {', '.join(chosen) or 'none'}"
            )
        if self.git is not None:
            if not self.git.strip():
                raise ValidationError("dynamo.source.git must be a non-empty repository URL")
            if self.rev is None:
                raise ValidationError("dynamo.source.git requires rev (commit SHA, tag, or refs/pull/<n>/head)")
            validate_ref(self.rev, where="dynamo.source")
        elif self.rev is not None or self.sha is not None or self.patches:
            raise ValidationError("dynamo.source.rev, sha, and patches only apply to a git source")
        if self.sha is not None and not is_commit_sha(self.sha):
            raise ValidationError(f"dynamo.source.sha must be a commit SHA, got {self.sha!r}")
        for name in ("pypi", "wheel"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValidationError(f"dynamo.source.{name} must be a non-empty version")

    @property
    def checkout(self) -> str | None:
        """Commit or ref to check out for a git source; None otherwise."""
        if self.git is None or self.rev is None:
            return None
        return self.sha or self.rev.strip()


def pin_source_revs(document: Any, *, resolve=None) -> list[str]:
    """Resolve every unpinned ``source.rev`` in a raw recipe document to a ``sha``, in place.

    Walks the whole document (plain, sweep, and override-format files alike),
    so it works on the comment-preserving ``ruamel`` tree before any reader
    sees it. A ``source`` mapping is any dict under the key ``source`` with
    ``git`` and ``rev`` and no ``sha``. A resolution failure is logged and left
    unpinned rather than blocking the submit; the install path then fetches
    the ref by name on the compute node.

    Returns one human-readable note per pinned source.
    """
    resolve = resolve or resolve_rev
    notes: list[str] = []

    def walk(node: Any, path: tuple[Any, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "source" and isinstance(value, dict) and value.get("git") and value.get("rev"):
                    if not value.get("sha"):
                        git, rev = str(value["git"]), str(value["rev"])
                        dotted = ".".join(str(p) for p in (*path, key))
                        if is_commit_sha(rev):
                            continue
                        try:
                            sha = resolve(git, rev)
                        except RuntimeError as exc:
                            logger.warning(
                                "%s: could not pin rev %r (%s); the job will fetch the ref by name and "
                                "any build cache keys on the ref, not the commit",
                                dotted,
                                rev,
                                exc,
                            )
                            continue
                        value["sha"] = sha
                        notes.append(f"{dotted}: {rev} -> {sha}")
                    continue
                if isinstance(value, dict | list):
                    walk(value, (*path, key))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, (*path, index))

    walk(document, ())
    return notes


__all__ = [
    "MOVING_REFS",
    "DynamoSourceConfig",
    "SourceConfig",
    "is_commit_sha",
    "pin_source_revs",
    "resolve_rev",
    "validate_ref",
]
