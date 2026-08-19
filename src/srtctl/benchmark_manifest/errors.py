# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured issue types the manifest validator emits.

Each failure mode has a stable ``IssueCode`` so tests (and CI) can assert on a
specific failure rather than matching free text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    ERROR = "error"  # blocks the handoff
    WARNING = "warning"  # allowed, but the manifest is weaker for it


class IssueCode(str, Enum):
    # structural
    SCHEMA_ERROR = "schema_error"  # failed to parse against the models
    UNSUPPORTED_MANIFEST_VERSION = "unsupported_manifest_version"  # major version this validator can't read

    # artifact pinning
    FLOATING_TAG = "floating_tag"  # a tag where a digest is required
    BAD_DIGEST_FORMAT = "bad_digest_format"  # not 'sha256:<64 hex>' (or a bad bare 64-hex sha256)
    UNPINNED_WHEEL = "unpinned_wheel"  # neither sha256 nor git_commit
    MISSING_SBOM = "missing_sbom"  # no SBOM recorded (warning)

    # source
    BAD_COMMIT = "bad_commit"  # not a 40-hex sha
    MISSING_COMMIT = "missing_commit"  # empty where required

    # model / checkpoint
    MODEL_NAME_MISMATCH = "model_name_mismatch"  # served != requested
    PRIVATE_PATH = "private_path"  # a local filesystem path where a shareable ref is required
    CHECKPOINT_UNDERSPECIFIED = "checkpoint_underspecified"  # neither HF id+rev nor OCI digest

    # runtime
    SILENT_BACKEND_FALLBACK = "silent_backend_fallback"  # backend.effective != backend.requested

    # results
    DANGLING_CONFIG_TO_PERF = "dangling_config_to_perf"  # a result point is not traceable / duplicate id

    # conversion
    RUNTIME_INSTALL_WHEEL = "runtime_install_wheel"  # wheel not baked (floating install)

    # capabilities
    INCOMPLETE_CAPABILITY = "incomplete_capability"  # declared available but no invocation, or section missing

    # online checks (not yet implemented)
    ONLINE_NOT_IMPLEMENTED = "online_not_implemented"


@dataclass
class Issue:
    code: IssueCode
    severity: Severity
    path: str  # dotted location within the manifest, e.g. "artifacts.base_image.digest"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.code.value} at {self.path}: {self.message}"


@dataclass
class Report:
    """Result of validating one manifest."""

    issues: list[Issue] = field(default_factory=list)

    def add(self, code: IssueCode, severity: Severity, path: str, message: str) -> None:
        self.issues.append(Issue(code=code, severity=severity, path=path, message=message))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """True when there are no error-severity issues (warnings are allowed)."""
        return not self.errors

    def codes(self) -> set[IssueCode]:
        return {i.code for i in self.issues}
