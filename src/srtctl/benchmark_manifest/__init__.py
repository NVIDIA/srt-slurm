# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Portable benchmark manifest.

A single, machine-readable file that fully describes a benchmark run so a
different team can reproduce the native baseline and build a matched image
with no private paths, out-of-band files, or tribal knowledge.

This package provides:
  - models.py    : the manifest schema, as Pydantic v2 models (source of truth).
  - validate.py  : a validator that layers cross-field consistency checks on top
                   of schema conformance and reports structured issues.
  - errors.py    : the issue/severity/code types the validator emits.

A generator (emit a manifest from a completed run's artifacts) is a separate,
not-yet-built piece; see docs. The schema and validator here define the target
it must produce.
"""

from srtctl.benchmark_manifest.errors import Issue, IssueCode, Report, Severity
from srtctl.benchmark_manifest.models import BenchmarkManifest

__all__ = [
    "BenchmarkManifest",
    "Issue",
    "IssueCode",
    "Report",
    "Severity",
]
