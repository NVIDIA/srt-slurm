# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the portable benchmark manifest validator.

The shipped reference example is the positive fixture. Negative cases are built
by loading that example and mutating exactly one field, so each test isolates a
single failure mode and asserts its specific IssueCode.
"""

import copy
import json
from pathlib import Path

import pytest
import yaml

from srtctl.benchmark_manifest.errors import IssueCode, Report, Severity
from srtctl.benchmark_manifest.export_schema import build_schema
from srtctl.benchmark_manifest.models import BenchmarkManifest
from srtctl.benchmark_manifest.validate import check_consistency, load_manifest, validate_file

REPO_ROOT = Path(__file__).parent.parent
EXAMPLE = REPO_ROOT / "examples/benchmark-manifest/dsv4-pro-8k1k-b300/manifest.yaml"
COMMITTED_SCHEMA = REPO_ROOT / "src/srtctl/benchmark_manifest/schema/benchmark-manifest.schema.json"

HEX64 = "a" * 64  # a well-formed bare sha256
HEX40 = "b" * 40  # a well-formed commit sha


@pytest.fixture
def example_dict() -> dict:
    return yaml.safe_load(EXAMPLE.read_text())


def _report_for(manifest_dict: dict) -> Report:
    report = Report()
    manifest = BenchmarkManifest.model_validate(manifest_dict)
    check_consistency(manifest, report)
    return report


def _consistency_codes(manifest_dict: dict) -> set[IssueCode]:
    """Parse a manifest dict and run only the consistency checks; return all codes."""
    return _report_for(manifest_dict).codes()


def _error_codes(manifest_dict: dict) -> set[IssueCode]:
    """The exact set of ERROR-severity codes — used to assert a mutation is isolated."""
    return {i.code for i in _report_for(manifest_dict).errors}


# ---------------------------------------------------------------------------
# positive
# ---------------------------------------------------------------------------


def test_reference_example_passes():
    report = validate_file(EXAMPLE)
    assert report.ok, f"reference example should validate cleanly, got: {[str(i) for i in report.issues]}"
    assert not report.errors


def test_reference_example_parses_into_models(example_dict):
    manifest = BenchmarkManifest.model_validate(example_dict)
    assert manifest.metadata.name == "dsv4-pro-8k1k-b300"


# ---------------------------------------------------------------------------
# negative — one failure mode per test
# ---------------------------------------------------------------------------


def test_floating_tag_rejected(example_dict):
    example_dict["artifacts"]["base_image"]["digest"] = "1.3.0rc23"  # a tag, not a digest
    assert IssueCode.FLOATING_TAG in _consistency_codes(example_dict)


def test_bad_digest_format_rejected(example_dict):
    example_dict["artifacts"]["base_image"]["digest"] = "sha256:deadbeef"  # too short
    assert IssueCode.BAD_DIGEST_FORMAT in _consistency_codes(example_dict)


def test_unpinned_wheel_rejected(example_dict):
    example_dict["artifacts"]["wheels"][0].pop("sha256")  # only a version string left
    assert IssueCode.UNPINNED_WHEEL in _consistency_codes(example_dict)


def test_bad_commit_rejected(example_dict):
    example_dict["source"]["commit"] = "main"  # not a 40-hex sha
    assert IssueCode.BAD_COMMIT in _consistency_codes(example_dict)


def test_model_name_mismatch_rejected(example_dict):
    example_dict["model"]["served"] = "some-other-name"
    assert IssueCode.MODEL_NAME_MISMATCH in _consistency_codes(example_dict)


def test_private_checkpoint_path_rejected(example_dict):
    example_dict["model"]["checkpoint"] = {"hf_id": "/lustre/local/ckpt/dsv4-mxfp4", "hf_revision": "x"}
    assert IssueCode.PRIVATE_PATH in _consistency_codes(example_dict)


def test_checkpoint_underspecified_rejected(example_dict):
    example_dict["model"]["checkpoint"] = {}  # neither HF id+rev nor OCI digest
    assert IssueCode.CHECKPOINT_UNDERSPECIFIED in _consistency_codes(example_dict)


def test_silent_backend_fallback_rejected(example_dict):
    example_dict["runtime"]["backends"]["moe"]["effective"] = "TRTLLM"
    example_dict["runtime"]["backends"]["moe"]["requested"] = "MEGAMOE_DEEPGEMM"
    assert IssueCode.SILENT_BACKEND_FALLBACK in _consistency_codes(example_dict)


def test_runtime_install_wheel_rejected(example_dict):
    example_dict["conversion"]["inputs"]["dynamo_wheel"] = {"name": "ai-dynamo", "version": "1.3.0.dev0"}
    assert IssueCode.RUNTIME_INSTALL_WHEEL in _consistency_codes(example_dict)


def test_duplicate_point_id_rejected(example_dict):
    pt = copy.deepcopy(example_dict["results"]["baseline"][0])
    example_dict["results"]["baseline"].append(pt)  # same point_id twice
    assert IssueCode.DANGLING_CONFIG_TO_PERF in _consistency_codes(example_dict)


def test_incomplete_capability_rejected(example_dict):
    # Declared available but no invocation documented.
    example_dict["capabilities"]["worker_without_frontend"] = {"available": True}
    assert IssueCode.INCOMPLETE_CAPABILITY in _consistency_codes(example_dict)


# ---------------------------------------------------------------------------
# structural (schema layer)
# ---------------------------------------------------------------------------


def test_unknown_key_is_schema_error(example_dict, tmp_path):
    example_dict["unexpected_field"] = 1
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(example_dict))
    report = validate_file(p)
    assert IssueCode.SCHEMA_ERROR in report.codes()
    assert not report.ok


def test_missing_required_section_is_schema_error(example_dict, tmp_path):
    del example_dict["runtime"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(example_dict))
    report = Report()
    assert load_manifest(p, report) is None
    assert IssueCode.SCHEMA_ERROR in report.codes()


# ---------------------------------------------------------------------------
# online stub
# ---------------------------------------------------------------------------


def test_online_checks_flag_not_implemented():
    report = validate_file(EXAMPLE, online=True)
    assert IssueCode.ONLINE_NOT_IMPLEMENTED in report.codes()
    # It's a warning, not an error — the manifest still passes.
    assert report.ok
    assert any(i.severity is Severity.WARNING for i in report.issues)


# ---------------------------------------------------------------------------
# accept-alternative pin branches (guard against false-rejection regressions)
# ---------------------------------------------------------------------------


def test_wheel_pinned_by_git_commit_only_accepted(example_dict):
    example_dict["artifacts"]["wheels"][0] = {"name": "ai-dynamo", "git_commit": HEX40}
    assert IssueCode.UNPINNED_WHEEL not in _consistency_codes(example_dict)


def test_checkpoint_pinned_by_oci_digest_only_accepted(example_dict):
    example_dict["model"]["checkpoint"] = {"oci_digest": f"sha256:{HEX64}"}
    codes = _consistency_codes(example_dict)
    assert IssueCode.CHECKPOINT_UNDERSPECIFIED not in codes
    assert IssueCode.PRIVATE_PATH not in codes
    assert IssueCode.FLOATING_TAG not in codes


def test_checkpoint_oci_digest_with_repo_prefix_accepted(example_dict):
    example_dict["model"]["checkpoint"] = {"oci_digest": f"nvcr.io/nvidia/dsv4@sha256:{HEX64}"}
    assert not _error_codes(example_dict)


# ---------------------------------------------------------------------------
# pin/hash format enforcement (finding: inconsistent enforcement)
# ---------------------------------------------------------------------------


def test_wheel_garbage_sha256_rejected(example_dict):
    example_dict["artifacts"]["wheels"][0]["sha256"] = "deadbeef"  # present but not 64-hex
    assert IssueCode.BAD_DIGEST_FORMAT in _consistency_codes(example_dict)


def test_wheel_garbage_git_commit_rejected(example_dict):
    example_dict["artifacts"]["wheels"][0] = {"name": "ai-dynamo", "git_commit": "main"}
    assert IssueCode.BAD_COMMIT in _consistency_codes(example_dict)


def test_effective_config_garbage_sha256_rejected(example_dict):
    example_dict["runtime"]["effective_config"]["sha256"] = "TODO"
    assert IssueCode.BAD_DIGEST_FORMAT in _consistency_codes(example_dict)


def test_bundle_empty_sha256_rejected(example_dict):
    example_dict["source"]["bundle"][0]["sha256"] = ""
    assert IssueCode.BAD_DIGEST_FORMAT in _consistency_codes(example_dict)


def test_sbom_garbage_sha256_rejected(example_dict):
    example_dict["artifacts"]["sbom"]["sha256"] = "not-a-hash"
    assert IssueCode.BAD_DIGEST_FORMAT in _consistency_codes(example_dict)


def test_checkpoint_oci_floating_tag_rejected(example_dict):
    example_dict["model"]["checkpoint"] = {"oci_digest": "myreg.io/dsv4:latest"}
    assert IssueCode.FLOATING_TAG in _consistency_codes(example_dict)


def test_dynamo_wheel_garbage_sha256_rejected(example_dict):
    example_dict["conversion"]["inputs"]["dynamo_wheel"]["sha256"] = "deadbeef"
    assert IssueCode.BAD_DIGEST_FORMAT in _consistency_codes(example_dict)


# ---------------------------------------------------------------------------
# commit checks: conversion image + compat window + missing commit
# ---------------------------------------------------------------------------


def test_conversion_image_floating_tag_rejected(example_dict):
    example_dict["conversion"]["inputs"]["trtllm_image_digest"] = "release:1.3.0rc23"
    assert IssueCode.FLOATING_TAG in _consistency_codes(example_dict)


def test_missing_commit_rejected(example_dict):
    example_dict["source"]["commit"] = ""
    assert IssueCode.MISSING_COMMIT in _consistency_codes(example_dict)


def test_compat_min_commit_bad_rejected(example_dict):
    example_dict["conversion"]["inputs"]["compat_min_commit"] = "main"
    assert IssueCode.BAD_COMMIT in _consistency_codes(example_dict)


def test_compat_max_commit_bad_rejected(example_dict):
    example_dict["conversion"]["inputs"]["compat_max_commit"] = "whenever"
    assert IssueCode.BAD_COMMIT in _consistency_codes(example_dict)


def test_compat_max_commit_head_ref_accepted(example_dict):
    example_dict["conversion"]["inputs"]["compat_max_commit"] = "HEAD@2026-01-01"
    assert IssueCode.BAD_COMMIT not in _consistency_codes(example_dict)


# ---------------------------------------------------------------------------
# manifest version
# ---------------------------------------------------------------------------


def test_unsupported_manifest_version_rejected(example_dict):
    example_dict["manifest_version"] = "2.0"
    assert IssueCode.UNSUPPORTED_MANIFEST_VERSION in _error_codes(example_dict)


def test_missing_manifest_version_is_schema_error(example_dict, tmp_path):
    del example_dict["manifest_version"]  # now required
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(example_dict))
    report = validate_file(p)
    assert IssueCode.SCHEMA_ERROR in report.codes()
    assert not report.ok


# ---------------------------------------------------------------------------
# private-path heuristic: mount-relative form
# ---------------------------------------------------------------------------


def test_relative_private_checkpoint_path_rejected(example_dict):
    example_dict["model"]["checkpoint"] = {"hf_id": "lustre/local/ckpt/dsv4-mxfp4", "hf_revision": "x"}
    assert IssueCode.PRIVATE_PATH in _consistency_codes(example_dict)


# ---------------------------------------------------------------------------
# empty point_id branch
# ---------------------------------------------------------------------------


def test_empty_point_id_rejected(example_dict):
    example_dict["results"]["baseline"][0]["point_id"] = ""
    assert IssueCode.DANGLING_CONFIG_TO_PERF in _consistency_codes(example_dict)


# ---------------------------------------------------------------------------
# warning-severity branches (non-blocking)
# ---------------------------------------------------------------------------


def test_missing_sbom_warns_but_passes(example_dict):
    example_dict["artifacts"].pop("sbom")
    report = _report_for(example_dict)
    assert IssueCode.MISSING_SBOM in report.codes()
    assert report.ok  # warning only
    assert not any(i.severity is Severity.ERROR and i.code is IssueCode.MISSING_SBOM for i in report.issues)


def test_missing_capabilities_warns_but_passes(example_dict):
    example_dict.pop("capabilities")
    report = _report_for(example_dict)
    incomplete = [i for i in report.issues if i.code is IssueCode.INCOMPLETE_CAPABILITY]
    assert incomplete and all(i.severity is Severity.WARNING for i in incomplete)
    assert report.ok


# ---------------------------------------------------------------------------
# isolation: a single mutation yields exactly its own error (no spurious extras)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda d: d["model"].__setitem__("served", "other"), IssueCode.MODEL_NAME_MISMATCH),
        (
            lambda d: d["runtime"]["backends"]["moe"].update({"requested": "A", "effective": "B"}),
            IssueCode.SILENT_BACKEND_FALLBACK,
        ),
        (lambda d: d["source"].__setitem__("commit", "main"), IssueCode.BAD_COMMIT),
    ],
)
def test_single_mutation_is_isolated(example_dict, mutate, expected):
    mutate(example_dict)
    assert _error_codes(example_dict) == {expected}


# ---------------------------------------------------------------------------
# schema-drift guard: the committed JSON schema matches the models
# ---------------------------------------------------------------------------


def test_committed_schema_matches_models():
    committed = json.loads(COMMITTED_SCHEMA.read_text())
    assert build_schema() == committed, (
        "Committed JSON schema is stale. Regenerate:\n"
        "  uv run python -m srtctl.benchmark_manifest.export_schema "
        "-o src/srtctl/benchmark_manifest/schema/benchmark-manifest.schema.json"
    )
