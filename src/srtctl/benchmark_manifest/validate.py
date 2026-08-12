# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark manifest validator.

Two layers:

1. Schema conformance (``load_manifest``): parse YAML into the Pydantic models.
   Any structural/type error becomes a single SCHEMA_ERROR issue with the
   pydantic detail attached.

2. Semantic consistency (``check_consistency``): the policy checks that make a
   handoff portable — every artifact pinned by digest, served == requested,
   backend.effective == requested, no private paths, config -> perf traceable.
   These are where the plain-English "the validator rejects a non-portable
   handoff" failures come from.

An ``--online`` mode (resolve a digest / HF revision / git commit against the
network) is declared but NOT implemented yet — it emits a single
ONLINE_NOT_IMPLEMENTED warning so the gap is visible rather than silently
skipped.

Run:
    python -m srtctl.benchmark_manifest.validate path/to/manifest.yaml
    srtctl-validate-manifest path/to/manifest.yaml        # once installed
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from srtctl.benchmark_manifest.errors import IssueCode, Report, Severity
from srtctl.benchmark_manifest.models import MANIFEST_VERSION, BenchmarkManifest

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")  # a bare sha256, no 'sha256:' prefix
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
# A Hugging Face repo id is 'namespace/name' — exactly one slash, no path syntax.
_HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
# Escape hatch for a moving upper bound on the compatibility window.
_HEAD_REF_RE = re.compile(r"^HEAD@")
# Heuristic for "this is a local filesystem path, not a shareable reference".
_PATH_LIKE_RE = re.compile(r"^(/|\./|\.\./|~)")


# ---------------------------------------------------------------------------
# layer 1: schema conformance
# ---------------------------------------------------------------------------


def load_manifest(path: Path, report: Report) -> BenchmarkManifest | None:
    """Load + parse a manifest. Returns None (and records SCHEMA_ERROR) on failure."""
    try:
        raw = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, OSError) as exc:
        # OSError covers a directory, an unreadable file, etc. — record it rather
        # than let it crash the CLI and abort the rest of a multi-file batch.
        report.add(IssueCode.SCHEMA_ERROR, Severity.ERROR, "<file>", f"could not read manifest: {exc}")
        return None
    if not isinstance(raw, dict):
        report.add(IssueCode.SCHEMA_ERROR, Severity.ERROR, "<file>", "Top-level manifest must be a mapping.")
        return None
    try:
        return BenchmarkManifest.model_validate(raw)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            report.add(IssueCode.SCHEMA_ERROR, Severity.ERROR, loc or "<root>", err["msg"])
        return None


# ---------------------------------------------------------------------------
# layer 2: semantic consistency
# ---------------------------------------------------------------------------


def _check_digest(ref_desc: str, path: str, digest: str, report: Report) -> None:
    if _DIGEST_RE.match(digest):
        return
    # A value that starts with 'sha256:' was meant to be a digest but is malformed;
    # anything else isn't a digest at all — most likely a floating tag.
    if digest.startswith("sha256:"):
        report.add(
            IssueCode.BAD_DIGEST_FORMAT,
            Severity.ERROR,
            path,
            f"{ref_desc} digest '{digest}' is not 'sha256:<64 hex>'.",
        )
    else:
        report.add(
            IssueCode.FLOATING_TAG,
            Severity.ERROR,
            path,
            f"{ref_desc} is pinned by a tag ('{digest}'); pin it by an immutable sha256 digest.",
        )


def _check_commit(path: str, commit: str, report: Report) -> None:
    if not commit:
        report.add(IssueCode.MISSING_COMMIT, Severity.ERROR, path, "commit is empty.")
    elif not _SHA1_RE.match(commit):
        report.add(
            IssueCode.BAD_COMMIT,
            Severity.ERROR,
            path,
            f"commit '{commit}' is not a full 40-hex sha.",
        )


def _check_sha256(ref_desc: str, path: str, value: str, report: Report) -> None:
    """A bare sha256 hash (no 'sha256:' prefix) — used for wheels, files, and the SBOM."""
    if not _SHA256_RE.match(value):
        report.add(
            IssueCode.BAD_DIGEST_FORMAT,
            Severity.ERROR,
            path,
            f"{ref_desc} sha256 '{value}' is not 64 lowercase hex chars.",
        )


def _check_pin_formats(path: str, sha256: str | None, git_commit: str | None, report: Report) -> None:
    """Format-check whichever pin a wheel provides (presence is checked separately)."""
    if sha256:
        _check_sha256("wheel", f"{path}.sha256", sha256, report)
    if git_commit and not _SHA1_RE.match(git_commit):
        report.add(
            IssueCode.BAD_COMMIT,
            Severity.ERROR,
            f"{path}.git_commit",
            f"git_commit '{git_commit}' is not a full 40-hex sha.",
        )


def _check_oci_digest(path: str, value: str, report: Report) -> None:
    """An OCI reference must be pinned by digest: 'sha256:<64 hex>' or 'repo@sha256:<64 hex>'."""
    digest_part = value.rsplit("@", 1)[-1]
    if _DIGEST_RE.match(digest_part):
        return
    if digest_part.startswith("sha256:"):
        report.add(IssueCode.BAD_DIGEST_FORMAT, Severity.ERROR, path, f"OCI digest '{value}' is not 'sha256:<64 hex>'.")
    else:
        report.add(
            IssueCode.FLOATING_TAG,
            Severity.ERROR,
            path,
            f"'{value}' is not pinned by a digest; use a 'sha256:<64 hex>' (or 'repo@sha256:...') reference.",
        )


def check_consistency(m: BenchmarkManifest, report: Report) -> None:
    """Run the portability policy checks against a parsed manifest."""

    # --- version: this validator only understands its own major version ---
    _check_version(m, report)

    # --- artifacts: everything pinned by digest ---
    _check_digest("base image", "artifacts.base_image.digest", m.artifacts.base_image.digest, report)
    for i, wheel in enumerate(m.artifacts.wheels):
        if not (wheel.sha256 or wheel.git_commit):
            report.add(
                IssueCode.UNPINNED_WHEEL,
                Severity.ERROR,
                f"artifacts.wheels[{i}]",
                f"wheel '{wheel.name}' has neither sha256 nor git_commit; a version string is not a pin.",
            )
        _check_pin_formats(f"artifacts.wheels[{i}]", wheel.sha256, wheel.git_commit, report)
    if m.artifacts.sbom is None:
        report.add(
            IssueCode.MISSING_SBOM,
            Severity.WARNING,
            "artifacts.sbom",
            "No SBOM recorded; nested-dependency mismatches will not be catchable.",
        )
    else:
        _check_sha256("SBOM", "artifacts.sbom.sha256", m.artifacts.sbom.sha256, report)

    # --- source ---
    _check_commit("source.commit", m.source.commit, report)
    for i, f in enumerate(m.source.bundle):
        _check_sha256("bundle file", f"source.bundle[{i}].sha256", f.sha256, report)

    # --- runtime: the effective-config reference must be a real hash ---
    _check_sha256("effective config", "runtime.effective_config.sha256", m.runtime.effective_config.sha256, report)

    # --- conversion inputs ---
    _check_digest(
        "conversion base image",
        "conversion.inputs.trtllm_image_digest",
        m.conversion.inputs.trtllm_image_digest,
        report,
    )
    _check_commit("conversion.inputs.trtllm_source_commit", m.conversion.inputs.trtllm_source_commit, report)
    _check_commit("conversion.inputs.compat_min_commit", m.conversion.inputs.compat_min_commit, report)
    _check_compat_max("conversion.inputs.compat_max_commit", m.conversion.inputs.compat_max_commit, report)
    dw = m.conversion.inputs.dynamo_wheel
    if not (dw.sha256 or dw.git_commit):
        report.add(
            IssueCode.RUNTIME_INSTALL_WHEEL,
            Severity.ERROR,
            "conversion.inputs.dynamo_wheel",
            "Dynamo wheel is not pinned (no sha256/git_commit); bake a pinned wheel at build time "
            "instead of a floating runtime install.",
        )
    _check_pin_formats("conversion.inputs.dynamo_wheel", dw.sha256, dw.git_commit, report)

    # --- model / checkpoint ---
    if m.model.served != m.model.requested:
        report.add(
            IssueCode.MODEL_NAME_MISMATCH,
            Severity.ERROR,
            "model",
            f"served ('{m.model.served}') != requested ('{m.model.requested}').",
        )
    _check_checkpoint(m, report)

    # --- runtime: silent backend fallback ---
    for name, sel in (("moe", m.runtime.backends.moe), ("comm", m.runtime.backends.comm)):
        if sel.effective != sel.requested:
            report.add(
                IssueCode.SILENT_BACKEND_FALLBACK,
                Severity.ERROR,
                f"runtime.backends.{name}",
                f"effective backend ('{sel.effective}') != requested ('{sel.requested}'); "
                "the run did not use the configured backend.",
            )

    # --- results: every point carries a stable, unique label within this run ---
    # A manifest describes one fully-pinned configuration, so point_id is the label
    # that identifies a point within that run (e.g. a concurrency). We enforce that
    # it is present and unique; we do not resolve it against a separate config table.
    seen: set[str] = set()
    for i, pt in enumerate(m.results.baseline):
        if not pt.point_id:
            report.add(
                IssueCode.DANGLING_CONFIG_TO_PERF,
                Severity.ERROR,
                f"results.baseline[{i}]",
                "result point has no point_id, so it cannot be identified.",
            )
        elif pt.point_id in seen:
            report.add(
                IssueCode.DANGLING_CONFIG_TO_PERF,
                Severity.ERROR,
                f"results.baseline[{i}].point_id",
                f"duplicate point_id '{pt.point_id}'.",
            )
        seen.add(pt.point_id)

    # --- capabilities: declared-but-incomplete ---
    _check_capabilities(m, report)


def _check_checkpoint(m: BenchmarkManifest, report: Report) -> None:
    ckpt = m.model.checkpoint
    has_hf = bool(ckpt.hf_id and ckpt.hf_revision)
    has_oci = bool(ckpt.oci_digest)
    # An hf_id must be a shareable 'namespace/name' id, not a filesystem path.
    # This also catches mount-relative paths ('lustre/ckpts/...') that a leading-
    # slash heuristic would miss, since a real HF id has exactly one slash.
    if ckpt.hf_id and not _HF_ID_RE.match(ckpt.hf_id):
        report.add(
            IssueCode.PRIVATE_PATH,
            Severity.ERROR,
            "model.checkpoint.hf_id",
            f"'{ckpt.hf_id}' is not a Hugging Face 'namespace/name' id; it looks like a local path.",
        )
    if ckpt.oci_digest:
        if _PATH_LIKE_RE.match(ckpt.oci_digest):
            report.add(
                IssueCode.PRIVATE_PATH,
                Severity.ERROR,
                "model.checkpoint.oci_digest",
                f"'{ckpt.oci_digest}' looks like a local path; use an OCI 'sha256:<64 hex>' reference.",
            )
        else:
            _check_oci_digest("model.checkpoint.oci_digest", ckpt.oci_digest, report)
    if not (has_hf or has_oci):
        report.add(
            IssueCode.CHECKPOINT_UNDERSPECIFIED,
            Severity.ERROR,
            "model.checkpoint",
            "checkpoint needs either (hf_id + hf_revision) or oci_digest.",
        )


def _check_version(m: BenchmarkManifest, report: Report) -> None:
    supported_major = MANIFEST_VERSION.split(".")[0]
    major = m.manifest_version.split(".")[0]
    if major != supported_major:
        report.add(
            IssueCode.UNSUPPORTED_MANIFEST_VERSION,
            Severity.ERROR,
            "manifest_version",
            f"manifest_version '{m.manifest_version}' is not supported; this validator understands "
            f"{supported_major}.x (current schema {MANIFEST_VERSION}).",
        )


def _check_compat_max(path: str, value: str, report: Report) -> None:
    """The window's upper bound is a 40-hex commit or the 'HEAD@<date>' escape hatch."""
    if not value:
        report.add(IssueCode.MISSING_COMMIT, Severity.ERROR, path, "compat_max_commit is empty.")
    elif not (_HEAD_REF_RE.match(value) or _SHA1_RE.match(value)):
        report.add(
            IssueCode.BAD_COMMIT,
            Severity.ERROR,
            path,
            f"'{value}' is not a 40-hex sha or a 'HEAD@<date>' reference.",
        )


def _check_capabilities(m: BenchmarkManifest, report: Report) -> None:
    if m.capabilities is None:
        report.add(
            IssueCode.INCOMPLETE_CAPABILITY,
            Severity.WARNING,
            "capabilities",
            "No Dynamo capability declarations; worker-without-frontend / prefill-only-discovery are unstated.",
        )
        return
    for name in ("worker_without_frontend", "prefill_only_discovery"):
        decl = getattr(m.capabilities, name)
        if decl.available and not decl.invocation:
            report.add(
                IssueCode.INCOMPLETE_CAPABILITY,
                Severity.ERROR,
                f"capabilities.{name}",
                "declared available but no invocation is documented.",
            )


# ---------------------------------------------------------------------------
# layer 3: online checks (NOT implemented)
# ---------------------------------------------------------------------------


def check_online(m: BenchmarkManifest, report: Report) -> None:
    """Resolve digests / HF revisions / commits against the network.

    TODO: not implemented. When built, this must:
      - resolve artifacts.base_image.digest against the registry (pull-ability),
      - resolve model.checkpoint HF id+revision (or OCI digest),
      - resolve source.commit in the srt-slurm repo,
      - verify trtllm_source_commit falls within [compat_min_commit, compat_max_commit]
        (needs the git DAG, so it cannot be decided statically).
    Until then, emit a visible warning rather than silently passing.
    """
    report.add(
        IssueCode.ONLINE_NOT_IMPLEMENTED,
        Severity.WARNING,
        "<online>",
        "--online checks (registry/HF/git reachability) are not implemented yet.",
    )


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def validate_file(path: Path, online: bool = False) -> Report:
    report = Report()
    manifest = load_manifest(path, report)
    if manifest is None:
        return report  # structural failure; consistency checks would be noise
    check_consistency(manifest, report)
    if online:
        check_online(manifest, report)
    return report


def _format_report(path: Path, report: Report) -> str:
    lines = [f"manifest: {path}"]
    if not report.issues:
        lines.append("  OK — no issues.")
    for issue in report.issues:
        lines.append(f"  {issue}")
    n_err, n_warn = len(report.errors), len(report.warnings)
    lines.append(f"  => {n_err} error(s), {n_warn} warning(s)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a portable benchmark manifest.")
    parser.add_argument("manifest", type=Path, nargs="+", help="Manifest YAML file(s) to validate.")
    parser.add_argument("--online", action="store_true", help="Also run network reachability checks (not implemented).")
    args = parser.parse_args(argv)

    exit_code = 0
    for path in args.manifest:
        if not path.exists():
            print(f"manifest: {path}\n  [error] file not found", file=sys.stderr)
            exit_code = 1
            continue
        report = validate_file(path, online=args.online)
        print(_format_report(path, report))
        if not report.ok:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
