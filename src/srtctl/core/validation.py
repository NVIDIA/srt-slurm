# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pre-submit validation for recipe artifacts.

Checks that model paths exist, container images are real, and HuggingFace/Docker
registry references resolve. All checks are fault-tolerant — they run in a
background thread after job submission and never block or fail the submit.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from srtctl.core.config import (
    generate_override_configs,
    resolve_config_with_defaults,
)

if TYPE_CHECKING:
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 2.0  # Fast enough for live networks, doesn't block long on air-gapped clusters


@dataclass(frozen=True)
class ValidationResult:
    """Result of a single validation check."""

    check: str
    ok: bool
    message: str


@dataclass(frozen=True)
class PreflightIssue:
    code: str
    field: str
    message: str


@dataclass(frozen=True)
class PreflightResolution:
    field: str
    raw: str | None
    resolved: str | None
    source: str
    ok: bool
    message: str


@dataclass(frozen=True)
class PreflightResult:
    variant: str
    ok: bool
    model: PreflightResolution
    container: PreflightResolution
    errors: list[PreflightIssue]

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "ok": self.ok,
            "model": self.model.__dict__,
            "container": self.container.__dict__,
            "errors": [issue.__dict__ for issue in self.errors],
        }


def _expand_path(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def _check_path(path_str: str, *, expect: str) -> tuple[bool, str]:
    path = Path(path_str).resolve()
    if not path.exists():
        return False, f"not found: {path}"
    if expect == "dir" and not path.is_dir():
        return False, f"not a directory: {path}"
    if expect == "file" and not path.is_file():
        return False, f"not a file: {path}"
    return True, f"exists: {path}"


def _preflight_model(
    raw_config: dict[str, Any],
    resolved_config: dict[str, Any],
    cluster_config: dict[str, Any] | None,
) -> tuple[PreflightResolution, list[PreflightIssue]]:
    raw = raw_config.get("model", {}).get("path")
    resolved = resolved_config.get("model", {}).get("path")
    aliases = (cluster_config or {}).get("model_paths") or {}
    source = "srtslurm.yaml:model_paths" if raw in aliases else "literal"

    if not raw or not resolved:
        issue = PreflightIssue(
            code="model-missing",
            field="model.path",
            message="model.path is required",
        )
        return (
            PreflightResolution(
                field="model.path",
                raw=raw,
                resolved=resolved,
                source=source,
                ok=False,
                message=issue.message,
            ),
            [issue],
        )

    ok, detail = _check_path(_expand_path(resolved), expect="dir")
    if ok:
        return (
            PreflightResolution(
                field="model.path",
                raw=raw,
                resolved=str(Path(_expand_path(resolved)).resolve()),
                source=source,
                ok=True,
                message=detail,
            ),
            [],
        )

    if source == "srtslurm.yaml:model_paths":
        message = (
            f"Model alias '{raw}' resolved to '{resolved}', but that path is unavailable. "
            "Pull or register the model yourself before submitting."
        )
    else:
        message = (
            f"Model '{raw}' is not a local model path and is not defined in srtslurm.yaml "
            "model_paths. Pull or register the model yourself before submitting."
        )
    issue = PreflightIssue(
        code="model-not-available",
        field="model.path",
        message=message,
    )
    return (
        PreflightResolution(
            field="model.path",
            raw=raw,
            resolved=resolved,
            source=source,
            ok=False,
            message=message,
        ),
        [issue],
    )


def _preflight_container(
    raw_config: dict[str, Any],
    resolved_config: dict[str, Any],
    cluster_config: dict[str, Any] | None,
) -> tuple[PreflightResolution, list[PreflightIssue]]:
    raw = raw_config.get("model", {}).get("container")
    resolved = resolved_config.get("model", {}).get("container")
    aliases = (cluster_config or {}).get("containers") or {}
    source = "srtslurm.yaml:containers" if raw in aliases else "literal"

    if not raw or not resolved:
        issue = PreflightIssue(
            code="container-missing",
            field="model.container",
            message="model.container is required",
        )
        return (
            PreflightResolution(
                field="model.container",
                raw=raw,
                resolved=resolved,
                source=source,
                ok=False,
                message=issue.message,
            ),
            [issue],
        )

    ok, detail = _check_path(_expand_path(resolved), expect="file")
    if ok:
        return (
            PreflightResolution(
                field="model.container",
                raw=raw,
                resolved=str(Path(_expand_path(resolved)).resolve()),
                source=source,
                ok=True,
                message=detail,
            ),
            [],
        )

    if source == "srtslurm.yaml:containers":
        message = (
            f"Container alias '{raw}' resolved to '{resolved}', but that file is unavailable. "
            "Provide or register the container yourself before submitting."
        )
    else:
        message = (
            f"Container '{raw}' is not a local container path and is not defined in "
            "srtslurm.yaml containers. Provide or register the container yourself before submitting."
        )
    issue = PreflightIssue(
        code="container-not-available",
        field="model.container",
        message=message,
    )
    return (
        PreflightResolution(
            field="model.container",
            raw=raw,
            resolved=resolved,
            source=source,
            ok=False,
            message=message,
        ),
        [issue],
    )


def validate_topology(resources: dict[str, Any] | None) -> list[PreflightIssue]:
    """Catch semantically wrong resources blocks that pass the marshmallow schema.

    The schema accepts any combination of prefill_*, decode_*, and agg_* fields,
    so configs like ``prefill_workers: 0`` with ``decode_workers: 1`` look valid
    but express "disaggregated with no prefill" — which is really aggregated and
    should use agg_nodes/agg_workers instead.
    """
    if not resources:
        return [
            PreflightIssue(
                code="topology-missing",
                field="resources",
                message=(
                    "resources block is empty. Set either disaggregated "
                    "(prefill_nodes/prefill_workers + decode_nodes/decode_workers) "
                    "or aggregated (agg_nodes + agg_workers)."
                ),
            )
        ]

    prefill_nodes = resources.get("prefill_nodes")
    decode_nodes = resources.get("decode_nodes")
    prefill_workers = resources.get("prefill_workers")
    decode_workers = resources.get("decode_workers")
    agg_nodes = resources.get("agg_nodes")
    agg_workers = resources.get("agg_workers")

    disagg_fields = {
        "prefill_nodes": prefill_nodes,
        "decode_nodes": decode_nodes,
        "prefill_workers": prefill_workers,
        "decode_workers": decode_workers,
    }
    agg_fields = {"agg_nodes": agg_nodes, "agg_workers": agg_workers}

    has_disagg = any(v is not None for v in disagg_fields.values())
    has_agg = any(v is not None for v in agg_fields.values())

    if has_disagg and has_agg:
        set_disagg = sorted(k for k, v in disagg_fields.items() if v is not None)
        set_agg = sorted(k for k, v in agg_fields.items() if v is not None)
        return [
            PreflightIssue(
                code="topology-mixed",
                field="resources",
                message=(
                    f"Mixes disaggregated fields ({', '.join(set_disagg)}) with aggregated fields "
                    f"({', '.join(set_agg)}). Use disaggregated (prefill_*/decode_*) "
                    "or aggregated (agg_*), not both."
                ),
            )
        ]

    issues: list[PreflightIssue] = []

    if has_disagg:
        pf_workers = prefill_workers or 0
        dc_workers = decode_workers or 0
        pf_nodes = prefill_nodes or 0
        dc_nodes = decode_nodes or 0

        if pf_workers == 0 and dc_workers == 0:
            issues.append(
                PreflightIssue(
                    code="topology-no-workers",
                    field="resources",
                    message=(
                        "Disaggregated resources block has no workers: prefill_workers and "
                        "decode_workers are both 0/null. Set both, or switch to aggregated "
                        "mode with agg_nodes + agg_workers."
                    ),
                )
            )
        elif pf_workers == 0:
            issues.append(
                PreflightIssue(
                    code="topology-aggregated-style",
                    field="resources.prefill_workers",
                    message=(
                        "prefill_workers is 0 in a disaggregated-style resources block. "
                        f"For a single-side topology on {dc_nodes} node(s) with {dc_workers} "
                        f"worker(s), use aggregated mode: `agg_nodes: {dc_nodes}, "
                        f"agg_workers: {dc_workers}` (remove prefill_*/decode_*)."
                    ),
                )
            )
        elif dc_workers == 0:
            issues.append(
                PreflightIssue(
                    code="topology-aggregated-style",
                    field="resources.decode_workers",
                    message=(
                        "decode_workers is 0 in a disaggregated-style resources block. "
                        f"For a single-side topology on {pf_nodes} node(s) with {pf_workers} "
                        f"worker(s), use aggregated mode: `agg_nodes: {pf_nodes}, "
                        f"agg_workers: {pf_workers}` (remove prefill_*/decode_*)."
                    ),
                )
            )
        return issues

    if has_agg:
        ag_workers = agg_workers or 0
        ag_nodes = agg_nodes or 0
        if ag_workers == 0:
            issues.append(
                PreflightIssue(
                    code="topology-no-workers",
                    field="resources.agg_workers",
                    message="agg_workers must be > 0 in aggregated mode.",
                )
            )
        if ag_nodes == 0:
            issues.append(
                PreflightIssue(
                    code="topology-no-nodes",
                    field="resources.agg_nodes",
                    message="agg_nodes must be > 0 in aggregated mode.",
                )
            )
        return issues

    return [
        PreflightIssue(
            code="topology-missing",
            field="resources",
            message=(
                "No topology set. Set either disaggregated "
                "(prefill_nodes/prefill_workers + decode_nodes/decode_workers) "
                "or aggregated (agg_nodes + agg_workers)."
            ),
        )
    ]


def preflight_config_variants(
    raw_config: dict[str, Any],
    *,
    cluster_config: dict[str, Any] | None = None,
    selector: str | None = None,
) -> list[PreflightResult]:
    active_cluster_config = cluster_config
    variants = (
        generate_override_configs(raw_config, selector=selector) if "base" in raw_config else [("base", raw_config)]
    )
    results: list[PreflightResult] = []
    for suffix, variant in variants:
        resolved = resolve_config_with_defaults(variant, active_cluster_config)
        model, model_issues = _preflight_model(variant, resolved, active_cluster_config)
        container, container_issues = _preflight_container(variant, resolved, active_cluster_config)
        topology_issues = validate_topology(variant.get("resources"))
        issues = [*model_issues, *container_issues, *topology_issues]
        results.append(
            PreflightResult(
                variant=suffix,
                ok=not issues,
                model=model,
                container=container,
                errors=issues,
            )
        )
    return results


def validate_local_path(name: str, path: str) -> ValidationResult:
    """Check that a local file or directory exists."""
    try:
        p = Path(path)
        if not p.exists():
            return ValidationResult(name, False, f"not found: {path}")
        if p.is_dir():
            file_count = 0
            total_bytes = 0
            for f in p.rglob("*"):
                if f.is_file():
                    file_count += 1
                    total_bytes += f.stat().st_size
            return ValidationResult(name, True, f"{file_count} files, {total_bytes / 1e9:.1f}GB")
        size_gb = p.stat().st_size / 1e9
        return ValidationResult(name, True, f"{size_gb:.1f}GB")
    except Exception as e:
        return ValidationResult(name, False, f"check failed: {e}")


def validate_hf_model(name: str | None, revision: str | None) -> ValidationResult:
    """Check that a HuggingFace model exists (HTTP HEAD, 5s timeout)."""
    if not name:
        return ValidationResult("hf_model", True, "skipped (no model.name)")
    try:
        resp = requests.head(f"https://huggingface.co/api/models/{name}", timeout=_HTTP_TIMEOUT)
        if resp.status_code == 200:
            msg = f"{name} exists"
            if revision:
                rev_resp = requests.head(
                    f"https://huggingface.co/api/models/{name}/revision/{revision}",
                    timeout=_HTTP_TIMEOUT,
                )
                if rev_resp.status_code == 200:
                    msg += f", revision {revision[:12]} verified"
                else:
                    return ValidationResult("hf_model", False, f"revision {revision[:12]} not found")
            return ValidationResult("hf_model", True, msg)
        if resp.status_code == 401:
            return ValidationResult("hf_model", True, f"{name} exists (gated)")
        if resp.status_code == 404:
            return ValidationResult("hf_model", False, f"{name} not found on HuggingFace")
        return ValidationResult("hf_model", False, f"unexpected status {resp.status_code}")
    except requests.Timeout:
        return ValidationResult("hf_model", False, "HuggingFace check timed out")
    except Exception as e:
        return ValidationResult("hf_model", False, f"HuggingFace check failed: {e}")


def validate_docker_image(image: str | None, digest: str | None) -> ValidationResult:
    """Check that a Docker image exists on the registry (HTTP HEAD, 5s timeout)."""
    if not image:
        return ValidationResult("docker_image", True, "skipped (no container_image)")
    try:
        # Parse image into repo:tag
        if ":" in image:
            repo, tag = image.rsplit(":", 1)
        else:
            repo, tag = image, "latest"

        # Handle Docker Hub (no registry prefix)
        if "/" not in repo or (repo.count("/") == 1 and "." not in repo.split("/")[0]):
            if "/" not in repo:
                repo = f"library/{repo}"
            url = f"https://registry.hub.docker.com/v2/{repo}/manifests/{tag}"
        else:
            # Other registries (nvcr.io, ghcr.io, etc.)
            registry, repo_path = repo.split("/", 1)
            url = f"https://{registry}/v2/{repo_path}/manifests/{tag}"

        resp = requests.head(
            url,
            headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"},
            timeout=_HTTP_TIMEOUT,
        )
        if resp.status_code == 200:
            msg = f"{image} exists"
            if digest:
                remote_digest = resp.headers.get("Docker-Content-Digest", "")
                if remote_digest and remote_digest != digest:
                    return ValidationResult("docker_image", False, "digest mismatch (tag may have been re-pushed)")
                elif remote_digest:
                    msg += ", digest verified"
            return ValidationResult("docker_image", True, msg)
        if resp.status_code == 404:
            return ValidationResult("docker_image", False, f"{image} not found")
        if resp.status_code == 401:
            return ValidationResult("docker_image", True, f"{image} exists (auth required)")
        return ValidationResult("docker_image", False, f"unexpected status {resp.status_code}")
    except requests.Timeout:
        return ValidationResult("docker_image", False, "Docker registry check timed out")
    except Exception as e:
        return ValidationResult("docker_image", False, f"Docker check failed: {e}")


def run_all_validations(config: SrtConfig) -> list[ValidationResult]:
    """Run all applicable validation checks. Never raises."""
    results: list[ValidationResult] = []

    # Local model path
    try:
        results.append(validate_local_path("model_path", config.model.path))
    except Exception as e:
        results.append(ValidationResult("model_path", False, f"check failed: {e}"))

    # Local container path
    try:
        results.append(validate_local_path("container_path", config.model.container))
    except Exception as e:
        results.append(ValidationResult("container_path", False, f"check failed: {e}"))

    # HuggingFace model (from identity block)
    try:
        hf_repo = None
        hf_rev = None
        if config.identity and config.identity.model:
            hf_repo = config.identity.model.repo
            hf_rev = config.identity.model.revision
        results.append(validate_hf_model(hf_repo, hf_rev))
    except Exception as e:
        results.append(ValidationResult("hf_model", False, f"check failed: {e}"))

    return results


def _format_validation_results(results: list[ValidationResult]) -> str:
    """Format validation results for console output."""
    lines = ["Validation:"]
    for r in results:
        icon = "ok" if r.ok else "WARN"
        lines.append(f"  [{icon}] {r.check}: {r.message}")
    return "\n".join(lines)


def run_validations_background(config: SrtConfig) -> threading.Thread:
    """Run all validations in a daemon background thread. Never blocks."""

    def _run():
        try:
            results = run_all_validations(config)
            output = _format_validation_results(results)
            logger.info("\n%s", output)
        except Exception as e:
            logger.debug("Background validation failed: %s", e)

    thread = threading.Thread(target=_run, daemon=True, name="srtctl-validation")
    thread.start()
    return thread
