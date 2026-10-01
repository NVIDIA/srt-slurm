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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from marshmallow import ValidationError as MarshmallowValidationError

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


def _is_registry_uri(value: str) -> bool:
    # Mirrors the runtime classification in runtime.py (RuntimeContext.from_config):
    # anything not starting with "/" or "./" is forwarded to ``srun --container-image``,
    # which Pyxis/enroot pulls on first use. The ":" guard distinguishes a URI
    # (registry/...:tag or scheme://...) from a typo'd local relative path.
    return not value.startswith(("/", "./")) and ":" in value


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

    # HuggingFace model IDs (e.g. "hf:meta-llama/Llama-3.1-8B").  Mirrors
    # the runtime classification in runtime.py (RuntimeContext.from_config),
    # which strips the prefix and hands the model ID to the framework — the
    # framework downloads via HF_HOME at serve time.  Preflight cannot
    # filesystem-check a remote ID, so accept and let runtime fail loudly
    # if the ID is bogus.
    if isinstance(raw, str) and raw.startswith("hf:"):
        return (
            PreflightResolution(
                field="model.path",
                raw=raw,
                resolved=raw,
                source="huggingface",
                ok=True,
                message=f"HuggingFace model ID: {raw[3:]}",
            ),
            [],
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

    # Container image URIs (e.g. "nvcr.io/nvidia/sglang-runtime:0.8.1",
    # "vllm/vllm-openai:latest", "docker://...").
    if isinstance(raw, str) and _is_registry_uri(raw):
        return (
            PreflightResolution(
                field="model.container",
                raw=raw,
                resolved=raw,
                source="container-uri",
                ok=True,
                message=f"Container image URI: {raw}",
            ),
            [],
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


_TACHOMETER_IMAGE_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("observability.tachometer.dcgm_exporter.container_image", ("dcgm_exporter", "container_image")),
    ("observability.tachometer.node_exporter.container_image", ("node_exporter", "container_image")),
)
_POWER_IMAGE_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("telemetry.dcgm_exporter.container_image", ("dcgm_exporter", "container_image")),
)


def _preflight_telemetry(
    raw_config: dict[str, Any],
    resolved_config: dict[str, Any],
    cluster_config: dict[str, Any] | None,
) -> list[PreflightIssue]:
    aliases = (cluster_config or {}).get("containers") or {}
    issues: list[PreflightIssue] = []

    raw_observability = raw_config.get("observability") or {}
    resolved_observability = resolved_config.get("observability") or {}
    sources = (
        (
            "Tachometer",
            raw_observability.get("tachometer") if isinstance(raw_observability, dict) else None,
            resolved_observability.get("tachometer") if isinstance(resolved_observability, dict) else None,
            _TACHOMETER_IMAGE_FIELDS,
            "tachometer-container-not-available",
        ),
        (
            "Telemetry",
            raw_config.get("telemetry"),
            resolved_config.get("telemetry"),
            _POWER_IMAGE_FIELDS,
            "telemetry-container-not-available",
        ),
    )

    for label, raw_block, resolved_block, fields, code in sources:
        if not isinstance(resolved_block, dict) or not resolved_block.get("enabled"):
            continue
        raw_block = raw_block if isinstance(raw_block, dict) else {}
        for field, path in fields:
            resolved_value: Any = resolved_block
            raw_value: Any = raw_block
            for key in path:
                resolved_value = (resolved_value or {}).get(key) if isinstance(resolved_value, dict) else None
                raw_value = (raw_value or {}).get(key) if isinstance(raw_value, dict) else None
            if not resolved_value or _is_registry_uri(str(resolved_value)):
                continue

            ok, _ = _check_path(_expand_path(resolved_value), expect="file")
            if ok:
                continue

            if raw_value in aliases:
                message = (
                    f"{label} alias '{raw_value}' resolved to '{resolved_value}', but that file is unavailable. "
                    "Provide or register the container yourself before submitting."
                )
            else:
                message = (
                    f"{label} container '{resolved_value}' is not a local container path and is not defined "
                    "in srtslurm.yaml containers. Provide or register the container yourself before submitting."
                )
            issues.append(PreflightIssue(code=code, field=field, message=message))

    return issues


def validate_topology(roles: Mapping[str, Any] | None, *, service_nodes: int | None = None) -> list[PreflightIssue]:
    """Catch semantically wrong ``roles:`` blocks that pass the marshmallow schema.

    The schema accepts any mix of the prefill, decode, and agg roles, so a recipe
    with a prefill role of 0 workers next to a decode role looks valid but
    expresses "disaggregated with no prefill", which is really an aggregated
    deployment and should declare ``roles.agg`` instead.
    """
    present = (
        {role: spec for role, spec in roles.items() if isinstance(spec, dict)} if isinstance(roles, Mapping) else {}
    )
    # Services that own nodes (pools) add to the allocation next to the roles. A
    # recipe with pools and no roles is a services-only job and needs no topology here.
    if not present:
        if service_nodes is not None:
            return []
        return [
            PreflightIssue(
                code="topology-missing",
                field="roles",
                message=(
                    "No roles declared. Set roles.prefill and roles.decode (nodes + workers) for a "
                    "disaggregated deployment, or roles.agg (nodes + workers) for an aggregated one."
                ),
            )
        ]

    def count(spec: dict[str, Any], key: str) -> int:
        value = spec.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    disagg = [role for role in ("prefill", "decode") if role in present]
    agg = present.get("agg")
    if disagg and agg is not None:
        return [
            PreflightIssue(
                code="topology-mixed",
                field="roles",
                message=(
                    f"Mixes the disaggregated roles ({', '.join(disagg)}) with the aggregated role (agg). "
                    "Declare prefill and decode, or agg, not both."
                ),
            )
        ]

    issues: list[PreflightIssue] = []

    if disagg:
        prefill = present.get("prefill", {})
        decode = present.get("decode", {})
        pf_workers = count(prefill, "workers")
        dc_workers = count(decode, "workers")

        if pf_workers == 0 and dc_workers == 0:
            issues.append(
                PreflightIssue(
                    code="topology-no-workers",
                    field="roles",
                    message=(
                        "The disaggregated roles have no workers: roles.prefill.workers and "
                        "roles.decode.workers are both 0/null. Set both, or declare roles.agg instead."
                    ),
                )
            )
        elif pf_workers == 0:
            issues.append(
                PreflightIssue(
                    code="topology-aggregated-style",
                    field="roles.prefill.workers",
                    message=(
                        "roles.prefill has no workers next to a decode role. For a single-side deployment "
                        f"on {decode.get('nodes')} node(s) with {dc_workers} worker(s), declare roles.agg "
                        f"(nodes: {decode.get('nodes')}, workers: {dc_workers}) and drop prefill/decode."
                    ),
                )
            )
        elif dc_workers == 0:
            issues.append(
                PreflightIssue(
                    code="topology-aggregated-style",
                    field="roles.decode.workers",
                    message=(
                        "roles.decode has no workers next to a prefill role. For a single-side deployment "
                        f"on {prefill.get('nodes')} node(s) with {pf_workers} worker(s), declare roles.agg "
                        f"(nodes: {prefill.get('nodes')}, workers: {pf_workers}) and drop prefill/decode."
                    ),
                )
            )
        return issues

    assert agg is not None
    if count(agg, "workers") == 0:
        issues.append(
            PreflightIssue(
                code="topology-no-workers",
                field="roles.agg.workers",
                message="roles.agg.workers must be > 0.",
            )
        )
    if count(agg, "nodes") == 0:
        issues.append(
            PreflightIssue(
                code="topology-no-nodes",
                field="roles.agg.nodes",
                message="roles.agg.nodes must be > 0.",
            )
        )
    return issues


def _declared_service_nodes(services: Any) -> int | None:
    """Nodes the recipe's services own through ``services[].nodes``, summed; None when none do."""
    if not isinstance(services, list):
        return None
    counts = [entry["nodes"] for entry in services if isinstance(entry, dict) and entry.get("nodes") is not None]
    return sum(counts) if counts else None
    for entry in services:
        if isinstance(entry, dict) and entry.get("nodes") is not None:
            return entry["nodes"]
    return None


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
    from srtctl.core.schema import SrtConfig

    results: list[PreflightResult] = []
    for suffix, variant in variants:
        try:
            resolved = resolve_config_with_defaults(variant, active_cluster_config)
            SrtConfig.Schema().load(resolved)
        except (TypeError, ValueError, MarshmallowValidationError) as exc:
            # A pre-2.0 layout, a malformed block, or any rule the schema enforces is a
            # finding for this variant, not a crash of the whole preflight.
            unresolved = PreflightResolution(
                field="recipe", raw=None, resolved=None, source="unresolved", ok=False, message=str(exc)
            )
            results.append(
                PreflightResult(
                    variant=suffix,
                    ok=False,
                    model=unresolved,
                    container=unresolved,
                    errors=[PreflightIssue(code="recipe-rejected", field="schema", message=str(exc))],
                )
            )
            continue
        model, model_issues = _preflight_model(variant, resolved, active_cluster_config)
        container, container_issues = _preflight_container(variant, resolved, active_cluster_config)
        topology_issues = validate_topology(
            resolved.get("roles"), service_nodes=_declared_service_nodes(resolved.get("services"))
        )
        telemetry_issues = _preflight_telemetry(variant, resolved, active_cluster_config)
        issues = [*model_issues, *container_issues, *topology_issues, *telemetry_issues]
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
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
        return ValidationResult("docker_image", False, f"Docker check failed: {e}")


def run_all_validations(config: SrtConfig) -> list[ValidationResult]:
    """Run all applicable validation checks. Never raises."""
    results: list[ValidationResult] = []

    # Local model path
    try:
        results.append(validate_local_path("model_path", config.model.path))
    except Exception as e:  # noqa: BLE001
        results.append(ValidationResult("model_path", False, f"check failed: {e}"))

    # Local container path
    try:
        results.append(validate_local_path("container_path", config.model.container))
    except Exception as e:  # noqa: BLE001
        results.append(ValidationResult("container_path", False, f"check failed: {e}"))

    # HuggingFace model (from identity block)
    try:
        hf_repo = None
        hf_rev = None
        if config.identity and config.identity.model:
            hf_repo = config.identity.model.repo
            hf_rev = config.identity.model.revision
        results.append(validate_hf_model(hf_repo, hf_rev))
    except Exception as e:  # noqa: BLE001
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
        except Exception as e:  # noqa: BLE001
            logger.debug("Background validation failed: %s", e)

    thread = threading.Thread(target=_run, daemon=True, name="srtctl-validation")
    thread.start()
    return thread
