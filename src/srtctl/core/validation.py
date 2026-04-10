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
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import requests

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


def validate_local_path(name: str, path: str) -> ValidationResult:
    """Check that a local file or directory exists."""
    try:
        p = Path(path)
        if not p.exists():
            return ValidationResult(name, False, f"not found: {path}")
        if p.is_dir():
            file_count = sum(1 for _ in p.rglob("*") if _.is_file())
            total_gb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
            return ValidationResult(name, True, f"{file_count} files, {total_gb:.1f}GB")
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
