# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``srtctl install <model>`` — one-shot model + container provisioning.

The command is SLURM-first: it submits an sbatch wrapper job that runs the
download + import + alias-registration flow on a compute node.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import re

from rich.console import Console

from srtctl.core.config import load_cluster_config
from srtctl.install.container import container_filename
from srtctl.install.model import model_storage_dirname
from srtctl.install.registry import ModelInstallSpec
from srtctl.install.setup import srtctl_root_from_package
from srtctl.install.slurm import submit_install_job

console = Console()
logger = logging.getLogger(__name__)

_SAFE_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_MODEL_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_SAFE_CONTAINER_IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:+-]*$")


def add_install_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the ``install`` subcommand on the top-level ``srtctl`` parser."""
    parser = subparsers.add_parser(
        "install",
        help="Download model + container and register aliases in srtslurm.yaml",
        description=(
            "One-shot provisioning for a model: submits a SLURM job that downloads HF weights, "
            "imports the container .sqsh, and updates srtslurm.yaml."
        ),
    )
    parser.add_argument(
        "model",
        help="Model install name (used for job naming and log labels), e.g. glm5",
    )
    parser.add_argument(
        "--hf-repo-id",
        required=True,
        help="Hugging Face repo id to download, e.g. nvidia/GLM-5-NVFP4",
    )
    parser.add_argument(
        "--model-alias",
        required=True,
        help="Alias key to write under srtslurm.yaml:model_paths",
    )
    parser.add_argument(
        "--container-image",
        required=True,
        help="Container image reference to import with enroot (e.g. nvcr.io/org/image:tag)",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Install base directory (default: <srtctl_root>/install)",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=None,
        help=(
            "Path to the venv to activate on the compute node. "
            "Must already exist with srtctl + huggingface_hub installed for the "
            "compute-node architecture. Default: <srtctl_root>/.venv"
        ),
    )
    parser.add_argument(
        "--strict-auth-preflight",
        action="store_true",
        help=(
            "Fail early when common nvcr.io credential files are missing. "
            "Default behavior is warning-only and attempts pull anyway."
        ),
    )
    return parser


def _resolve_paths(spec: ModelInstallSpec, srtctl_root: Path, base: Path | None) -> tuple[Path, Path, Path]:
    install_base = base if base is not None else (srtctl_root / "install")
    model_dir = install_base / "models" / model_storage_dirname(spec.hf_repo_id)
    container_path = install_base / "containers" / container_filename(spec.container_image)
    return install_base, model_dir, container_path


def _load_cluster_config_for_install(srtctl_root: Path) -> dict[str, object] | None:
    """Load cluster config anchored to this srtctl checkout.

    The generic loader searches from CWD; for `srtctl install` we want stable
    behavior even when invoked from another directory (for example from LLMB
    staging scripts).
    """
    srtslurm_yaml = srtctl_root / "srtslurm.yaml"
    if not srtslurm_yaml.exists():
        return load_cluster_config()

    old = os.environ.get("SRTSLURM_CONFIG")
    os.environ["SRTSLURM_CONFIG"] = str(srtslurm_yaml)
    try:
        return load_cluster_config()
    finally:
        if old is None:
            os.environ.pop("SRTSLURM_CONFIG", None)
        else:
            os.environ["SRTSLURM_CONFIG"] = old


def _resolve_spec(args: argparse.Namespace) -> ModelInstallSpec:
    """Resolve install spec from explicit CLI values."""
    if not (args.hf_repo_id and args.model_alias and args.container_image):
        raise ValueError("--hf-repo-id, --model-alias, and --container-image are required.")

    spec = ModelInstallSpec(
        name=args.model,
        hf_repo_id=args.hf_repo_id,
        model_alias=args.model_alias,
        container_image=args.container_image,
        default_recipe="",
        description=f"Custom install for {args.model}",
    )
    _validate_spec(spec)
    return spec


def _validate_spec(spec: ModelInstallSpec) -> None:
    """Validate user-controllable fields before generating shell scripts."""
    if not _SAFE_MODEL_NAME_RE.fullmatch(spec.name):
        raise ValueError(
            f"Invalid model name {spec.name!r}. Allowed characters: letters, numbers, dot, underscore, hyphen."
        )
    if not _SAFE_HF_REPO_RE.fullmatch(spec.hf_repo_id):
        raise ValueError(
            f"Invalid hf repo id {spec.hf_repo_id!r}. Expected format 'org/repo' with safe URL characters."
        )
    if not _SAFE_MODEL_ALIAS_RE.fullmatch(spec.model_alias):
        raise ValueError(
            f"Invalid model alias {spec.model_alias!r}. Allowed characters: letters, numbers, /, dot, underscore, hyphen."
        )
    if not _SAFE_CONTAINER_IMAGE_RE.fullmatch(spec.container_image):
        raise ValueError(
            f"Invalid container image {spec.container_image!r}. Allowed characters: letters, numbers, /, :, @, +, dot, underscore, hyphen."
        )


def _submit_as_slurm_job(
    spec: ModelInstallSpec,
    srtctl_root: Path,
    install_base: Path,
    venv_path: Path,
    *,
    strict_auth_preflight: bool,
) -> int:
    """Generate + submit the sbatch wrapper. Returns CLI exit code."""
    cluster_config = _load_cluster_config_for_install(srtctl_root)
    if "HF_TOKEN" not in os.environ:
        console.print(
            "[bold red]HF_TOKEN is not set in this shell.[/] "
            "Export it before submitting: `export HF_TOKEN=hf_xxx`."
        )
        return 2

    try:
        sub = submit_install_job(
            spec=spec,
            srtctl_root=srtctl_root,
            install_base=install_base,
            venv_path=venv_path,
            cluster_config=cluster_config,
            strict_auth_preflight=strict_auth_preflight,
        )
    except RuntimeError as exc:
        console.print(f"[bold red]{exc}[/]")
        return 2

    console.print()
    console.print(f"[bold green]✓ Submitted SLURM job {sub.job_id}[/]")
    console.print(f"  script : {sub.script_path}")
    console.print(f"  log    : {sub.log_path}")
    console.print()
    console.print("[bold]Monitor:[/]")
    console.print(f"  squeue -u $USER -j {sub.job_id}")
    console.print(f"  tail -f {sub.log_path}")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    """Handler invoked by ``srtctl install`` argparse dispatch."""
    try:
        spec = _resolve_spec(args)
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/]")
        return 2

    srtctl_root = srtctl_root_from_package()
    install_base, _, _ = _resolve_paths(spec, srtctl_root, args.base)
    inside_slurm_job = bool(os.environ.get("SLURM_JOB_ID"))
    if inside_slurm_job:
        console.print(
            "[bold red]Detected SLURM_JOB_ID in environment.[/] "
            "Run `srtctl install` from a login node (outside an active job) so it can submit sbatch."
        )
        return 2

    venv_path = args.venv if args.venv is not None else (srtctl_root / ".venv")
    console.print(f"[bold cyan]Submitting {spec.name} install as a SLURM job[/]")
    console.print(f"  srtctl root  : {srtctl_root}")
    console.print(f"  install base : {install_base}")
    console.print(f"  venv (compute): {venv_path}")
    return _submit_as_slurm_job(
        spec,
        srtctl_root,
        install_base,
        venv_path,
        strict_auth_preflight=args.strict_auth_preflight,
    )
