#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified job submission interface for srtctl.

This is the main entrypoint for submitting benchmarks via YAML configs.

Usage:
    srtctl apply -f config.yaml                     # Submit job
    srtctl apply -f config.yaml -o /path/to/logs   # Submit with custom output dir
    srtctl dry-run -f sweep.yaml --sweep            # Dry run sweep
"""

import argparse
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

# Import from srtctl modules
from srtctl.core.config import (
    find_cluster_config_path,
    generate_override_configs,
    get_cluster_aliases,
    get_srtslurm_setting,
    load_cluster_config,
    load_config,
    resolve_config_with_defaults,
)
from srtctl.core.fingerprint import (
    capture_fingerprint,
    check_against_fingerprint,
    diff_fingerprints,
    format_check_results,
    format_diff,
)
from srtctl.core.lockfile import load_lockfile_fingerprints
from srtctl.core.schema import SrtConfig
from srtctl.core.status import create_job_record
from srtctl.core.validation import preflight_config_variants

console = Console()
logger = logging.getLogger(__name__)

# Populated by submit_with_orchestrator on successful submission. Consumed by
# main() when --json is set so callers get one JSON line per submitted job.
_submissions: list[dict] = []


def _record_submission(data: dict) -> None:
    _submissions.append(data)


def _format_preflight_error(label: str, results: list[Any]) -> str:
    lines = [f"Preflight failed for {label}:"]
    for result in results:
        for issue in result.errors:
            lines.append(f"- {issue.field}: {issue.message}")
    return "\n".join(lines)


def _assert_preflight_passed(raw_config: dict[str, Any], *, label: str) -> None:
    results = preflight_config_variants(
        raw_config,
        cluster_config=load_cluster_config(),
    )
    failed = [result for result in results if not result.ok]
    if failed:
        raise ValueError(_format_preflight_error(label, failed))


def _install_mock_submit_patches() -> list:
    """Stub the subset of `submit_with_orchestrator` that reaches real infra.

    - `subprocess.run(["sbatch", ...])` is replaced with a fake that returns a
      synthetic job id so the rest of the submit flow continues through the
      real config write + metadata + _record_submission path.
    - `validate_setup` and `create_job_record` are stubbed so mock runs do not
      probe the cluster install or POST to a real status endpoint.
    """
    from unittest.mock import patch

    original_run = subprocess.run

    def _fake_subprocess_run(cmd, *args, **kwargs):
        is_sbatch = (
            isinstance(cmd, list | tuple)
            and len(cmd) > 0
            and (cmd[0] == "sbatch" or (isinstance(cmd[0], str) and cmd[0].endswith("sbatch")))
        )
        if not is_sbatch:
            return original_run(cmd, *args, **kwargs)
        import time as _time

        job_id = str(400_000 + int(_time.time() * 100) % 100_000)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=f"Submitted batch job {job_id}\n",
            stderr="",
        )

    patchers = [
        patch("subprocess.run", _fake_subprocess_run),
        patch("srtctl.cli.submit.validate_setup"),
        patch("srtctl.cli.submit.create_job_record"),
    ]
    for p in patchers:
        p.start()
    return patchers


def _spawn_mock_worker(submission: dict, tick_s: float) -> None:
    """Detach a `srtctl.cli.mock_worker` subprocess to drive the full orchestrator.

    Writes worker stdout+stderr to <output_dir>/mock_worker.log so the parent
    process can exit cleanly while the child keeps ticking.
    """
    output_dir = Path(submission["output_dir"])
    config_path = submission["config_path"]
    job_id = submission["slurm_job_id"]
    log_path = output_dir / "mock_worker.log"
    log_fh = log_path.open("w")
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "srtctl.cli.mock_worker",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--job-id",
            str(job_id),
            "--tick-s",
            str(tick_s),
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def get_job_name(config: SrtConfig) -> str:
    """Get job name, using RUNNER_NAME if available, otherwise config name.

    This allows multi-runner setups to have unique job names for cleanup.

    Args:
        config: SrtConfig with the base job name

    Returns:
        Job name: RUNNER_NAME if set, otherwise config.name
    """
    runner_name = os.environ.get("RUNNER_NAME")
    if runner_name:
        return runner_name
    return config.name


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def show_config_details(config: SrtConfig) -> None:
    """Display container mounts and environment variables for dry-run verification.

    Shows all mounts (from built-in defaults, srtslurm.yaml, and recipe) and all
    environment variables (global and backend per-mode) so users can verify their
    config is correct before submitting.
    """
    # --- Container Mounts ---
    mounts_table = Table(title="Container Mounts", show_lines=False, pad_edge=False)
    mounts_table.add_column("Source", style="dim", width=14)
    mounts_table.add_column("Host Path", style="green")
    mounts_table.add_column("Container Path", style="cyan")

    # Built-in mounts (always present at runtime)
    model_path = os.path.expandvars(config.model.path)
    mounts_table.add_row("built-in", model_path, "/model")
    mounts_table.add_row("built-in", "<log_dir>", "/logs")

    # Cluster-level mounts from srtslurm.yaml
    cluster_mounts = get_srtslurm_setting("default_mounts")
    if cluster_mounts:
        for host_path, container_path in cluster_mounts.items():
            expanded = os.path.expandvars(host_path)
            mounts_table.add_row("srtslurm.yaml", expanded, container_path)

    # Recipe extra_mount (simple string mounts)
    if config.extra_mount:
        for mount_spec in config.extra_mount:
            parts = mount_spec.split(":", 1)
            if len(parts) == 2:
                mounts_table.add_row("recipe", parts[0], parts[1])
            else:
                mounts_table.add_row("recipe", mount_spec, mount_spec)

    # Recipe container_mounts (FormattablePath mounts)
    if config.container_mounts:
        for host_template, container_template in config.container_mounts.items():
            mounts_table.add_row("recipe", str(host_template), str(container_template))

    console.print(Panel(mounts_table, border_style="green"))

    # --- Environment Variables ---
    has_env = bool(config.environment)
    backend = config.backend
    mode_envs: list[tuple[str, dict[str, str]]] = []
    for mode_name, attr in [
        ("prefill", "prefill_environment"),
        ("decode", "decode_environment"),
        ("aggregated", "aggregated_environment"),
    ]:
        env = getattr(backend, attr, {})
        if env:
            has_env = True
            mode_envs.append((mode_name, dict(env)))
    if config.benchmark.env:
        has_env = True
        mode_envs.append(("benchmark", dict(config.benchmark.env)))

    if has_env:
        env_table = Table(title="Environment Variables", show_lines=False, pad_edge=False)
        env_table.add_column("Scope", style="dim", width=14)
        env_table.add_column("Variable", style="yellow")
        env_table.add_column("Value", style="white")

        for var, val in sorted(config.environment.items()):
            env_table.add_row("global", var, val)

        for mode_name, env in mode_envs:
            for var, val in sorted(env.items()):
                env_table.add_row(mode_name, var, val)

        console.print(Panel(env_table, border_style="yellow"))
    else:
        console.print("[dim]No custom environment variables configured.[/]")

    # --- srun options ---
    if config.srun_options:
        opts = " ".join(f"--{k} {v}" if v else f"--{k}" for k, v in config.srun_options.items())
        console.print(f"[dim]srun options:[/] {opts}")

    if config.benchmark.type == "custom" or config.telemetry.enabled:
        details = Table(title="Execution Extensions", show_lines=False, pad_edge=False)
        details.add_column("Area", style="dim", width=14)
        details.add_column("Setting", style="yellow")
        details.add_column("Value", style="white")

        if config.benchmark.type == "custom":
            details.add_row("benchmark", "type", config.benchmark.type)
            if config.benchmark.command:
                details.add_row("benchmark", "command", config.benchmark.command)
            if config.benchmark.container_image:
                details.add_row("benchmark", "container_image", config.benchmark.container_image)

        if config.telemetry.enabled:
            details.add_row("telemetry", "provider", config.telemetry.provider.value)
            details.add_row("telemetry", "container_image", config.telemetry.container_image or "<unset>")
            details.add_row("telemetry", "storage_subdir", config.telemetry.storage_subdir)
            details.add_row("telemetry", "frequency", str(config.telemetry.default_frequency))

        console.print(Panel(details, border_style="blue"))


def validate_setup(srtctl_source: Path) -> None:
    """Validate that make setup has been run and required binaries exist.

    Checks for NATS, etcd, and compute-arch uv binaries. Raises SystemExit
    with a clear error message if anything is missing.
    """
    missing = []

    configs_dir = srtctl_source / "configs"
    if not (configs_dir / "nats-server").exists():
        missing.append("configs/nats-server")
    if not (configs_dir / "etcd").exists():
        missing.append("configs/etcd")
    if not (srtctl_source / "bin" / "uv").exists():
        missing.append("bin/uv (compute-arch uv)")

    if missing:
        console.print(f"\n[red bold]ERROR:[/] Required binaries not found in {srtctl_source}:")
        for m in missing:
            console.print(f"  [red]✗[/] {m}")
        console.print("\nRun [bold]make setup ARCH=<compute_arch>[/] first:")
        console.print(f"  cd {srtctl_source}")
        console.print("  make setup ARCH=aarch64  [dim]# for GB200/Grace compute nodes[/]")
        console.print("  make setup ARCH=x86_64   [dim]# for x86_64 compute nodes[/]\n")
        raise SystemExit(1)


def generate_minimal_sbatch_script(
    config: SrtConfig,
    config_path: Path,
    setup_script: str | None = None,
    output_dir: Path | None = None,
    runtime_config_filename: str = "config.yaml",
) -> str:
    """Generate minimal sbatch script that calls the Python orchestrator.

    The orchestrator runs INSIDE the container on the head node.
    srtctl is pip-installed inside the container at job start.

    Args:
        config: Typed SrtConfig
        config_path: Path to the YAML config file
        setup_script: Optional setup script override (passed via env var)
        output_dir: Custom output directory (CLI flag, highest priority)
        runtime_config_filename: Config file name under OUTPUT_DIR used by do_sweep

    Returns:
        Rendered sbatch script as string
    """
    from jinja2 import Environment, FileSystemLoader

    # Find template directory and srtctl source
    # Templates are now in src/srtctl/templates/
    template_dir = Path(__file__).parent.parent / "templates"

    srtctl_root = get_srtslurm_setting("srtctl_root")
    # srtctl source is the parent of src/srtctl (i.e., the repo root)
    srtctl_source = Path(srtctl_root) if srtctl_root else Path(__file__).parent.parent.parent.parent

    # Determine output base directory
    # Priority: CLI -o flag > srtslurm.yaml output_dir > srtctl_root/outputs
    if output_dir:
        output_base = str(output_dir.resolve())
    else:
        custom_output_dir = get_srtslurm_setting("output_dir")
        if custom_output_dir:
            output_base = str(Path(os.path.expandvars(custom_output_dir)).resolve())
        else:
            output_base = str((srtctl_source / "outputs").resolve())

    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("job_script_minimal.j2")

    total_nodes = config.resources.total_nodes
    # Add extra node for dedicated etcd/nats infrastructure
    if config.infra.etcd_nats_dedicated_node:
        total_nodes += 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve container image path (expand aliases from srtslurm.yaml)
    container_image = os.path.expandvars(config.model.container)

    job_name = get_job_name(config)

    rendered = template.render(
        job_name=job_name,
        total_nodes=total_nodes,
        gpus_per_node=config.resources.gpus_per_node,
        backend_type=config.backend_type,
        account=config.slurm.account or os.environ.get("SLURM_ACCOUNT", "default"),
        partition=config.slurm.partition or os.environ.get("SLURM_PARTITION", "default"),
        time_limit=config.slurm.time_limit or "01:00:00",
        config_path=str(config_path.resolve()),
        runtime_config_filename=runtime_config_filename,
        timestamp=timestamp,
        use_gpus_per_node_directive=get_srtslurm_setting("use_gpus_per_node_directive", True),
        use_segment_sbatch_directive=get_srtslurm_setting("use_segment_sbatch_directive", True),
        use_exclusive_sbatch_directive=get_srtslurm_setting("use_exclusive_sbatch_directive", False),
        sbatch_directives=config.sbatch_directives,
        container_image=container_image,
        srtctl_source=str(srtctl_source.resolve()),
        output_base=output_base,
        setup_script=setup_script,
    )

    return rendered


def _print_running_summary(config: SrtConfig, console: Console) -> None:
    """Print what's being run and identity verification status."""
    console.print()
    console.print("[bold]Running:[/]")
    console.print(f"  Model:     {config.model.path}")
    console.print(f"  Container: {config.model.container}")
    console.print(f"  Backend:   {config.backend_type}")
    console.print(f"  Benchmark: {config.benchmark.type}")

    has_identity = config.identity and (
        (config.identity.model and (config.identity.model.repo or config.identity.model.revision))
        or (config.identity.container and config.identity.container.image)
        or config.identity.frameworks
    )
    if has_identity:
        id_fields = []
        if config.identity.model and config.identity.model.repo:
            id_fields.append(f"model={config.identity.model.repo}")
        if config.identity.model and config.identity.model.revision:
            id_fields.append(f"rev={config.identity.model.revision[:12]}")
        if config.identity.container and config.identity.container.image:
            # Shorten long registry URIs for display
            img = config.identity.container.image
            if len(img) > 50:
                img = "..." + img[-47:]
            id_fields.append(f"container={img}")
        for name, ver in (config.identity.frameworks or {}).items():
            id_fields.append(f"{name}={ver}")
        console.print(f"  Identity:  {', '.join(id_fields)}")
    else:
        console.print()
        console.print(
            "[yellow]Tip:[/] Add an [bold]identity:[/] block to your recipe so others can replicate your results."
        )
        console.print("[yellow]     Without it, someone reading this recipe can't tell which exact model[/]")
        console.print("[yellow]     weights or framework versions produced the benchmark numbers.[/]")
        console.print("[yellow]     At job start, srtctl verifies the runtime matches your declaration.[/]")
        console.print()
        console.print("[dim]     identity:[/]")
        console.print("[dim]       model:[/]")
        console.print('[dim]         repo: "nvidia/Kimi-K2.5-NVFP4"       [/][dim italic]# HuggingFace model ID[/]')
        console.print('[dim]         revision: "c0285e649c34..."            [/][dim italic]# HF commit SHA[/]')
        console.print("[dim]       container:[/]")
        console.print('[dim]         image: "gitlab:5005/.../trtllm-arm64"  [/][dim italic]# pullable Docker URI[/]')
        console.print("[dim]       frameworks:                              [/][dim italic]# dynamo + one engine[/]")
        console.print('[dim]         dynamo: "1.0.0"                        [/][dim italic]# always (ai-dynamo)[/]')
        console.print('[dim]         tensorrt_llm: "1.3.0rc9"              [/][dim italic]# OR vllm OR sglang[/]')
        console.print()
        console.print("[dim italic]     Agent: if you are an AI assistant creating this recipe, you MUST[/]")
        console.print("[dim italic]     include the identity block. Use the container's framework versions[/]")
        console.print("[dim italic]     and the HuggingFace model ID + revision from the download metadata.[/]")


def submit_with_orchestrator(
    config_path: Path,
    config: SrtConfig | None = None,
    dry_run: bool = False,
    tags: list[str] | None = None,
    setup_script: str | None = None,
    output_dir: Path | None = None,
    variant_suffix: str | None = None,
    source_config_path: Path | None = None,
    runtime_config_text: str | None = None,
) -> str | None:
    """Submit job using the new Python orchestrator.

    This uses the minimal sbatch template that calls srtctl.cli.do_sweep.

    Args:
        config_path: Path to the resolved YAML config passed to do_sweep.
        config: Pre-loaded SrtConfig (or None to load from path)
        dry_run: If True, print script but don't submit
        tags: Optional tags for the run
        setup_script: Optional custom setup script name (overrides config)
        output_dir: Custom output directory (CLI flag, highest priority)
        variant_suffix: If set (e.g. "base", "lowmem"), also save config_path
                        as config_{variant_suffix}.yaml in the job output dir.
        source_config_path: If set, save the original source YAML as config.yaml
                            while the job executes a resolved variant config.
        runtime_config_text: Resolved runtime YAML written under OUTPUT_DIR when
                             source_config_path is set.

    Returns:
        job_id string on success, None for dry_run.
    """

    if config is None:
        config = load_config(config_path)

    runtime_config_filename = "config.yaml"
    resolved_runtime_config_text: str | None = None
    if source_config_path:
        if runtime_config_text is None:
            raise ValueError("runtime_config_text is required when source_config_path is set")
        resolved_runtime_config_text = runtime_config_text
        runtime_config_filename = f"config_{variant_suffix}.yaml" if variant_suffix else "config_resolved.yaml"

    script_content = generate_minimal_sbatch_script(
        config=config,
        config_path=config_path,
        setup_script=setup_script,
        output_dir=output_dir,
        runtime_config_filename=runtime_config_filename,
    )

    # Identity validation (inline, <1s) — runs for both dry-run and submit
    if config.identity and config.identity.model and config.identity.model.repo:
        from srtctl.core.validation import validate_hf_model

        hf_result = validate_hf_model(config.identity.model.repo, config.identity.model.revision)
        if hf_result.ok:
            console.print(f"[green]✓[/] HF model: {hf_result.message}")
        else:
            console.print(f"[yellow]⚠ HF model: {hf_result.message}[/]")

    if dry_run:
        console.print()
        console.print(
            Panel(
                "[bold]🔍 DRY-RUN[/] [dim](orchestrator mode)[/]",
                title=config.name,
                border_style="yellow",
            )
        )
        console.print()
        syntax = Syntax(script_content, "bash", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Generated sbatch Script", border_style="cyan"))
        console.print()
        show_config_details(config)

        # Show running summary + identity in dry-run too
        _print_running_summary(config, console)
        return

    # Validate setup before submitting (not during dry-run)
    srtctl_root = get_srtslurm_setting("srtctl_root")
    srtctl_source = Path(srtctl_root) if srtctl_root else Path(__file__).parent.parent.parent.parent
    validate_setup(srtctl_source)

    # Write script to temp file
    fd, script_path = tempfile.mkstemp(suffix=".slurm", prefix="srtctl_", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    console.print(f"[bold cyan]🚀 Submitting:[/] {config.name}")
    logging.debug(f"Script: {script_path}")

    keep_script = False
    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True,
        )

        job_id = result.stdout.strip().split()[-1]

        # Determine output directory
        # Priority: CLI -o flag > srtslurm.yaml output_dir > srtctl_root/outputs
        if output_dir:
            job_output_dir = output_dir / job_id
        else:
            custom_output_dir = get_srtslurm_setting("output_dir")
            if custom_output_dir:
                job_output_dir = Path(os.path.expandvars(custom_output_dir)) / job_id
            else:
                srtctl_root = get_srtslurm_setting("srtctl_root")
                srtctl_source = Path(srtctl_root) if srtctl_root else Path(__file__).parent.parent.parent.parent
                job_output_dir = srtctl_source / "outputs" / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(source_config_path or config_path, job_output_dir / "config.yaml")
        if source_config_path:
            assert resolved_runtime_config_text is not None
            runtime_config_path = job_output_dir / runtime_config_filename
            runtime_config_path.write_text(resolved_runtime_config_text)
        shutil.copy(script_path, job_output_dir / "sbatch_script.sh")

        job_name = get_job_name(config)

        # Build comprehensive job metadata
        metadata: dict[str, Any] = {
            "version": "2.0",
            "orchestrator": True,
            "job_id": job_id,
            "job_name": job_name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Model info
            "model": {
                "path": config.model.path,
                "container": config.model.container,
                "precision": config.model.precision,
            },
            # Resource allocation
            "resources": {
                "gpu_type": config.resources.gpu_type,
                "gpus_per_node": config.resources.gpus_per_node,
                "prefill_nodes": config.resources.prefill_nodes,
                "decode_nodes": config.resources.decode_nodes,
                "prefill_workers": config.resources.num_prefill,
                "decode_workers": config.resources.num_decode,
                "agg_workers": config.resources.num_agg,
            },
            # Backend and frontend
            "backend_type": config.backend_type,
            "frontend_type": config.frontend.type,
            # Benchmark config
            "benchmark": {
                "type": config.benchmark.type,
                "isl": config.benchmark.isl,
                "osl": config.benchmark.osl,
            },
        }
        if tags:
            metadata["tags"] = tags
        if config.setup_script:
            metadata["setup_script"] = config.setup_script

        with open(job_output_dir / f"{job_id}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        _record_submission(
            {
                "status": "submitted",
                "slurm_job_id": job_id,
                "job_name": job_name,
                "output_dir": str(job_output_dir),
                "metadata_path": str(job_output_dir / f"{job_id}.json"),
                "config_path": str(config_path),
                "tags": list(tags) if tags else None,
            }
        )

        # Report to status API (fire-and-forget, silent on failure)
        # Note: tags are already included in metadata dict above
        create_job_record(
            reporting=config.reporting,
            job_id=job_id,
            job_name=job_name,
            cluster=get_srtslurm_setting("cluster"),
            recipe=str(config_path),
            metadata=metadata,
        )

        console.print(f"[bold green]✅ Job {job_id} submitted![/]")
        console.print(f"[dim]📁 Logs:[/] {job_output_dir}/logs")
        console.print(f"[dim]📋 Monitor:[/] tail -f {job_output_dir}/logs/sweep_{job_id}.log")
        console.print(f"[dim]📊 Queue:[/] squeue --job {job_id}")

        _print_running_summary(config, console)

        return job_id

    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ sbatch failed:[/] {e.stderr}")
        keep_script = True
        raise
    finally:
        if not keep_script:
            with contextlib.suppress(OSError):
                os.remove(script_path)
    return None


def submit_single(
    config_path: Path | None = None,
    config: SrtConfig | None = None,
    dry_run: bool = False,
    setup_script: str | None = None,
    tags: list[str] | None = None,
    output_dir: Path | None = None,
    variant_suffix: str | None = None,
    source_config_path: Path | None = None,
    runtime_config_text: str | None = None,
    enforce_preflight: bool = True,
) -> str | None:
    """Submit a single job from YAML config.

    Uses the orchestrator by default. This is the recommended submission method.

    Args:
        config_path: Path to YAML config file
        config: Pre-loaded SrtConfig (or None if loading from path)
        dry_run: If True, don't submit to SLURM
        setup_script: Optional custom setup script name
        tags: Optional list of tags
        output_dir: Custom output directory (CLI flag, highest priority)
        variant_suffix: If set, also save config as config_{suffix}.yaml in job output dir.
        source_config_path: If set, saved as config.yaml while execution uses the
                            resolved variant config.
        runtime_config_text: Resolved runtime YAML written under OUTPUT_DIR for
                             override submissions.

    Returns:
        job_id string on success, None for dry_run.
    """
    if config is None and config_path:
        config = load_config(config_path)

    if config is None:
        raise ValueError("Either config_path or config must be provided")

    if runtime_config_text is not None:
        raw_config = yaml.safe_load(runtime_config_text)
    elif config_path is not None:
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)
    else:
        raw_config = SrtConfig.Schema().dump(config)

    if enforce_preflight:
        _assert_preflight_passed(raw_config, label=str(config_path or "<inline-config>"))

    # Always use orchestrator mode
    return submit_with_orchestrator(
        config_path=config_path or Path("./config.yaml"),
        config=config,
        dry_run=dry_run,
        tags=tags,
        setup_script=setup_script,
        output_dir=output_dir,
        variant_suffix=variant_suffix,
        source_config_path=source_config_path,
        runtime_config_text=runtime_config_text,
    )


def is_sweep_config(config_path: Path) -> bool:
    """Check if config file is a sweep config by looking for 'sweep' section."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return "sweep" in config if config else False
    except Exception:
        return False


def submit_sweep(
    config_path: Path,
    dry_run: bool = False,
    setup_script: str | None = None,
    tags: list[str] | None = None,
    output_dir: Path | None = None,
):
    """Submit parameter sweep.

    Args:
        config_path: Path to sweep YAML config
        dry_run: If True, don't submit to SLURM
        setup_script: Optional custom setup script name
        tags: Optional list of tags
        output_dir: Custom output directory (CLI flag, highest priority)
    """
    from srtctl.core.sweep import generate_sweep_configs

    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)

    configs = generate_sweep_configs(sweep_config)

    # Display sweep table
    table = Table(title=f"Sweep: {sweep_config.get('name', 'unnamed')} ({len(configs)} jobs)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Job Name", style="green")
    table.add_column("Parameters", style="yellow")

    for i, (config_dict, params) in enumerate(configs, 1):
        job_name = config_dict.get("name", f"job_{i}")
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        table.add_row(str(i), job_name, params_str)

    console.print()
    console.print(table)
    console.print()

    if dry_run:
        console.print(
            Panel(
                "[bold yellow]🔍 DRY-RUN MODE[/]",
                subtitle=f"{len(configs)} jobs",
                border_style="yellow",
            )
        )

        sweep_dir = Path.cwd() / "dry-runs" / f"{sweep_config['name']}_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        with open(sweep_dir / "sweep_config.yaml", "w") as f:
            yaml.dump(sweep_config, f, default_flow_style=False)

        for i, (config_dict, _params) in enumerate(configs, 1):
            job_name = config_dict.get("name", f"job_{i}")
            job_dir = sweep_dir / f"job_{i:03d}_{job_name}"
            job_dir.mkdir(exist_ok=True)
            with open(job_dir / "config.yaml", "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)

        console.print(f"[dim]📁 Output:[/] {sweep_dir}")
        return

    # Real submission with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Submitting jobs...", total=len(configs))

        for i, (config_dict, _params) in enumerate(configs, 1):
            job_name = config_dict.get("name", f"job_{i}")
            progress.update(task, description=f"[{i}/{len(configs)}] {job_name}")

        # Save temp config and submit
        fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", prefix="srtctl_sweep_", text=True)
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(config_dict, f)

            config = load_config(Path(temp_config_path))
            submit_single(
                config_path=Path(temp_config_path),
                config=config,
                dry_run=False,
                setup_script=setup_script,
                tags=tags,
                output_dir=output_dir,
            )
        finally:
            with contextlib.suppress(OSError):
                os.remove(temp_config_path)

            progress.advance(task)

    console.print(f"\n[bold green]✅ Sweep complete![/] Submitted {len(configs)} jobs.")


def find_yaml_files(directory: Path) -> list[Path]:
    """Recursively find all YAML files in a directory.

    Args:
        directory: Directory to search

    Returns:
        Sorted list of YAML file paths
    """
    yaml_files = list(directory.rglob("*.yaml")) + list(directory.rglob("*.yml"))
    return sorted(set(yaml_files))


def submit_directory(
    directory: Path,
    dry_run: bool = False,
    setup_script: str | None = None,
    tags: list[str] | None = None,
    force_sweep: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Submit all YAML configs in a directory recursively.

    Args:
        directory: Directory containing YAML config files
        dry_run: If True, don't submit to SLURM
        setup_script: Optional custom setup script name
        tags: Optional list of tags
        force_sweep: If True, treat all configs as sweeps
        output_dir: Custom output directory (CLI flag, highest priority)
    """
    yaml_files = find_yaml_files(directory)

    if not yaml_files:
        console.print(f"[bold yellow]⚠️  No YAML files found in:[/] {directory}")
        return

    console.print(f"[bold cyan]📁 Found {len(yaml_files)} YAML file(s) in:[/] {directory}")
    console.print()

    # Display table of files to be processed
    table = Table(title=f"Configs to {'validate' if dry_run else 'submit'}")
    table.add_column("#", style="dim", width=4)
    table.add_column("File", style="green")
    table.add_column("Type", style="yellow")

    for i, yaml_file in enumerate(yaml_files, 1):
        relative_path = yaml_file.relative_to(directory)
        if is_override_config(yaml_file):
            config_type = "override"
        elif force_sweep or is_sweep_config(yaml_file):
            config_type = "sweep"
        else:
            config_type = "single"
        table.add_row(str(i), str(relative_path), config_type)

    console.print(table)
    console.print()

    # Process each file
    success_count = 0
    error_count = 0

    for i, yaml_file in enumerate(yaml_files, 1):
        relative_path = yaml_file.relative_to(directory)
        console.print(f"[bold]({i}/{len(yaml_files)})[/] Processing: {relative_path}")

        try:
            if is_override_config(yaml_file):
                submit_override(yaml_file, dry_run=dry_run, setup_script=setup_script, tags=tags, output_dir=output_dir)
            elif force_sweep or is_sweep_config(yaml_file):
                submit_sweep(yaml_file, dry_run=dry_run, setup_script=setup_script, tags=tags, output_dir=output_dir)
            else:
                submit_single(
                    config_path=yaml_file, dry_run=dry_run, setup_script=setup_script, tags=tags, output_dir=output_dir
                )
            success_count += 1
        except Exception as e:
            console.print(f"[bold red]  ❌ Error:[/] {e}")
            logging.debug("Full traceback:", exc_info=True)
            error_count += 1

        console.print()

    # Summary
    if dry_run:
        console.print(f"[bold green]✅ Validated {success_count} config(s)[/]", end="")
    else:
        console.print(f"[bold green]✅ Submitted {success_count} job(s)[/]", end="")

    if error_count > 0:
        console.print(f" [bold red]({error_count} failed)[/]")
    else:
        console.print()


def parse_config_arg(arg: str) -> tuple[Path, str | None]:
    """Parse -f argument, supporting path:selector format.

    Args:
        arg: CLI argument value, e.g.:
             "config.yaml"
             "config.yaml:base"
             "config.yaml:override_tp64"
             "config.yaml:override_mtp*"
             "config.yaml:zip_override_tp_sweep"
             "config.yaml:zip_override_tp_sweep[0]"

    Returns:
        (config_path, selector) — selector is None when submitting all variants
    """
    if ":" in arg:
        path_str, selector = arg.rsplit(":", 1)
        if not path_str.strip():
            raise ValueError("Invalid config path in selector syntax.")
        valid = bool(
            selector == "base"
            or re.fullmatch(r"override_\S+", selector)
            or re.fullmatch(r"zip_override_[\w-]+", selector)
            or re.fullmatch(r"zip_override_[\w-]+\[\d+\]", selector)
            or ("*" in selector or "?" in selector)
        )
        if not valid:
            raise ValueError(
                f"Invalid selector '{selector}'. "
                "Must be 'base', 'override_<name>', 'zip_override_<name>', "
                "'zip_override_<name>[N]', or a glob pattern like '*mtp*'."
            )
        return Path(path_str), selector
    return Path(arg), None


def is_override_config(config_path: Path) -> bool:
    """Check if a YAML file uses override format (has a 'base' top-level key)."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception:
        logger.debug(f"Failed to parse YAML while checking override format: {config_path}", exc_info=True)
        return False
    if not isinstance(config, dict):
        return False
    return "base" in config


@contextlib.contextmanager
def materialize_config_path(config_path: Path):
    """Stage stdin-backed configs to a temporary YAML file for repeated reads."""
    if str(config_path) not in {"-", "/dev/stdin"}:
        yield config_path
        return

    payload = sys.stdin.read()
    if not payload.strip():
        raise ValueError("No YAML received on stdin")

    fd, temp_path = tempfile.mkstemp(
        suffix=".yaml",
        prefix="srtctl_stdin_",
        text=True,
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        yield Path(temp_path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(temp_path)


def submit_override(
    config_path: Path,
    selector: str | None = None,
    dry_run: bool = False,
    setup_script: str | None = None,
    tags: list[str] | None = None,
    output_dir: Path | None = None,
) -> None:
    """Expand an override config file and submit each variant.

    Loads the raw YAML, expands base + override_* via generate_override_configs(),
    then routes each variant through submit_sweep or submit_single.

    Args:
        config_path: Path to override YAML file
        selector: Optional selector ("base", "override_xxx", or None for all)
        dry_run: If True, print config but don't submit
        setup_script: Optional custom setup script name
        tags: Optional list of tags
        output_dir: Custom output directory
    """
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    override_configs = generate_override_configs(raw_config, selector=selector)

    if dry_run:
        base_name = raw_config["base"].get("name", "unnamed")
        selector_info = f", selector: {selector}" if selector else ""
        console.print()
        console.print(
            Panel(
                f"[bold]Override Config:[/] {base_name} ({len(override_configs)} variant{'s' if len(override_configs) != 1 else ''}{selector_info})",
                border_style="cyan",
            )
        )
        console.print()

    from srtctl.core.config import resolve_override_yaml
    from srtctl.core.yaml_utils import dump_yaml_with_comments

    resolved_variants = resolve_override_yaml(config_path, selector=selector)

    for i, (suffix, config_cm) in enumerate(resolved_variants, 1):
        variant_label = "base" if suffix == "base" else f"override_{suffix}"
        job_name = config_cm.get("name", "unnamed")
        runtime_config_text = dump_yaml_with_comments(config_cm)
        if runtime_config_text is None:
            raise RuntimeError("dump_yaml_with_comments returned None unexpectedly")

        if dry_run:
            console.print(f"[bold cyan][{i}/{len(override_configs)}][/] {variant_label}: {job_name}")

        logger.info(f"Override variant: {variant_label} -> {job_name}")

        resolved_config = resolve_config_with_defaults(yaml.safe_load(runtime_config_text), load_cluster_config())
        config = SrtConfig.Schema().load(resolved_config)

        if "sweep" in config_cm:
            fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", prefix="srtctl_override_", text=True)
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(runtime_config_text)
                submit_sweep(
                    config_path=Path(temp_config_path),
                    dry_run=dry_run,
                    setup_script=setup_script,
                    tags=tags,
                    output_dir=output_dir,
                )
            finally:
                with contextlib.suppress(OSError):
                    os.remove(temp_config_path)
        else:
            submit_single(
                config_path=config_path,
                config=config,
                dry_run=dry_run,
                setup_script=setup_script,
                tags=tags,
                output_dir=output_dir,
                variant_suffix=suffix,
                source_config_path=config_path,
                runtime_config_text=runtime_config_text,
            )


def resolve_override_cmd(
    config_path: Path,
    selector: str | None = None,
    stdout: bool = False,
) -> None:
    """Resolve an override config and write the specialised YAML file(s).

    Unlike ``submit_override``, this command only generates the resolved
    YAML — it does not submit any jobs. Field order follows the base config,
    with any override-only keys appended at the end. Comments from the
    source file are preserved.

    Args:
        config_path: Path to the override YAML file.
        selector: Optional variant selector (same syntax as apply -f file:selector).
        stdout: When True, print the resolved YAML to stdout instead of writing files.
    """
    from srtctl.core.config import resolve_override_yaml
    from srtctl.core.yaml_utils import dump_yaml_with_comments

    variants = resolve_override_yaml(config_path, selector=selector)

    if stdout:
        for i, (suffix, cm) in enumerate(variants):
            if len(variants) > 1:
                if i > 0:
                    print()
                print(f"# --- {suffix} ---")
            text = dump_yaml_with_comments(cm)
            print(text, end="")
        return

    written: list[Path] = []
    for suffix, cm in variants:
        out_path = config_path.parent / f"{config_path.stem}_{suffix}.yaml"
        with open(out_path, "w") as f:
            dump_yaml_with_comments(cm, f)
        written.append(out_path)

    for p in written:
        console.print(f"[green]Wrote:[/] {p}")


def main():
    # If no args at all, launch interactive mode
    if len(sys.argv) == 1:
        from srtctl.cli.interactive import run_interactive

        sys.exit(run_interactive())

    setup_logging()

    parser = argparse.ArgumentParser(
        description="srtctl - SLURM job submission",
        epilog="""Examples:
  srtctl                                         # Interactive mode
  srtctl apply -f config.yaml                    # Submit job
  srtctl apply -f ./configs/                     # Submit all YAMLs in directory
  srtctl apply -f config.yaml --sweep            # Submit sweep
  srtctl preflight -f config.yaml                # Check model/container availability
  srtctl dry-run -f config.yaml                  # Dry run
  srtctl resolve-override -f config.yaml         # Resolve override YAML (no submit)
  srtctl resolve-override -f config.yaml --stdout  # Print to stdout
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(p):
        p.add_argument(
            "-f",
            "--file",
            type=str,
            required=True,
            dest="config",
            help="YAML config file, directory, or file:selector for overrides",
        )
        p.add_argument("-o", "--output", type=Path, dest="output_dir", help="Custom output directory for job logs")
        p.add_argument("--sweep", action="store_true", help="Force sweep mode")
        p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")

    apply_parser = subparsers.add_parser("apply", help="Submit job(s) to SLURM")
    add_common_args(apply_parser)
    apply_parser.add_argument("--setup-script", type=str, help="Custom setup script in configs/")
    apply_parser.add_argument("--tags", type=str, help="Comma-separated tags")
    apply_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit one JSON line per submission on stdout; prose output goes to stderr.",
    )
    apply_parser.add_argument(
        "--mock",
        action="store_true",
        dest="mock_mode",
        help=(
            "Stub sbatch and spawn a detached mock worker that runs the full "
            "SweepOrchestrator locally. For testing external harnesses without "
            "cluster access."
        ),
    )
    apply_parser.add_argument(
        "--mock-tick-s",
        type=float,
        default=0.2,
        dest="mock_tick_s",
        help="Per-phase wall time used by the detached mock worker.",
    )

    dry_run_parser = subparsers.add_parser("dry-run", help="Validate without submitting")
    add_common_args(dry_run_parser)

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Check model and container availability without submitting",
    )
    preflight_parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        dest="config",
        help="YAML config file, or file:selector for overrides",
    )

    resolve_parser = subparsers.add_parser(
        "resolve-override",
        help="Resolve override YAML into specialised files without submitting",
    )
    resolve_parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        dest="config",
        help="Override YAML file, or file:selector",
    )
    resolve_parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print resolved YAML to stdout instead of writing files",
    )

    aliases_parser = subparsers.add_parser(
        "cluster-aliases",
        help="Print model/container aliases from srtslurm.yaml",
    )
    aliases_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output aliases as JSON",
    )

    # Fingerprint comparison: srtctl diff <path_a> <path_b>
    diff_parser = subparsers.add_parser("diff", help="Compare fingerprints from two runs")
    diff_parser.add_argument("path_a", type=Path, help="First output dir or lockfile")
    diff_parser.add_argument("path_b", type=Path, help="Second output dir or lockfile")
    diff_parser.add_argument("--verbose", action="store_true", help="Show all package changes")

    # Environment check: srtctl check <path>
    check_parser = subparsers.add_parser("check", help="Check environment against a fingerprint")
    check_parser.add_argument("path", type=Path, help="Lockfile or output dir to check against")
    check_parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    args = parser.parse_args()

    json_mode = bool(getattr(args, "json_output", False))
    mock_mode = bool(getattr(args, "mock_mode", False))
    # Always rebind the module console on each invocation so json-mode prose
    # goes to stderr and non-json prose returns to stdout. Save the original
    # so we can restore it on exit — direct library callers of submit_single /
    # submit_override (tests, etc.) must not see a leaked stderr binding.
    global console
    _original_console = console
    console = Console(file=sys.stderr) if json_mode else Console()

    def restore_console() -> None:
        global console
        console = _original_console

    if json_mode:
        _submissions.clear()

    _mock_patch_teardowns: list = []
    if mock_mode:
        _mock_patch_teardowns = _install_mock_submit_patches()

    # Handle diff and check commands first (they don't use -f/config)
    if args.command == "diff":
        fps_a = load_lockfile_fingerprints(args.path_a)
        fps_b = load_lockfile_fingerprints(args.path_b)
        if fps_a is None or fps_b is None:
            missing = []
            if fps_a is None:
                missing.append(str(args.path_a))
            if fps_b is None:
                missing.append(str(args.path_b))
            console.print(f"[bold red]Could not load fingerprints from:[/] {', '.join(missing)}")
            sys.exit(1)

        # Diff each worker against its counterpart
        all_workers = sorted(set(fps_a.keys()) | set(fps_b.keys()))
        for worker in all_workers:
            if worker not in fps_a:
                console.print(f"\n[bold]{worker}:[/] only in {args.path_b}")
                continue
            if worker not in fps_b:
                console.print(f"\n[bold]{worker}:[/] only in {args.path_a}")
                continue
            diff = diff_fingerprints(fps_a[worker], fps_b[worker])
            console.print(f"\n[bold]{worker}:[/]")
            console.print(format_diff(diff, verbose=args.verbose))
        restore_console()
        return

    if args.command == "check":
        import json as json_mod

        fps = load_lockfile_fingerprints(args.path)
        if fps is None:
            console.print(f"[bold red]Could not load fingerprints from:[/] {args.path}")
            sys.exit(1)

        # Capture current environment once, reuse for all worker checks
        current_fp = capture_fingerprint()
        all_results = []
        for worker in sorted(fps.keys()):
            results = check_against_fingerprint(fps[worker], current_fp)
            if results:
                all_results.extend(results)
                console.print(f"\n[bold]{worker}:[/]")
                if args.json_output:
                    console.print(
                        json_mod.dumps(
                            [{"field": r.field, "status": r.status.value, "message": r.message} for r in results],
                            indent=2,
                        )
                    )
                else:
                    console.print(format_check_results(results))
        if not all_results:
            console.print(format_check_results([]))
        restore_console()
        sys.exit(1 if all_results else 0)

    if args.command == "cluster-aliases":
        payload = get_cluster_aliases()
        if args.json_output:
            sys.stdout.write(json.dumps(payload) + "\n")
            sys.stdout.flush()
            restore_console()
            return

        config_path = find_cluster_config_path()
        console.print(f"[bold]Cluster config:[/] {config_path if config_path else '<not found>'}")
        table = Table(title="Cluster Aliases")
        table.add_column("Type", style="cyan")
        table.add_column("Alias", style="green")
        table.add_column("Resolved Value", style="yellow")
        for alias, value in sorted(payload["model_paths"].items()):
            table.add_row("model_paths", alias, value)
        for alias, value in sorted(payload["containers"].items()):
            table.add_row("containers", alias, value)
        if not payload["model_paths"] and not payload["containers"]:
            table.add_row("-", "-", "No aliases configured")
        console.print()
        console.print(table)
        restore_console()
        return

    # Parse config arg: supports path:selector format for overrides
    config_path, selector = parse_config_arg(args.config)

    is_dry_run = args.command == "dry-run"
    tags = [t.strip() for t in (getattr(args, "tags", "") or "").split(",") if t.strip()] or None

    try:
        with materialize_config_path(config_path) as effective_config_path:
            if not effective_config_path.exists():
                console.print(f"[bold red]Config not found:[/] {config_path}")
                sys.exit(1)

            # resolve-override has its own simple dispatch path
            if args.command == "resolve-override":
                if not is_override_config(effective_config_path):
                    console.print(f"[bold red]Error:[/] {config_path} is not an override config (missing 'base' key)")
                    sys.exit(1)
                resolve_override_cmd(
                    effective_config_path,
                    selector=selector,
                    stdout=getattr(args, "stdout", False),
                )
                restore_console()
                return

            if args.command == "preflight":
                if effective_config_path.is_dir():
                    raise ValueError("preflight currently expects a file, not a directory")
                with open(effective_config_path) as f:
                    raw_config = yaml.safe_load(f)
                results = preflight_config_variants(
                    raw_config,
                    cluster_config=load_cluster_config(),
                    selector=selector,
                )
                for result in results:
                    icon = "[green]✓[/]" if result.ok else "[red]✗[/]"
                    console.print(f"{icon} {result.variant}")
                    console.print(f"  model.path: {result.model.message}")
                    console.print(f"  model.container: {result.container.message}")
                if any(not result.ok for result in results):
                    raise ValueError(
                        _format_preflight_error(
                            str(config_path),
                            [result for result in results if not result.ok],
                        )
                    )
                restore_console()
                return

            setup_script = getattr(args, "setup_script", None)
            output_dir = getattr(args, "output_dir", None)

            # Handle directory input
            if effective_config_path.is_dir():
                if selector:
                    logger.warning(f"Selector ':{selector}' ignored for directory input")
                submit_directory(
                    effective_config_path,
                    dry_run=is_dry_run,
                    setup_script=setup_script,
                    tags=tags,
                    force_sweep=args.sweep,
                    output_dir=output_dir,
                )
            elif is_override_config(effective_config_path):
                submit_override(
                    effective_config_path,
                    selector=selector,
                    dry_run=is_dry_run,
                    setup_script=setup_script,
                    tags=tags,
                    output_dir=output_dir,
                )
            else:
                if selector:
                    logger.warning(f"Selector ':{selector}' ignored — config is not an override file")
                is_sweep = args.sweep or is_sweep_config(effective_config_path)
                if is_sweep:
                    submit_sweep(
                        effective_config_path,
                        dry_run=is_dry_run,
                        setup_script=setup_script,
                        tags=tags,
                        output_dir=output_dir,
                    )
                else:
                    submit_single(
                        config_path=effective_config_path,
                        dry_run=is_dry_run,
                        setup_script=setup_script,
                        tags=tags,
                        output_dir=output_dir,
                        enforce_preflight=not (mock_mode or is_dry_run),
                    )
    except Exception as e:
        # Restore subprocess.run etc. before we exit so in-process test
        # invocations don't leak patches across runs.
        for patcher in _mock_patch_teardowns:
            with contextlib.suppress(Exception):
                patcher.stop()
        _mock_patch_teardowns = []
        restore_console()
        if json_mode:
            sys.stdout.write(json.dumps({"status": "error", "error": str(e)}) + "\n")
            sys.stdout.flush()
            logging.debug("Full traceback:", exc_info=True)
            sys.exit(1)
        console.print(f"[bold red]Error:[/] {e}")
        logging.debug("Full traceback:", exc_info=True)
        sys.exit(1)

    # Mock-mode post-submit: spawn the detached orchestrator worker so the
    # real SweepOrchestrator runs against the output_dir that submit just
    # wrote to. Tear down sbatch patches AFTER the spawn so the spawned
    # subprocess inherits a clean environment (Popen itself isn't patched).
    if mock_mode:
        for submission in _submissions:
            _spawn_mock_worker(submission, tick_s=float(args.mock_tick_s))
        for patcher in _mock_patch_teardowns:
            with contextlib.suppress(Exception):
                patcher.stop()
        _mock_patch_teardowns = []

    if json_mode:
        if not _submissions:
            sys.stdout.write(json.dumps({"status": "no-submissions"}) + "\n")
        else:
            for record in _submissions:
                sys.stdout.write(json.dumps(record) + "\n")
        sys.stdout.flush()

    # Restore the pre-main console binding so direct library callers aren't
    # affected by this invocation's json-mode rebinding.
    restore_console()


if __name__ == "__main__":
    main()
