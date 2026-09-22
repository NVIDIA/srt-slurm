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
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from srtctl.backends import VLLMMooncakeKVStoreConfig, VLLMProtocol
from srtctl.core.config import (
    expand_engine_config_defaults,
    generate_override_configs,
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
from srtctl.core.git_state import (
    GIT_STATE_FILENAME,
    git_snapshot_sources_from_extra_mounts,
    write_git_state_snapshot,
)
from srtctl.core.lockfile import load_lockfile_fingerprints
from srtctl.core.runtime import Nodes
from srtctl.core.schema import SrtConfig, installs_dynamo
from srtctl.core.status import create_job_record
from srtctl.core.validation import preflight_config_variants
from srtctl.frontends.dynamo import ROUTER_POLICY_CONFIG_CONTAINER_PATH
from srtctl.ports import FRONTEND_PUBLIC_PORT, MOONCAKE_MASTER_PORT
from srtctl.runtime_scripts.dynamo_wheels import arch_from_binary, detect_target_arch
from srtctl.status_server.server import add_arguments as add_status_server_arguments
from srtctl.status_server.server import serve as serve_status_server

console = Console()
logger = logging.getLogger(__name__)

# Populated by submit_with_orchestrator on successful submission. Consumed by
# main() when --json is set so callers get one JSON line per submitted job.
_submissions: list[dict] = []

# Rendered --set/--unset overrides for the current invocation; echoed into every
# submission record so a caller can see exactly what was applied.
_active_overrides: list[str] = []
# "dynamo.source: refs/pull/14000/head -> <sha>" notes from pinning source revs at
# submit time; echoed the same way so a caller can see exactly what will build.
_pinned_sources: list[str] = []


def _record_submission(data: dict) -> None:
    if _active_overrides:
        data["applied_overrides"] = list(_active_overrides)
    if _pinned_sources:
        data["pinned_sources"] = list(_pinned_sources)
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


def _host_setup_source(config: SrtConfig) -> str:
    """Where the effective host_setup came from: the recipe or srtslurm.yaml.

    resolve_config_with_defaults only injects default_host_setup when the recipe
    omits the block entirely, so an exact match against the cluster default
    identifies it. Worth showing: an unexpected `sudo` in the dry-run output is
    much easier to chase when you know which file to open.
    """
    default = get_srtslurm_setting("default_host_setup")
    if isinstance(default, dict):
        commands = list(default.get("commands") or [])
        teardown = list(default.get("teardown") or [])
        if commands == config.host_setup.commands and teardown == config.host_setup.teardown:
            return "srtslurm.yaml (default_host_setup)"
    return "recipe"


def _engine_bool(value: object) -> str:
    """Render an engine-yaml boolean the way the YAML file will spell it."""
    return "unset" if value is None else str(value).lower()


def _metrics_suffix(service) -> str:
    """`` metrics=[name]:<port><path>[@first]`` per endpoint tachometer will scrape (the recipe's or the kind's)."""
    from srtctl.services import get_service_kind

    endpoints = get_service_kind(service.type).metrics(service)
    if not endpoints:
        return ""
    parts = [
        f"{endpoint.name or ''}:{endpoint.port}{endpoint.path}" + ("@first" if endpoint.nodes == "first" else "")
        for endpoint in endpoints
    ]
    return " metrics=" + ",".join(parts)


def show_config_details(config: SrtConfig) -> None:
    """Display container mounts and environment variables for dry-run verification.

    Shows all mounts (from built-in defaults, srtslurm.yaml, and recipe) and all
    environment variables (global and backend per-mode) so users can verify their
    config is correct before submitting.
    """
    if config.frontend.type == "dynamo" and not config.dynamo.sidecar:
        from srtctl.backends.trtllm import TRTLLMProtocol

        if isinstance(config.backend, TRTLLMProtocol):
            descriptions = {
                "--publish-metrics": "metrics only",
                "--publish-events-and-metrics": "metrics and KV events",
            }
            publication = [f"{flag} ({descriptions[flag]})" for flag in config.backend.dynamo_metrics_flags]
            disabled_by = (
                "publish_events_and_metrics"
                if config.backend.publish_events_and_metrics is False
                else "publish_metrics"
            )
            console.print(
                Panel(
                    "\n".join(publication) or f"No publication flag (backend.{disabled_by}: false)",
                    title="Dynamo TRT-LLM Metrics",
                    border_style="cyan",
                )
            )

    from srtctl.backends.trtllm import TRTLLMProtocol

    if isinstance(config.backend, TRTLLMProtocol):
        # Engine-yaml statistics keys srtctl defaults at config load
        # (expand_trtllm_engine_defaults, expand_trtllm_serve_defaults,
        # expand_observability). Shown for both frontends so a run that expects
        # the iteration-level trtllm_* gauges can see before submitting that
        # enable_iter_perf_stats is off.
        modes = ("prefill", "decode") if config.resources.is_disaggregated else ("agg",)
        rows = []
        for mode in modes:
            section = config.backend.get_config_for_mode(mode)
            rows.append(
                f"{mode}: enable_iter_perf_stats={_engine_bool(section.get('enable_iter_perf_stats'))}, "
                f"return_perf_metrics={_engine_bool(section.get('return_perf_metrics'))}"
            )
        rows.append(
            "(engine yaml; the iteration-level trtllm_* gauges and the dashboard's KV-utilisation panels "
            "need enable_iter_perf_stats: true)"
        )
        console.print(Panel("\n".join(rows), title="TRT-LLM Engine Statistics", border_style="cyan"))

    if config.frontend.type == "vllm":
        from srtctl.backends.vllm import find_vllm_orchestration_recipe_flags

        if isinstance(config.backend, VLLMProtocol):
            orchestration_flags = find_vllm_orchestration_recipe_flags(config.backend)
            if orchestration_flags:
                for mode_name, flag_name in orchestration_flags:
                    console.print(
                        "[yellow]WARNING:[/] "
                        f"vllm_config.{mode_name}.{flag_name} is set in the recipe but srtslurm "
                        "derives this from the job topology at runtime; remove it from the recipe "
                        "to avoid confusion (the configured value is ignored)."
                    )

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
                expanded_host = os.path.expanduser(os.path.expandvars(parts[0]))
                mounts_table.add_row("recipe", expanded_host, parts[1])
            else:
                expanded_host = os.path.expanduser(os.path.expandvars(mount_spec))
                mounts_table.add_row("recipe", expanded_host, mount_spec)

    # Recipe container_mounts (FormattablePath mounts)
    if config.container_mounts:
        for host_template, container_template in config.container_mounts.items():
            mounts_table.add_row("recipe", str(host_template), str(container_template))

    # InferenceX workspace, mounted by RuntimeContext.from_config when
    # INFMAX_WORKSPACE is set in the submitting environment (core/runtime.py).
    #
    # It comes from the environment rather than the recipe, which is exactly why it has
    # to be shown: nothing in the config file mentions it, so a reader comparing recipe
    # to dry-run sees a complete picture and is wrong. Recipes whose benchmark command
    # lives under /infmax-workspace (the agentic suites) fail with exit 127 twelve
    # minutes in when it is missing, and the dry-run gave no hint either way -- the
    # table listed every other mount, which made its absence read as "no such mount
    # exists" rather than "not set".
    infmax_ws = os.environ.get("INFMAX_WORKSPACE")
    if infmax_ws:
        mounts_table.add_row("INFMAX_WORKSPACE", infmax_ws, "/infmax-workspace")
    elif "/infmax-workspace" in str(config.benchmark.command or ""):
        mounts_table.add_row(
            "[red]MISSING[/]",
            "[red]INFMAX_WORKSPACE is not set in this environment[/]",
            "[red]/infmax-workspace[/]",
        )

    console.print(Panel(mounts_table, border_style="green"))

    if not infmax_ws and "/infmax-workspace" in str(config.benchmark.command or ""):
        console.print(
            "[red bold]ERROR:[/] this recipe runs its benchmark from /infmax-workspace, "
            "but INFMAX_WORKSPACE is not set, so that mount will be absent and the "
            "benchmark command will fail with exit 127 after the workers have loaded. "
            "Export INFMAX_WORKSPACE=<path to the InferenceX checkout> before submitting."
        )

    # --- SLURM heterogeneous job structure ---
    het_components = config.resources.het_components(
        infra_dedicated=config.infra.etcd_nats_dedicated_node,
        cluster_default=get_srtslurm_setting("use_het_jobs", False),
    )
    if het_components is not None:
        het_table = Table(title="SLURM Heterogeneous Job", show_lines=False, pad_edge=False)
        het_table.add_column("Group", style="dim", width=5)
        het_table.add_column("Side", style="cyan", width=8)
        het_table.add_column("Nodes", style="white", justify="right", width=6)
        het_table.add_column("Segment", style="white", justify="right", width=8)
        het_table.add_column("GPUs/node", style="white", justify="right", width=10)
        het_table.add_column("Infra", style="dim")
        for c in het_components:
            infra_note = "first node" if c.name == "prefill" and config.infra.etcd_nats_dedicated_node else ""
            het_table.add_row(
                str(c.group),
                c.name,
                str(c.nodes),
                str(c.segment),
                str(c.gpus_per_node),
                infra_note,
            )
        console.print(Panel(het_table, border_style="magenta"))

    # --- Environment Variables ---
    dynamo_environment = config.dynamo.get_wheel_environment()
    has_env = bool(config.environment or dynamo_environment)
    backend = config.backend
    mode_envs: list[tuple[str, dict[str, str]]] = []
    for mode_name, env in [
        ("prefill", backend.prefill_environment),
        ("decode", backend.decode_environment),
        ("aggregated", backend.aggregated_environment),
    ]:
        if env:
            has_env = True
            mode_envs.append((mode_name, dict(env)))
    if config.benchmark.env:
        has_env = True
        mode_envs.append(("benchmark", dict(config.benchmark.env)))

    mooncake_cfg = backend.mooncake_kv_store
    if mooncake_cfg is not None and mooncake_cfg.env:
        has_env = True
        mode_envs.append(("mooncake", dict(mooncake_cfg.env)))

    if has_env:
        env_table = Table(title="Environment Variables", show_lines=False, pad_edge=False)
        env_table.add_column("Scope", style="dim", width=14)
        env_table.add_column("Variable", style="yellow")
        env_table.add_column("Value", style="white")

        for var, val in sorted(dynamo_environment.items()):
            env_table.add_row("dynamo", var, val)

        for var, val in sorted(config.environment.items()):
            env_table.add_row("global", var, val)

        for mode_name, env in mode_envs:
            for var, val in sorted(env.items()):
                env_table.add_row(mode_name, var, val)

        console.print(Panel(env_table, border_style="yellow"))
    else:
        console.print("[dim]No custom environment variables configured.[/]")

    # --- Shadow engine recovery (engine.failover, vLLM + Dynamo GPU Memory Service) ---
    failover = config.backend.failover
    if failover is not None:
        from srtctl.backends.vllm import FAILOVER_LOCK_FILENAME, failover_root

        root = failover_root(failover.shared_dir, "<job_id>")
        restart_roles = [
            role
            for role in ("prefill", "decode", "agg")
            if getattr(getattr(config.resources, f"{role}_restart", None), "enabled", False)
        ]
        relaunch = (
            f"roles.{{{','.join(restart_roles)}}}.restart relaunches an exited engine in place as the new shadow"
            if restart_roles
            else "none: an exited engine's step ends and roles.<role>.critical decides (add roles.<role>.restart)"
        )
        lines = [
            f"engines per worker: {failover.engines_per_worker} (engine 0 + {failover.shadow_engines} shadow)",
            (
                f"shared dir: {root}/<role>_<index>/  (gms_*.sock, {FAILOVER_LOCK_FILENAME}; node-local, every "
                "container on the node)"
            ),
            (
                "per worker and node: the gms service (one instance per worker, listed under Services) then steps "
                "<role>_<index>_<node> and <role>_<index>_<node>_e<k>"
            ),
            "engine flags: --load-format gms --gms-shadow-mode (no --device-ids; CUDA_VISIBLE_DEVICES is pinned)",
            (
                "engine env: ENGINE_ID, GMS_SOCKET_DIR, FAILOVER_LOCK_PATH, DYN_VLLM_GMS_SHADOW_MODE=true, "
                "DYN_SYSTEM_STARTING_HEALTH_STATUS=notready"
            ),
            f"engine relaunch: {relaunch}",
        ]
        console.print(Panel("\n".join(lines), title="Shadow Engine Recovery (engine.failover)", border_style="magenta"))
        console.print(
            "[yellow]NOTE:[/] two engines share each GPU: size gpu-memory-utilization so the active engine's KV "
            "cache leaves room for the shadow's CUDA context, graphs and communicator buffers. To test a failover "
            "kill the engine process, not its step (see docs/shadow-engine-recovery.md)."
        )

    # --- Host setup (runs on the bare node, outside the container) ---
    if config.host_setup.enabled:
        host_table = Table(title="Host Setup (outside container)", show_lines=False, pad_edge=False)
        host_table.add_column("Phase", style="dim", width=10)
        host_table.add_column("Command", style="white")
        for command in config.host_setup.commands:
            host_table.add_row("setup", command)
        for command in config.host_setup.teardown:
            host_table.add_row("teardown", command)
        console.print(Panel(host_table, border_style="red"))

        setup = config.host_setup
        source = _host_setup_source(config)
        console.print(
            f"[dim]host_setup:[/] nodes={setup.nodes} "
            f"ignore_failure={str(setup.ignore_failure).lower()} "
            f"timeout_seconds={setup.timeout_seconds} [dim]source: {source}[/]"
        )
        if any("sudo" in command for command in [*setup.commands, *setup.teardown]):
            console.print(
                "[yellow]NOTE:[/] host_setup runs as you, not root. Confirm passwordless sudo on a "
                "compute node first (`srun --jobid <job> --overlap -w <node> sudo -n true`); "
                "a sudo that prompts will hang until timeout_seconds and fail the job."
            )
        if setup.commands and not setup.teardown:
            console.print(
                "[yellow]NOTE:[/] no host_setup.teardown configured — any node state set here "
                "outlives this allocation and is inherited by the next job on these nodes."
            )

    # --- post_eval (RUN_EVAL / EVAL_ONLY dispatch) ---
    if config.post_eval.passthrough_env or config.post_eval.command:
        console.print("[bold cyan]Post-eval dispatch:[/]")
        if config.post_eval.command:
            console.print(f"    [yellow]command:[/] {shlex.join(config.post_eval.command)}", crop=False)
        if config.post_eval.passthrough_env:
            console.print(f"    [yellow]passthrough_env:[/] {', '.join(config.post_eval.passthrough_env)}", crop=False)

    # --- services (see docs/services.md) ---
    # Plain lines, not a Table: repo URLs and long argv overflow a narrow console and a
    # Table would wrap or truncate them. crop=False keeps each value intact on one line.
    from srtctl.services.implicit import effective_services
    from srtctl.services.registry import get_service_kind

    effective = effective_services(config)
    if effective:
        console.print("[bold cyan]Services:[/]")
        for entry in effective:
            service = entry.service
            console.print(
                f"  [cyan]{service.name}[/] [dim]type={service.type} placement={service.effective_placement}"
                f"{' per=worker' if service.effective_per == 'worker' else ''} "
                f"start={service.effective_start} critical={str(service.effective_critical).lower()}"
                f"{f' nodes={service.nodes}' if service.nodes is not None else ''}"
                f"{' terminal' if service.terminal else ''}"
                f"{_metrics_suffix(service)}[/]"
            )
            if entry.implicit:
                console.print(
                    f"    [yellow]implied by:[/] {entry.reason} (declare a service named {service.name} to change it)"
                )
            if service.external:
                console.print(f"    [yellow]external:[/] {service.external} (not launched)", crop=False)
                continue
            console.print(f"    [yellow]command:[/] {shlex.join(service.preview_command())}", crop=False)
            container = service.container or get_service_kind(service.type).container_fallback(config)
            console.print(f"    [yellow]container:[/] {container or '<job container>'}")
            if service.source is not None:
                console.print(f"    [yellow]source:[/] {service.source.git} @ {service.source.rev}", crop=False)
                if service.source.path:
                    console.print(f"    [yellow]source.path:[/] {service.source.path}")
            if service.build_command:
                console.print(f"    [yellow]build_command:[/] {shlex.join(service.build_command)}", crop=False)
            if service.readiness is not None:
                console.print(f"    [yellow]readiness:[/] {service.readiness.describe()}")
            elif get_service_kind(service.type).default_readiness_ports:
                ports = ", ".join(f"tcp/{p}" for p in get_service_kind(service.type).default_readiness_ports)
                console.print(f"    [yellow]readiness:[/] {ports} (kind default)")
            if service.options:
                console.print(f"    [yellow]options:[/] {service.options}", crop=False)
            if service.preamble:
                console.print(f"    [yellow]preamble:[/] {service.preamble.strip()}", crop=False)
            if service.type not in ("etcd", "nats"):  # the discovery plane never gets its own address
                console.print(f"    [yellow]inherit_discovery_env:[/] {str(service.inherit_discovery_env).lower()}")
            for var, val in sorted(service.env.items()):
                console.print(f"    [yellow]env.{var}:[/] {val}", crop=False)

    # --- srun options ---
    if config.srun_options:
        opts = " ".join(f"--{k}={v}" if v else f"--{k}" for k, v in config.srun_options.items())
        console.print(f"[dim]srun options:[/] {opts}")

    # Dynamo install runs apt-get/pip as root inside the container, so srtctl injects
    # ENROOT_REMAP_ROOT=yes (via srun --export) on the worker + dynamo-frontend launches.
    if installs_dynamo(config):
        console.print(
            "[dim]srun --export (dynamo install):[/] ALL,ENROOT_REMAP_ROOT=yes [dim](workers + dynamo frontend)[/]"
        )
        source = config.dynamo.source
        if source is not None and source.git:
            console.print(f"[dim]dynamo source:[/] {source.git} @ {source.rev}", crop=False)
            if source.sha:
                console.print(f"[dim]dynamo source sha:[/] {source.sha}", crop=False)
            else:
                console.print("[dim]dynamo source sha:[/] resolved from rev at submit (srtctl apply)")
        elif source is not None and source.pypi:
            console.print(f"[dim]dynamo source:[/] PyPI ai-dynamo=={source.pypi}")
        elif source is not None and source.wheel:
            console.print(f"[dim]dynamo source:[/] staged wheel ai-dynamo=={source.wheel}")

    # --- nodes: who owns what (engine roles, service pools) ---
    if config.pool_services:
        console.print("[bold cyan]Nodes:[/]")
        console.print(f"  engine roles: {config.engine_node_count}")
        for svc in config.pool_services:
            console.print(f"  pool {svc.name} ({svc.type}): {svc.nodes}")
        console.print(f"  total: {config.total_nodes}")

    show_extensions = (
        config.benchmark.type == "custom"
        or config.benchmark.container_image
        or config.observability.enabled
        or config.observability.tachometer.enabled
        or config.telemetry.enabled
        or mooncake_cfg is not None
        or config.profiling.enabled
        or config.frontend.worker_selection is not None
    )
    if show_extensions:
        details = Table(title="Execution Extensions", show_lines=False, pad_edge=False)
        details.add_column("Area", style="dim", width=14)
        details.add_column("Setting", style="yellow")
        details.add_column("Value", style="white")

        if config.benchmark.type == "custom":
            details.add_row("benchmark", "type", config.benchmark.type)
            if config.benchmark.command:
                details.add_row("benchmark", "command", config.benchmark.command)

        # Surface a non-default benchmark container regardless of type — accuracy
        # benchmarks like AIME (run via type: custom + the NeMo Skills container)
        # need this visible at submit time so operators can verify the alias
        # resolved to the expected sqsh / URI.
        if config.benchmark.container_image:
            details.add_row("benchmark", "container_image", config.benchmark.container_image)

        profiling = config.profiling
        # Other extensions can enable this section without enabling profiling.
        if profiling.enabled:
            details.add_row("profiling", "type", profiling.type)
            if profiling.is_nsys:
                details.add_row("profiling", "nsys_trace", profiling.nsys_trace)
                details.add_row("profiling", "capture_range_end", profiling.capture_range_end)
                fork_setting = (
                    "dynamo default"
                    if profiling.trace_fork_before_exec is None
                    else str(profiling.trace_fork_before_exec).lower()
                )
                details.add_row("profiling", "trace_fork_before_exec", fork_setting)
                if profiling.nsys_library_paths:
                    details.add_row(
                        "profiling",
                        "nsys_library_paths",
                        ":".join(profiling.nsys_library_paths),
                    )
                for mode, phase in (
                    ("prefill", profiling.prefill),
                    ("decode", profiling.decode),
                    ("aggregated", profiling.aggregated),
                ):
                    if phase is not None and not profiling.is_nsys_time:
                        target = (
                            "all physical processes"
                            if phase.capture_scope == "all"
                            else f"worker {phase.worker_index}, rank {phase.worker_rank}"
                        )
                        details.add_row(
                            "profiling",
                            f"{mode} target",
                            target,
                        )

        if config.observability.enabled:
            settings = config.observability.nsys
            state = (
                "enabled"
                if config.observability_nsys_enabled
                else ("superseded by profiling" if profiling.enabled else "disabled")
            )
            details.add_row("observability", "nsys", state)
            if config.observability_nsys_enabled:
                targets = "all worker processes/ranks"
                if config.frontend.type == "dynamo":
                    targets += " + Dynamo frontends"
                details.add_row("observability", "nsys targets", targets)
                details.add_row("observability", "nsys binary", profiling.nsys_binary)
                details.add_row("observability", "nsys trace", "NVTX (no CUDA tracing)")
                window = (
                    "after warmup until workload completes (client start/stop hooks)"
                    if settings.capture_window == "measured_workload"
                    else "process launch until teardown"
                )
                details.add_row("observability", "nsys capture_window", settings.capture_window)
                details.add_row("observability", "nsys capture", window)
                if settings.capture_window == "measured_workload":
                    details.add_row("observability", "SRT_NSYS_CONTROL_SCRIPT", "/srtctl-runtime/nsys_window.py")
                    details.add_row("observability", "SRT_NSYS_CONTROL_DIR", "/logs/profiles/.control")
                details.add_row("observability", "nsys CPU sampling", "process-tree (every target)")
                details.add_row("observability", "nsys report timeout", f"{settings.report_timeout_secs}s")
                details.add_row("observability", "nsys reports", "<log_dir>/profiles/{prefill,decode,agg,frontend}/")
                details.add_row("observability", "nsys env", "DYN_ENABLE_RUST_NVTX=1")
                if config.backend_type == "trtllm":
                    details.add_row(
                        "observability", "nsys TRT-LLM env", "TLLM_PROFILE_LOG_RANKS=all; TLLM_LLMAPI_ENABLE_NVTX=1"
                    )
                if settings.nvtx_injection_path:
                    details.add_row("observability", "NVTX_INJECTION64_PATH", settings.nvtx_injection_path)

        tachometer = config.observability.tachometer
        if config.observability.tachometer_enabled:
            details.add_row("observability", "tachometer", "enabled")
            details.add_row("observability", "storage_subdir", tachometer.storage_subdir)
            details.add_row("observability", "collect_interval_ms", str(tachometer.collect_interval_ms))
            details.add_row("observability", "binary_path", tachometer.binary_path)
            if config.telemetry.enabled:
                details.add_row("observability", "dcgm_exporter", "shared with power telemetry")
            elif tachometer.resolved_dcgm_exporter is not None:
                dcgm = tachometer.resolved_dcgm_exporter
                details.add_row("observability", "dcgm_exporter", f"{dcgm.container_image} :{dcgm.port}")
            if tachometer.resolved_node_exporter is not None:
                node = tachometer.resolved_node_exporter
                details.add_row("observability", "node_exporter", f"{node.container_image} :{node.port}")
            if tachometer.resolved_process_exporter is not None:
                proc = tachometer.resolved_process_exporter
                launch = f"host binary {proc.binary}" if proc.binary else proc.container_image
                details.add_row("observability", "process_exporter", f"{launch} :{proc.port}")

        if config.telemetry.enabled:
            exporter = config.telemetry.dcgm_exporter
            details.add_row("telemetry", "provider", "dcgm-power")
            details.add_row("telemetry", "required", str(config.telemetry.required))
            details.add_row("telemetry", "artifacts", f"<log_dir>/{config.telemetry.storage_subdir}")
            if exporter is not None:
                details.add_row("telemetry", "dcgm_exporter", f"{exporter.container_image} (port {exporter.port})")

            cpu_exporter = config.telemetry.cpu_power_exporter
            if cpu_exporter is not None:
                details.add_row(
                    "telemetry", "cpu_power_exporter", f"{cpu_exporter.port} (source {cpu_exporter.source})"
                )

            cpu_power = config.telemetry.cpu_power
            if cpu_power.enabled:
                details.add_row(
                    "telemetry",
                    "cpu_power",
                    f"host collector (source {cpu_power.source}, <log_dir>/{cpu_power.storage_subdir}"
                    f"{', required' if cpu_power.required else ''})",
                )

        if config.frontend.worker_selection is not None:
            details.add_row("frontend", "router_policy_config", f"{ROUTER_POLICY_CONFIG_CONTAINER_PATH} (auto)")
            details.add_row(
                "frontend",
                "worker_selection",
                yaml.safe_dump(config.frontend.worker_selection, sort_keys=False).rstrip(),
            )

        if mooncake_cfg is not None:
            details.add_row("mooncake", "container", mooncake_cfg.container or "<job container>")
            if isinstance(mooncake_cfg, VLLMMooncakeKVStoreConfig) and mooncake_cfg.device_names_by_gpu:
                details.add_row("mooncake", "device_names_by_gpu", str(mooncake_cfg.device_names_by_gpu))
                details.add_row("mooncake", "process config", "/logs/mooncake_store_config_gpu<physical-ids>.json")
            details.add_row("mooncake", "master_port", f"{MOONCAKE_MASTER_PORT} (auto)")
            if mooncake_cfg.master_extra_args:
                details.add_row("mooncake", "master_extra_args", shlex.join(mooncake_cfg.master_extra_args))
            if isinstance(backend, VLLMProtocol):
                # vLLM workers need MOONCAKE_CONFIG_PATH pointing at a JSON file
                # — srtslurm writes this at job start. Show the resolved JSON
                # so operators can sanity-check protocol/device_name/sizes
                # before submitting. infra IP is unknown until allocation, so
                # use a placeholder for master_server_address.
                store_cfg = backend.build_mooncake_store_config("<infra_ip>")
                details.add_row(
                    "mooncake",
                    "store_config",
                    json.dumps(store_cfg, indent=2),
                )
                details.add_row(
                    "mooncake",
                    "MOONCAKE_CONFIG_PATH",
                    "/logs/mooncake_store_config.json (auto)",
                )

        console.print(Panel(details, border_style="blue"))


def _cpu_power_exporter_problem(srtctl_source: Path) -> str | None:
    """Why the installed exporter could not run on the compute nodes, if it could not.

    Existence alone is not enough: a partial download leaves a file srun cannot
    execute, and a checkout carried between architectures leaves one built for
    the wrong machine. Either way the failure surfaces only once the allocation
    is already running. The intended architecture is the compute architecture
    make setup ARCH= installed, not this submit host, which is routinely a
    different machine; when neither can be read, nothing is claimed.
    """
    label = "bin/cpu-power-exporter (compute-arch ACPI CPU power exporter)"
    exporter = srtctl_source / "bin" / "cpu-power-exporter"
    if not exporter.is_file():
        return label
    if not os.access(exporter, os.X_OK):
        return f"{label} — present but not executable"
    installed = arch_from_binary(exporter)
    target = detect_target_arch(srtctl_source)
    if installed is not None and installed != target:
        return f"{label} — built for {installed}, but the compute nodes are {target}"
    return None


def validate_setup(srtctl_source: Path, config: SrtConfig | None = None) -> None:
    """Validate that make setup has been run and required binaries exist.

    Checks for NATS, etcd, Tachometer, and compute-arch uv binaries. Raises SystemExit
    with a clear error message if anything is missing.

    cpu-power-exporter is only required by recipes that configure
    telemetry.cpu_power_exporter; every other recipe submits without it.
    """
    missing = []

    configs_dir = srtctl_source / "configs"
    if not (configs_dir / "nats-server").exists():
        missing.append("configs/nats-server")
    if not (configs_dir / "etcd").exists():
        missing.append("configs/etcd")
    if not (srtctl_source / "bin" / "uv").exists():
        missing.append("bin/uv (compute-arch uv)")
    if not (srtctl_source / "bin" / "tachometer-scraper").exists():
        missing.append("bin/tachometer-scraper (compute-arch Tachometer scraper)")
    cpu_power_enabled = (
        config is not None and config.telemetry.enabled and config.telemetry.cpu_power_exporter is not None
    )
    if cpu_power_enabled:
        problem = _cpu_power_exporter_problem(srtctl_source)
        if problem is not None:
            missing.append(problem)

    if missing:
        console.print(f"\n[red bold]ERROR:[/] Required binaries not found in {srtctl_source}:")
        for m in missing:
            console.print(f"  [red]✗[/] {m}")
        console.print("\nRun [bold]make setup ARCH=<compute_arch>[/] first:")
        console.print(f"  cd {srtctl_source}")
        console.print("  make setup ARCH=aarch64  [dim]# for GB200/Grace compute nodes[/]")
        console.print("  make setup ARCH=x86_64   [dim]# for x86_64 compute nodes[/]\n")
        raise SystemExit(1)

    # Optional: the default process exporter is host-native and skipped at launch
    # (with a warning in the sweep log) when its binary is absent. Surface that at
    # submit time so the gap is not discovered after the run.
    if not (configs_dir / "process-exporter").exists():
        console.print(
            "[yellow]WARNING:[/] configs/process-exporter not found; Tachometer will run without per-process/"
            "per-thread CPU telemetry. Re-run [bold]make setup ARCH=<compute_arch>[/] to install it."
        )


def generate_minimal_sbatch_script(
    config: SrtConfig,
    config_path: Path,
    setup_script: str | None = None,
    output_dir: Path | None = None,
    runtime_config_filename: str = "config.yaml",
    serve_only: bool = False,
    staged_config_dir: Path | None = None,
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
        serve_only: Keep the inference endpoint running without launching a benchmark
        staged_config_dir: Directory holding the recipe YAML(s) the job copies into its own
            OUTPUT_DIR at start. Set by ``srtctl render``, whose script is submitted by
            someone else, so the copy ``srtctl apply`` does after sbatch never happens.

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

    het_components = config.resources.het_components(
        infra_dedicated=config.infra.etcd_nats_dedicated_node,
        cluster_default=get_srtslurm_setting("use_het_jobs", False),
    )
    if het_components is not None and (config.frontend.dedicated_node or config.benchmark.client_dedicated_node):
        # SrtConfig validation only catches resources.het_jobs: true explicitly
        # set in the recipe — it can't see a cluster-level use_het_jobs default,
        # which is only resolved here via het_components(). Catch the combo now,
        # before sbatch submits a heterogeneous allocation that Nodes.from_slurm
        # will then reject at job startup after the nodes are already granted.
        raise ValueError(
            "frontend.dedicated_node/benchmark.client_dedicated_node are not supported with heterogeneous "
            "SLURM jobs, and this job resolved to heterogeneous (either resources.het_jobs: true or the "
            "cluster's use_het_jobs default)"
        )
    # For het jobs the sum is informational only — the template iterates het_components
    # and ignores total_nodes when het_components is set.
    total_nodes = planned_total_nodes(config) if het_components is None else sum(c.nodes for c in het_components)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Resolve container image path (expand aliases from srtslurm.yaml)
    container_image = os.path.expandvars(config.model.container)

    job_name = get_job_name(config)
    config_environment = config.dynamo.get_wheel_environment()
    config_environment.update(config.environment)

    rendered = template.render(
        job_name=job_name,
        total_nodes=total_nodes,
        het_components=het_components,
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
        serve_only=serve_only,
        staged_config_dir=str(staged_config_dir.resolve()) if staged_config_dir else None,
        config_environment={key: shlex.quote(str(value)) for key, value in config_environment.items()},
    )

    return rendered


def _print_running_summary(config: SrtConfig, console: Console, *, serve_only: bool = False) -> None:
    """Print what's being run and identity verification status."""
    console.print()
    console.print("[bold]Running:[/]")
    console.print(f"  Model:     {config.model.path}")
    console.print(f"  Container: {config.model.container}")
    console.print(f"  Backend:   {config.backend_type}")
    if serve_only:
        console.print("  Mode:      Serve only (no benchmark)")
    else:
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


def planned_total_nodes(config: SrtConfig) -> int:
    """Nodes a non-heterogeneous job asks Slurm for: engine roles, pools, plus one per
    dedicated role (etcd/nats, frontend, benchmark client), or one shared node when
    ``benchmark.colocate_with_frontend`` folds the dedicated roles together."""
    total_nodes = config.total_nodes
    num_dedicated_roles = sum(
        (
            config.infra.etcd_nats_dedicated_node,
            config.frontend.dedicated_node,
            config.benchmark.client_dedicated_node,
        )
    )
    if num_dedicated_roles > 0:
        total_nodes += 1 if config.benchmark.colocate_with_frontend else num_dedicated_roles
    return total_nodes


def render_placement(config: SrtConfig) -> dict[str, Any]:
    """What an external launcher needs to know about the job it is about to submit.

    Written next to a rendered script as ``render.json``. The node indices are positions
    in the allocation's nodelist (``scontrol show hostnames`` order), computed with the
    same rules the orchestrator applies at job start; they are ``None`` for
    heterogeneous jobs, whose components are addressed differently.
    """
    het = config.resources.het_components(
        infra_dedicated=config.infra.etcd_nats_dedicated_node,
        cluster_default=get_srtslurm_setting("use_het_jobs", False),
    )
    if het is None:
        total_nodes = planned_total_nodes(config)
        head, client = Nodes.planned_role_indices(
            total_nodes,
            frontend_dedicated_node=config.frontend.dedicated_node,
            client_dedicated_node=config.benchmark.client_dedicated_node,
            etcd_nats_dedicated_node=config.infra.etcd_nats_dedicated_node,
            colocate_dedicated_nodes=config.benchmark.colocate_with_frontend,
        )
        indices: dict[str, int | None] = {"frontend_node_index": head, "client_node_index": client}
    else:
        total_nodes = sum(c.nodes for c in het)
        indices = {"frontend_node_index": None, "client_node_index": None}
    return {
        "schema_version": 1,
        "name": config.name,
        "total_nodes": total_nodes,
        "heterogeneous": het is not None,
        **indices,
        "frontend_port": FRONTEND_PUBLIC_PORT,
        "served_model_name": config.served_model_name,
        "benchmark_type": config.benchmark.type,
        "frontend_type": config.frontend.type,
    }


def _render_to_dir(
    render_dir: Path,
    script_content: str,
    config: SrtConfig,
    *,
    source_config_path: Path,
    runtime_config_filename: str,
    runtime_config_text: str | None,
) -> str:
    """Write the sbatch script and the recipe(s) it stages into render_dir.

    A rendered script is submitted by someone else, so it has to carry everything
    ``submit_with_orchestrator`` would otherwise arrange after sbatch returns: the
    recipe under OUTPUT_DIR (the template copies it from render_dir at job start) and
    the git-state snapshot of any mounted checkouts. The script path is the last line
    on stdout so a caller can do ``sbatch --parsable "$(srtctl render ...)"``.
    """
    render_dir.mkdir(parents=True, exist_ok=True)
    staged_recipe = render_dir / "config.yaml"
    if not (staged_recipe.exists() and staged_recipe.samefile(source_config_path)):
        shutil.copy(source_config_path, staged_recipe)
    if runtime_config_text is not None and runtime_config_filename != "config.yaml":
        (render_dir / runtime_config_filename).write_text(runtime_config_text)
    git_sources = git_snapshot_sources_from_extra_mounts(config)
    if git_sources:
        write_git_state_snapshot(render_dir / GIT_STATE_FILENAME, git_sources)
    script_path = render_dir / "sbatch_script.sh"
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    placement = {**render_placement(config), "script": str(script_path.resolve())}
    (render_dir / "render.json").write_text(json.dumps(placement, indent=2) + "\n")
    console.print(f"[bold cyan]📝 Rendered:[/] {config.name} -> {script_path}")
    _print_running_summary(config, console)
    print(str(script_path.resolve()), flush=True)
    return str(script_path.resolve())


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
    serve_only: bool = False,
    render_dir: Path | None = None,
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
        serve_only: Keep the inference endpoint running without launching a benchmark.
        render_dir: Write the sbatch script and staged recipe here instead of submitting.
            The script is self-contained: whoever runs ``sbatch`` on it gets the same
            job ``srtctl apply`` would have submitted.

    Returns:
        job_id string on success, the rendered script path when render_dir is set,
        None for dry_run.
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
        serve_only=serve_only,
        staged_config_dir=render_dir,
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
        _print_running_summary(config, console, serve_only=serve_only)
        return

    # Validate setup before submitting (not during dry-run)
    srtctl_root = get_srtslurm_setting("srtctl_root")
    srtctl_source = Path(srtctl_root) if srtctl_root else Path(__file__).parent.parent.parent.parent
    validate_setup(srtctl_source)

    if render_dir is not None:
        return _render_to_dir(
            render_dir,
            script_content,
            config,
            source_config_path=source_config_path or config_path,
            runtime_config_filename=runtime_config_filename,
            runtime_config_text=resolved_runtime_config_text,
        )

    # Write script to temp file
    fd, script_path = tempfile.mkstemp(suffix=".slurm", prefix="srtctl_", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    console.print(f"[bold cyan]🚀 Submitting:[/] {config.name}")
    logger.debug("Script: %s", script_path)

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
        git_sources = git_snapshot_sources_from_extra_mounts(config)
        if git_sources:
            write_git_state_snapshot(job_output_dir / GIT_STATE_FILENAME, git_sources)

        job_name = get_job_name(config)

        # Build comprehensive job metadata
        metadata: dict[str, Any] = {
            "version": "2.0",
            "orchestrator": True,
            "job_id": job_id,
            "job_name": job_name,
            "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
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
                "agg_nodes": config.resources.agg_nodes,
                "prefill_workers": config.resources.num_prefill,
                "decode_workers": config.resources.num_decode,
                "agg_workers": config.resources.num_agg,
                "gpus_per_prefill": config.resources.gpus_per_prefill,
                "gpus_per_decode": config.resources.gpus_per_decode,
                "gpus_per_agg": config.resources.gpus_per_agg,
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
        if serve_only:
            metadata["serve_only"] = True
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

        log_dir = f"{job_output_dir}/logs"
        os.makedirs(log_dir, exist_ok=True)
        log = f"{log_dir}/sweep_{job_id}.log"
        result = subprocess.run(
            ["touch", log],
            check=False,
        )

        console.print(f"[bold green]✅ Job {job_id} submitted![/]")
        console.print(f"[dim]📁 Logs:[/] {log_dir}")
        console.print(f"[dim]📋 Monitor:[/] tail -f {log}")
        console.print(f"[dim]📊 Queue:[/] squeue --job {job_id}")

        _print_running_summary(config, console, serve_only=serve_only)

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
    serve_only: bool = False,
    render_dir: Path | None = None,
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
        serve_only: Keep the inference endpoint running without launching a benchmark.

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
        serve_only=serve_only,
        render_dir=render_dir,
    )


def is_sweep_config(config_path: Path) -> bool:
    """Check if config file is a sweep config by looking for 'sweep' section."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return "sweep" in config if config else False
    except Exception:  # noqa: BLE001
        return False


def submit_sweep(
    config_path: Path,
    dry_run: bool = False,
    setup_script: str | None = None,
    tags: list[str] | None = None,
    output_dir: Path | None = None,
    enforce_preflight: bool = True,
):
    """Submit parameter sweep.

    Args:
        config_path: Path to sweep YAML config
        dry_run: If True, don't submit to SLURM
        setup_script: Optional custom setup script name
        tags: Optional list of tags
        output_dir: Custom output directory (CLI flag, highest priority)
        enforce_preflight: When False, skip the pre-submit model/container/telemetry
            FS checks for every variant (propagated to submit_single).
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

        sweep_dir = (
            Path.cwd()
            / "dry-runs"
            / f"{sweep_config['name']}_sweep_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
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
                    enforce_preflight=enforce_preflight,
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
    enforce_preflight: bool = True,
) -> None:
    """Submit all YAML configs in a directory recursively.

    Args:
        directory: Directory containing YAML config files
        dry_run: If True, don't submit to SLURM
        setup_script: Optional custom setup script name
        tags: Optional list of tags
        force_sweep: If True, treat all configs as sweeps
        output_dir: Custom output directory (CLI flag, highest priority)
        enforce_preflight: When False, skip the pre-submit model/container/telemetry
            FS checks for every config (propagated to submit_single / submit_sweep /
            submit_override).
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
                submit_override(
                    yaml_file,
                    dry_run=dry_run,
                    setup_script=setup_script,
                    tags=tags,
                    output_dir=output_dir,
                    enforce_preflight=enforce_preflight,
                )
            elif force_sweep or is_sweep_config(yaml_file):
                submit_sweep(
                    yaml_file,
                    dry_run=dry_run,
                    setup_script=setup_script,
                    tags=tags,
                    output_dir=output_dir,
                    enforce_preflight=enforce_preflight,
                )
            else:
                submit_single(
                    config_path=yaml_file,
                    dry_run=dry_run,
                    setup_script=setup_script,
                    tags=tags,
                    output_dir=output_dir,
                    enforce_preflight=enforce_preflight,
                )
            success_count += 1
        except Exception as e:
            console.print(f"[bold red]  ❌ Error:[/] {e}")
            logger.debug("Full traceback:", exc_info=True)
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
def materialize_config_path(config_path: Path, overrides: Sequence[Any] = (), *, pin_sources: bool = False):
    """Stage stdin-backed, --set/--unset-modified, or source-pinned configs to a temporary YAML file.

    Overrides are applied to the raw document (comments preserved) before any
    reader sees it, so they take effect identically for plain, sweep, and
    override-format files and end up in the config.yaml copied into the job
    output directory. With ``pin_sources`` (submit only), every ``source.rev``
    that is not already a commit is resolved with ``git ls-remote`` and recorded
    as ``source.sha`` in that same document, so the job builds exactly the
    commit the lockfile names. The source file is never modified.
    """
    from_stdin = str(config_path) in {"-", "/dev/stdin"}
    if not from_stdin and not overrides and not pin_sources:
        yield config_path
        return
    if not from_stdin and (not config_path.exists() or config_path.is_dir()):
        if config_path.is_dir() and overrides:
            raise ValueError("--set/--unset apply to a single config file, not a directory")
        yield config_path  # let the caller report the missing file, or handle the directory
        return

    payload = sys.stdin.read() if from_stdin else config_path.read_text()
    if not payload.strip():
        raise ValueError("No YAML received on stdin" if from_stdin else f"{config_path} is empty")

    changed = from_stdin
    if overrides or pin_sources:
        from srtctl.core.yaml_utils import dump_yaml_with_comments, load_yaml_text_with_comments

        document = load_yaml_text_with_comments(payload)
        if overrides:
            from srtctl.core.overrides import apply_overrides_to_recipe

            for entry in apply_overrides_to_recipe(document, overrides):
                logger.info("Applied override: %s", entry)
            changed = True
        if pin_sources and "source" in payload:
            from srtctl.core.source import pin_source_revs

            pinned = pin_source_revs(document)
            for entry in pinned:
                logger.info("Pinned source: %s", entry)
            _pinned_sources.extend(pinned)
            changed = changed or bool(pinned)
        if changed:
            payload = dump_yaml_with_comments(document) or ""

    if not changed:
        yield config_path
        return

    fd, temp_path = tempfile.mkstemp(
        suffix=".yaml",
        prefix="srtctl_stdin_" if from_stdin else "srtctl_override_",
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
    enforce_preflight: bool = True,
    serve_only: bool = False,
    render_dir: Path | None = None,
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
        enforce_preflight: When False, skip the pre-submit model/container/telemetry
            FS checks for every expanded variant (propagated to submit_single /
            submit_sweep).
        serve_only: Keep the inference endpoint running without launching a benchmark.
    """
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    override_configs = generate_override_configs(raw_config, selector=selector)
    if render_dir is not None and len(override_configs) != 1:
        raise ValueError(
            "render needs exactly one variant: pass a selector (-f config.yaml:base or -f config.yaml:override_name) "
            f"instead of rendering {len(override_configs)} variants into one directory"
        )

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
    if serve_only and len(resolved_variants) != 1:
        raise ValueError("--serve-only requires an override selector that resolves to exactly one job")

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
        # Same expansions load_config applies, so the dry-run details and the
        # sbatch-time config match what the in-job loader will run with.
        expand_engine_config_defaults(resolved_config)
        config = SrtConfig.Schema().load(resolved_config)

        if "sweep" in config_cm:
            if serve_only:
                raise ValueError("--serve-only does not support sweep configs")
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
                    enforce_preflight=enforce_preflight,
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
                enforce_preflight=enforce_preflight,
                serve_only=serve_only,
                render_dir=render_dir,
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
  srtctl apply -f config.yaml --serve-only       # Serve until cancelled; do not benchmark
  srtctl apply -f ./configs/                     # Submit all YAMLs in directory
  srtctl apply -f config.yaml --sweep            # Submit sweep
  srtctl preflight -f config.yaml                # Check model/container availability
  srtctl dry-run -f config.yaml                  # Dry run
  srtctl render -f config.yaml --to ./rendered   # Write the sbatch script for someone else to submit
  srtctl resolve-override -f config.yaml         # Resolve override YAML (no submit)
  srtctl resolve-override -f config.yaml --stdout  # Print to stdout
  srtctl monitor                                 # Live job dashboard
  srtctl monitor --outputs /path/to/outputs      # Dashboard with custom outputs dir
  srtctl view /path/to/run-output                # Local ruter route-decision viewer
  srtctl status-server --host 0.0.0.0            # Local status collector for reporting.status.endpoint
  srtctl schema-docs [--check]                   # Regenerate (or verify) docs/schema-reference.md + docs/legacy-v1.md
  srtctl migrate -f config.yaml --in-place       # Upgrade a recipe to the current schema version
  srtctl migrate -f recipes/ --verify            # Prove v1 and migrated v2 recipes resolve identically
  srtctl skill --target claude                   # Install the srtctl agent skill into this project
  srtctl --version                               # Version (from the git tag), commit, schema and lockfile versions
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # `srtctl --version`: package version (from the git tag), commit, and the protocol versions.
    from srtctl.version import version_info

    parser.add_argument("--version", action="version", version=str(version_info()))

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Offline artifact generation is explicit and never part of the job lifecycle.
    from srtctl.dsight.cli import add_commands as add_dashboard_commands

    add_dashboard_commands(
        subparsers.add_parser(
            "dsight", aliases=["dashboard"], help="Build and query an offline inference trace dashboard"
        )
    )

    def add_override_args(p):
        p.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            dest="set_overrides",
            help=(
                "Override a recipe value by dotted path before validation (repeatable), e.g. "
                "--set health_check.max_attempts=720 or --set 'backend.sglang_config.prefill.dist-timeout=1800'. "
                "Values parse as YAML scalars or lists; mappings stay literal strings. "
                "On override files the value is written into base and every variant."
            ),
        )
        p.add_argument(
            "--unset",
            action="append",
            default=[],
            metavar="KEY",
            dest="unset_overrides",
            help="Remove a recipe key by dotted path before validation (repeatable), e.g. --unset health_check",
        )

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
        add_override_args(p)

    apply_parser = subparsers.add_parser("apply", help="Submit job(s) to SLURM")
    add_common_args(apply_parser)
    apply_parser.add_argument("--setup-script", type=str, help="Custom setup script in configs/")
    apply_parser.add_argument("--tags", type=str, help="Comma-separated tags")
    apply_parser.add_argument(
        "--serve-only",
        action="store_true",
        help="Deploy the inference endpoint without running a benchmark; keep serving until the job is cancelled.",
    )
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
    apply_parser.add_argument(
        "--no-preflight",
        action="store_true",
        dest="no_preflight",
        help=(
            "Skip the pre-submit model.path / model.container / telemetry filesystem "
            "checks. Useful when those paths only exist on compute nodes (e.g. node-local "
            "NVMe like /scratch/models/...) and not on the node invoking srtctl. The "
            "framework itself will still fail loudly at runtime if a path is genuinely "
            "missing on the compute node."
        ),
    )

    dry_run_parser = subparsers.add_parser("dry-run", help="Validate without submitting")
    add_common_args(dry_run_parser)
    render_parser = subparsers.add_parser(
        "render",
        help="Write a self-contained sbatch script instead of submitting it",
        description=(
            "Render the exact sbatch script `srtctl apply` would submit, plus the staged recipe, into "
            "--to DIR, and print the script path as the last line of stdout. For launchers that must own "
            "the sbatch call themselves (e.g. a harness whose contract is `exec sbatch --parsable ...`)."
        ),
    )
    add_common_args(render_parser)
    render_parser.add_argument("--to", type=Path, required=True, dest="render_dir", help="Directory to render into")
    render_parser.add_argument("--setup-script", type=str, help="Custom setup script in configs/")
    render_parser.add_argument(
        "--serve-only",
        action="store_true",
        help="Render a serve-only job: deploy the endpoint and keep serving until the job is cancelled.",
    )
    render_parser.add_argument(
        "--no-preflight",
        action="store_true",
        dest="no_preflight",
        help="Skip the pre-render model.path / model.container / telemetry filesystem checks.",
    )

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
    add_override_args(preflight_parser)

    monitor_parser = subparsers.add_parser("monitor", help="Live dashboard for srt-slurm jobs", add_help=False)
    monitor_parser.add_argument("args", nargs=argparse.REMAINDER)

    view_parser = subparsers.add_parser("view", help="Serve the local ruter route-decision viewer")
    view_parser.add_argument(
        "root", nargs="?", type=Path, default=Path("."), help="srt-slurm run directory or logs/.ruter"
    )
    view_parser.add_argument("--port", type=int, default=8877, help="Loopback port (default: 8877)")
    view_parser.add_argument("--refresh", action="store_true", help="Reparse logs before loading the viewer")

    status_server_parser = subparsers.add_parser(
        "status-server",
        help="Run the native status collector that reporting.status.endpoint can point at",
    )
    add_status_server_arguments(status_server_parser)

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
    add_override_args(resolve_parser)

    # Fingerprint comparison: srtctl diff <path_a> <path_b>
    diff_parser = subparsers.add_parser("diff", help="Compare fingerprints from two runs")
    diff_parser.add_argument("path_a", type=Path, help="First output dir or lockfile")
    diff_parser.add_argument("path_b", type=Path, help="Second output dir or lockfile")
    diff_parser.add_argument("--verbose", action="store_true", help="Show all package changes")

    # Environment check: srtctl check <path>
    check_parser = subparsers.add_parser("check", help="Check environment against a fingerprint")
    check_parser.add_argument("path", type=Path, help="Lockfile or output dir to check against")
    check_parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    # Generated schema reference: srtctl schema-docs [--check] [--output PATH]
    schema_docs_parser = subparsers.add_parser(
        "schema-docs",
        help="Regenerate docs/schema-reference.md (2.0 layout) and docs/legacy-v1.md (v1 layout) from the code",
    )
    schema_docs_parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if either checked-in document is stale instead of rewriting them (used by CI)",
    )
    schema_docs_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the schema reference to this path instead of docs/schema-reference.md (legacy-v1.md lands beside it)",
    )

    # Recipe migration: srtctl migrate -f recipe.yaml [--in-place | --output PATH]
    skill_parser = subparsers.add_parser(
        "skill",
        help="Install the in-package agent skill (how to drive srtctl) for Claude Code, Codex, or Cursor",
    )
    skill_parser.add_argument(
        "--target",
        choices=["claude", "codex", "cursor"],
        required=True,
        help="Which agent's project skill layout to write",
    )
    skill_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root to install under (default: the current directory)",
    )
    skill_parser.add_argument(
        "--print", action="store_true", dest="print_only", help="Print the skill instead of writing it"
    )

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Upgrade a recipe (plain, override, or lock file) to the current schema version",
    )
    migrate_parser.add_argument(
        "-f",
        "--file",
        type=Path,
        required=True,
        action="append",
        dest="migrate_files",
        help="Recipe YAML to migrate; a directory is walked recursively (repeatable)",
    )
    migrate_parser.add_argument("--in-place", action="store_true", help="Rewrite the file(s) instead of printing")
    migrate_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the migrated recipe to this path (single file only; default: print to stdout)",
    )
    migrate_parser.add_argument(
        "--verify",
        action="store_true",
        help="Do not write: migrate in memory and prove the v1 and v2 recipes resolve identically (golden equality)",
    )

    args = parser.parse_args()

    if args.command in ("dsight", "dashboard"):
        from srtctl.dsight.cli import run as run_dashboard

        raise SystemExit(run_dashboard(args))

    json_mode = bool(getattr(args, "json_output", False))
    render_dir: Path | None = getattr(args, "render_dir", None)
    mock_mode = bool(getattr(args, "mock_mode", False))
    serve_only = bool(getattr(args, "serve_only", False))
    if serve_only and mock_mode:
        parser.error("--serve-only cannot be combined with --mock")
    if serve_only and getattr(args, "sweep", False):
        parser.error("--serve-only does not support sweeps")

    # Always rebind the module console on each invocation so json-mode prose
    # goes to stderr and non-json prose returns to stdout. Save the original
    # so we can restore it on exit — direct library callers of submit_single /
    # submit_override (tests, etc.) must not see a leaked stderr binding.
    global console
    _original_console = console
    # stdout is the machine-readable channel for --json and for render (whose last
    # line is the script path a caller feeds to sbatch); prose goes to stderr there.
    console = Console(file=sys.stderr) if (json_mode or render_dir is not None) else Console()

    def restore_console() -> None:
        global console
        console = _original_console

    if json_mode:
        _submissions.clear()

    from srtctl.core.overrides import parse_overrides

    try:
        overrides = parse_overrides(getattr(args, "set_overrides", None), getattr(args, "unset_overrides", None))
    except ValueError as exc:
        parser.error(str(exc))
    global _active_overrides
    _active_overrides = [override.render() for override in overrides]

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

    if args.command == "schema-docs":
        from srtctl.core.schema_docs import (
            DEFAULT_OUTPUT,
            legacy_output_for,
            schema_reference_is_current,
            write_schema_reference,
        )

        output = args.output or DEFAULT_OUTPUT
        if args.check:
            if schema_reference_is_current(output):
                console.print(f"[green]✓[/] {output} and {legacy_output_for(output)} are up to date")
                restore_console()
                return
            console.print(
                f"[bold red]✗[/] {output} or {legacy_output_for(output)} is stale; "
                "run `srtctl schema-docs` and commit the result"
            )
            restore_console()
            sys.exit(1)
        written = write_schema_reference(output)
        console.print(f"[green]✓[/] Wrote {written} and {legacy_output_for(written)}")
        restore_console()
        return

    if args.command == "skill":
        from srtctl.skills import install_skill, render_skill

        if args.print_only:
            print(render_skill(args.target))
            restore_console()
            return
        written = install_skill(args.target, args.root)
        console.print(f"[green]✓[/] Wrote {written}")
        restore_console()
        return

    if args.command == "migrate":
        from srtctl.core.migrate import migrate_recipe_file, recipe_files, verify_migration_file

        files = recipe_files(args.migrate_files)
        if not files:
            console.print("[bold red]No recipe files found[/]")
            sys.exit(1)
        if args.verify:
            counts: dict[str, int] = {"ok": 0, "mismatch": 0, "skipped": 0, "error": 0}
            for path in files:
                outcome = verify_migration_file(path)
                counts[outcome.status] += 1
                if outcome.status == "ok":
                    console.print(f"[green]✓[/] {path} ({outcome.variants} variant(s) resolve identically)")
                elif outcome.status == "skipped":
                    console.print(f"[yellow]-[/] {path}: skipped, {outcome.detail}")
                else:
                    console.print(f"[bold red]✗[/] {path}: {outcome.detail}")
            console.print(
                f"\n{counts['ok']} identical, {counts['mismatch']} mismatched, "
                f"{counts['skipped']} skipped (v1 does not load), {counts['error']} unreadable"
            )
            restore_console()
            sys.exit(1 if counts["mismatch"] or counts["error"] else 0)
        if not args.in_place and args.output is None and len(files) > 1:
            console.print("[bold red]Error:[/] printing to stdout needs a single file; use --in-place for many")
            sys.exit(1)
        if args.output is not None and len(files) > 1:
            console.print("[bold red]Error:[/] --output takes a single file")
            sys.exit(1)
        failed = 0
        for path in files:
            try:
                result = migrate_recipe_file(path, in_place=args.in_place, output=args.output)
            except Exception as exc:  # noqa: BLE001 - one unreadable recipe must not stop a directory run
                failed += 1
                detail = next(
                    (line for line in str(exc).splitlines() if "duplicate key" in line), str(exc).splitlines()[0]
                )
                console.print(f"[bold red]✗[/] {path}: not migrated: {detail} (fix the recipe, then re-run)")
                continue
            if not args.in_place and args.output is None:
                sys.stdout.write(result.text)
                sys.stdout.flush()
            else:
                target = path if args.in_place else args.output
                detail = ", ".join(result.notes) if result.notes else "already current"
                console.print(f"[green]✓[/] {target}: schema {result.from_version} -> {result.to_version} ({detail})")
        if failed:
            console.print(f"\n{len(files) - failed} migrated, {failed} not migrated")
            restore_console()
            sys.exit(1)
        restore_console()
        return

    if args.command == "monitor":
        from srtctl.cli.monitor import main as _monitor_main

        sys.argv = [sys.argv[0]] + (args.args or [])
        _monitor_main()
        return

    if args.command == "view":
        from srtctl.ruter.view import main as _view_main

        view_args = [str(args.root), "--port", str(args.port)]
        if args.refresh:
            view_args.append("--refresh")
        _view_main(view_args)
        return

    if args.command == "status-server":
        serve_status_server(
            host=args.host,
            port=args.port,
            db_path=args.db,
            token_env=args.token_env,
            read_token_env=args.read_token_env,
            allow_unauthenticated=args.allow_unauthenticated,
            cors_origins=args.cors_origin,
        )
        return

    # Parse config arg: supports path:selector format for overrides
    config_path, selector = parse_config_arg(args.config)

    is_dry_run = args.command == "dry-run"
    tags = [t.strip() for t in (getattr(args, "tags", "") or "").split(",") if t.strip()] or None

    try:
        with materialize_config_path(
            config_path, overrides, pin_sources=args.command == "apply"
        ) as effective_config_path:
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

            # --no-preflight is only registered on the apply parser, so
            # dry-run / preflight / resolve-override won't carry it. Default
            # to False on those subcommands; dry-run already implies no
            # enforcement via the is_dry_run branch below.
            no_preflight = getattr(args, "no_preflight", False)
            # srtslurm.yaml `preflight: false` turns the check off cluster-wide
            # (paths that exist only on compute nodes); same effect as the flag.
            cluster_preflight = get_srtslurm_setting("preflight", True)
            if cluster_preflight is False and not (mock_mode or is_dry_run or no_preflight):
                logger.info("preflight skipped: srtslurm.yaml sets preflight: false")
            enforce_preflight = not (mock_mode or is_dry_run or no_preflight or cluster_preflight is False)

            # Handle directory input
            if effective_config_path.is_dir():
                if serve_only:
                    raise ValueError("--serve-only expects a single config file, not a directory")
                if render_dir is not None:
                    raise ValueError("render expects a single config file, not a directory")
                if selector:
                    logger.warning(f"Selector ':{selector}' ignored for directory input")
                submit_directory(
                    effective_config_path,
                    dry_run=is_dry_run,
                    setup_script=setup_script,
                    tags=tags,
                    force_sweep=args.sweep,
                    output_dir=output_dir,
                    enforce_preflight=enforce_preflight,
                )
            elif is_override_config(effective_config_path):
                submit_override(
                    effective_config_path,
                    selector=selector,
                    dry_run=is_dry_run,
                    setup_script=setup_script,
                    tags=tags,
                    output_dir=output_dir,
                    enforce_preflight=enforce_preflight,
                    serve_only=serve_only,
                    render_dir=render_dir,
                )
            else:
                if selector:
                    logger.warning(f"Selector ':{selector}' ignored — config is not an override file")
                is_sweep = args.sweep or is_sweep_config(effective_config_path)
                if is_sweep:
                    if serve_only:
                        raise ValueError("--serve-only does not support sweep configs")
                    if render_dir is not None:
                        raise ValueError("render does not support sweep configs; render one variant at a time")
                    submit_sweep(
                        effective_config_path,
                        dry_run=is_dry_run,
                        setup_script=setup_script,
                        tags=tags,
                        output_dir=output_dir,
                        enforce_preflight=enforce_preflight,
                    )
                else:
                    submit_single(
                        config_path=effective_config_path,
                        dry_run=is_dry_run,
                        setup_script=setup_script,
                        tags=tags,
                        output_dir=output_dir,
                        enforce_preflight=enforce_preflight,
                        serve_only=serve_only,
                        render_dir=render_dir,
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
            logger.debug("Full traceback:", exc_info=True)
            sys.exit(1)
        console.print(f"[bold red]Error:[/] {e}")
        logger.debug("Full traceback:", exc_info=True)
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
