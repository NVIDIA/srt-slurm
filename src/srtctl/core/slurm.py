# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SLURM utilities for job management and process launching.

This module consolidates all SLURM-related functionality:
- Environment: get_slurm_job_id, get_slurm_nodelist
- Network: get_hostname_ip, get_node_ips
- Process launching: start_srun_process, run_command
- Container utilities: get_container_mounts_str
"""

import logging
import os
import socket
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .ip_utils import get_node_ip

logger = logging.getLogger(__name__)


def _get_cluster_bash_preamble() -> str | None:
    """Look up the cluster-wide default_bash_preamble.

    Imported lazily to avoid a circular dependency (config.py imports schema,
    which transitively imports from this module's siblings).
    """
    from .config import get_srtslurm_setting

    value = get_srtslurm_setting("default_bash_preamble")
    return value if isinstance(value, str) and value else None


# ============================================================================
# SLURM Environment
# ============================================================================


def get_slurm_job_id() -> str | None:
    """Get the current SLURM job ID from environment."""
    return os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")


def get_slurm_nodelist() -> list[str]:
    """Get list of nodes from SLURM_NODELIST environment variable.

    Returns:
        List of node hostnames, or empty list if not in SLURM.
    """
    return _expand_nodelist(os.environ.get("SLURM_NODELIST", ""))


def _expand_nodelist(nodelist_raw: str) -> list[str]:
    """Expand a SLURM ranged nodelist via ``scontrol show hostnames``."""
    if not nodelist_raw:
        return []

    try:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist_raw],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("\n")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: try simple parsing for non-ranged formats
        return [nodelist_raw]


def get_slurm_het_nodelists() -> list[list[str]] | None:
    """Per-component nodelists for a SLURM heterogeneous job, else None.

    Returns one expanded nodelist per het component when ``SLURM_HET_SIZE`` is
    set to a value greater than 1. Returns None for non-het jobs so callers can
    fall back to ``get_slurm_nodelist()``.
    """
    het_size_raw = os.environ.get("SLURM_HET_SIZE", "")
    if not het_size_raw:
        return None
    try:
        het_size = int(het_size_raw)
    except ValueError:
        return None
    if het_size < 2:
        return None

    groups: list[list[str]] = []
    for i in range(het_size):
        nodelist_raw = os.environ.get(f"SLURM_JOB_NODELIST_HET_GROUP_{i}", "")
        groups.append(_expand_nodelist(nodelist_raw))
    return groups


# ============================================================================
# Network Resolution
# ============================================================================


def get_hostname_ip(hostname: str, network_interface: str | None = None) -> str:
    """The address other processes in the job use to reach ``hostname``, from the selected launcher."""
    from .launcher import get_launcher

    return get_launcher().node_ip(hostname, network_interface)


def resolve_slurm_hostname_ip(hostname: str, network_interface: str | None = None) -> str:
    """Resolve hostname to routable IP address.

    Uses multiple resolution strategies:
    1. If inside a SLURM job, use srun to get the real IP from the target node
    2. Fall back to socket.gethostbyname() (may return loopback on some systems)

    Args:
        hostname: Node hostname to resolve
        network_interface: Optional network interface to prefer

    Returns:
        IP address as string
    """
    # If we're inside a SLURM allocation, use srun-based resolution
    # This gets the actual routable IP from the target node
    slurm_job_id = get_slurm_job_id()
    if slurm_job_id:
        ip = get_node_ip(hostname, slurm_job_id, network_interface)
        if ip:
            return ip
        logger.warning(
            "srun-based IP resolution failed for %s, falling back to socket resolution",
            hostname,
        )

    # Fallback to socket resolution
    try:
        ip = socket.gethostbyname(hostname)
        # Warn if we got a loopback address
        if ip.startswith("127."):
            logger.warning(
                "socket.gethostbyname returned loopback %s for %s - this may cause cross-node issues",
                ip,
                hostname,
            )
        return ip
    except socket.gaierror:
        # Return hostname as-is (may be IP already)
        return hostname


def get_node_ips(
    nodes: list[str],
    slurm_job_id: str | None = None,
    network_interface: str | None = None,
) -> dict[str, str]:
    """Get IP addresses for multiple SLURM nodes.

    Args:
        nodes: List of node hostnames
        slurm_job_id: SLURM job ID for srun context
        network_interface: Specific network interface to use

    Returns:
        Dict mapping node hostname to IP address
    """
    ips = {}
    for node in nodes:
        ip = get_node_ip(node, slurm_job_id, network_interface)
        if ip:
            ips[node] = ip
        else:
            logger.warning("Could not resolve IP for node %s", node)
    return ips


# ============================================================================
# Process Launching
# ============================================================================

# enroot env var that remaps the unprivileged user to root inside the container
# at container-creation time. Injected (via srun --export) only on launches that
# install dynamo, whose cold build needs apt-get/pip-to-system as root. Passing an
# env var (not the pyxis --container-remap-root flag) degrades gracefully: srun
# never parses it, so an unsupporting cluster no-ops instead of failing the step.
CONTAINER_REMAP_ROOT_EXPORT = {"ENROOT_REMAP_ROOT": "yes"}


def start_srun_process(
    command: list[str],
    *,
    nodes: int = 1,
    ntasks: int = 1,
    cpus_per_task: int | None = None,
    nodelist: Sequence[str] | None = None,
    output: str | None = None,
    container_image: str | None = None,
    container_mounts: dict[Path, Path] | None = None,
    env_to_pass_through: list[str] | None = None,
    env_to_set: dict[str, str] | None = None,
    env_to_unset: list[str] | None = None,
    bash_preamble: str | None = None,
    srun_options: dict[str, str] | None = None,
    srun_export_env: dict[str, str] | None = None,
    overlap: bool = True,
    use_bash_wrapper: bool = True,
    mpi: str | None = None,
    oversubscribe: bool = False,
    cpu_bind: str | None = None,
    het_group: int | None = None,
    step_name: str | None = None,
) -> subprocess.Popen:
    """Start a process through the selected launcher (``srun`` by default; see ``core/launcher.py``).

    This is the central function for launching all processes.
    It handles container mounts, environment variables, and output redirection.

    Args:
        command: Command to run as list of strings
        nodes: Number of nodes (default: 1)
        ntasks: Number of tasks (default: 1)
        cpus_per_task: CPUs per task (optional)
        nodelist: Specific nodes to run on (optional)
        output: Output file path (optional)
        container_image: Container image path (optional)
        container_mounts: Dict of host_path -> container_path mounts
        env_to_pass_through: Environment variable names to pass through
        env_to_set: Environment variables to set (name -> value)
        env_to_unset: Environment variable names to unset before the preamble and command
        bash_preamble: Bash commands to run before the main command
        step_name: Name the Slurm step (``srun --job-name``) so it can be found in
            ``squeue --steps`` and signalled with ``scancel --signal`` later. SIGTERM
            to the srun process itself only aborts the step (the task is SIGKILLed).
        srun_options: Additional srun options as dict
        srun_export_env: Env vars to set in the srun *task* environment (rendered as
            ``--export=ALL,K=V,...``). Unlike env_to_set (which exports inside the
            container after it starts), these reach the container runtime at creation
            time — required for vars like ENROOT_REMAP_ROOT that enroot reads up front.
        overlap: Use --overlap flag (default: True)
        use_bash_wrapper: Wrap command in bash -c (default: True)
        mpi: MPI type (e.g., "pmix" for TRTLLM)
        oversubscribe: Use --oversubscribe flag (for MPI jobs)
        cpu_bind: CPU binding mode (e.g., "verbose,none" for TRTLLM)

    Returns:
        subprocess.Popen object for the srun process

    Example:
        proc = start_srun_process(
            command=["python3", "-m", "dynamo.sglang", "--model-path", "/model"],
            nodelist=["node1"],
            container_image="/containers/sglang.sqsh",
            container_mounts={Path("/models/llama"): Path("/model")},
            env_to_set={"NATS_SERVER": "nats://node1:4222"},
        )
    """
    from .launcher import LaunchSpec, get_launcher

    return get_launcher().launch(
        LaunchSpec(
            command=command,
            nodes=nodes,
            ntasks=ntasks,
            cpus_per_task=cpus_per_task,
            nodelist=nodelist,
            output=output,
            container_image=container_image,
            container_mounts=container_mounts,
            env_to_pass_through=env_to_pass_through,
            env_to_set=env_to_set,
            env_to_unset=env_to_unset,
            bash_preamble=bash_preamble,
            srun_options=srun_options,
            srun_export_env=srun_export_env,
            overlap=overlap,
            use_bash_wrapper=use_bash_wrapper,
            mpi=mpi,
            oversubscribe=oversubscribe,
            cpu_bind=cpu_bind,
            het_group=het_group,
            step_name=step_name,
        )
    )


def run_command(
    command: str,
    background: bool = False,
    stdout=None,
    stderr=None,
) -> subprocess.Popen | int:
    """Run a shell command.

    Args:
        command: Command string to run
        background: If True, return Popen object; if False, wait and return exit code
        stdout: Optional stdout file handle
        stderr: Optional stderr file handle

    Returns:
        Popen object if background=True, exit code if background=False
    """
    logger.debug("Running command: %s", command)

    if background:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=stdout or subprocess.DEVNULL,
            stderr=stderr or subprocess.DEVNULL,
        )
        return proc
    else:
        result = subprocess.run(command, shell=True, check=False)
        return result.returncode


# ============================================================================
# Container Utilities
# ============================================================================


def get_container_mounts_str(mounts: dict[Path, Path]) -> str:
    """Convert container mounts dict to comma-separated string.

    Args:
        mounts: Dict mapping host paths to container paths

    Returns:
        Comma-separated string for --container-mounts
    """
    return ",".join(f"{host}:{container}" for host, container in mounts.items())
