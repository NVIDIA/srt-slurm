# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shlex
import socket
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .ip_utils import get_node_ip

logger = logging.getLogger(__name__)


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
    nodelist_raw = os.environ.get("SLURM_NODELIST", "")
    if not nodelist_raw:
        return []

    # Use scontrol to expand the nodelist
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


# ============================================================================
# Network Resolution
# ============================================================================


def get_hostname_ip(hostname: str, network_interface: str | None = None) -> str:
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
    bash_preamble: str | None = None,
    srun_options: dict[str, str] | None = None,
    overlap: bool = True,
    use_bash_wrapper: bool = True,
    mpi: str | None = None,
    oversubscribe: bool = False,
    cpu_bind: str | None = None,
) -> subprocess.Popen:
    """Start a process via srun with container support.

    This is the central function for launching all srun processes.
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
        bash_preamble: Bash commands to run before the main command
        srun_options: Additional srun options as dict
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
    srun_cmd = ["srun"]

    # ensures srun runs in the same job context
    slurm_job_id = get_slurm_job_id()
    if slurm_job_id:
        srun_cmd.extend(["--jobid", slurm_job_id])

    # Basic options
    if overlap:
        srun_cmd.append("--overlap")

    # MPI options (for TRTLLM)
    if mpi:
        srun_cmd.extend(["--mpi", mpi])
    if oversubscribe:
        srun_cmd.append("--oversubscribe")
    if cpu_bind:
        srun_cmd.extend(["--cpu-bind", cpu_bind])

    srun_cmd.extend(["--nodes", str(nodes)])
    srun_cmd.extend(["--ntasks", str(ntasks)])

    if cpus_per_task:
        srun_cmd.extend(["--cpus-per-task", str(cpus_per_task)])

    if nodelist:
        srun_cmd.extend(["--nodelist", ",".join(nodelist)])

    if output:
        srun_cmd.extend(["--output", output])

    # Container options
    if container_image:
        srun_cmd.extend(["--container-image", str(container_image)])
        srun_cmd.append("--no-container-entrypoint")
        srun_cmd.append("--no-container-mount-home")

        if container_mounts:
            mount_str = ",".join(f"{host}:{container}" for host, container in container_mounts.items())
            srun_cmd.extend(["--container-mounts", mount_str])

    # Additional srun options
    if srun_options:
        for key, value in srun_options.items():
            if value:
                srun_cmd.extend([f"--{key}", value])
            else:
                srun_cmd.append(f"--{key}")

    # Build the actual command to run
    if use_bash_wrapper:
        # Build bash command with environment setup
        bash_parts = []

        # Export environment variables
        if env_to_set:
            for name, value in env_to_set.items():
                bash_parts.append(f"export {name}={shlex.quote(value)}")

        # Add preamble if provided. It runs after exports so setup/fingerprint
        # hooks observe the same environment as the main command.
        if bash_preamble:
            bash_parts.append(bash_preamble)

        # Add the main command
        bash_parts.append(shlex.join(command))

        # Join with && for sequential execution
        bash_command = " && ".join(bash_parts)
        srun_cmd.extend(["bash", "-c", bash_command])
    else:
        srun_cmd.extend(command)

    logger.info("srun command: %s", shlex.join(srun_cmd))

    # Start the process
    proc = subprocess.Popen(
        srun_cmd,
        stdout=subprocess.PIPE if not output else None,
        stderr=subprocess.STDOUT if not output else None,
        env=None,  # Inherit environment
    )

    return proc


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
        result = subprocess.run(command, shell=True)
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
