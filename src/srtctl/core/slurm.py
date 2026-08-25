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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SrunSpec:
    """Declarative inputs for one ``srun`` launch.

    Stage builders return this object so normal execution and launch-script
    rendering consume the same command definition.
    """

    command: tuple[str, ...]
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int | None = None
    nodelist: tuple[str, ...] | None = None
    output: str | None = None
    container_image: str | None = None
    container_mounts: dict[Path, Path] | None = None
    env_to_pass_through: tuple[str, ...] | None = None
    env_to_set: dict[str, str] | None = None
    env_to_unset: tuple[str, ...] | None = None
    bash_preamble: str | None = None
    srun_options: dict[str, str] | None = None
    srun_export_env: dict[str, str] | None = None
    overlap: bool = True
    use_bash_wrapper: bool = True
    mpi: str | None = None
    oversubscribe: bool = False
    cpu_bind: str | None = None
    het_group: int | None = None

    def argv(self, *, job_id: str | None = None) -> list[str]:
        """Render this launch for a specific allocation or portable replay."""
        return build_srun_command(
            list(self.command),
            job_id=job_id,
            nodes=self.nodes,
            ntasks=self.ntasks,
            cpus_per_task=self.cpus_per_task,
            nodelist=self.nodelist,
            output=self.output,
            container_image=self.container_image,
            container_mounts=self.container_mounts,
            env_to_pass_through=list(self.env_to_pass_through) if self.env_to_pass_through else None,
            env_to_set=self.env_to_set,
            env_to_unset=list(self.env_to_unset) if self.env_to_unset else None,
            bash_preamble=self.bash_preamble,
            srun_options=self.srun_options,
            srun_export_env=self.srun_export_env,
            overlap=self.overlap,
            use_bash_wrapper=self.use_bash_wrapper,
            mpi=self.mpi,
            oversubscribe=self.oversubscribe,
            cpu_bind=self.cpu_bind,
            het_group=self.het_group,
        )


def start_srun_spec(
    spec: SrunSpec,
    *,
    launcher: Callable[..., subprocess.Popen] | None = None,
) -> subprocess.Popen:
    """Launch a stage-built spec in the current allocation."""
    launch = launcher or start_srun_process
    return launch(
        list(spec.command),
        nodes=spec.nodes,
        ntasks=spec.ntasks,
        cpus_per_task=spec.cpus_per_task,
        nodelist=spec.nodelist,
        output=spec.output,
        container_image=spec.container_image,
        container_mounts=spec.container_mounts,
        env_to_pass_through=list(spec.env_to_pass_through) if spec.env_to_pass_through else None,
        env_to_set=spec.env_to_set,
        env_to_unset=list(spec.env_to_unset) if spec.env_to_unset else None,
        bash_preamble=spec.bash_preamble,
        srun_options=spec.srun_options,
        srun_export_env=spec.srun_export_env,
        overlap=spec.overlap,
        use_bash_wrapper=spec.use_bash_wrapper,
        mpi=spec.mpi,
        oversubscribe=spec.oversubscribe,
        cpu_bind=spec.cpu_bind,
        het_group=spec.het_group,
    )


def build_srun_command(
    command: list[str],
    *,
    job_id: str | None = None,
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
) -> list[str]:
    """Build an ``srun`` argv without launching it.

    Keeping construction pure lets runtime execution and portable launch-script
    rendering share the exact same command path. ``job_id`` is explicit: pass
    the current allocation ID for normal execution, or ``None`` for a command
    that inherits whichever allocation is active when it is run.

    Args:
        command: Command to run as list of strings
        job_id: Optional allocation ID to pass via ``--jobid``.
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
        Fully constructed ``srun`` argv.

    Example:
        command = build_srun_command(
            command=["python3", "-m", "dynamo.sglang", "--model-path", "/model"],
            job_id=None,
            nodelist=["node1"],
            container_image="/containers/sglang.sqsh",
            container_mounts={Path("/models/llama"): Path("/model")},
            env_to_set={"NATS_SERVER": "nats://node1:4222"},
        )
    """
    srun_cmd = ["srun"]

    # ensures srun runs in the same job context
    if job_id:
        srun_cmd.extend(["--jobid", job_id])

    # Basic options
    if overlap:
        srun_cmd.append("--overlap")

    # MPI options (for TRTLLM)
    if mpi:
        srun_cmd.extend(["--mpi", mpi])
    if oversubscribe:
        srun_cmd.append("--oversubscribe")
    if cpu_bind:
        srun_cmd.append(f"--cpu-bind={cpu_bind}")

    srun_cmd.extend(["--nodes", str(nodes)])
    srun_cmd.extend(["--ntasks", str(ntasks)])

    if cpus_per_task:
        srun_cmd.extend(["--cpus-per-task", str(cpus_per_task)])

    if nodelist:
        srun_cmd.extend(["--nodelist", ",".join(nodelist)])

    # Route this srun to a specific component of a SLURM heterogeneous job.
    # Omitted (None) for non-het jobs; safe to always pass-through from callers.
    if het_group is not None:
        srun_cmd.append(f"--het-group={het_group}")

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

    if srun_options:
        for key, value in srun_options.items():
            if value:
                srun_cmd.append(f"--{key}={value}")
            else:
                srun_cmd.append(f"--{key}")

    # Set env vars in the task environment so the container runtime (enroot/pyxis)
    # sees them at container-creation time. Prefix ALL to preserve srun's normal
    # full-environment propagation and only add these on top.
    if srun_export_env:
        exports = ",".join(f"{k}={v}" for k, v in srun_export_env.items())
        srun_cmd.append(f"--export=ALL,{exports}")

    # Build the actual command to run
    if use_bash_wrapper:
        # Build bash command with environment setup
        bash_parts = []

        # Export environment variables
        if env_to_set:
            for name, value in env_to_set.items():
                bash_parts.append(f"export {name}={shlex.quote(value)}")

        # Cluster-wide preamble (e.g. ulimits) runs first so it applies to
        # exports, the local preamble, and the main command alike.
        cluster_preamble = _get_cluster_bash_preamble()
        if cluster_preamble:
            bash_parts.insert(0, cluster_preamble)

        # Explicitly clear inherited variables after setting worker-specific
        # values so the preamble and main command see the intended environment.
        if env_to_unset:
            for name in env_to_unset:
                bash_parts.append(f"unset -- {shlex.quote(name)}")

        # Add per-call preamble if provided. It runs after exports/unsets so
        # setup / fingerprint hooks observe the same environment as the main command.
        if bash_preamble:
            bash_parts.append(bash_preamble)

        # Add the main command
        bash_parts.append(shlex.join(command))

        # Join with && for sequential execution
        bash_command = " && ".join(bash_parts)
        srun_cmd.extend(["bash", "-c", bash_command])
    else:
        cluster_preamble = _get_cluster_bash_preamble()
        if cluster_preamble:
            logger.warning(
                "Cluster default_bash_preamble is set but this srun bypasses the bash wrapper "
                "(use_bash_wrapper=False); preamble will not be applied. command=%s",
                shlex.join(command),
            )
        srun_cmd.extend(command)

    return srun_cmd


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
) -> subprocess.Popen:
    """Build and start an ``srun`` child in the current allocation."""
    srun_cmd = build_srun_command(
        command,
        job_id=get_slurm_job_id(),
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
    )

    # Every worker srun line is multi-KB once fingerprint capture is inlined.
    logger.debug("srun command: %s", shlex.join(srun_cmd))
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
