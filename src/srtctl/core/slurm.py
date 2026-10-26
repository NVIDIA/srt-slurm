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

import hashlib
import logging
import os
import shlex
import socket
import subprocess
import threading
import time
from _thread import LockType
from collections.abc import Sequence
from contextlib import ExitStack
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

DEFAULT_CONTAINER_PREPARE_TIMEOUT_SECONDS = 1800.0
_CONTAINER_PREPARE_GUARD = threading.Lock()
_CONTAINER_PREPARE_LOCKS: dict[tuple[str, str], LockType] = {}
_PREPARED_CONTAINER_NODES: set[tuple[str, str]] = set()
_FAILED_CONTAINER_NODES: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class ContainerName:
    """A resolved Pyxis name and whether srtctl owns its preparation."""

    name: str | None
    prepare: bool = False


def _prepare_shared_container(
    *,
    image: str | Path,
    name: str,
    nodes: Sequence[str] | None,
    output: str | None,
    mounts: dict[Path, Path] | None,
    options: dict[str, str] | None,
    export_env: dict[str, str] | None,
    het_group: int | None,
    step_name: str | None,
    timeout_seconds: float | None = None,
) -> None:
    """Prepare missing rootfs copies together; only overlapping names/nodes wait."""
    keys = [(name, node) for node in sorted(set(nodes or ("",)))]
    timeout = DEFAULT_CONTAINER_PREPARE_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    deadline = time.monotonic() + timeout
    with _CONTAINER_PREPARE_GUARD:
        locks = [_CONTAINER_PREPARE_LOCKS.setdefault(key, threading.Lock()) for key in keys]
    try:
        with ExitStack() as stack:
            # Stable acquisition order also handles partially overlapping node sets.
            for lock in locks:
                if not lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
                    raise TimeoutError("timed out waiting for another container preparation")
                stack.callback(lock.release)
            for key in keys:
                if key in _FAILED_CONTAINER_NODES:
                    raise OSError(f"previous preparation failed: {_FAILED_CONTAINER_NODES[key]}")
            pending = [key for key in keys if key not in _PREPARED_CONTAINER_NODES]
            if not pending:
                return
            # Preparation needs one task per node, independent of the worker's
            # MPI rank layout. Reuse the normal launcher for all other options.
            prep_options = {
                key: value
                for key, value in (options or {}).items()
                if key not in {"distribution", "nodes", "ntasks", "ntasks-per-node", "nodelist", "job-name"}
            }
            prep_options.update({"container-name": name, "ntasks-per-node": "1"})
            try:
                proc = start_srun_process(
                    ["true"],
                    nodes=len(pending),
                    ntasks=len(pending),
                    nodelist=[node for _, node in pending] if nodes else None,
                    output=output,
                    container_image=image,
                    container_mounts=mounts,
                    srun_options=prep_options,
                    srun_export_env=export_env,
                    het_group=het_group,
                    step_name=f"{step_name or 'srtctl'}.container-prepare",
                    use_bash_wrapper=False,
                )
                try:
                    proc.communicate(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                    raise
                if proc.returncode:
                    raise subprocess.CalledProcessError(proc.returncode, proc.args)
            except (subprocess.SubprocessError, OSError) as exc:
                # A failed extraction may leave a partial rootfs. Waiting launches
                # must fail too rather than treating its existence as success.
                _FAILED_CONTAINER_NODES.update({key: str(exc) for key in pending})
                raise
            # Without an explicit node list, Slurm may select a different node
            # next time, so that placement cannot be cached.
            if nodes:
                _PREPARED_CONTAINER_NODES.update(pending)
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError(
            f"Container preparation failed for step {step_name or '<unnamed>'}, image {image}, "
            f"nodes {','.join(nodes) if nodes else '<Slurm-selected>'}; log: {output or '<orchestrator output>'}: {exc}"
        ) from exc


def shared_container_name(container_image: str | Path, *, remap_root: bool = False, job_id: str | None = None) -> str:
    """Name a job's shared image rootfs, separating incompatible root-remap modes.

    Adapted from srt-slurm PR #484 (318c460f7846e396a03f701acfd97f67e24ded45).
    The job ID prevents concurrent jobs from attaching to each other's rootfs;
    cleanup of named containers depends on the cluster's Pyxis container scope.
    """
    key = f"{container_image}|remap_root" if remap_root else str(container_image)
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return f"srtctl_{digest}_{job_id or get_slurm_job_id() or 'nojob'}"


def resolve_container_name(
    container_image: str | Path | None,
    *,
    reuse_container: bool = False,
    srun_options: dict[str, str] | None = None,
    srun_export_env: dict[str, str] | None = None,
    job_id: str | None = None,
) -> ContainerName:
    """Resolve an explicit name or the opt-in job/image name used by every launch."""
    if not container_image:
        return ContainerName(None)
    options = srun_options or {}
    explicit_name = options.get("container-name")
    if explicit_name:
        return ContainerName(explicit_name)
    if not reuse_container:
        return ContainerName(None)
    remap_value = (srun_export_env or {}).get("ENROOT_REMAP_ROOT", os.environ.get("ENROOT_REMAP_ROOT", "no"))
    remap_root = remap_value.lower() in {"1", "yes", "true", "y"}
    if "container-remap-root" in options:
        remap_root = True
    if "no-container-remap-root" in options:
        remap_root = False
    # Share the rootfs while giving each step its own mounts and namespaces.
    return ContainerName(
        f"{shared_container_name(container_image, remap_root=remap_root, job_id=job_id)}:no_exec", True
    )


def start_srun_process(
    command: list[str],
    *,
    nodes: int = 1,
    ntasks: int = 1,
    cpus_per_task: int | None = None,
    nodelist: Sequence[str] | None = None,
    output: str | None = None,
    container_image: str | Path | None = None,
    container_mounts: dict[Path, Path] | None = None,
    reuse_container: bool = False,
    container_prepare_nodes: Sequence[str] | None = None,
    container_prepare_timeout_seconds: float | None = None,
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
        reuse_container: Reuse one named Pyxis rootfs per job/image/remap mode (default: False).
        container_prepare_nodes: Prepare this image on these nodes together before its first launch.
            Defaults to nodelist. Later launches reuse the completed preparation.
        container_prepare_timeout_seconds: Combined preparation and lock-wait budget.
            Job launches pass their resolved health-check budget; standalone callers default to 30 minutes.
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
        srun_cmd.append(f"--cpu-bind={cpu_bind}")

    # Arbitrary layouts derive their node count from the repeated host list.
    if not srun_options or srun_options.get("distribution") != "arbitrary":
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
        # Stream task output with O_APPEND so another process on the shared
        # filesystem can inject lines into a live log without srun overwriting
        # them on its next write — this is what lets the benchmark mark its
        # stages inside the worker logs. Log files are unique per job, so append
        # behaves like truncate on the first open.
        srun_cmd.extend(["--open-mode", "append"])

    # Container options
    resolved_container_name = resolve_container_name(
        container_image,
        reuse_container=reuse_container,
        srun_options=srun_options,
        srun_export_env=srun_export_env,
        job_id=slurm_job_id,
    )
    if container_image:
        srun_cmd.extend(["--container-image", str(container_image)])
        if resolved_container_name.name:
            srun_cmd.append(f"--container-name={resolved_container_name.name}")
        srun_cmd.append("--no-container-entrypoint")
        srun_cmd.append("--no-container-mount-home")

        if container_mounts:
            mount_str = ",".join(f"{host}:{container}" for host, container in container_mounts.items())
            srun_cmd.extend(["--container-mounts", mount_str])

    if srun_options:
        for key, value in srun_options.items():
            if key == "container-name" and resolved_container_name.name:
                continue
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

    if resolved_container_name.prepare:
        assert container_image is not None and resolved_container_name.name is not None
        _prepare_shared_container(
            image=container_image,
            name=resolved_container_name.name,
            nodes=container_prepare_nodes or nodelist,
            output=output,
            mounts=container_mounts,
            options=srun_options,
            export_env=srun_export_env,
            het_group=het_group,
            step_name=step_name,
            timeout_seconds=container_prepare_timeout_seconds,
        )

    if step_name:
        srun_cmd.append(f"--job-name={step_name}")

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

        # exec the main command so it replaces bash as the step's task: srun forwards
        # SIGTERM to the task, and a bash -c parent would hold the signal until its
        # child exited, so the child was only ever SIGKILLed at the cleanup timeout
        # (tachometer never compacted, workers never shut down cleanly).
        bash_parts.append("exec " + shlex.join(command))

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

    # Demoted to debug — every worker srun line is multi-KB once the
    # fingerprint heredoc is inlined (see core/fingerprint.generate_capture_script),
    # which dominates the orchestrator log. Re-enable with `--verbose` / by setting
    # the srtctl logger to DEBUG when troubleshooting srun arg construction.
    logger.debug("srun command: %s", shlex.join(srun_cmd))

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
