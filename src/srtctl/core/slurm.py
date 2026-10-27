# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SLURM utilities for job management and process launching.

This module consolidates all SLURM-related functionality:
- Environment: get_slurm_job_id, get_slurm_nodelist, get_batch_host
- Allocation: shrink_job_allocation
- Network: get_hostname_ip, get_node_ips
- Process launching: start_srun_process, run_command
- Container utilities: get_container_mounts_str
"""

import hashlib
import logging
import os
import re
import shlex
import socket
import subprocess
import tempfile
import threading
import time
from _thread import LockType
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .ip_utils import get_local_ip, get_node_ip

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


def get_batch_host() -> str | None:
    """Node running the batch script: ``SLURMD_NODENAME``, else the first allocated node.

    Slurm runs the batch script on the first node of a non-het allocation, so
    the fallback agrees with ``SLURMD_NODENAME`` whenever both are available.
    Returns None outside a Slurm job.
    """
    node = os.environ.get("SLURMD_NODENAME")
    if node:
        return node
    nodelist = get_slurm_nodelist()
    return nodelist[0] if nodelist and nodelist[0] else None


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
# Allocation Shrink
# ============================================================================

# Variables Slurm's resize script sets to the new size, each only when it was already set.
_RESIZE_NODELIST_VARS = ("SLURM_NODELIST", "SLURM_JOB_NODELIST")
_RESIZE_NODE_COUNT_VARS = ("SLURM_NNODES", "SLURM_JOB_NUM_NODES")
# Unset when SLURM_TASKS_PER_NODE is set: scontrol cannot recompute them for the new size.
_RESIZE_UNSET_VARS = ("SLURM_NPROCS", "SLURM_NTASKS", "SLURM_TASKS_PER_NODE")
# Every name scontrol's resize script may set or unset (src/scontrol/update_job.c _update_job_size);
# any other name in the file is ignored.
_RESIZE_SCRIPT_VARS = frozenset(
    (*_RESIZE_NODELIST_VARS, *_RESIZE_NODE_COUNT_VARS, "SLURM_JOB_CPUS_PER_NODE", *_RESIZE_UNSET_VARS)
)
_RESIZE_EXPORT_RE = re.compile(r'^export\s+([A-Za-z_][A-Za-z0-9_]*)=(?:"([^"]*)"|(\S*))\s*$')
_RESIZE_UNSET_RE = re.compile(r"^unset\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")
# One group of SLURM_JOB_CPUS_PER_NODE: "64" or "64(x3)".
_CPU_GROUP_RE = re.compile(r"^(\d+)(?:\(x(\d+)\))?$")


@dataclass(frozen=True)
class AllocationShrink:
    """Outcome of ``shrink_job_allocation``.

    Attributes:
        requested: ``scontrol update`` was invoked (False when the shrink was skipped).
        succeeded: scontrol accepted the new node list.
        retained: The node the job was asked to keep.
        released: Nodes given back to the scheduler; empty unless ``succeeded``.
        verified: ``squeue`` reported exactly ``retained`` after the update.
        reason: Skip reason, scontrol error, or verification warning; None on a clean verified success.
        duration_s: Wall time spent in ``shrink_job_allocation``.
    """

    requested: bool
    succeeded: bool
    retained: str
    released: tuple[str, ...]
    verified: bool
    reason: str | None
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, ``released`` as a list."""
        data = asdict(self)
        data["released"] = list(self.released)
        return data


def _parse_resize_script(path: Path) -> list[tuple[str, str | None]] | None:
    """Parse scontrol's ``slurm_job_<id>_resize.sh`` into (name, value) pairs; value None means unset.

    The file is parsed, never executed. Recognises ``export K="V"``, ``export K=V``
    and ``unset K`` for the variables scontrol writes (``_RESIZE_SCRIPT_VARS``) and
    ignores every other line and name. Returns None when the file is missing,
    unreadable, or holds no accepted line, so the caller applies the equivalent
    update itself.
    """
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return None
    ops: list[tuple[str, str | None]] = []
    for line in text.splitlines():
        line = line.strip()
        if (match := _RESIZE_EXPORT_RE.match(line)) and match.group(1) in _RESIZE_SCRIPT_VARS:
            value = match.group(2) if match.group(2) is not None else match.group(3)
            ops.append((match.group(1), value))
        elif (match := _RESIZE_UNSET_RE.match(line)) and match.group(1) in _RESIZE_SCRIPT_VARS:
            ops.append((match.group(1), None))
    return ops or None


def _expand_cpus_per_node(compressed: str) -> list[str]:
    """Expand Slurm's compressed per-node count, ``"72,64(x3)"`` -> ``["72", "64", "64", "64"]``."""
    counts: list[str] = []
    for group in compressed.split(","):
        match = _CPU_GROUP_RE.match(group.strip())
        if match is None:
            raise ValueError(f"unrecognised SLURM_JOB_CPUS_PER_NODE group {group!r}")
        counts.extend([match.group(1)] * int(match.group(2) or 1))
    return counts


def _kept_node_cpus(compressed: str, nodelist: Sequence[str], keep: str) -> str:
    """The kept node's entry in ``SLURM_JOB_CPUS_PER_NODE``, which lists counts in nodelist order.

    Falls back to the first group's count when the value does not expand to one
    entry per node.
    """
    first = compressed.split(",")[0].split("(")[0].strip()
    try:
        counts = _expand_cpus_per_node(compressed)
    except ValueError:
        return first
    if len(counts) != len(nodelist) or keep not in nodelist:
        return first
    return counts[list(nodelist).index(keep)]


def _fallback_resize_ops(keep: str, nodelist: Sequence[str]) -> list[tuple[str, str | None]]:
    """The update Slurm's resize script makes for a one-node allocation, computed from ``os.environ``.

    ``nodelist`` is the allocation before the shrink, used to find the kept node's CPU count.
    """
    ops: list[tuple[str, str | None]] = []
    for name in _RESIZE_NODELIST_VARS:
        if name in os.environ:
            ops.append((name, keep))
    for name in _RESIZE_NODE_COUNT_VARS:
        if name in os.environ:
            ops.append((name, "1"))
    cpus = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if cpus is not None:
        ops.append(("SLURM_JOB_CPUS_PER_NODE", _kept_node_cpus(cpus, nodelist, keep)))
    if "SLURM_TASKS_PER_NODE" in os.environ:
        ops.extend((name, None) for name in _RESIZE_UNSET_VARS)
    return ops


def _apply_resize(script: Path, keep: str, nodelist: Sequence[str]) -> None:
    """Apply scontrol's resize script to ``os.environ``, or the equivalent fallback when it is absent.

    scontrol exits 0 even when it could not write the script, and writes none
    without ``SLURM_JOBID`` in its environment, so a missing file is expected.
    """
    ops = _parse_resize_script(script)
    for name, value in ops if ops is not None else _fallback_resize_ops(keep, nodelist):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _squeue_nodelist(job_id: str, timeout: float) -> list[str] | None:
    """The job's current nodes from ``squeue -h -j <id> -o %N``, or None when squeue fails."""
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%N"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            return None
        raw = (result.stdout or "").strip()
        return _expand_nodelist(raw) if raw else None
    except Exception:  # noqa: BLE001 - verification is advisory
        return None


def shrink_job_allocation(job_id: str, keep: str, *, timeout: float = 30.0) -> AllocationShrink:
    """Shrink a running job to the single node ``keep``, returning the other nodes to the scheduler.

    Runs ``scontrol update JobId=<id> NodeList=<keep>`` from a scratch directory
    (scontrol writes ``slurm_job_<id>_resize.{sh,csh}`` into its working
    directory), checks the result with ``squeue``, then applies that resize
    script to ``os.environ`` so every later srun inherits the new node list and
    counts. Slurm refuses the update under ``SchedulerParameters=disable_job_shrink``,
    when ``keep`` is not the batch host, or for arbitrary task distribution; the
    environment is then left untouched. Never raises.

    ``squeue`` decides when it answers: exactly ``[keep]`` is a verified release
    and any other node list is a failure, because slurmctld accepts an update to
    the job's submitted ``--nodelist`` as a no-op (job_mgr.c:13314-13317). When
    squeue cannot answer, scontrol's exit 0 stands as an unverified release.
    After a scontrol timeout only an exact ``[keep]`` counts, since slurmctld may
    have applied the shrink before the client gave up.
    """
    start = time.monotonic()
    requested = False

    def result(
        *,
        succeeded: bool = False,
        released: tuple[str, ...] = (),
        verified: bool = False,
        reason: str | None,
    ) -> AllocationShrink:
        return AllocationShrink(
            requested=requested,
            succeeded=succeeded,
            retained=keep,
            released=released,
            verified=verified,
            reason=reason,
            duration_s=time.monotonic() - start,
        )

    try:
        current = get_slurm_nodelist()
        if keep not in current:
            return result(reason=f"{keep} is not in the job's nodelist ({','.join(current) or 'empty'})")
        if current == [keep]:
            return result(reason=f"allocation is already only {keep}; nothing to release")
        released = tuple(node for node in current if node != keep)

        with tempfile.TemporaryDirectory(prefix="srtctl-resize-", ignore_cleanup_errors=True) as scratch:
            resize_script = Path(scratch) / f"slurm_job_{job_id}_resize.sh"
            requested = True
            try:
                proc = subprocess.run(
                    ["scontrol", "update", f"JobId={job_id}", f"NodeList={keep}"],
                    cwd=scratch,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                if _squeue_nodelist(job_id, timeout) != [keep]:
                    return result(reason=f"scontrol update timed out after {timeout:g}s")
                _apply_resize(resize_script, keep, current)
                return result(
                    succeeded=True,
                    released=released,
                    verified=True,
                    reason=f"scontrol update timed out after {timeout:g}s, but squeue shows the job on {keep} only",
                )
            except OSError as e:
                return result(reason=f"scontrol update failed to run: {e}")
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip() or "no output"
                return result(reason=f"scontrol update exited {proc.returncode}: {detail}")

            nodes_after = _squeue_nodelist(job_id, timeout)
            if nodes_after is not None and nodes_after != [keep]:
                return result(
                    reason=f"scontrol update exited 0 but squeue still shows {','.join(nodes_after)}, expected {keep}"
                )
            _apply_resize(resize_script, keep, current)
    except Exception as e:  # noqa: BLE001 - the caller is a cleanup path
        return result(reason=f"shrink failed: {e}")

    if nodes_after is None:
        return result(succeeded=True, released=released, reason="squeue could not confirm the new nodelist")
    return result(succeeded=True, released=released, verified=True, reason=None)


# ============================================================================
# Network Resolution
# ============================================================================


def _is_loopback_ip(ip: str) -> bool:
    """True for IPv4 loopback, including Ubuntu's 127.0.1.1 hostname alias."""
    return ip.startswith("127.")


def _hostname_keys(hostname: str) -> set[str]:
    """Short and FQDN spellings of a hostname, for local-node comparison."""
    name = hostname.strip().lower()
    if not name:
        return set()
    return {name, name.split(".", 1)[0]}


def _is_local_hostname(hostname: str) -> bool:
    """True when ``hostname`` names this process's node."""
    local_names = (
        socket.gethostname(),
        socket.getfqdn(),
        os.environ.get("SLURMD_NODENAME", ""),
        os.environ.get("HOSTNAME", ""),
    )
    local_keys: set[str] = set()
    for name in local_names:
        local_keys |= _hostname_keys(name)
    return bool(_hostname_keys(hostname) & local_keys)


def get_hostname_ip(hostname: str, network_interface: str | None = None) -> str:
    """Resolve hostname to a routable IP address.

    Inside a SLURM job, query the target node with srun. If that fails and the
    hostname is this node, use get_local_ip (ip route / hostname -I). DNS via
    socket.gethostbyname is last and is ignored when it returns loopback:
    Ubuntu /etc/hosts often maps the node's own name to 127.0.1.1, which peers
    cannot use as a torch TCPStore address.

    Args:
        hostname: Node hostname to resolve
        network_interface: Optional network interface to prefer

    Returns:
        Routable IP address as string

    Raises:
        RuntimeError: If the only address found is loopback
    """
    slurm_job_id = get_slurm_job_id()
    if slurm_job_id:
        ip = get_node_ip(hostname, slurm_job_id, network_interface)
        if ip and not _is_loopback_ip(ip):
            return ip
        logger.warning(
            "srun-based IP resolution failed for %s, trying local/socket fallbacks",
            hostname,
        )

    if _is_local_hostname(hostname):
        ip = get_local_ip(network_interface)
        if ip and not _is_loopback_ip(ip):
            return ip

    try:
        ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        # Return hostname as-is (may be IP already)
        return hostname
    if not _is_loopback_ip(ip):
        return ip

    raise RuntimeError(f"Could not resolve a routable IP for {hostname}; refusing loopback")


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
