# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the orchestrator's processes run: a Slurm allocation, or Docker on this machine.

The ``SweepOrchestrator`` asks a ``Launcher`` for everything it used to ask Slurm
for directly: start a process (``launch``), signal a named step (``signal_step``),
list running steps (``list_step_ids``), the job's nodes (``nodes``) and their
addresses (``node_ip``), the job id (``job_id``), and how a rendered job script is
submitted (``submit`` / ``start``). ``SlurmLauncher`` is the default and keeps the
srun/sbatch/scancel behavior; ``DockerLauncher`` (``core/docker.py``) runs one
job on the current machine, with containers under ``docker run``.

The launcher is picked by the ``launcher`` key of ``srtslurm.yaml``
(``get_launcher``). Nothing outside the launcher modules branches on that name.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from srtctl.core import slurm as _slurm

if TYPE_CHECKING:
    from srtctl.core.runtime import Nodes, RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

DEFAULT_LAUNCHER = "slurm"


@dataclass(frozen=True)
class LaunchSpec:
    """One process launch, independent of where it runs. Fields mirror ``start_srun_process``."""

    command: list[str]
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int | None = None
    nodelist: Sequence[str] | None = None
    output: str | None = None
    container_image: str | Path | None = None
    container_mounts: dict[Path, Path] | None = None
    # Slurm/pyxis: share one named rootfs per job/image/remap mode across launches, and
    # prepare it on these nodes before the first launch (see ``start_srun_process``).
    reuse_container: bool = False
    container_prepare_nodes: Sequence[str] | None = None
    container_prepare_timeout_seconds: float | None = None
    env_to_pass_through: list[str] | None = None
    env_to_set: dict[str, str] | None = None
    env_to_unset: list[str] | None = None
    bash_preamble: str | None = None
    srun_options: dict[str, str] | None = None
    srun_export_env: dict[str, str] | None = None
    overlap: bool = True
    use_bash_wrapper: bool = True
    mpi: str | None = None
    oversubscribe: bool = False
    cpu_bind: str | None = None
    het_group: int | None = None
    step_name: str | None = None


class Launcher(ABC):
    """How and where the orchestrator's processes run."""

    name: ClassVar[str]

    def build_task_command(self, spec: LaunchSpec) -> list[str]:
        """The command the task runs: the bash wrapper around ``spec.command``, or the command itself.

        The wrapper exports ``env_to_set``, runs the cluster ``default_bash_preamble``,
        unsets ``env_to_unset``, runs the per-call preamble, then ``exec``s the command
        so it replaces bash and receives SIGTERM directly: a ``bash -c`` parent would
        hold the signal until its child exited, so the child was only ever SIGKILLed
        at the cleanup timeout.
        """
        cluster_preamble = _slurm._get_cluster_bash_preamble()
        if not spec.use_bash_wrapper:
            if cluster_preamble:
                logger.warning(
                    "Cluster default_bash_preamble is set but this launch bypasses the bash wrapper "
                    "(use_bash_wrapper=False); preamble will not be applied. command=%s",
                    shlex.join(spec.command),
                )
            return list(spec.command)

        bash_parts = []
        if cluster_preamble:
            bash_parts.append(cluster_preamble)
        if spec.env_to_set:
            for name, value in spec.env_to_set.items():
                bash_parts.append(f"export {name}={shlex.quote(value)}")
        if spec.env_to_unset:
            for name in spec.env_to_unset:
                bash_parts.append(f"unset -- {shlex.quote(name)}")
        if spec.bash_preamble:
            bash_parts.append(spec.bash_preamble)
        bash_parts.append("exec " + shlex.join(spec.command))
        return ["bash", "-c", " && ".join(bash_parts)]

    @abstractmethod
    def launch(self, spec: LaunchSpec) -> subprocess.Popen:
        """Start ``spec`` and return the local process that tracks it."""

    @abstractmethod
    def list_step_ids(
        self, job_id: str | None = None, *, unique_names: set[str] | None = None
    ) -> dict[str, str] | None:
        """``{step name: id}`` for this job's running named steps; None when it cannot be asked.

        ``unique_names`` are step names that must occur once; a duplicate makes the
        listing ambiguous and the result None.
        """

    @abstractmethod
    def signal_step(
        self, step_name: str, sig: str = "TERM", *, step_ids: dict[str, str] | None = None, full: bool = True
    ) -> bool:
        """Deliver ``sig`` to the task of the step named ``step_name``; True when delivered.

        False tells the caller to signal the tracking process directly.
        """

    @abstractmethod
    def nodes(
        self,
        *,
        frontend_dedicated_node: bool = False,
        client_dedicated_node: bool = False,
        etcd_nats_dedicated_node: bool = False,
        colocate_dedicated_nodes: bool = True,
        engine_nodes: int | None = None,
        pools: Sequence[tuple[str, int]] = (),
    ) -> Nodes:
        """The job's nodes, carved into head, infra, worker and pool roles."""

    @abstractmethod
    def node_ip(self, hostname: str, network_interface: str | None = None) -> str:
        """The address other processes in the job use to reach ``hostname``."""

    @abstractmethod
    def job_id(self) -> str | None:
        """The running job's id, or None outside a job."""

    @abstractmethod
    def submit(self, script_path: Path) -> str:
        """Hand the rendered job script over and return the new job id."""

    def start(self, job_id: str, job_output_dir: Path) -> int | None:
        """Run a submitted job once its output directory is staged; None when the scheduler runs it."""
        return None

    @abstractmethod
    def shell_command(
        self, runtime: RuntimeContext, node: str, *, image: str | None = None, container_name: str | None = None
    ) -> str:
        """A command a user can paste to get a shell next to the job's processes on ``node``.

        ``image`` and ``container_name`` are the worker's container when they differ from
        the job's (role images, shared pyxis rootfs); None means the job defaults.
        """

    @abstractmethod
    def status_hint(self, job_id: str) -> str:
        """A command that shows whether ``job_id`` is still running."""

    def validate(self, config: SrtConfig) -> list[str]:
        """Reasons ``config`` cannot run under this launcher; empty when it can."""
        return []


class SlurmLauncher(Launcher):
    """Processes are Slurm steps (``srun``) inside an ``sbatch`` allocation."""

    name = "slurm"

    def launch(self, spec: LaunchSpec) -> subprocess.Popen:
        srun_cmd = ["srun"]

        # Run in the same job context.
        slurm_job_id = _slurm.get_slurm_job_id()
        if slurm_job_id:
            srun_cmd.extend(["--jobid", slurm_job_id])

        if spec.overlap:
            srun_cmd.append("--overlap")

        # MPI options (for TRTLLM)
        if spec.mpi:
            srun_cmd.extend(["--mpi", spec.mpi])
        if spec.oversubscribe:
            srun_cmd.append("--oversubscribe")
        if spec.cpu_bind:
            srun_cmd.append(f"--cpu-bind={spec.cpu_bind}")

        # Arbitrary layouts derive their node count from the repeated host list.
        if not spec.srun_options or spec.srun_options.get("distribution") != "arbitrary":
            srun_cmd.extend(["--nodes", str(spec.nodes)])
        srun_cmd.extend(["--ntasks", str(spec.ntasks)])

        if spec.cpus_per_task:
            srun_cmd.extend(["--cpus-per-task", str(spec.cpus_per_task)])

        if spec.nodelist:
            srun_cmd.extend(["--nodelist", ",".join(spec.nodelist)])

        # Route this srun to a specific component of a SLURM heterogeneous job.
        if spec.het_group is not None:
            srun_cmd.append(f"--het-group={spec.het_group}")

        if spec.output:
            srun_cmd.extend(["--output", spec.output])
            # Stream task output with O_APPEND so another process on the shared
            # filesystem can inject lines into a live log without srun overwriting
            # them on its next write — this is what lets the benchmark mark its
            # stages inside the worker logs. Log files are unique per job, so append
            # behaves like truncate on the first open.
            srun_cmd.extend(["--open-mode", "append"])

        # Container options
        resolved_container_name = _slurm.resolve_container_name(
            spec.container_image,
            reuse_container=spec.reuse_container,
            srun_options=spec.srun_options,
            srun_export_env=spec.srun_export_env,
            job_id=slurm_job_id,
        )
        if spec.container_image:
            srun_cmd.extend(["--container-image", str(spec.container_image)])
            if resolved_container_name.name:
                srun_cmd.append(f"--container-name={resolved_container_name.name}")
            srun_cmd.append("--no-container-entrypoint")
            srun_cmd.append("--no-container-mount-home")
            if spec.container_mounts:
                mount_str = ",".join(f"{host}:{container}" for host, container in spec.container_mounts.items())
                srun_cmd.extend(["--container-mounts", mount_str])

        if spec.srun_options:
            for key, value in spec.srun_options.items():
                if key == "container-name" and resolved_container_name.name:
                    continue
                if value:
                    srun_cmd.append(f"--{key}={value}")
                else:
                    srun_cmd.append(f"--{key}")

        # Set env vars in the task environment so the container runtime (enroot/pyxis)
        # sees them at container-creation time. Prefix ALL to preserve srun's normal
        # full-environment propagation and only add these on top.
        if spec.srun_export_env:
            exports = ",".join(f"{k}={v}" for k, v in spec.srun_export_env.items())
            srun_cmd.append(f"--export=ALL,{exports}")

        if resolved_container_name.prepare:
            assert spec.container_image is not None and resolved_container_name.name is not None
            _slurm._prepare_shared_container(
                image=spec.container_image,
                name=resolved_container_name.name,
                nodes=spec.container_prepare_nodes or spec.nodelist,
                output=spec.output,
                mounts=spec.container_mounts,
                options=spec.srun_options,
                export_env=spec.srun_export_env,
                het_group=spec.het_group,
                step_name=spec.step_name,
                timeout_seconds=spec.container_prepare_timeout_seconds,
            )

        if spec.step_name:
            srun_cmd.append(f"--job-name={spec.step_name}")

        srun_cmd.extend(self.build_task_command(spec))

        # Demoted to debug — every worker srun line is multi-KB once the
        # fingerprint heredoc is inlined (see core/fingerprint.generate_capture_script),
        # which dominates the orchestrator log. Re-enable with `--verbose` / by setting
        # the srtctl logger to DEBUG when troubleshooting srun arg construction.
        logger.debug("srun command: %s", shlex.join(srun_cmd))

        return subprocess.Popen(
            srun_cmd,
            stdout=subprocess.PIPE if not spec.output else None,
            stderr=subprocess.STDOUT if not spec.output else None,
            env=None,  # Inherit environment
        )

    def list_step_ids(
        self, job_id: str | None = None, *, unique_names: set[str] | None = None
    ) -> dict[str, str] | None:
        job_id = job_id or _slurm.get_slurm_job_id()
        if not job_id or shutil.which("squeue") is None:
            return None
        try:
            result = subprocess.run(
                ["squeue", "--steps", f"--jobs={job_id}", "--noheader", "--format=%i %j"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("squeue --steps failed: %s", exc)
            return None
        if result.returncode != 0:
            logger.warning("squeue --steps exited %d: %s", result.returncode, result.stderr.strip())
            return None
        steps: dict[str, str] = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                if unique_names is not None and parts[1] in unique_names and parts[1] in steps:
                    logger.warning("Ambiguous Slurm step name %s", parts[1])
                    return None
                steps.setdefault(parts[1], parts[0])
        return steps

    def signal_step(
        self, step_name: str, sig: str = "TERM", *, step_ids: dict[str, str] | None = None, full: bool = True
    ) -> bool:
        """``scancel --signal=<sig> [--full] <job>.<step>``.

        ``srun`` turns a SIGTERM aimed at itself into a step abort that SIGKILLs the
        task, so a process that must flush on SIGTERM has to be signalled through
        Slurm. With ``full=False`` only the tasks receive the signal.
        """
        if shutil.which("scancel") is None:
            return False  # not under Slurm (tests, the mock): the caller signals srun directly
        if step_ids is not None:
            step_id = step_ids.get(step_name)
        else:
            steps = self.list_step_ids()
            step_id = steps.get(step_name) if steps else None
        if step_id is None:
            logger.warning("No running step named %s found; falling back to signalling srun", step_name)
            return False
        try:
            result = subprocess.run(
                ["scancel", f"--signal={sig}", *(["--full"] if full else []), step_id],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("scancel --signal=%s %s failed: %s", sig, step_id, exc)
            return False
        if result.returncode != 0:
            logger.warning(
                "scancel --signal=%s %s exited %d: %s", sig, step_id, result.returncode, result.stderr.strip()
            )
            return False
        logger.info("Sent SIG%s to step %s (%s)", sig, step_id, step_name)
        return True

    def nodes(
        self,
        *,
        frontend_dedicated_node: bool = False,
        client_dedicated_node: bool = False,
        etcd_nats_dedicated_node: bool = False,
        colocate_dedicated_nodes: bool = True,
        engine_nodes: int | None = None,
        pools: Sequence[tuple[str, int]] = (),
    ) -> Nodes:
        from srtctl.core.runtime import Nodes

        return Nodes.from_slurm(
            frontend_dedicated_node=frontend_dedicated_node,
            client_dedicated_node=client_dedicated_node,
            etcd_nats_dedicated_node=etcd_nats_dedicated_node,
            colocate_dedicated_nodes=colocate_dedicated_nodes,
            engine_nodes=engine_nodes,
            pools=pools,
        )

    def node_ip(self, hostname: str, network_interface: str | None = None) -> str:
        return _slurm.resolve_slurm_hostname_ip(hostname, network_interface)

    def job_id(self) -> str | None:
        return _slurm.get_slurm_job_id()

    def submit(self, script_path: Path) -> str:
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True, check=True)
        return result.stdout.strip().split()[-1]

    def shell_command(
        self, runtime: RuntimeContext, node: str, *, image: str | None = None, container_name: str | None = None
    ) -> str:
        args = [f"--container-image={image or runtime.container_image}"]
        if container_name:
            args.append(f"--container-name={container_name}")
        if runtime.container_mounts:
            mounts = ",".join(f"{src}:{dst}" for src, dst in runtime.container_mounts.items())
            args.append(f"--container-mounts={mounts}")
        return f"srun {shlex.join(args)} --jobid {runtime.job_id} -w {node} --overlap --pty bash"

    def status_hint(self, job_id: str) -> str:
        return f"squeue --job {job_id}"


@cache
def launchers() -> dict[str, type[Launcher]]:
    """Every launcher by its ``srtslurm.yaml`` name."""
    from srtctl.core.docker import DockerLauncher

    return {cls.name: cls for cls in (SlurmLauncher, DockerLauncher)}


@cache
def _launcher(name: str) -> Launcher:
    return launchers()[name]()


def get_launcher() -> Launcher:
    """The launcher named by ``launcher`` in ``srtslurm.yaml`` (default ``slurm``)."""
    from srtctl.core.config import get_srtslurm_setting

    name = get_srtslurm_setting("launcher") or DEFAULT_LAUNCHER
    if name not in launchers():
        raise ValueError(f"srtslurm.yaml launcher: {name!r} is not one of {', '.join(sorted(launchers()))}")
    return _launcher(name)
