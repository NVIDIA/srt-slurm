# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every single-machine launcher shares: one job on the machine that ran ``srtctl apply``.

``LocalLauncher`` owns the step registry, process-group signalling, placement on this
host, foreground job execution and the checks that keep a recipe on one machine.
``DockerLauncher`` (``core/docker.py``) runs containers under ``docker run`` on top of
it; ``HostLauncher`` (``core/host.py``) runs every process straight on the host.
"""

from __future__ import annotations

import atexit
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.launcher import Launcher, LaunchSpec

if TYPE_CHECKING:
    from srtctl.core.runtime import Nodes
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

JOB_ID_ENV = "SRTCTL_JOB_ID"

# srun/pyxis options a recipe may carry that have no single-machine meaning (or are the launcher's own job).
_IGNORED_SRUN_OPTIONS = frozenset(
    {
        "mem",
        "container-name",
        "container-remap-root",
        "no-container-remap-root",
        "container-mount-home",
        "no-container-mount-home",
        "container-writable",
        "container-readonly",
        "distribution",
        "exclusive",
    }
)


def _killpg(pgid: int, sig: signal.Signals) -> None:
    """Signal a process group; one that is already gone is not an error."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        logger.warning("killpg %s %d failed: %s", sig.name, pgid, exc)


class _ProcessGroup(subprocess.Popen):
    """A host process started as a process group of its own (``start_new_session``); the group ends with it.

    When a Slurm step's task exits, Slurm stops what is left of the step, and when a
    container's PID 1 exits, the kernel kills the rest of the container. On the host
    nothing would, so an engine's children (vLLM's EngineCore, SGLang's schedulers)
    could outlive it holding GPU memory and ports. The first ``poll`` or ``wait`` that
    sees this process exit SIGKILLs whatever is left of its group. The group id cannot
    have been reused by then: the exited process kept its pid until that same call
    reaped it, and any member still running keeps the group.
    """

    _group_ended = False

    def poll(self) -> int | None:
        returncode = super().poll()
        if returncode is not None:
            self._end_group()
        return returncode

    def wait(self, timeout: float | None = None) -> int:
        returncode = super().wait(timeout)
        self._end_group()
        return returncode

    def _end_group(self) -> None:
        if self._group_ended:
            return
        self._group_ended = True
        _killpg(self.pid, signal.SIGKILL)


@dataclass
class _LocalStep:
    popen: subprocess.Popen
    container: str | None  # docker container name; None for a host process


class LocalLauncher(Launcher):
    """One job on the current machine; subclasses decide how a single process starts.

    Only single-node jobs whose launches are one task each are supported. Each named
    step is tracked here, so it can be listed and signalled without a scheduler.
    """

    def __init__(self) -> None:
        self._steps: dict[str, _LocalStep] = {}
        # Every host process this launcher started, named or not, for stop_leftovers.
        self._groups: list[subprocess.Popen] = []
        self._lock = threading.Lock()
        self._counter = 0
        self._warned_srun_options: set[str] = set()

    # -- launching -----------------------------------------------------------

    def _check_single_host(self, spec: LaunchSpec) -> None:
        if spec.nodes > 1 or spec.ntasks > 1:
            raise ValueError(
                f"launcher: {self.name} runs one task per launch; got nodes={spec.nodes} ntasks={spec.ntasks} "
                f"for {spec.step_name or shlex.join(spec.command[:3])}"
            )
        this_host = {socket.gethostname(), "localhost"}
        foreign = [n for n in spec.nodelist or () if n not in this_host]
        if foreign:
            raise ValueError(f"launcher: {self.name} cannot place a process on {', '.join(foreign)}")

    def _container_workdir(self, spec: LaunchSpec) -> str | None:
        """The recipe's ``srun_options: container-workdir``; other unknown options are reported once.

        Pyxis options either have no single-machine counterpart or are handled by the
        launcher itself; an unknown one is reported so a recipe that depends on it is
        not silently changed.
        """
        workdir: str | None = None
        unknown: list[str] = []
        for key, value in (spec.srun_options or {}).items():
            if key == "container-workdir":
                workdir = str(value) if value else None
            elif key not in _IGNORED_SRUN_OPTIONS:
                unknown.append(key)
        if unknown:
            with self._lock:
                new = [k for k in unknown if k not in self._warned_srun_options]
                self._warned_srun_options.update(new)
            if new:
                logger.warning("launcher: %s ignores srun_options %s", self.name, ", ".join(sorted(new)))
        return workdir

    def _start(
        self,
        cmd: list[str],
        spec: LaunchSpec,
        *,
        container: str | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> subprocess.Popen:
        """Run ``cmd`` in a process group of its own and record it under ``spec.step_name``.

        A host process (no ``container``) is a ``_ProcessGroup``: its group ends with it.
        """
        logger.info("%s launcher command: %s", self.name, shlex.join(cmd))
        popen_class = subprocess.Popen if container is not None else _ProcessGroup
        if spec.output:
            Path(spec.output).parent.mkdir(parents=True, exist_ok=True)
            with open(spec.output, "w") as out:
                # start_new_session: a host process gets its own process group so the
                # whole tree can be signalled; docker run forwards signals itself.
                popen = popen_class(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True)
        else:
            popen = popen_class(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True
            )
        with self._lock:
            if container is None:
                if not self._groups:
                    atexit.register(self._at_exit)
                self._groups.append(popen)
            if spec.step_name:
                self._steps[spec.step_name] = _LocalStep(popen=popen, container=container)
        return popen

    def _at_exit(self) -> None:
        """When the orchestrator exits, once this launcher has started a host process.

        The run's finally calls stop_leftovers, but the host teardown and post-processing
        launch after it, and a signal can end the orchestrator while they run.
        """
        self.stop_leftovers()

    def stop_leftovers(self, grace: float = 10.0) -> None:
        """SIGTERM every host process group still running, then SIGKILL what outlives ``grace`` seconds.

        Runs after the orchestrator stopped the processes it registered: a launch it never
        registered, or a group whose stop did not land, would otherwise outlive the job,
        which neither Slurm nor Docker allows. Polling a group whose process already exited
        ends what is left of it (``_ProcessGroup``).
        """
        with self._lock:
            running = [popen for popen in self._groups if popen.poll() is None]
        if not running:
            return
        logger.warning(
            "Stopping %d process group(s) still running after cleanup: %s",
            len(running),
            ", ".join(str(popen.pid) for popen in running),
        )
        for popen in running:
            _killpg(popen.pid, signal.SIGTERM)
        deadline = time.monotonic() + grace
        for popen in running:
            try:
                popen.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                _killpg(popen.pid, signal.SIGKILL)
                try:
                    popen.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error("Process group %d survived SIGKILL", popen.pid)

    # -- signalling ----------------------------------------------------------

    def list_step_ids(
        self, job_id: str | None = None, *, unique_names: set[str] | None = None
    ) -> dict[str, str] | None:
        # Step names are the registry keys, so they are unique by construction.
        with self._lock:
            return {
                name: step.container or str(step.popen.pid)
                for name, step in self._steps.items()
                if step.popen.poll() is None
            }

    def _signal_container(self, container: str, sig: str) -> bool:
        """Deliver ``sig`` to a container this launcher started; only container launchers start any."""
        return False

    def signal_step(
        self, step_name: str, sig: str = "TERM", *, step_ids: dict[str, str] | None = None, full: bool = True
    ) -> bool:
        """``killpg`` for a host process; a container step goes through ``_signal_container``."""
        with self._lock:
            step = self._steps.get(step_name)
        if step is None or step.popen.poll() is not None:
            return False
        if step.container is not None:
            if not self._signal_container(step.container, sig):
                return False
        else:
            try:
                os.killpg(step.popen.pid, signal.Signals[f"SIG{sig}"])
            except (ProcessLookupError, PermissionError, KeyError) as exc:
                logger.warning("killpg SIG%s %s failed: %s", sig, step_name, exc)
                return False
        logger.info("Sent SIG%s to %s", sig, step_name)
        return True

    # -- placement -----------------------------------------------------------

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

        return Nodes.from_nodelist(
            [socket.gethostname()],
            frontend_dedicated_node=frontend_dedicated_node,
            client_dedicated_node=client_dedicated_node,
            etcd_nats_dedicated_node=etcd_nats_dedicated_node,
            colocate_dedicated_nodes=colocate_dedicated_nodes,
            engine_nodes=engine_nodes,
            pools=pools,
        )

    def node_ip(self, hostname: str, network_interface: str | None = None) -> str:
        # Every process (and every --network host container) runs on this one machine.
        return "127.0.0.1"

    def job_id(self) -> str | None:
        return os.environ.get(JOB_ID_ENV)

    # -- submission ----------------------------------------------------------

    def submit(self, script_path: Path) -> str:
        now = time.time()
        return f"{self.name}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{int(now * 1000) % 1000:03d}"

    def _job_environment(self) -> dict[str, str]:
        """Variables the job script gets besides this environment and the job id."""
        return {}

    def start(self, job_id: str, job_output_dir: Path) -> int:
        """Run the staged job script in the foreground, teeing it to ``logs/sweep_<job_id>.log``.

        Runs in the foreground so a sweep's jobs take the machine's GPUs one at a time.
        """
        script = job_output_dir / "sbatch_script.sh"
        log_path = job_output_dir / "logs" / f"sweep_{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
        env.update(self._job_environment())
        env[JOB_ID_ENV] = job_id
        with open(log_path, "a") as log:
            proc = subprocess.Popen(
                ["bash", str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True
            )
            assert proc.stdout is not None
            while True:
                try:
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        sys.stderr.write(line)
                    break
                except KeyboardInterrupt:
                    # Ctrl+C also reached the orchestrator (same process group); keep
                    # streaming while it cleans up.
                    continue
            return proc.wait()

    # -- validation ----------------------------------------------------------

    def validate(self, config: SrtConfig) -> list[str]:
        """What keeps a recipe off one machine: more nodes, Slurm-only layouts, multi-task MPI steps."""
        from srtctl.core.config import get_srtslurm_setting

        errors = []
        topology = config.topology
        if config.total_nodes > 1:
            errors.append(f"needs {config.total_nodes} nodes; launcher: {self.name} runs on one machine")
        if topology.het_components(
            infra_dedicated=config.infra_dedicated_node,
            cluster_default=get_srtslurm_setting("use_het_jobs", False),
        ):
            errors.append("resources.het_jobs is a Slurm heterogeneous job; set it false")
        if config.frontend.placement.dedicated:
            errors.append("frontend.placement.dedicated needs a second node")
        if config.benchmark.placement.dedicated:
            errors.append("benchmark.placement.dedicated needs a second node")
        if config.infra_dedicated_node:
            errors.append("a dedicated infra node (etcd/NATS) needs a second node")
        if config.pool_services:
            errors.append("services[].nodes (pools) need nodes of their own")
        if config.backend.get_srun_config().launch_per_endpoint:
            errors.append(
                f"engine {config.backend_type} launches each endpoint as one multi-task MPI step, "
                f"which launcher: {self.name} does not support"
            )
        return errors
