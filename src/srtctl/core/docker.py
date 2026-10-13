# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``launcher: docker``: one job on the current machine, with containers under ``docker run``."""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
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
    from srtctl.core.runtime import Nodes, RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

# Docker container names: [a-zA-Z0-9][a-zA-Z0-9_.-]*
_CONTAINER_NAME_UNSAFE = re.compile(r"[^a-zA-Z0-9_.-]")
JOB_ID_ENV = "SRTCTL_JOB_ID"
JOB_LABEL = "srtctl.job"
STEP_LABEL = "srtctl.step"
# Host env forwarded into every container: pyxis hands a Slurm container the whole
# submit environment, docker hands it nothing.
_PASSTHROUGH_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NGC_API_KEY")


@dataclass
class _DockerStep:
    popen: subprocess.Popen
    container: str | None  # docker container name; None for a host process


class DockerLauncher(Launcher):
    """One job on the current machine: containers under ``docker run``, host commands under ``bash``.

    Every container shares the host network and IPC namespace and sees every GPU;
    each worker's ``CUDA_VISIBLE_DEVICES`` (set inside the container by the bash
    wrapper) selects its GPUs, exactly as it does inside a Slurm step. Only
    single-node jobs whose launches are one task each are supported.
    """

    name = "docker"

    def __init__(self) -> None:
        self._steps: dict[str, _DockerStep] = {}
        self._lock = threading.Lock()
        self._counter = 0

    # -- launching -----------------------------------------------------------

    def _job_label(self) -> str:
        return self.job_id() or "docker"

    def _container_name(self, step_name: str | None) -> str:
        # A relaunched step (restart, a second benchmark) gets a fresh name: the
        # previous ``--rm`` container may not be removed yet.
        with self._lock:
            self._counter += 1
            suffix = (
                step_name if step_name and step_name not in self._steps else f"{step_name or 'proc'}{self._counter}"
            )
        return _CONTAINER_NAME_UNSAFE.sub("_", f"srtctl_{self._job_label()}_{suffix}")

    @staticmethod
    def docker_image(image: str) -> str:
        """An enroot/pyxis image URI as docker expects it (``nvcr.io#nvidia/x`` -> ``nvcr.io/nvidia/x``)."""
        for prefix in ("docker://", "dockerd://"):
            image = image.removeprefix(prefix)
        return image.replace("#", "/", 1)

    def docker_command(self, spec: LaunchSpec, container_name: str) -> list[str]:
        from srtctl.core.config import get_srtslurm_setting

        assert spec.container_image
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--label",
            f"{JOB_LABEL}={self._job_label()}",
            "--gpus",
            "all",
            "--network",
            "host",
            "--ipc",
            "host",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "--entrypoint",
            "",
        ]
        if spec.step_name:
            cmd.extend(["--label", f"{STEP_LABEL}={spec.step_name}"])
        for host, container in (spec.container_mounts or {}).items():
            cmd.extend(["-v", f"{host}:{container}"])
        for name in (*_PASSTHROUGH_ENV, *(spec.env_to_pass_through or ())):
            if name in os.environ:
                cmd.extend(["-e", name])
        for key, value in (spec.srun_export_env or {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend(get_srtslurm_setting("docker_args") or [])
        cmd.append(self.docker_image(str(spec.container_image)))
        cmd.extend(self.build_task_command(spec))
        return cmd

    def _check_single_host(self, spec: LaunchSpec) -> None:
        if spec.nodes > 1 or spec.ntasks > 1:
            raise ValueError(
                f"launcher: docker runs one task per launch; got nodes={spec.nodes} ntasks={spec.ntasks} "
                f"for {spec.step_name or shlex.join(spec.command[:3])}"
            )
        this_host = {socket.gethostname(), "localhost"}
        foreign = [n for n in spec.nodelist or () if n not in this_host]
        if foreign:
            raise ValueError(f"launcher: docker cannot place a process on {', '.join(foreign)}")

    def launch(self, spec: LaunchSpec) -> subprocess.Popen:
        self._check_single_host(spec)
        container: str | None = None
        env: dict[str, str] | None = None
        if spec.container_image:
            container = self._container_name(spec.step_name)
            cmd = self.docker_command(spec, container)
        else:
            cmd = self.build_task_command(spec)
            if spec.srun_export_env:
                env = {**os.environ, **spec.srun_export_env}
        logger.info("docker launcher command: %s", shlex.join(cmd))

        if spec.output:
            Path(spec.output).parent.mkdir(parents=True, exist_ok=True)
            with open(spec.output, "w") as out:
                # start_new_session: a host process gets its own process group so the
                # whole tree can be signalled; docker run forwards signals itself.
                popen = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        else:
            popen = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
        if spec.step_name:
            with self._lock:
                self._steps[spec.step_name] = _DockerStep(popen=popen, container=container)
        return popen

    # -- signalling ----------------------------------------------------------

    def list_step_ids(self, job_id: str | None = None) -> dict[str, str] | None:
        with self._lock:
            return {
                name: step.container or str(step.popen.pid)
                for name, step in self._steps.items()
                if step.popen.poll() is None
            }

    def signal_step(
        self, step_name: str, sig: str = "TERM", *, step_ids: dict[str, str] | None = None, full: bool = True
    ) -> bool:
        """``docker kill --signal`` for a container, ``killpg`` for a host process.

        ``docker run`` forwards signals too, but ``docker kill`` reaches the
        container even if its client is gone.
        """
        with self._lock:
            step = self._steps.get(step_name)
        if step is None or step.popen.poll() is not None:
            return False
        if step.container is not None:
            if shutil.which("docker") is None:
                return False
            result = subprocess.run(
                ["docker", "kill", f"--signal={sig}", step.container],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0:
                logger.warning("docker kill --signal=%s %s failed: %s", sig, step.container, result.stderr.strip())
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
        # Every container runs with --network host on this one machine.
        return "127.0.0.1"

    def job_id(self) -> str | None:
        return os.environ.get(JOB_ID_ENV)

    # -- submission ----------------------------------------------------------

    def submit(self, script_path: Path) -> str:
        now = time.time()
        return f"docker-{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{int(now * 1000) % 1000:03d}"

    def start(self, job_id: str, job_output_dir: Path) -> int:
        """Run the staged job script in the foreground, teeing it to ``logs/sweep_<job_id>.log``.

        Runs in the foreground so a sweep's jobs take the machine's GPUs one at a time.
        """
        script = job_output_dir / "sbatch_script.sh"
        log_path = job_output_dir / "logs" / f"sweep_{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
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

    def shell_command(self, runtime: RuntimeContext, node: str) -> str:
        return f"docker exec -it $(docker ps -q --filter label={JOB_LABEL}={runtime.job_id} | head -1) bash"

    def status_hint(self, job_id: str) -> str:
        return f"docker ps --filter label={JOB_LABEL}={job_id}"

    # -- validation ----------------------------------------------------------

    def validate(self, config: SrtConfig) -> list[str]:
        from srtctl.core.config import get_srtslurm_setting

        errors = []
        topology = config.topology
        if config.total_nodes > 1:
            errors.append(f"needs {config.total_nodes} nodes; launcher: docker runs on one machine")
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
                "which launcher: docker does not support"
            )
        images = {"model.container": str(config.model.container)}
        images.update({f"roles.{role}.container": image for role, image in config.role_containers.items()})
        for key, image in images.items():
            if os.path.expandvars(image).startswith(("/", "./")):
                errors.append(f"{key} is an enroot image file ({image}); launcher: docker needs a docker image name")
        return errors
