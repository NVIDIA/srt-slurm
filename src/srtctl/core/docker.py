# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``launcher: docker``: one job on the current machine, with containers under ``docker run``."""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.launcher import LaunchSpec
from srtctl.core.local import LocalLauncher

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

# Docker container names: [a-zA-Z0-9][a-zA-Z0-9_.-]*
_CONTAINER_NAME_UNSAFE = re.compile(r"[^a-zA-Z0-9_.-]")
JOB_LABEL = "srtctl.job"
STEP_LABEL = "srtctl.step"
# Host env forwarded into every container: pyxis hands a Slurm container the whole
# submit environment, docker hands it nothing.
_PASSTHROUGH_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NGC_API_KEY")


def _gpus_flag_value(value: str) -> str:
    """Quote a ``--gpus`` value for Docker's CSV parser.

    ``--gpus device=2,3,4,5`` is split on commas into ``device=2`` plus bare
    fields that Docker reads as a GPU count ("cannot set both Count and
    DeviceIDs on device request"); the whole value must arrive wrapped in
    literal double quotes, as ``docker run --gpus '"device=2,3"'`` does.
    """
    value = str(value)
    if "," in value and not (value.startswith('"') and value.endswith('"')):
        return f'"{value}"'
    return value


class DockerLauncher(LocalLauncher):
    """One job on the current machine: containers under ``docker run``, host commands under ``bash``.

    Every container shares the host network and IPC namespace and sees the GPUs
    ``docker_gpus`` names (every GPU by default);
    each worker's ``CUDA_VISIBLE_DEVICES`` (set inside the container by the bash
    wrapper) selects its GPUs, exactly as it does inside a Slurm step. Only
    single-node jobs whose launches are one task each are supported.
    """

    name = "docker"

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
            _gpus_flag_value(get_srtslurm_setting("docker_gpus") or "all"),
            "--network",
            "host",
            "--ipc",
            "host",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            # Performance counters for nsys CPU sampling: a Slurm cluster allows them host-wide,
            # a stock Ubuntu host (perf_event_paranoid=4) refuses them to anyone without this.
            "--cap-add",
            "PERFMON",
            # NUMA affinity: SGLang/vLLM pin each rank's host memory and threads to its GPU's NUMA
            # node (set_mempolicy/sched_setaffinity); without this they log "User lacks permission
            # to set NUMA affinity" and run unpinned, which an srun step on a host without the
            # restriction does not.
            "--cap-add",
            "SYS_NICE",
            "--entrypoint",
            "",
        ]
        if spec.step_name:
            cmd.extend(["--label", f"{STEP_LABEL}={spec.step_name}"])
        mounts, pre_commands = self._mounts(spec)
        for host, container in mounts:
            cmd.extend(["-v", f"{host}:{container}"])
        workdir = self._container_workdir(spec)
        if workdir:
            cmd.extend(["-w", workdir])
        for name in (*_PASSTHROUGH_ENV, *(spec.env_to_pass_through or ())):
            if name in os.environ:
                cmd.extend(["-e", name])
        for key, value in (spec.srun_export_env or {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend(get_srtslurm_setting("docker_args") or [])
        cmd.append(self.docker_image(str(spec.container_image)))
        cmd.extend(self._with_pre_commands(self.build_task_command(spec), pre_commands))
        return cmd

    # How many directory entries to inspect for symlinks that leave a mount (HF snapshot dirs hold tens).
    _MOUNT_SCAN_LIMIT = 20_000

    def _mounts(self, spec: LaunchSpec) -> tuple[list[tuple[str, str]], list[str]]:
        """``(host, container)`` bind mounts plus the shell commands that complete them inside the container.

        A bind mount does not follow symlinks that point outside the mounted directory,
        and a HuggingFace snapshot directory (``models--org--name/snapshots/<rev>``) is
        exactly that: every weight file is a relative symlink into the hub's blob store.
        Mounted at ``/model`` on its own, the files resolve to nothing. For such a
        directory the launcher mounts the common root of the directory and its link
        targets at its own host path and creates the recipe's container path as a
        symlink to the directory, so the relative links resolve exactly as on the host.
        """
        mounts: list[tuple[str, str]] = []
        pre_commands: list[str] = []
        roots: set[str] = set()
        for host, container in (spec.container_mounts or {}).items():
            root = self._escape_root(Path(host))
            if root is None:
                mounts.append((str(host), str(container)))
                continue
            if root not in roots:
                roots.add(root)
                mounts.append((root, root))
            parent = os.path.dirname(str(container))
            if parent not in ("", "/"):
                pre_commands.append(f"mkdir -p {shlex.quote(parent)}")
            pre_commands.append(f"ln -sfnT {shlex.quote(os.path.realpath(host))} {shlex.quote(str(container))}")
            logger.info(
                "Mount %s -> %s holds symlinks into %s; mounting that root and linking %s inside the container",
                host,
                container,
                root,
                container,
            )
        return mounts, pre_commands

    def _escape_root(self, host: Path) -> str | None:
        """The directory that holds ``host`` and every target its symlinks escape to; None when nothing escapes."""
        if not host.is_dir():
            return None
        base = os.path.realpath(host)
        targets: set[str] = set()
        seen = 0
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            for name in (*dirnames, *filenames):
                seen += 1
                if seen > self._MOUNT_SCAN_LIMIT:
                    return self._common_root(base, targets)
                path = os.path.join(dirpath, name)
                if os.path.islink(path):
                    target = os.path.realpath(path)
                    if os.path.commonpath([base, target]) != base:
                        targets.add(target)
        return self._common_root(base, targets)

    @staticmethod
    def _common_root(base: str, targets: set[str]) -> str | None:
        if not targets:
            return None
        root = os.path.commonpath([base, *targets])
        if root == "/":
            logger.warning("Symlinks under %s point all over the filesystem; mounting it as is", base)
            return None
        return root

    @staticmethod
    def _with_pre_commands(task: list[str], pre_commands: list[str]) -> list[str]:
        """``task`` with ``pre_commands`` run first inside the same ``bash -c``."""
        if not pre_commands:
            return task
        setup = " && ".join(pre_commands)
        if len(task) == 3 and task[:2] == ["bash", "-c"]:
            return ["bash", "-c", f"{setup} && {task[2]}"]
        return ["bash", "-c", f"{setup} && exec {shlex.join(task)}"]

    def launch(self, spec: LaunchSpec) -> subprocess.Popen:
        self._check_single_host(spec)
        if spec.container_image:
            container = self._container_name(spec.step_name)
            return self._start(self.docker_command(spec, container), spec, container=container)
        env = {**os.environ, **spec.srun_export_env} if spec.srun_export_env else None
        return self._start(self.build_task_command(spec), spec, env=env)

    # -- signalling ----------------------------------------------------------

    def _signal_container(self, container: str, sig: str) -> bool:
        """``docker kill --signal``: ``docker run`` forwards signals too, but this reaches a container whose client is gone."""
        if shutil.which("docker") is None:
            return False
        result = subprocess.run(
            ["docker", "kill", f"--signal={sig}", container],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("docker kill --signal=%s %s failed: %s", sig, container, result.stderr.strip())
            return False
        return True

    # -- submission ----------------------------------------------------------

    def shell_command(
        self, runtime: RuntimeContext, node: str, *, image: str | None = None, container_name: str | None = None
    ) -> str:
        return f"docker exec -it $(docker ps -q --filter label={JOB_LABEL}={runtime.job_id} | head -1) bash"

    def status_hint(self, job_id: str) -> str:
        return f"docker ps --filter label={JOB_LABEL}={job_id}"

    # -- validation ----------------------------------------------------------

    def validate(self, config: SrtConfig) -> list[str]:
        errors = super().validate(config)
        images = {"model.container": str(config.model.container)}
        images.update({f"roles.{role}.container": image for role, image in config.role_containers.items()})
        for key, image in images.items():
            if os.path.expandvars(image).startswith(("/", "./")):
                errors.append(f"{key} is an enroot image file ({image}); launcher: docker needs a docker image name")
        return errors
