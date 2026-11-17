# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``launcher: host``: one job on the current machine, every process straight on the host.

For a machine with no container runtime and no Slurm (a Mac, a workstation, a VM
without the NVIDIA Container Toolkit): each launch runs the same bash-wrapped command
an ``srun`` step or ``docker run`` container would, as a process group on this host,
with the binaries on this machine's ``PATH``. Container images are not used.

The orchestrator still describes every launch with the container mounts it would
hand ``srun`` or ``docker run`` (the run's logs at ``/logs``, the model at ``/model``,
the repo's ``configs/`` at ``/configs``, ...). With no container to mount them in,
``ContainerPaths`` rewrites those container paths in the command, the exported
environment and the working directory to the host directories behind them.

A container also keeps whatever a step installs (``pip install`` in a benchmark
script, the Dynamo install, a setup script) out of the machine; on the host there is
no such boundary. ``host_environment`` hands every process this machine's Python
instead of srtctl's own virtualenv and stops ``pip`` from installing into a Python
that is not a virtualenv, and ``HostLauncher.validate`` turns away the steps that
would install outside Python altogether.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.launcher import LaunchSpec
from srtctl.core.local import LocalLauncher

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

# pip refuses to install into a Python that is not a virtualenv (``--user`` and
# ``--break-system-packages`` included), so no step can change this machine's system or
# Homebrew Python. uv already refuses without ``--system``. A value set in the user's own
# environment wins.
PIP_GUARD = ("PIP_REQUIRE_VIRTUALENV", "1")
# Variables ``uv run`` sets for srtctl's own interpreter.
_OWN_VENV_VARS = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")


def _same_path(a: str, b: str) -> bool:
    return os.path.realpath(os.path.expanduser(a)) == os.path.realpath(os.path.expanduser(b))


def host_environment(base: Mapping[str, str], own_venv: str | None) -> dict[str, str]:
    """The environment every ``launcher: host`` process starts from.

    ``base`` is the orchestrator's environment, which ``uv run`` gave srtctl's own
    virtualenv (``own_venv``): its ``bin`` first on ``PATH``, its path in
    ``VIRTUAL_ENV`` and ``UV_PROJECT_ENVIRONMENT``. Left that way, ``python3 -m
    dynamo.frontend`` would run srtctl's interpreter, which has no engine, and a step's
    ``uv sync`` or ``uv pip install`` would install into srtctl's environment. Its ``bin``
    moves to the end of ``PATH`` (srtctl's own console scripts, such as
    ``tachometer-scraper``, are still found) and the variables naming it are dropped.
    """
    env = dict(base)
    if own_venv:
        for name in _OWN_VENV_VARS:
            if name in env and _same_path(env[name], own_venv):
                del env[name]
        own_bin = os.path.join(own_venv, "bin")
        entries = env.get("PATH", "").split(os.pathsep)
        if any(entry and _same_path(entry, own_bin) for entry in entries):
            rest = [entry for entry in entries if not (entry and _same_path(entry, own_bin))]
            env["PATH"] = os.pathsep.join([*rest, own_bin])
    env.setdefault(*PIP_GUARD)
    return env


def _own_venv() -> str | None:
    """srtctl's own virtualenv, when the orchestrator runs from one (``uv run`` in the job script)."""
    return sys.prefix if sys.prefix != sys.base_prefix else None


# A container path starts a value or follows one of these: whitespace, a quote, or shell/flag
# punctuation (`--model=/model`, `PATH=/configs:$PATH`, `$(cat /logs/x)`).
_NOT_PATH_START = r"""[^\s"'`=:,;(\[{<>|&]"""


class ContainerPaths:
    """Rewrites the container paths of a launch's mounts to the host directories behind them.

    A container path is rewritten wherever it appears as a whole path, on its own or
    as the start of a longer one (``/logs`` and ``/logs/x``), inside any argument,
    including a ``bash -c`` script. It is left alone inside a longer path segment
    (``/models``, ``/data/logs``) or a URL (``http://host:8000/model``). Mounts whose
    container path is the host path, or ``/``, need no rewriting.
    """

    def __init__(self, mounts: Mapping[Path, Path] | None) -> None:
        self._host: dict[str, str] = {}
        for host, container in (mounts or {}).items():
            container_path = Path(container).as_posix()
            host_path = Path(host).as_posix()
            if container_path in ("", "/") or container_path == host_path:
                continue
            self._host[container_path] = host_path
        self._pattern: re.Pattern[str] | None = None
        if self._host:
            alternatives = "|".join(re.escape(c) for c in sorted(self._host, key=len, reverse=True))
            self._pattern = re.compile(rf"(?<!{_NOT_PATH_START})({alternatives})(?![\w.-])")

    def to_host(self, text: str) -> str:
        """``text`` with every mounted container path replaced by its host directory."""
        if self._pattern is None:
            return text
        return self._pattern.sub(lambda match: self._host[match.group(1)], text)


class HostLauncher(LocalLauncher):
    """One job on the current machine: every process runs on the host under ``bash``, without containers.

    Commands find their binaries on the ``PATH`` of the shell that ran ``srtctl apply``
    (see ``host_environment``), so the engine, the benchmark clients and anything
    ``host_setup`` installs must be on this machine. Each worker's
    ``CUDA_VISIBLE_DEVICES`` (set by the bash wrapper) still selects its GPUs.
    """

    name = "host"

    def __init__(self) -> None:
        super().__init__()
        self._ignored_images: set[str] = set()

    def host_command(self, spec: LaunchSpec) -> tuple[list[str], dict[str, str], str | None]:
        """The command, environment and working directory ``spec`` runs with on this host."""
        paths = ContainerPaths(spec.container_mounts)
        cmd = [paths.to_host(part) for part in self.build_task_command(spec)]
        env = host_environment(os.environ, _own_venv())
        env.update({key: paths.to_host(value) for key, value in (spec.srun_export_env or {}).items()})
        workdir = self._container_workdir(spec)
        return cmd, env, paths.to_host(workdir) if workdir else None

    def launch(self, spec: LaunchSpec) -> subprocess.Popen:
        self._check_single_host(spec)
        if spec.container_image:
            image = str(spec.container_image)
            with self._lock:
                first = image not in self._ignored_images
                self._ignored_images.add(image)
            if first:
                logger.info("launcher: host ignores container image %s; commands run with this machine's PATH", image)
        cmd, env, cwd = self.host_command(spec)
        return self._start(cmd, spec, env=env, cwd=cwd)

    def shell_command(
        self, runtime: RuntimeContext, node: str, *, image: str | None = None, container_name: str | None = None
    ) -> str:
        return f"cd {shlex.quote(str(runtime.log_dir))} && bash"

    def status_hint(self, job_id: str) -> str:
        # The orchestrator's command line names the job's output directory, which ends in the job id;
        # every other process of the job is its descendant.
        return f"ps -A -o pid,ppid,etime,command | grep -F -- {shlex.quote(job_id)} | grep -v grep"

    def validate(self, config: SrtConfig) -> list[str]:
        """The single-machine checks, plus the steps that would install into this machine rather than a container."""
        from srtctl.core.config import get_srtslurm_setting
        from srtctl.core.schema import installs_dynamo

        errors = super().validate(config)
        if installs_dynamo(config):
            errors.append(
                "frontend.type: dynamo installs ai-dynamo into the worker's Python before it starts "
                "(dynamo.install: true) and marks it done there; launcher: host has no container to "
                "throw away, so that would change your own environment and a later job would skip its "
                "install. Install ai-dynamo where your engine runs and set dynamo.install: false"
            )
        reporting = get_srtslurm_setting("reporting") or {}
        ai_analysis = reporting.get("ai_analysis") if isinstance(reporting, dict) else None
        if isinstance(ai_analysis, dict) and ai_analysis.get("enabled"):
            errors.append(
                "reporting.ai_analysis installs Claude Code and the GitHub CLI into your home directory "
                "with curl, which launcher: host does not do; set reporting.ai_analysis.enabled: false"
            )
        return errors
