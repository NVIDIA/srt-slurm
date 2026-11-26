# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``launcher: host``: one job on the current machine, every process straight on the host.

For a machine with no container runtime and no Slurm (a Mac, a workstation, a VM
without the NVIDIA Container Toolkit): each launch runs the same bash-wrapped command
an ``srun`` step or ``docker run`` container would, as a process group on this host,
with the binaries on this machine's ``PATH``. Container images are not used, so neither preflight
nor the job requires the recipe's image files to exist.

The orchestrator still describes every launch with the container mounts it would
hand ``srun`` or ``docker run`` (the run's logs at ``/logs``, the model at ``/model``,
the repo's ``configs/`` at ``/configs``, ...). With no container to mount them in,
``ContainerPaths`` rewrites those container paths in the command, the exported
environment and the working directory to the host directories behind them. A host
directory whose path a shell would split (a space, a quote) is reached through a
shell-safe symlink instead, since the rewritten path lands inside ``bash -c`` scripts.

A container also keeps whatever a step installs (``pip install`` in a benchmark
script, the Dynamo install, a setup script) out of the machine; on the host there is
no such boundary. ``host_environment`` hands every process the ``PATH`` of the shell
that ran ``srtctl apply`` instead of srtctl's own, and stops ``pip`` from installing
into a Python that is not a virtualenv, and ``HostLauncher.validate`` turns away the
steps that would install outside Python altogether.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
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
# The PATH of the shell that ran `srtctl apply`, handed to the job script before it (and
# `uv run`) put their own directories first, and that shell's VIRTUAL_ENV (empty when none
# was active), which `uv run` replaced with srtctl's.
HOST_PATH_ENV = "SRTCTL_HOST_PATH"
HOST_VIRTUAL_ENV_ENV = "SRTCTL_HOST_VIRTUAL_ENV"
# Variables ``uv run`` sets for srtctl's own interpreter.
_OWN_VENV_VARS = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")
# Characters a shell takes literally in an unquoted word (shlex.quote leaves such a string alone).
_SHELL_UNSAFE = re.compile(r"[^\w@%+=:,./-]", re.ASCII)


def _real(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def host_environment(base: Mapping[str, str], own_venv: str | None) -> dict[str, str]:
    """The environment every ``launcher: host`` process starts from.

    ``base`` is the orchestrator's environment. The job script put the checkout's
    ``bin/`` and ``~/.local/bin`` first on its ``PATH``, and ``uv run`` put srtctl's own
    virtualenv (``own_venv``) before them and named it in ``VIRTUAL_ENV`` and
    ``UV_PROJECT_ENVIRONMENT``. Left that way, ``python3 -m dynamo.frontend`` would run
    srtctl's interpreter (or a ``python3`` in ``~/.local/bin``), which has no engine, and
    a step's ``uv sync`` or ``uv pip install`` would install into srtctl's environment.

    So ``PATH`` starts again from ``HOST_PATH_ENV``, the one ``srtctl apply`` ran with;
    what the job script and ``uv run`` added follows it (srtctl's own tools, such as
    ``tachometer-scraper``, are still found), srtctl's ``bin`` last. The variables naming
    srtctl's virtualenv are dropped, and ``VIRTUAL_ENV`` names the user's virtualenv, so
    ``uv pip install`` targets the environment ``pip`` and ``python3`` come from: the one
    ``srtctl apply`` saw (``HOST_VIRTUAL_ENV_ENV``), or, when that was srtctl's own
    (``uv run srtctl apply`` replaces it), the virtualenv of the first ``python3`` on ``PATH``.
    """
    env = dict(base)
    path = env.get("PATH")
    entries = path.split(os.pathsep) if path else []
    host_path = env.pop(HOST_PATH_ENV, None)
    if host_path is not None:
        user = host_path.split(os.pathsep) if host_path else []
        entries = [*user, *(entry for entry in entries if entry not in user)]
    host_venv = env.pop(HOST_VIRTUAL_ENV_ENV, None)
    if own_venv:
        for name in _OWN_VENV_VARS:
            if name in env and _real(env[name]) == _real(own_venv):
                del env[name]
        own_bin = os.path.join(own_venv, "bin")
        own_real = _real(own_bin)
        rest = [entry for entry in entries if not (entry and _real(entry) == own_real)]
        if len(rest) != len(entries):
            entries = [*rest, own_bin]
    if host_venv and not (own_venv and _real(host_venv) == _real(own_venv)):
        env["VIRTUAL_ENV"] = host_venv
    elif "VIRTUAL_ENV" not in env:
        venv = _first_python_venv(entries)
        if venv is not None and not (own_venv and _real(venv) == _real(own_venv)):
            env["VIRTUAL_ENV"] = venv
    if entries or path is not None:
        env["PATH"] = os.pathsep.join(entries)
    env.setdefault(*PIP_GUARD)
    return env


def _first_python_venv(entries: list[str]) -> str | None:
    """The virtualenv the first ``python3`` on ``entries`` belongs to; None when it is not a virtualenv's."""
    for entry in entries:
        if entry and os.path.isfile(os.path.join(entry, "python3")):
            venv = os.path.dirname(os.path.abspath(entry))
            return venv if os.path.isfile(os.path.join(venv, "pyvenv.cfg")) else None
    return None


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
    uses_container_images = False
    supports_ssh = True

    def __init__(self) -> None:
        super().__init__()
        self._ignored_images: set[str] = set()
        self._aliases: dict[str, Path] = {}
        self._scratch_dir: Path | None = None
        self._scratch_lock = threading.Lock()

    def _at_exit(self) -> None:
        super()._at_exit()
        if self._scratch_dir is not None:
            # After the processes that use it are gone; removes the aliases, never what they point at.
            shutil.rmtree(self._scratch_dir, ignore_errors=True)

    def scratch_dir(self) -> Path:
        """A directory of the job's own: private (``mkdtemp``), shell-safe, removed when the orchestrator exits.

        It holds the path aliases (``shell_safe_path``) and is where a launch that would
        have run in a container starts: outside the srt-slurm checkout the orchestrator
        runs in, where a step's ``uv`` would find srtctl's own ``.venv``.
        """
        with self._scratch_lock:
            if self._scratch_dir is None:
                base = Path(tempfile.gettempdir())
                if _SHELL_UNSAFE.search(base.as_posix()):
                    base = Path("/tmp")
                prefix = _SHELL_UNSAFE.sub("_", f"srtctl-{self.job_id() or 'host'}-")
                path = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
                if _SHELL_UNSAFE.search(path.as_posix()):
                    shutil.rmtree(path, ignore_errors=True)
                    raise ValueError(f"launcher: host found no shell-safe temporary directory (tried {path})")
                self._scratch_dir = path
            return self._scratch_dir

    def shell_safe_path(self, host: Path) -> Path:
        """``host`` itself, or a symlink to it whose path a shell takes as one word.

        The rewritten container path lands in ``bash -c`` scripts, unquoted or inside
        quotes, where a space or a quote in it would split or end the word; the symlink
        (``<scratch_dir>/<hash>``) has neither. One per host path, for the job. A missing
        directory is created first, as ``docker run -v`` does, so the alias never dangles.
        """
        text = host.as_posix()
        if not _SHELL_UNSAFE.search(text):
            return host
        with self._lock:
            alias = self._aliases.get(text)
            if alias is None:
                target = os.path.abspath(text)
                if not os.path.lexists(target):
                    os.makedirs(target, exist_ok=True)
                alias = self.scratch_dir() / hashlib.sha256(os.fsencode(text)).hexdigest()[:16]
                os.symlink(target, alias)
                self._aliases[text] = alias
                logger.info("launcher: host reaches %s through %s, a path a shell takes as one word", text, alias)
        return alias

    def host_command(self, spec: LaunchSpec) -> tuple[list[str], dict[str, str], str | None]:
        """The command, environment and working directory ``spec`` runs with on this host."""
        mounts = {
            self.shell_safe_path(Path(host)): container for host, container in (spec.container_mounts or {}).items()
        }
        paths = ContainerPaths(mounts)
        cmd = [paths.to_host(part) for part in self.build_task_command(spec)]
        if not spec.use_bash_wrapper:
            # A missing binary then fails the process (exit 127, "not found" in its log), as in a
            # container or an srun step, instead of raising from Popen inside the orchestrator.
            cmd = ["bash", "-c", 'exec -- "$@"', "bash", *cmd]
        env = host_environment(os.environ, _own_venv())
        env.update({key: paths.to_host(value) for key, value in (spec.srun_export_env or {}).items()})
        workdir = self._container_workdir(spec)
        if workdir:
            cwd: str | None = paths.to_host(workdir)
        elif spec.container_image:
            cwd = str(self.scratch_dir())
        else:
            cwd = None  # srtctl's own host commands start where an srun step on the bare node would
        return cmd, env, cwd

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

    def _job_environment(self) -> dict[str, str]:
        # The job script and `uv run` put their own directories in front of PATH and replace
        # VIRTUAL_ENV; host_environment gives every process these back.
        return {HOST_PATH_ENV: os.environ.get("PATH", ""), HOST_VIRTUAL_ENV_ENV: os.environ.get("VIRTUAL_ENV", "")}

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

    def setup_problems(self, srtctl_source: Path, config: SrtConfig | None) -> list[str]:
        """uv to run the orchestrator, and the checkout files the recipe's services run.

        Those are etcd and NATS for the discovery plane and the OTel collector for a Dynamo
        frontend, unless the recipe points at external ones or switches them off; a missing
        one fails the job even when the service is not critical (the collector checks for
        its binary before launch). Not the rest of what ``make setup`` installs: those are
        Linux binaries, and without the Tachometer scraper or the exporters, which list no
        checkout files, only their telemetry is lost.
        """
        from srtctl.services.implicit import effective_services
        from srtctl.services.registry import get_service_kind

        problems = []
        # The job script looks uv up on PATH after putting the checkout's bin/ and ~/.local/bin
        # (where the standalone installer puts it) first.
        uv_homes = (srtctl_source / "bin" / "uv", Path.home() / ".local" / "bin" / "uv")
        if shutil.which("uv") is None and not any(path.exists() for path in uv_homes):
            problems.append("uv, which runs the orchestrator: not on PATH, bin/ or ~/.local/bin")
        for entry in effective_services(config) if config is not None else ():
            service = entry.service
            if service.external:
                continue
            problems.extend(
                f"{path}, which services[{service.name}] runs, built for this machine"
                for path in get_service_kind(service.type).checkout_files(service)
                if not (srtctl_source / path).exists()
            )
        return problems

    def setup_hint(self, srtctl_source: Path) -> list[str]:
        return [
            "Install them on this machine (uv from Homebrew or https://docs.astral.sh/uv/).",
            "See docs/cluster-config.md#launcher-host.",
        ]
