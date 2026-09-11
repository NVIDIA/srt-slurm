# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nsight-slurm ("Nsight Cloud for Slurm") job setup for ``profiling.type: nsight-slurm``.

The wrapper (gitlab-master.nvidia.com/mhallock/nsight-cloud-slurm) replaces ``srun``
for the profiled steps. Before the first worker step it needs the job-local
configuration written and, for a coordinator that outlives the individual steps,
an explicit ``coordinator start``. This mixin does that once per sweep and stops
the coordinator in the sweep's cleanup.

Everything is driven through the wrapper's CLI so its config-file format stays its
own business:

    nsight-slurm configure tool-path <nsys inside the container>
    nsight-slurm configure tool-command profile
    nsight-slurm configure profiling-mode <at-launch|manual|cuda-api>
    nsight-slurm configure tool-options <nsys profile options...>
    nsight-slurm configure report-output <log_dir>/nsight-slurm-reports
    nsight-slurm enable pyxis            # mounts the install RO, runs the connector in-container
    nsight-slurm coordinator start

The wrapper keys its state on ``SLURM_SUBMIT_DIR`` and ``SLURM_JOB_ID``. SLURM_SUBMIT_DIR
is redirected to the run's log dir (see ``ProfilingConfig.nsight_slurm_process_env``)
so ``<log_dir>/.nsight-slurm/jobs/<job>/`` holds the coordinator config, keys, state
and log next to the rest of the run.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

NSIGHT_SLURM_REPORT_SUBDIR = "nsight-slurm-reports"
NSIGHT_SLURM_LOG_NAME = "nsight-slurm.out"


class NsightSlurmStageMixin:
    """Mixin for SweepOrchestrator: nsight-slurm job configuration and coordinator lifecycle."""

    config: SrtConfig
    runtime: Any
    _nsight_slurm_ready: bool = False
    _nsight_slurm_coordinator_started: bool = False

    def _nsight_slurm_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.config.profiling.nsight_slurm_process_env(self.runtime.log_dir))
        head_node_ip = getattr(self.runtime, "head_node_ip", None)
        if head_node_ip:
            # The wrapper publishes the coordinator as tcp://${SLURMD_NODENAME:-fqdn}:<port>. Its ZMQ
            # sockets run with IPV6=1, and ZMQ connects to the FIRST address a name resolves to (glibc
            # returns v4-mapped addresses only when the name has no IPv6 at all). A node's own name can
            # resolve to a link-local IPv6 that is not connectable, so the connectors that run on the
            # coordinator's node would time out (observed on hecate, job 571147). Publishing the head
            # node's IPv4 literal sidesteps name resolution for every rank.
            env["SLURMD_NODENAME"] = str(head_node_ip)
        return env

    def _nsight_slurm_run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run one wrapper CLI command from the log dir, appending its output to nsight-slurm.out."""
        cmd = [self.config.profiling.nsight_slurm_bin(), *args]
        log_path = Path(self.runtime.log_dir) / NSIGHT_SLURM_LOG_NAME
        with open(log_path, "a") as log:
            log.write(f"$ {shlex.join(cmd)}\n")
            log.flush()
            result = subprocess.run(
                cmd,
                cwd=str(self.runtime.log_dir),
                env=self._nsight_slurm_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if check and result.returncode != 0:
            raise RuntimeError(f"{shlex.join(cmd)} failed with exit code {result.returncode}; see {log_path}")
        return result

    def ensure_nsight_slurm(self) -> None:
        """Write the job-local wrapper configuration and start the coordinator (idempotent)."""
        prof = self.config.profiling
        if not prof.is_nsight_slurm or self._nsight_slurm_ready:
            return
        launcher = Path(prof.nsight_slurm_bin())
        connector = Path(prof.nsight_slurm_bin("nsight-slurm-connector"))
        for exe in (launcher, connector):
            if not exe.is_file() or not os.access(exe, os.X_OK):
                raise RuntimeError(
                    f"nsight-slurm executable missing or not executable: {exe} "
                    "(profiling.nsight_slurm_home must be a uv tool install of nsight-slurm for the compute arch)"
                )
        report_root = Path(self.runtime.log_dir) / NSIGHT_SLURM_REPORT_SUBDIR
        report_root.mkdir(parents=True, exist_ok=True)

        logger.info(
            "nsight-slurm: configuring job (home=%s, mode=%s)", prof.nsight_slurm_home, prof.nsight_slurm_profiling_mode
        )
        self._nsight_slurm_run("configure", "tool-path", prof.nsight_slurm_tool_path)
        self._nsight_slurm_run("configure", "tool-command", "profile")
        self._nsight_slurm_run("configure", "profiling-mode", prof.nsight_slurm_profiling_mode)
        self._nsight_slurm_run("configure", "tool-options", *prof.nsight_slurm_effective_tool_options())
        self._nsight_slurm_run("configure", "report-output", str(report_root))
        if self.runtime.container_image:
            self._nsight_slurm_run("enable", "pyxis")
        self._nsight_slurm_run("configure")  # prints the effective configuration into the log
        self._nsight_slurm_run("coordinator", "start")
        self._nsight_slurm_coordinator_started = True
        self._nsight_slurm_ready = True
        logger.info(
            "nsight-slurm: coordinator started; worker steps launch via %s, reports under %s",
            shlex.join(prof.nsight_slurm_launcher()),
            report_root,
        )

    def stop_nsight_slurm(self) -> None:
        """Stop the explicit coordinator (best effort; called from the sweep's cleanup)."""
        if not self._nsight_slurm_coordinator_started:
            return
        try:
            self._nsight_slurm_run("coordinator", "stop", "--force", check=False)
        except Exception as exc:  # noqa: BLE001 - cleanup must never raise
            logger.warning("nsight-slurm: coordinator stop failed: %s", exc)
        self._nsight_slurm_coordinator_started = False

    def nsight_slurm_launch_kwargs(self) -> dict[str, Any]:
        """kwargs for start_srun_process when the step must go through the wrapper (else {})."""
        prof = self.config.profiling
        if not prof.is_nsight_slurm:
            return {}
        self.ensure_nsight_slurm()
        return {
            "srun_launcher": prof.nsight_slurm_launcher(),
            "launcher_env": prof.nsight_slurm_process_env(self.runtime.log_dir),
        }
