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
    nsight-slurm disable pyxis           # srtctl mounts the install/connector itself (single --container-mounts)
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
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from srtctl.core.processes import ProcessRegistry
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

NSIGHT_SLURM_REPORT_SUBDIR = "nsight-slurm-reports"
NSIGHT_SLURM_LOG_NAME = "nsight-slurm.out"
NSIGHT_SLURM_RUNTIME_SUBDIR = "nsight-slurm-runtime"


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
        # -o last: nsys writes each rank's report straight into the report root (see schema).
        self._nsight_slurm_run(
            "configure",
            "tool-options",
            *prof.nsight_slurm_effective_tool_options(),
            *prof.nsight_slurm_output_option(self.runtime.log_dir),
        )
        (Path(self.runtime.log_dir) / prof.NSIGHT_SLURM_DIRECT_REPORT_SUBDIR).mkdir(parents=True, exist_ok=True)
        self._nsight_slurm_run("configure", "report-output", str(report_root))
        # The connector writes nsys output into its runtime workspace and only copies it to the report
        # root when its state machine sees a collection stop. With the nsys 2026.3 agent the cuda-api
        # range states are not recognised (RangeCollection/RangeGeneration), so keep that workspace on
        # the shared filesystem (bind-mounted at a short in-container path) instead of the
        # container-local /tmp: reports then survive the step and flush_nsight_slurm() rescues them.
        # The in-container path (/nsrt, see ProfilingConfig) reaches the connector through the
        # launcher env (NSIGHT_SLURM_RUNTIME_DIR); it is bind-mounted from this host directory.
        (Path(self.runtime.log_dir) / NSIGHT_SLURM_RUNTIME_SUBDIR).mkdir(parents=True, exist_ok=True)
        # Not `enable pyxis`: it appends a second --container-mounts flag and pyxis keeps only the
        # last one, dropping /model, /logs and /configs (hecate jobs 571147/571265). srtctl mounts
        # what the connector needs itself (see ProfilingConfig.nsight_slurm_container_mounts).
        self._nsight_slurm_run("disable", "pyxis")
        self._nsight_slurm_run("configure")  # prints the effective configuration into the log
        self._nsight_slurm_run("coordinator", "start")
        self._nsight_slurm_coordinator_started = True
        self._nsight_slurm_ready = True
        logger.info(
            "nsight-slurm: coordinator started; worker steps launch via %s, reports under %s",
            shlex.join(prof.nsight_slurm_launcher()),
            report_root,
        )

    def flush_nsight_slurm(
        self,
        registry: ProcessRegistry | None,
        *,
        timeout_s: float = 240.0,
        first_wait_s: float = 45.0,
        settle_s: float = 20.0,
        poll_s: float = 5.0,
    ) -> int:
        """End the nsys sessions and wait for the reports BEFORE the worker steps are killed.

        In cuda-api mode a capture range closes at cudaProfilerStop, but the wrapper keeps the nsys
        session open for the next range (``--capture-range-end repeat``), so a report only exists
        once the session ends. Steps that die by SIGKILL (registry.cleanup + batch-script exit)
        lose everything they captured (hecate job 571904). Two graceful triggers, in order:

        1. ``nsight-slurm stop --job <id>``: coordinator -> connectors -> ``nsys stop``.
        2. If no report showed up: SIGTERM to the wrapper processes of the worker steps; the wrapper
           relays it to srun -> connector -> nsys, and nsys (``--kill none``) writes its report on
           SIGTERM ("Processing events... Generated: ...") while the application keeps running.

        Then wait for ``*.nsys-rep`` files under the report root to appear and settle. Returns the
        number of report files found. Best effort: never raises.
        """
        if not self._nsight_slurm_coordinator_started:
            return 0
        report_root = Path(self.runtime.log_dir) / NSIGHT_SLURM_REPORT_SUBDIR

        runtime_dir = Path(self.runtime.log_dir) / NSIGHT_SLURM_RUNTIME_SUBDIR

        def rescue_scratch_reports() -> int:
            """Copy nsys reports still sitting in the connectors' runtime workspaces into the report root."""
            if not runtime_dir.exists():
                return 0
            rescued = 0
            for src in runtime_dir.rglob("*.nsys-rep"):
                dst = report_root / "rescued" / src.relative_to(runtime_dir)
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    continue
                try:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    rescued += 1
                except OSError as exc:
                    logger.warning("nsight-slurm: could not rescue %s: %s", src, exc)
            return rescued

        direct_dir = Path(self.runtime.log_dir) / self.config.profiling.NSIGHT_SLURM_DIRECT_REPORT_SUBDIR

        def count_reports() -> int:
            return sum(sum(1 for _ in d.rglob("*.nsys-rep")) for d in (report_root, direct_dir) if d.exists())

        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id:
            try:
                self._nsight_slurm_run("stop", "--job", job_id, "--timeout", "30", check=False)
            except Exception as exc:  # noqa: BLE001 - teardown must never raise
                logger.warning("nsight-slurm: stop failed: %s", exc)

        deadline = time.monotonic() + timeout_s
        first_deadline = time.monotonic() + first_wait_s
        rescue_scratch_reports()
        baseline = last = count_reports()  # reports that existed before `stop` do not prove anything ended
        stable_since = time.monotonic()
        signalled = False
        while time.monotonic() < deadline:
            rescue_scratch_reports()  # cheap; connectors delete their workspace when they exit
            n = count_reports()
            now = time.monotonic()
            if n != last:
                last, stable_since = n, now
            new_reports = n > baseline
            if new_reports and now - stable_since >= settle_s and (signalled or now >= first_deadline):
                break
            if not signalled and now >= first_deadline and not new_reports:
                signalled = True
                self._nsight_slurm_signal_workers(registry)
            time.sleep(poll_s)
        rescued = rescue_scratch_reports()
        if rescued:
            logger.info("nsight-slurm: rescued %d report(s) from the runtime workspaces", rescued)
        last = count_reports()
        logger.info("nsight-slurm: %d report file(s) under %s / %s after flush", last, report_root, direct_dir)
        return last

    NSIGHT_SLURM_CONNECTOR_STEP_NAME: str = "nsight-slurm-connector"

    @staticmethod
    def _slurm_cmd(args: list[str]) -> str:
        """Run a Slurm CLI command on the orchestrator node and return its stdout (empty on failure)."""
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=60, check=False).stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("nsight-slurm: %s failed: %s", args[0], exc)
            return ""

    def _nsight_slurm_connector_steps(self, job_id: str) -> list[str]:
        """Step ids of the wrapper-launched worker steps (their task command is the connector)."""
        out = self._slurm_cmd(["squeue", "-s", "-j", job_id, "-h", "-o", "%i|%j"])
        steps = []
        for line in out.splitlines():
            step_id, _, name = line.strip().partition("|")
            if name == self.NSIGHT_SLURM_CONNECTOR_STEP_NAME and step_id:
                steps.append(step_id)
        return steps

    def _nsight_slurm_signal_workers(self, registry: ProcessRegistry | None) -> None:
        """SIGTERM the connector TASKS of the worker steps so nsys ends its session and writes its report.

        Not the wrapper processes: the wrapper relays a SIGTERM to its srun, and srun then cancels
        the step with SIGKILL ("STEP ... CANCELLED ... DUE to SIGNAL Killed", hecate job 576022), which
        loses the report. `scancel --signal=TERM <job>.<step>` reaches only the tasks, i.e. the
        connectors, which terminate nsys gracefully (``--kill none`` leaves the application running).
        """
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return
        steps = self._nsight_slurm_connector_steps(job_id)
        if not steps:
            logger.warning("nsight-slurm: no %s steps found to signal", self.NSIGHT_SLURM_CONNECTOR_STEP_NAME)
            return
        for step in steps:
            logger.info(
                "nsight-slurm: SIGTERM step %s (connector tasks) so its nsys sessions end and reports are written", step
            )
            self._slurm_cmd(["scancel", "--signal=TERM", step])

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
            # In srtctl's own (single) --container-mounts flag; see nsight_slurm_container_mounts.
            "extra_container_mounts": prof.nsight_slurm_container_mounts(self.runtime.log_dir),
        }
