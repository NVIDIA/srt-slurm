# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The companion job: the ``placement.node: dedicated`` nodes on ``slurm.dedicated_partition``.

They cannot be a het-job component: Slurm checks a het job's QOS limits against the summed TRES of
its components (``acct_policy_validate_het_job``), so a CPU QOS that caps GPUs at zero rejects a
GPU + CPU job. The batch script submits a second job instead; the orchestrator waits for it and
publishes it (``slurm.COMPANION_JOB_ENV``), and ``slurm.step_target`` runs the dedicated roles'
steps in it. A watchdog ends it once the main job has left the queue.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from srtctl.core.slurm import COMPANION_JOB_ENV, COMPANION_NODELIST_ENV, _companion_step_env, _squeue_nodelist

if TYPE_CHECKING:
    from srtctl.core.schema import DedicatedAllocation

logger = logging.getLogger(__name__)

START_TIMEOUT_SECONDS = 1800.0
POLL_SECONDS = 5.0
# With enable_stepmgr and no PrologFlags=DeferBatch, a just-RUNNING job refuses steps for a few seconds.
STEP_READY_TIMEOUT_SECONDS = 180.0
_ENDED = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "TIMEOUT"}
# Set by the batch script when it submitted the companion (job_script_minimal.j2).
SUBMITTED_ENV = "SRTCTL_COMPANION_SUBMITTED"
MAIN_JOB_ENV = "SRTCTL_MAIN_JOB_ID"
# Hold the nodes while the main job is in the queue; three misses in a row tolerate a slow squeue.
_WATCHDOG = (
    'misses=0; while [ "$misses" -lt 3 ]; do '
    f'if [ -n "$(squeue -h -j ${MAIN_JOB_ENV} -o %T 2>/dev/null)" ]; then misses=0; else misses=$((misses + 1)); fi; '
    "sleep 30; done"
)


def sbatch_command(
    allocation: DedicatedAllocation,
    *,
    main_job_id: str,
    job_name: str,
    account: str | None,
    time_limit: str,
    output: str,
) -> list[str]:
    """The ``sbatch`` call that submits the companion job."""
    options = {"account": account, **allocation.sbatch_directives} if account else allocation.sbatch_directives
    return [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--partition={allocation.partition}",
        f"--nodes={allocation.nodes}",
        f"--ntasks={allocation.nodes}",
        "--ntasks-per-node=1",
        f"--time={time_limit}",
        f"--output={output}",
        f"--export=ALL,{MAIN_JOB_ENV}={main_job_id}",
        *(f"--{key}={value}" if value else f"--{key}" for key, value in options.items()),
        "--wrap",
        _WATCHDOG,
    ]


def submit_script_line(allocation: DedicatedAllocation, *, job_name: str, account: str | None, time_limit: str) -> str:
    """``sbatch_command`` for the batch script, which runs it before its setup so the queue wait overlaps it."""
    command = sbatch_command(
        allocation, main_job_id="@MAIN@", job_name=job_name, account=account, time_limit=time_limit, output="@OUT@"
    )
    # Both placeholders sit in unquoted words, so the script's variables stay expandable.
    return shlex.join(command).replace("@MAIN@", '"${JOB_ID}"').replace("@OUT@", '"${LOG_DIR}/companion_%j.log"')


def _state(job_id: str) -> str:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"], check=False, capture_output=True, text=True, timeout=30
    )
    return result.stdout.strip() if result.returncode == 0 else ""


@dataclass(frozen=True)
class CompanionJob:
    """A running companion job and its nodes."""

    job_id: str
    nodes: tuple[str, ...]

    @classmethod
    def start(
        cls,
        allocation: DedicatedAllocation,
        *,
        main_job_id: str,
        job_name: str,
        account: str | None,
        time_limit: str,
        output: str,
    ) -> CompanionJob:
        """Adopt the companion the batch script submitted, else submit one; wait until it runs and publish it."""
        job_id = os.environ.get(SUBMITTED_ENV, "").strip()
        if not job_id:
            command = sbatch_command(
                allocation,
                main_job_id=main_job_id,
                job_name=job_name,
                account=account,
                time_limit=time_limit,
                output=output,
            )
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(f"companion job submission failed: {(result.stderr or result.stdout).strip()}")
            job_id = result.stdout.strip().split(";")[0]
        try:
            nodes = cls._wait_until_running(job_id)
            cls._wait_until_steps_run(job_id, nodes)
        except BaseException:
            subprocess.run(["scancel", job_id], check=False, capture_output=True, timeout=30)
            raise
        os.environ[COMPANION_JOB_ENV] = job_id
        os.environ[COMPANION_NODELIST_ENV] = ",".join(nodes)
        logger.info("Companion job %s running on %s", job_id, ",".join(nodes))
        return cls(job_id=job_id, nodes=tuple(nodes))

    @staticmethod
    def _wait_until_running(job_id: str) -> list[str]:
        deadline = time.monotonic() + START_TIMEOUT_SECONDS
        missing = 0
        while True:
            state = _state(job_id)
            if state == "RUNNING" and (nodes := _squeue_nodelist(job_id, 30)):
                return nodes
            missing = 0 if state else missing + 1
            if state in _ENDED or missing >= 6:
                raise RuntimeError(f"companion job {job_id} ended before it ran ({state or 'left the queue'})")
            if time.monotonic() > deadline:
                raise TimeoutError(f"companion job {job_id} did not start within {START_TIMEOUT_SECONDS:.0f}s")
            time.sleep(POLL_SECONDS)

    @staticmethod
    def _wait_until_steps_run(job_id: str, nodes: list[str]) -> None:
        """Retry a no-op step on every companion node until the job accepts steps."""
        n = str(len(nodes))
        probe = ["srun", "--jobid", job_id, "--overlap", "--nodes", n, "--ntasks", n, "--time=1", "true"]
        deadline = time.monotonic() + STEP_READY_TIMEOUT_SECONDS
        while (
            result := subprocess.run(
                probe, env=_companion_step_env(), check=False, capture_output=True, text=True, timeout=120
            )
        ).returncode:
            if time.monotonic() > deadline:
                raise RuntimeError(f"companion job {job_id} refuses steps: {result.stderr.strip()}")
            time.sleep(POLL_SECONDS)

    def release(self) -> None:
        """Cancel the companion job and withdraw it from the environment."""
        os.environ.pop(COMPANION_JOB_ENV, None)
        os.environ.pop(COMPANION_NODELIST_ENV, None)
        subprocess.run(["scancel", self.job_id], check=False, capture_output=True, timeout=30)
