# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`slurm.dedicated_partition`: the dedicated nodes as a companion job on another partition."""

import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.cli.submit import generate_minimal_sbatch_script, planned_total_nodes
from srtctl.core.companion import CompanionJob
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.launcher import LaunchSpec, SlurmLauncher
from srtctl.core.runtime import Nodes
from srtctl.core.schema import (
    BenchmarkConfig,
    FrontendConfig,
    ModelConfig,
    PlacementConfig,
    ResourceConfig,
    RoleConfig,
    SlurmConfig,
    SrtConfig,
)
from srtctl.core.slurm import COMPANION_JOB_ENV, COMPANION_NODELIST_ENV, StepTarget, step_target
from srtctl.services import ServiceConfig, ServicePlacementConfig

DEDICATED = PlacementConfig(node="dedicated")
COMPANION_ENV = {
    "SLURM_JOB_ID": "42",
    "SLURM_STEPMGR": "batch-host",
    "SLURM_GPUS_PER_NODE": "8",
    "SLURM_JOB_ACCOUNT": "acct",
    "SLURM_CONF": "/etc/slurm/slurm.conf",
    "HOME": "/home/u",
    COMPANION_JOB_ENV: "99",
    COMPANION_NODELIST_ENV: "cpu-1",
}


def _config(*, dedicated_partition: str | None = "cpu", infra: bool = False, colocate: bool = True, **directives):
    services = [ServiceConfig(name="etcd", type="etcd", placement=ServicePlacementConfig(node="dedicated"))]
    return SrtConfig(
        name="t",
        model=ModelConfig(path="/m", container="/c.sqsh", precision="fp8"),
        resources=ResourceConfig(gpu_type="gb300", gpus_per_node=4),
        slurm=SlurmConfig(account="acct", partition="batch", dedicated_partition=dedicated_partition),
        roles={"agg": RoleConfig(nodes=2, workers=2)},
        services=services if infra else [],
        frontend=FrontendConfig(type="dynamo", placement=DEDICATED),
        benchmark=BenchmarkConfig(type="manual", placement=DEDICATED, colocate_with_frontend=colocate),
        dedicated_sbatch_directives=directives,
    )


def test_allocation():
    assert _config(dedicated_partition=None).dedicated_allocation is None
    assert planned_total_nodes(_config(dedicated_partition=None)) == 3
    assert _config(dedicated_partition="batch").dedicated_allocation is None
    colocated = _config(infra=True).dedicated_allocation
    assert (colocated.partition, colocated.nodes, colocated.sbatch_directives) == (
        "cpu",
        1,
        {"exclusive": "", "mem": "0"},
    )
    assert _config(infra=True, colocate=False).dedicated_allocation.nodes == 3
    assert _config(mem="64G").dedicated_allocation.sbatch_directives["mem"] == "64G"


def test_cluster_defaults_apply_and_the_recipe_wins():
    cluster = {"default_dedicated_partition": "cpu", "default_dedicated_sbatch_directives": {"qos": "cpu-short"}}
    resolved = resolve_config_with_defaults({"schema": 2, "name": "t"}, cluster)
    assert resolved["slurm"]["dedicated_partition"] == "cpu"
    assert resolved["dedicated_sbatch_directives"] == {"qos": "cpu-short"}
    recipe = {"schema": 2, "name": "t", "dedicated_sbatch_directives": {"qos": "cpu-normal"}}
    assert resolve_config_with_defaults(recipe, cluster)["dedicated_sbatch_directives"] == {"qos": "cpu-normal"}


def test_batch_script_asks_for_engine_nodes_and_requests_the_companion_before_setup():
    script = generate_minimal_sbatch_script(_config(qos="cpu-short"), Path("/tmp/t.yaml"))
    assert "#SBATCH --nodes=2" in script and "hetjob" not in script
    line = next(line for line in script.splitlines() if line.startswith("export SRTCTL_COMPANION_SUBMITTED="))
    assert script.index(line) < script.index("SRTCTL_OUTPUT_DIR")
    for part in ("--partition=cpu", "--exclusive", "--qos=cpu-short", '--export=ALL,SRTCTL_MAIN_JOB_ID="${JOB_ID}"'):
        assert part in line
    assert "squeue -h -j $SRTCTL_MAIN_JOB_ID" in line  # the watchdog
    no_companion = generate_minimal_sbatch_script(_config(dedicated_partition=None), Path("/tmp/t.yaml"))
    assert "SRTCTL_COMPANION_SUBMITTED" not in no_companion


def _fake_slurm(states: list[str], step_refusals: int = 0):
    calls: list[list[str]] = []

    def run(cmd, **_kwargs):
        nonlocal step_refusals
        calls.append(cmd)
        if cmd[0] == "sbatch":
            return subprocess.CompletedProcess(cmd, 0, "123\n", "")
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, states.pop(0) + "\n", "")
        if cmd[0] == "srun" and step_refusals:
            step_refusals -= 1
            return subprocess.CompletedProcess(cmd, 1, "", "srun: error: Invalid job id specified")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return calls, run


def _start(run, env: dict[str, str] | None = None) -> CompanionJob:
    with (
        patch.dict(os.environ, env or {}),
        patch("srtctl.core.companion.subprocess.run", side_effect=run),
        patch("srtctl.core.companion._squeue_nodelist", return_value=["cpu-1"]),
        patch("srtctl.core.companion.time.sleep"),
    ):
        companion = CompanionJob.start(
            _config().dedicated_allocation,
            main_job_id="42",
            job_name="t",
            account="acct",
            time_limit="01:00:00",
            output="o",
        )
        published = (os.environ[COMPANION_JOB_ENV], os.environ[COMPANION_NODELIST_ENV])
        companion.release()
        assert COMPANION_JOB_ENV not in os.environ
    assert published == (companion.job_id, "cpu-1")
    return companion


def test_start_submits_waits_until_steps_run_and_releases():
    # A job reported RUNNING before its prolog finishes refuses steps for a few seconds.
    calls, run = _fake_slurm(["PENDING", "RUNNING"], step_refusals=2)
    assert _start(run) == CompanionJob("123", ("cpu-1",))
    assert len([cmd for cmd in calls if cmd[0] == "srun"]) == 3
    assert calls[-1] == ["scancel", "123"]


def test_start_adopts_the_companion_the_batch_script_submitted():
    calls, run = _fake_slurm(["RUNNING"])
    assert _start(run, {"SRTCTL_COMPANION_SUBMITTED": "555"}).job_id == "555"
    assert not any(cmd[0] == "sbatch" for cmd in calls)


def test_a_companion_that_never_runs_is_cancelled():
    calls, run = _fake_slurm(["PENDING", "CANCELLED"])
    with pytest.raises(RuntimeError, match="ended before it ran"):
        _start(run)
    assert ["scancel", "123"] in calls


def test_companion_steps_run_in_the_companion_with_a_clean_slurm_environment():
    with patch.dict(os.environ, {k: v for k, v in COMPANION_ENV.items() if "COMPANION" not in k}, clear=True):
        assert step_target(1) == StepTarget("42", 1, None)  # no companion: a het group
    with patch.dict(os.environ, COMPANION_ENV, clear=True):
        assert step_target(0) == StepTarget("42", None, None)
        spec = LaunchSpec(command=["true"], nodes=1, ntasks=1, nodelist=["cpu-1"], het_group=1)
        with patch("srtctl.core.launcher.subprocess.Popen", return_value=MagicMock()) as popen:
            SlurmLauncher().launch(spec)
    command, env = popen.call_args.args[0], popen.call_args.kwargs["env"]
    assert command[command.index("--jobid") + 1] == "99"
    assert not any(arg.startswith("--het-group") for arg in command)
    assert {k for k in env if k.startswith("SLURM_")} == {"SLURM_CONF", "SLURM_ACCOUNT"}
    assert env["SLURM_ACCOUNT"] == "acct"


def test_graceful_shutdown_finds_steps_in_the_companion():
    result = subprocess.CompletedProcess([], 0, "42.3 worker\n99.0 frontend_0\n", "")
    with (
        patch.dict(os.environ, COMPANION_ENV, clear=True),
        patch("srtctl.core.launcher.shutil.which", return_value="/usr/bin/squeue"),
        patch("srtctl.core.launcher.subprocess.run", return_value=result) as run,
    ):
        assert SlurmLauncher().list_step_ids() == {"worker": "42.3", "frontend_0": "99.0"}
    assert "--jobs=42,99" in run.call_args.args[0]


@pytest.mark.parametrize("colocate", [True, False])
def test_dedicated_roles_take_the_companion_nodes(colocate):
    config = _config(infra=True, colocate=colocate)
    companion = ["cpu-1"] if colocate else ["cpu-1", "cpu-2", "cpu-3"]
    with (
        patch.dict(os.environ, {COMPANION_JOB_ENV: "99", COMPANION_NODELIST_ENV: ",".join(companion)}),
        patch("srtctl.core.runtime.get_slurm_nodelist", return_value=["gpu-1", "gpu-2"]),
    ):
        nodes = Nodes.from_slurm(
            frontend_dedicated_node=True,
            client_dedicated_node=True,
            etcd_nats_dedicated_node=True,
            colocate_dedicated_nodes=config.benchmark.colocate_with_frontend,
        )
    assert nodes.worker == ("gpu-1", "gpu-2")
    assert {nodes.head, nodes.bench, nodes.infra} == set(companion)
    assert (nodes.het_group_for("cpu-1"), nodes.het_group_for("gpu-1")) == (1, 0)
    assert not nodes.prefill_decode_split


def test_every_srun_on_a_dedicated_node_targets_the_companion():
    sys.path.insert(0, str(Path(__file__).parent))
    from launch_snapshots import EXAMPLES_DIR, render_launch_plan

    plan = render_launch_plan(EXAMPLES_DIR / "features" / "dedicated-partition.yaml")
    assert "# exit_code: 0" in plan
    steps = {}
    for block in plan.split("\n## ")[1:]:
        nodelist = re.search(r"^nodelist: (.+)$", block, re.MULTILINE)
        het_group = re.search(r"^het_group: (\d+)$", block, re.MULTILINE)
        steps[block.splitlines()[0]] = (nodelist and nodelist.group(1), het_group and int(het_group.group(1)))
    on_cpu = {name: group for name, (nodelist, group) in steps.items() if nodelist and "mock-cpu-" in nodelist}
    assert {"frontend_0", "benchmark"} <= on_cpu.keys() and set(on_cpu.values()) == {1}
    assert all(not (nodelist and "mock-cpu-" in nodelist) for name, (nodelist, _) in steps.items() if "agg" in name)
