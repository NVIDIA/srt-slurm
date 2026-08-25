# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for opt-in recording of realized Slurm launch commands."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.core.launch_plan import configure_launch_plan
from srtctl.core.schema import ClusterConfig, OutputConfig
from srtctl.core.slurm import start_srun_process


@pytest.fixture(autouse=True)
def _reset_launch_plan():
    configure_launch_plan(None)
    yield
    configure_launch_plan(None)


def _launch(tmp_path: Path, **kwargs) -> Path:
    plan_dir = tmp_path / "logs" / "launch-plan"
    configure_launch_plan(plan_dir, job_id="42042")
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="42042"),
        patch("srtctl.core.slurm._get_cluster_bash_preamble", return_value=None),
        patch("subprocess.Popen", return_value=MagicMock()),
    ):
        start_srun_process(["python3", "-m", "worker"], **kwargs)
    return plan_dir


def test_launch_plan_is_opt_in_for_recipe_and_cluster() -> None:
    assert OutputConfig().record_launch_plan is False
    assert OutputConfig(record_launch_plan=True).record_launch_plan is True
    assert ClusterConfig().record_launch_plan is False
    assert ClusterConfig(record_launch_plan=True).record_launch_plan is True


def test_disabled_recorder_writes_nothing(tmp_path: Path) -> None:
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="42042"),
        patch("srtctl.core.slurm._get_cluster_bash_preamble", return_value=None),
        patch("subprocess.Popen", return_value=MagicMock()),
    ):
        start_srun_process(["python3", "-m", "worker"])

    assert not (tmp_path / "launch-plan").exists()


def test_records_exact_realized_srun_as_executable_script(tmp_path: Path) -> None:
    output = tmp_path / "logs" / "node-a_prefill_w0.out"
    plan_dir = _launch(
        tmp_path,
        nodelist=["node-a"],
        output=str(output),
        container_image="/containers/vllm.sqsh",
        container_mounts={Path("/models/model"): Path("/model")},
        env_to_set={"NCCL_DEBUG": "INFO"},
        het_group=0,
    )

    scripts = list(plan_dir.glob("*.sh"))
    assert [path.name for path in scripts] == ["001-node-a_prefill_w0.sh"]
    script = scripts[0].read_text()
    assert scripts[0].stat().st_mode & 0o100
    assert "exec srun --jobid 42042 --overlap" in script
    assert "--nodelist node-a" in script
    assert "--het-group=0" in script
    assert "--container-image /containers/vllm.sqsh" in script
    assert "export NCCL_DEBUG=INFO" in script
    assert "python3 -m worker" in script
    syntax = subprocess.run(["bash", "-n", str(scripts[0])], capture_output=True, text=True, check=False)
    assert syntax.returncode == 0, syntax.stderr

    manifest = json.loads((plan_dir / "manifest.json").read_text())
    assert manifest["job_id"] == "42042"
    assert manifest["steps"][0]["script"] == scripts[0].name
    assert manifest["steps"][0]["nodelist"] == ["node-a"]
    assert manifest["steps"][0]["container_image"] == "/containers/vllm.sqsh"


def test_secret_environment_values_are_not_persisted(tmp_path: Path) -> None:
    plan_dir = _launch(
        tmp_path,
        output=str(tmp_path / "logs" / "benchmark.out"),
        env_to_set={"HF_TOKEN": "hf-super-secret", "PUBLIC_VALUE": "visible"},
        srun_export_env={"AWS_SECRET_ACCESS_KEY": "aws-super-secret", "ENROOT_REMAP_ROOT": "yes"},
    )

    script = next(plan_dir.glob("*.sh")).read_text()
    manifest_text = (plan_dir / "manifest.json").read_text()
    combined = script + manifest_text
    assert "hf-super-secret" not in combined
    assert "aws-super-secret" not in combined
    assert 'export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN to replay this step}"' in script
    assert "--export=ALL,AWS_SECRET_ACCESS_KEY,ENROOT_REMAP_ROOT=yes" in script
    assert "export PUBLIC_VALUE=visible" in script

    manifest = json.loads(manifest_text)
    assert manifest["steps"][0]["required_secret_environment"] == ["AWS_SECRET_ACCESS_KEY", "HF_TOKEN"]


def test_multiple_commands_have_stable_order_and_distinct_files(tmp_path: Path) -> None:
    plan_dir = _launch(tmp_path, output=str(tmp_path / "logs" / "infra.out"))
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="42042"),
        patch("srtctl.core.slurm._get_cluster_bash_preamble", return_value=None),
        patch("subprocess.Popen", return_value=MagicMock()),
    ):
        start_srun_process(["python3", "bench.py"], output=str(tmp_path / "logs" / "benchmark.out"))

    manifest = json.loads((plan_dir / "manifest.json").read_text())
    assert [entry["sequence"] for entry in manifest["steps"]] == [1, 2]
    assert [entry["script"] for entry in manifest["steps"]] == ["001-infra.sh", "002-benchmark.sh"]
