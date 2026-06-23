# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SLURM command construction."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from srtctl.cli.mixins.worker_stage import WorkerStageMixin
from srtctl.core.schema import ObservabilityConfig
from srtctl.core.slurm import get_slurm_het_nodelists, start_srun_process


def _built_bash_command(mock_popen: MagicMock) -> str:
    srun_cmd = mock_popen.call_args.args[0]
    assert srun_cmd[-3:-1] == ["bash", "-c"]
    return srun_cmd[-1]


def test_start_srun_exports_env_before_preamble() -> None:
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="12345"),
        patch("srtctl.core.slurm._get_cluster_bash_preamble", return_value=None),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value = MagicMock()
        start_srun_process(
            ["python3", "-m", "server"],
            env_to_set={"NCCL_DEBUG": "INFO"},
            bash_preamble="echo preamble",
        )

    bash_cmd = _built_bash_command(mock_popen)
    assert bash_cmd.index("export NCCL_DEBUG=INFO") < bash_cmd.index("echo preamble")
    assert bash_cmd.index("echo preamble") < bash_cmd.index("python3 -m server")


def test_cluster_bash_preamble_runs_before_exports_and_local_preamble() -> None:
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="12345"),
        patch(
            "srtctl.core.slurm._get_cluster_bash_preamble",
            return_value="ulimit -n 1048576",
        ),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value = MagicMock()
        start_srun_process(
            ["python3", "-m", "server"],
            env_to_set={"NCCL_DEBUG": "INFO"},
            bash_preamble="echo local",
        )

    bash_cmd = _built_bash_command(mock_popen)
    # ulimit must come first so it applies to everything downstream.
    assert bash_cmd.index("ulimit -n 1048576") < bash_cmd.index("export NCCL_DEBUG=INFO")
    assert bash_cmd.index("export NCCL_DEBUG=INFO") < bash_cmd.index("echo local")
    assert bash_cmd.index("echo local") < bash_cmd.index("python3 -m server")


def test_cluster_bash_preamble_applied_when_only_cluster_set() -> None:
    """Cluster preamble alone should land in the bash wrapper even with no local preamble or env."""
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="12345"),
        patch(
            "srtctl.core.slurm._get_cluster_bash_preamble",
            return_value="ulimit -n 1048576",
        ),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value = MagicMock()
        start_srun_process(["python3", "-m", "server"])

    bash_cmd = _built_bash_command(mock_popen)
    assert bash_cmd.startswith("ulimit -n 1048576 && python3 -m server")


def test_cluster_bash_preamble_warns_when_bash_wrapper_disabled(caplog) -> None:
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="12345"),
        patch(
            "srtctl.core.slurm._get_cluster_bash_preamble",
            return_value="ulimit -n 1048576",
        ),
        patch("subprocess.Popen") as mock_popen,
        caplog.at_level("WARNING", logger="srtctl.core.slurm"),
    ):
        mock_popen.return_value = MagicMock()
        start_srun_process(["/bin/node_exporter"], use_bash_wrapper=False)

    srun_cmd = mock_popen.call_args.args[0]
    # Distroless path runs the binary directly; preamble cannot apply.
    assert "bash" not in srun_cmd
    assert any("default_bash_preamble" in record.message for record in caplog.records)


def test_srun_options_use_equals_separator() -> None:
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="12345"),
        patch("srtctl.core.slurm._get_cluster_bash_preamble", return_value=None),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value = MagicMock()
        start_srun_process(
            ["python3", "-m", "server"],
            srun_options={"cpu-bind": "none", "export": "ALL", "exclusive": ""},
        )

    srun_cmd = mock_popen.call_args.args[0]
    assert "--cpu-bind=none" in srun_cmd
    assert "--export=ALL" in srun_cmd
    assert "--exclusive" in srun_cmd


def test_wrapped_nonfatal_hook_does_not_mask_prior_preamble_failure() -> None:
    bash_cmd = "false && ( false || true ) && echo main"

    result = subprocess.run(["bash", "-c", bash_cmd], capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "main" not in result.stdout


def test_worker_stage_wraps_nonfatal_fingerprint_hook(tmp_path: Path) -> None:
    backend = MagicMock()
    backend.build_worker_command.return_value = ["python3", "-m", "worker"]
    backend.get_environment_for_mode.return_value = {}
    backend.get_process_environment.return_value = {}

    mixin = WorkerStageMixin()
    mixin.config = SimpleNamespace(
        setup_script="setup.sh",
        frontend=SimpleNamespace(type="sglang"),
        dynamo=SimpleNamespace(install=False),
        observability=ObservabilityConfig(),
        profiling=SimpleNamespace(enabled=False, is_nsys=False),
        backend=backend,
    )
    mixin.runtime = SimpleNamespace(
        log_dir=tmp_path,
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        network_interface=None,
        nodes=SimpleNamespace(infra="infra-node", worker=["node-a"]),
        gpus_per_node=8,
        environment={},
        container_image=Path("/container.sqsh"),
        container_mounts={},
        srun_options=[],
    )
    process = SimpleNamespace(
        endpoint_mode="prefill",
        endpoint_index=0,
        node="node-a",
        sys_port=5000,
        gpu_indices=list(range(8)),
        cuda_visible_devices="0,1,2,3,4,5,6,7",
        het_group=None,
    )

    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_worker(process, [process])

    bash_preamble = mock_srun.call_args.kwargs["bash_preamble"]
    assert "setup.sh" in bash_preamble
    assert "/configs/patches/${setup_script}" in bash_preamble
    assert bash_preamble.endswith("&& ( fingerprint || true )")


# ---- Heterogeneous-job nodelist parsing ----


def test_get_slurm_het_nodelists_returns_none_without_het_size() -> None:
    with patch.dict("os.environ", {}, clear=False):
        # Make sure SLURM_HET_SIZE is unset
        import os

        os.environ.pop("SLURM_HET_SIZE", None)
        assert get_slurm_het_nodelists() is None


def test_get_slurm_het_nodelists_returns_none_for_size_one() -> None:
    with patch.dict("os.environ", {"SLURM_HET_SIZE": "1"}):
        assert get_slurm_het_nodelists() is None


def test_get_slurm_het_nodelists_expands_two_groups() -> None:
    env = {
        "SLURM_HET_SIZE": "2",
        "SLURM_JOB_NODELIST_HET_GROUP_0": "gb200-[01-03]",
        "SLURM_JOB_NODELIST_HET_GROUP_1": "gb200-[04-05]",
    }

    def mock_run(cmd, **kwargs):
        result = MagicMock()
        # cmd[-1] is the raw nodelist passed to `scontrol show hostnames`
        nodelist_raw = cmd[-1]
        if nodelist_raw == "gb200-[01-03]":
            result.stdout = "gb200-01\ngb200-02\ngb200-03\n"
        elif nodelist_raw == "gb200-[04-05]":
            result.stdout = "gb200-04\ngb200-05\n"
        else:
            raise AssertionError(f"unexpected nodelist {nodelist_raw}")
        result.returncode = 0
        return result

    with patch.dict("os.environ", env), patch("subprocess.run", side_effect=mock_run):
        groups = get_slurm_het_nodelists()
    assert groups == [["gb200-01", "gb200-02", "gb200-03"], ["gb200-04", "gb200-05"]]


def test_start_srun_emits_het_group_flag() -> None:
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="12345"),
        patch("srtctl.core.slurm._get_cluster_bash_preamble", return_value=None),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value = MagicMock()
        start_srun_process(["echo", "hi"], het_group=1)

    srun_cmd = mock_popen.call_args.args[0]
    assert "--het-group=1" in srun_cmd


def test_start_srun_omits_het_group_when_none() -> None:
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="12345"),
        patch("srtctl.core.slurm._get_cluster_bash_preamble", return_value=None),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value = MagicMock()
        start_srun_process(["echo", "hi"])  # default het_group=None

    srun_cmd = mock_popen.call_args.args[0]
    for arg in srun_cmd:
        assert not str(arg).startswith("--het-group")
