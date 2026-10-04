# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``core/launcher.py``: launcher selection, the local launcher, and its validation."""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.core import launcher as launcher_mod
from srtctl.core.launcher import LaunchSpec, LocalLauncher, SlurmLauncher, get_launcher
from srtctl.core.runtime import Nodes
from srtctl.core.schema import SrtConfig
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.mock import MockOptions, run_mock_sweep

SINGLE_NODE_DISAGG = {
    "schema": 2,
    "name": "local-disagg",
    "model": {"path": "hf:fake/model", "container": "vllm/vllm-openai:latest", "precision": "bf16"},
    "resources": {"gpu_type": "h100", "gpus_per_node": 8},
    "frontend": {"type": "vllm-router", "enable_multiple_frontends": False},
    "engine": {"type": "vllm", "connector": "nixl"},
    "roles": {
        "prefill": {"nodes": 1, "workers": 1, "gpus": 1},
        "decode": {"nodes": "colocate", "workers": 1, "gpus": 1},
    },
    "benchmark": {"type": "custom", "command": "echo fake-benchmark"},
}


def _config(tmp_path: Path, data: dict) -> SrtConfig:
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(data))
    return SrtConfig.from_yaml(path)


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    """Write ``srtslurm.yaml`` with the given settings and point SRTSLURM_CONFIG at it."""

    def write(**settings) -> Path:
        path = tmp_path / "srtslurm.yaml"
        path.write_text(yaml.safe_dump(settings))
        monkeypatch.setenv("SRTSLURM_CONFIG", str(path))
        return path

    return write


@pytest.fixture
def local(monkeypatch) -> LocalLauncher:
    monkeypatch.setenv("SRTCTL_JOB_ID", "local-test")
    return LocalLauncher()


class TestSelection:
    def test_slurm_is_the_default(self, cluster):
        cluster()
        assert isinstance(get_launcher(), SlurmLauncher)

    def test_local_from_cluster_config(self, cluster):
        cluster(launcher="local")
        assert isinstance(get_launcher(), LocalLauncher)

    def test_unknown_name_fails(self):
        with (
            patch("srtctl.core.config.get_srtslurm_setting", return_value="kubernetes"),
            pytest.raises(ValueError, match="kubernetes"),
        ):
            get_launcher()

    def test_start_srun_process_goes_through_the_launcher(self):
        fake = MagicMock()
        with patch.object(launcher_mod, "get_launcher", return_value=fake):
            start_srun_process(["echo", "hi"], step_name="s", env_to_set={"A": "1"})
        spec = fake.launch.call_args.args[0]
        assert spec == LaunchSpec(command=["echo", "hi"], step_name="s", env_to_set={"A": "1"})

    def test_hostname_resolution_is_loopback_under_local(self, cluster):
        cluster(launcher="local")
        assert get_hostname_ip("any-host", "ib0") == "127.0.0.1"


class TestLocalDockerCommand:
    def test_container_runs_on_host_network_with_every_gpu(self, local, cluster):
        cluster(launcher="local", default_bash_preamble="ulimit -n 4096", local_docker_args=["--user", "1000:1000"])
        spec = LaunchSpec(
            command=["vllm", "serve", "/model"],
            container_image="nvcr.io#nvidia/vllm:25.01",
            container_mounts={Path("/data/model"): Path("/model")},
            env_to_set={"CUDA_VISIBLE_DEVICES": "2,3"},
            env_to_unset=["NCCL_DEBUG"],
            srun_export_env={"ENROOT_REMAP_ROOT": "yes"},
            step_name="worker_prefill_0",
        )
        cmd = local.docker_command(spec, "srtctl_local-test_worker_prefill_0")
        assert cmd[:5] == ["docker", "run", "--rm", "--name", "srtctl_local-test_worker_prefill_0"]
        joined = " ".join(cmd)
        for flag in ("--gpus all", "--network host", "--ipc host", "-v /data/model:/model", "--entrypoint "):
            assert flag in joined
        assert "--label srtctl.job=local-test" in joined
        assert "--label srtctl.step=worker_prefill_0" in joined
        assert "-e ENROOT_REMAP_ROOT=yes" in joined
        image_at = cmd.index("nvcr.io/nvidia/vllm:25.01")
        assert cmd[image_at - 2 : image_at] == ["--user", "1000:1000"]
        assert cmd[image_at + 1 : image_at + 3] == ["bash", "-c"]
        assert cmd[-1] == (
            "ulimit -n 4096 && export CUDA_VISIBLE_DEVICES=2,3 && unset -- NCCL_DEBUG && exec vllm serve /model"
        )

    def test_bash_wrapper_matches_slurm(self, cluster):
        cluster(default_bash_preamble="ulimit -n 4096")
        spec = LaunchSpec(command=["python", "-m", "x"], env_to_set={"A": "a b"}, bash_preamble="echo hi")
        assert LocalLauncher().build_task_command(spec) == SlurmLauncher().build_task_command(spec)

    @pytest.mark.parametrize("image", ["docker://nvcr.io/nvidia/x:1", "nvcr.io#nvidia/x:1", "nvcr.io/nvidia/x:1"])
    def test_enroot_image_uris_become_docker_references(self, image):
        assert LocalLauncher.docker_image(image) == "nvcr.io/nvidia/x:1"

    def test_multi_task_launch_is_rejected(self, local):
        with pytest.raises(ValueError, match="one task per launch"):
            local.launch(LaunchSpec(command=["true"], ntasks=4))

    def test_other_hosts_are_rejected(self, local):
        with pytest.raises(ValueError, match="other-node"):
            local.launch(LaunchSpec(command=["true"], nodelist=["other-node"]))

    def test_container_step_is_signalled_with_docker_kill(self, local, cluster):
        cluster(launcher="local")
        popen = MagicMock()
        popen.poll.return_value = None
        with (
            patch("srtctl.core.launcher.subprocess.Popen", return_value=popen),
            patch("srtctl.core.launcher.shutil.which", return_value="/usr/bin/docker"),
            patch("srtctl.core.launcher.subprocess.run", return_value=MagicMock(returncode=0)) as run,
        ):
            local.launch(LaunchSpec(command=["sleep", "60"], container_image="img:1", step_name="tachometer"))
            assert local.list_step_ids() == {"tachometer": "srtctl_local-test_tachometer"}
            assert local.signal_step("tachometer", "TERM")
        assert run.call_args.args[0] == ["docker", "kill", "--signal=TERM", "srtctl_local-test_tachometer"]

    def test_relaunched_step_gets_a_fresh_container_name(self, local, cluster):
        cluster(launcher="local")
        popen = MagicMock()
        popen.poll.return_value = None
        with patch("srtctl.core.launcher.subprocess.Popen", return_value=popen) as run:
            for _ in range(2):
                local.launch(LaunchSpec(command=["true"], container_image="img:1", step_name="benchmark"))
        names = [call.args[0][call.args[0].index("--name") + 1] for call in run.call_args_list]
        assert names[0] == "srtctl_local-test_benchmark"
        assert names[1] != names[0]
        assert local.list_step_ids() == {"benchmark": names[1]}

    def test_launch_command_is_logged_at_info(self, local, cluster, caplog):
        cluster(launcher="local")
        with patch("srtctl.core.launcher.subprocess.Popen", return_value=MagicMock()), caplog.at_level("INFO"):
            local.launch(LaunchSpec(command=["true"], container_image="img:1", step_name="w"))
        assert any("local command: docker run" in r.getMessage() for r in caplog.records)


class TestLocalHostProcesses:
    def test_host_command_runs_the_wrapper_and_writes_output(self, local, cluster, tmp_path):
        cluster(launcher="local")
        out = tmp_path / "logs" / "host.out"
        proc = local.launch(
            LaunchSpec(
                command=["bash", "-c", 'echo "$GREETING $EXPORTED"'],
                env_to_set={"GREETING": "hello"},
                srun_export_env={"EXPORTED": "world"},
                output=str(out),
            )
        )
        assert proc.wait(timeout=10) == 0
        assert out.read_text().strip() == "hello world"

    def test_named_host_step_is_signalled_through_its_process_group(self, local, cluster, tmp_path):
        cluster(launcher="local")
        proc = local.launch(LaunchSpec(command=["sleep", "60"], step_name="sleeper", output=str(tmp_path / "s.out")))
        try:
            assert local.list_step_ids() == {"sleeper": str(proc.pid)}
            assert local.signal_step("sleeper", "TERM")
            assert proc.wait(timeout=10) == -15
            assert local.list_step_ids() == {}
            assert not local.signal_step("sleeper", "TERM")
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_unknown_step_falls_back_to_the_caller(self, local):
        assert not local.signal_step("never-launched", "TERM")


class TestLocalPlacement:
    def test_every_role_is_this_host(self, local):
        nodes = local.nodes()
        host = socket.gethostname()
        assert (nodes.head, nodes.bench, nodes.infra, nodes.worker) == (host, host, host, (host,))

    def test_dedicated_roles_need_a_second_node(self):
        with pytest.raises(ValueError, match="at least 2 nodes"):
            Nodes.from_nodelist(["only-node"], frontend_dedicated_node=True)

    def test_job_id_comes_from_srtctl_job_id(self, local):
        assert local.job_id() == "local-test"

    def test_submit_hands_out_distinct_ids(self, local, tmp_path):
        first = local.submit(tmp_path / "s.sh")
        time.sleep(0.002)
        assert first.startswith("local-") and first != local.submit(tmp_path / "s.sh")

    def test_start_runs_the_staged_script_without_slurm(self, local, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "999")
        (tmp_path / "sbatch_script.sh").write_text('echo "job=${SLURM_JOB_ID:-$SRTCTL_JOB_ID}"\nexit 3\n')
        assert local.start("local-1", tmp_path) == 3
        assert (tmp_path / "logs" / "sweep_local-1.log").read_text() == "job=local-1\n"


class TestLocalValidation:
    def test_single_node_disagg_is_accepted(self, tmp_path, cluster):
        cluster(launcher="local")
        assert LocalLauncher().validate(_config(tmp_path, SINGLE_NODE_DISAGG)) == []

    def test_multi_node_and_enroot_files_are_rejected(self, tmp_path, cluster):
        cluster(launcher="local")
        image = tmp_path / "vllm.sqsh"
        image.write_text("")
        data = {
            **SINGLE_NODE_DISAGG,
            "model": {**SINGLE_NODE_DISAGG["model"], "container": str(image)},
            "roles": {"prefill": {"nodes": 1, "workers": 1}, "decode": {"nodes": 1, "workers": 1}},
        }
        problems = LocalLauncher().validate(_config(tmp_path, data))
        assert any("needs 2 nodes" in p for p in problems)
        assert any("enroot image file" in p for p in problems)

    def test_mpi_step_engines_are_rejected(self, tmp_path, cluster):
        cluster(launcher="local")
        data = {
            **SINGLE_NODE_DISAGG,
            "frontend": {"type": "dynamo"},
            "engine": "trtllm",
            "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 4}},
        }
        problems = LocalLauncher().validate(_config(tmp_path, data))
        assert any("multi-task MPI step" in p for p in problems)

    def test_slurm_accepts_everything(self, tmp_path):
        assert SlurmLauncher().validate(_config(tmp_path, SINGLE_NODE_DISAGG)) == []

    def test_dry_run_refuses_an_unsupported_recipe(self, tmp_path, cluster):
        from srtctl.cli.submit import submit_with_orchestrator

        cluster(launcher="local")
        data = {**SINGLE_NODE_DISAGG, "roles": {"agg": {"nodes": 2, "workers": 1}}}
        path = tmp_path / "recipe.yaml"
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError, match="cannot run under launcher: local"):
            submit_with_orchestrator(config_path=path, config=SrtConfig.from_yaml(path), dry_run=True)


def test_mock_sweep_under_local_launcher_places_everything_on_this_host(tmp_path, cluster):
    """The real orchestrator, with only process launches faked, carves its nodes from this machine."""
    cluster(launcher="local")
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(SINGLE_NODE_DISAGG))
    launches: list[dict] = []
    exit_code = run_mock_sweep(
        config_path=path,
        output_dir=tmp_path / "out",
        job_id="local-mock",
        options=MockOptions(phase_pause_s=0, on_srun=launches.append),
    )
    assert exit_code == 0
    host = socket.gethostname()
    placed = {node for launch in launches for node in launch.get("nodelist") or ()}
    assert placed == {host}


def test_subprocess_is_not_left_running(local, cluster, tmp_path):
    cluster(launcher="local")
    proc = local.launch(LaunchSpec(command=["true"], step_name="quick", output=str(tmp_path / "q.out")))
    assert proc.wait(timeout=10) == 0
    assert isinstance(proc, subprocess.Popen)
