# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``core/launcher.py``: launcher selection, the Docker and host launchers, and their validation."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.core import launcher as launcher_mod
from srtctl.core.docker import DockerLauncher
from srtctl.core.host import ContainerPaths, HostLauncher, host_environment
from srtctl.core.launcher import LaunchSpec, SlurmLauncher, get_launcher, launchers
from srtctl.core.runtime import Nodes
from srtctl.core.schema import SrtConfig
from srtctl.core.slurm import get_hostname_ip, start_srun_process
from srtctl.mock import MockOptions, run_mock_sweep

SINGLE_NODE_DISAGG = {
    "schema": 2,
    "name": "docker-disagg",
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
def docker(monkeypatch) -> DockerLauncher:
    monkeypatch.setenv("SRTCTL_JOB_ID", "docker-test")
    return DockerLauncher()


class TestSelection:
    def test_slurm_is_the_default(self, cluster):
        cluster()
        assert isinstance(get_launcher(), SlurmLauncher)

    def test_docker_from_cluster_config(self, cluster):
        cluster(launcher="docker")
        assert isinstance(get_launcher(), DockerLauncher)

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

    def test_hostname_resolution_is_loopback_under_docker(self, cluster):
        cluster(launcher="docker")
        assert get_hostname_ip("any-host", "ib0") == "127.0.0.1"


class TestLocalDockerCommand:
    def test_container_runs_on_host_network_with_every_gpu(self, docker, cluster):
        cluster(launcher="docker", default_bash_preamble="ulimit -n 4096", docker_args=["--user", "1000:1000"])
        spec = LaunchSpec(
            command=["vllm", "serve", "/model"],
            container_image="nvcr.io#nvidia/vllm:25.01",
            container_mounts={Path("/data/model"): Path("/model")},
            env_to_set={"CUDA_VISIBLE_DEVICES": "2,3"},
            env_to_unset=["NCCL_DEBUG"],
            srun_export_env={"ENROOT_REMAP_ROOT": "yes"},
            step_name="worker_prefill_0",
        )
        cmd = docker.docker_command(spec, "srtctl_docker-test_worker_prefill_0")
        assert cmd[:5] == ["docker", "run", "--rm", "--name", "srtctl_docker-test_worker_prefill_0"]
        joined = " ".join(cmd)
        for flag in (
            "--gpus all",
            "--network host",
            "--ipc host",
            "--cap-add PERFMON",
            "--cap-add SYS_NICE",
            "-v /data/model:/model",
            "--entrypoint ",
        ):
            assert flag in joined
        assert "--label srtctl.job=docker-test" in joined
        assert "--label srtctl.step=worker_prefill_0" in joined
        assert "-e ENROOT_REMAP_ROOT=yes" in joined
        image_at = cmd.index("nvcr.io/nvidia/vllm:25.01")
        assert cmd[image_at - 2 : image_at] == ["--user", "1000:1000"]
        assert cmd[image_at + 1 : image_at + 3] == ["bash", "-c"]
        assert cmd[-1] == (
            "ulimit -n 4096 && export CUDA_VISIBLE_DEVICES=2,3 && unset -- NCCL_DEBUG && exec vllm serve /model"
        )

    def test_docker_gpus_setting_replaces_all(self, docker, cluster):
        cluster(launcher="docker", docker_gpus="device=2,3,4,5")
        spec = LaunchSpec(command=["true"], container_image="img:1", env_to_set={"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
        cmd = docker.docker_command(spec, "c")
        # Docker splits --gpus on commas unless the whole value is wrapped in literal quotes.
        assert cmd[cmd.index("--gpus") + 1] == '"device=2,3,4,5"'
        assert "all" not in cmd

    def test_docker_gpus_single_device_or_all_is_not_quoted(self, docker, cluster):
        cluster(launcher="docker", docker_gpus="device=3")
        cmd = docker.docker_command(LaunchSpec(command=["true"], container_image="img:1"), "c")
        assert cmd[cmd.index("--gpus") + 1] == "device=3"
        cluster(launcher="docker")
        cmd = docker.docker_command(LaunchSpec(command=["true"], container_image="img:1"), "c")
        assert cmd[cmd.index("--gpus") + 1] == "all"

    def test_container_workdir_becomes_w_and_other_srun_options_are_dropped(self, docker, cluster, caplog):
        cluster(launcher="docker")
        spec = LaunchSpec(
            command=["true"],
            container_image="img:1",
            srun_options={"container-workdir": "/infmax-workspace", "mem": "0", "container-name": "x", "gres": "gpu:4"},
        )
        with caplog.at_level("WARNING", logger="srtctl.core.docker"):
            cmd = docker.docker_command(spec, "c")
            docker.docker_command(spec, "c2")
        assert cmd[cmd.index("-w") + 1] == "/infmax-workspace"
        assert not any(flag.startswith("--mem") or flag == "--gres" for flag in cmd)
        warnings = [r.message for r in caplog.records if "ignores srun_options" in r.message]
        assert warnings == ["launcher: docker ignores srun_options gres"]  # once, and only the unknown key

    def test_hf_snapshot_mount_brings_its_blob_store(self, docker, cluster, tmp_path):
        """A snapshot dir is a farm of relative symlinks into the hub's blobs; the mount must keep them working."""
        cluster(launcher="docker")
        hub = tmp_path / "hub"
        snapshot = hub / "models--org--name" / "snapshots" / "rev"
        snapshot.mkdir(parents=True)
        (hub / "models--org--name" / "blobs").mkdir()
        (hub / "blobs" / "ab").mkdir(parents=True)
        (hub / "blobs" / "ab" / "abc").write_text("weights")
        (hub / "models--org--name" / "blobs" / "abc").symlink_to("../../blobs/ab/abc")
        (snapshot / "model.safetensors").symlink_to("../../blobs/abc")
        (snapshot / "config.json").write_text("{}")
        plain = tmp_path / "logs"
        plain.mkdir()
        spec = LaunchSpec(
            command=["serve", "/model"],
            container_image="img:1",
            container_mounts={snapshot: Path("/model"), plain: Path("/logs")},
        )
        cmd = docker.docker_command(spec, "c")
        joined = " ".join(cmd)
        real_hub = os.path.realpath(hub)
        assert f"-v {real_hub}:{real_hub}" in joined
        assert f"-v {snapshot}:/model" not in joined
        assert f"-v {plain}:/logs" in joined
        assert cmd[-1] == f"ln -sfnT {os.path.realpath(snapshot)} /model && exec serve /model"

    def test_plain_directory_mounts_are_untouched(self, docker, cluster, tmp_path):
        cluster(launcher="docker")
        data = tmp_path / "data"
        data.mkdir()
        (data / "inside").write_text("x")
        (data / "link").symlink_to("inside")  # stays within the mount
        spec = LaunchSpec(command=["true"], container_image="img:1", container_mounts={data: Path("/data")})
        cmd = docker.docker_command(spec, "c")
        assert f"-v {data}:/data" in " ".join(cmd)
        assert cmd[-1] == "exec true"

    def test_bash_wrapper_matches_slurm(self, cluster):
        cluster(default_bash_preamble="ulimit -n 4096")
        spec = LaunchSpec(command=["python", "-m", "x"], env_to_set={"A": "a b"}, bash_preamble="echo hi")
        assert DockerLauncher().build_task_command(spec) == SlurmLauncher().build_task_command(spec)

    @pytest.mark.parametrize("image", ["docker://nvcr.io/nvidia/x:1", "nvcr.io#nvidia/x:1", "nvcr.io/nvidia/x:1"])
    def test_enroot_image_uris_become_docker_references(self, image):
        assert DockerLauncher.docker_image(image) == "nvcr.io/nvidia/x:1"

    def test_multi_task_launch_is_rejected(self, docker):
        with pytest.raises(ValueError, match="one task per launch"):
            docker.launch(LaunchSpec(command=["true"], ntasks=4))

    def test_other_hosts_are_rejected(self, docker):
        with pytest.raises(ValueError, match="other-node"):
            docker.launch(LaunchSpec(command=["true"], nodelist=["other-node"]))

    def test_container_step_is_signalled_with_docker_kill(self, docker, cluster):
        cluster(launcher="docker")
        popen = MagicMock()
        popen.poll.return_value = None
        with (
            patch("srtctl.core.docker.subprocess.Popen", return_value=popen),
            patch("srtctl.core.docker.shutil.which", return_value="/usr/bin/docker"),
            patch("srtctl.core.docker.subprocess.run", return_value=MagicMock(returncode=0)) as run,
        ):
            docker.launch(LaunchSpec(command=["sleep", "60"], container_image="img:1", step_name="tachometer"))
            assert docker.list_step_ids() == {"tachometer": "srtctl_docker-test_tachometer"}
            assert docker.signal_step("tachometer", "TERM")
        assert run.call_args.args[0] == ["docker", "kill", "--signal=TERM", "srtctl_docker-test_tachometer"]

    def test_relaunched_step_gets_a_fresh_container_name(self, docker, cluster):
        cluster(launcher="docker")
        popen = MagicMock()
        popen.poll.return_value = None
        with patch("srtctl.core.docker.subprocess.Popen", return_value=popen) as run:
            for _ in range(2):
                docker.launch(LaunchSpec(command=["true"], container_image="img:1", step_name="benchmark"))
        names = [call.args[0][call.args[0].index("--name") + 1] for call in run.call_args_list]
        assert names[0] == "srtctl_docker-test_benchmark"
        assert names[1] != names[0]
        assert docker.list_step_ids() == {"benchmark": names[1]}

    def test_launch_command_is_logged_at_info(self, docker, cluster, caplog):
        cluster(launcher="docker")
        with patch("srtctl.core.docker.subprocess.Popen", return_value=MagicMock()), caplog.at_level("INFO"):
            docker.launch(LaunchSpec(command=["true"], container_image="img:1", step_name="w"))
        assert any("docker launcher command: docker run" in r.getMessage() for r in caplog.records)


class TestLocalHostProcesses:
    def test_host_command_runs_the_wrapper_and_writes_output(self, docker, cluster, tmp_path):
        cluster(launcher="docker")
        out = tmp_path / "logs" / "host.out"
        proc = docker.launch(
            LaunchSpec(
                command=["bash", "-c", 'echo "$GREETING $EXPORTED"'],
                env_to_set={"GREETING": "hello"},
                srun_export_env={"EXPORTED": "world"},
                output=str(out),
            )
        )
        assert proc.wait(timeout=10) == 0
        assert out.read_text().strip() == "hello world"

    def test_named_host_step_is_signalled_through_its_process_group(self, docker, cluster, tmp_path):
        cluster(launcher="docker")
        proc = docker.launch(LaunchSpec(command=["sleep", "60"], step_name="sleeper", output=str(tmp_path / "s.out")))
        try:
            assert docker.list_step_ids() == {"sleeper": str(proc.pid)}
            assert docker.signal_step("sleeper", "TERM")
            assert proc.wait(timeout=10) == -15
            assert docker.list_step_ids() == {}
            assert not docker.signal_step("sleeper", "TERM")
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_unknown_step_falls_back_to_the_caller(self, docker):
        assert not docker.signal_step("never-launched", "TERM")


class TestLocalPlacement:
    def test_every_role_is_this_host(self, docker):
        nodes = docker.nodes()
        host = socket.gethostname()
        assert (nodes.head, nodes.bench, nodes.infra, nodes.worker) == (host, host, host, (host,))

    def test_dedicated_roles_need_a_second_node(self):
        with pytest.raises(ValueError, match="at least 2 nodes"):
            Nodes.from_nodelist(["only-node"], frontend_dedicated_node=True)

    def test_job_id_comes_from_srtctl_job_id(self, docker):
        assert docker.job_id() == "docker-test"

    def test_submit_hands_out_distinct_ids(self, docker, tmp_path):
        first = docker.submit(tmp_path / "s.sh")
        time.sleep(0.002)
        assert first.startswith("docker-") and first != docker.submit(tmp_path / "s.sh")

    def test_start_runs_the_staged_script_without_slurm(self, docker, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "999")
        (tmp_path / "sbatch_script.sh").write_text('echo "job=${SLURM_JOB_ID:-$SRTCTL_JOB_ID}"\nexit 3\n')
        assert docker.start("docker-1", tmp_path) == 3
        assert (tmp_path / "logs" / "sweep_docker-1.log").read_text() == "job=docker-1\n"


class TestLocalValidation:
    def test_single_node_disagg_is_accepted(self, tmp_path, cluster):
        cluster(launcher="docker")
        assert DockerLauncher().validate(_config(tmp_path, SINGLE_NODE_DISAGG)) == []

    def test_multi_node_and_enroot_files_are_rejected(self, tmp_path, cluster):
        cluster(launcher="docker")
        image = tmp_path / "vllm.sqsh"
        image.write_text("")
        data = {
            **SINGLE_NODE_DISAGG,
            "model": {**SINGLE_NODE_DISAGG["model"], "container": str(image)},
            "roles": {"prefill": {"nodes": 1, "workers": 1}, "decode": {"nodes": 1, "workers": 1}},
        }
        problems = DockerLauncher().validate(_config(tmp_path, data))
        assert any("needs 2 nodes" in p for p in problems)
        assert any("enroot image file" in p for p in problems)

    def test_mpi_step_engines_are_rejected(self, tmp_path, cluster):
        cluster(launcher="docker")
        data = {
            **SINGLE_NODE_DISAGG,
            "frontend": {"type": "dynamo"},
            "engine": "trtllm",
            "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 4}},
        }
        problems = DockerLauncher().validate(_config(tmp_path, data))
        assert any("multi-task MPI step" in p for p in problems)

    def test_slurm_accepts_everything(self, tmp_path):
        assert SlurmLauncher().validate(_config(tmp_path, SINGLE_NODE_DISAGG)) == []

    def test_dry_run_refuses_an_unsupported_recipe(self, tmp_path, cluster):
        from srtctl.cli.submit import submit_with_orchestrator

        cluster(launcher="docker")
        data = {**SINGLE_NODE_DISAGG, "roles": {"agg": {"nodes": 2, "workers": 1}}}
        path = tmp_path / "recipe.yaml"
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError, match="cannot run under launcher: docker"):
            submit_with_orchestrator(config_path=path, config=SrtConfig.from_yaml(path), dry_run=True)


def test_mock_sweep_under_docker_launcher_places_everything_on_this_host(tmp_path, cluster):
    """The real orchestrator, with only process launches faked, carves its nodes from this machine."""
    cluster(launcher="docker")
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(SINGLE_NODE_DISAGG))
    launches: list[dict] = []
    exit_code = run_mock_sweep(
        config_path=path,
        output_dir=tmp_path / "out",
        job_id="docker-mock",
        options=MockOptions(phase_pause_s=0, on_srun=launches.append),
    )
    assert exit_code == 0
    host = socket.gethostname()
    placed = {node for launch in launches for node in launch.get("nodelist") or ()}
    assert placed == {host}


@pytest.fixture
def host(monkeypatch) -> HostLauncher:
    monkeypatch.setenv("SRTCTL_JOB_ID", "host-test")
    return HostLauncher()


class TestHostSelection:
    def test_host_from_cluster_config(self, cluster):
        cluster(launcher="host")
        assert isinstance(get_launcher(), HostLauncher)

    def test_every_launcher_name_is_an_allowed_cluster_value(self):
        from typing import get_args, get_type_hints

        from srtctl.core.schema import ClusterConfig

        allowed = set(get_args(get_type_hints(ClusterConfig)["launcher"]))
        assert set(launchers()) == allowed == {"slurm", "docker", "host"}

    def test_hostname_resolution_is_loopback_under_host(self, cluster):
        cluster(launcher="host")
        assert get_hostname_ip("any-host", "ib0") == "127.0.0.1"


HOST_MOUNTS = {
    Path("/home/me/outputs/7/logs"): Path("/logs"),
    Path("/data/qwen"): Path("/model"),
    Path("/home/me/srt-slurm/configs"): Path("/configs"),
    Path("/scratch/stage"): Path("/scratch/stage"),
}


class TestContainerPaths:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("/model", "/data/qwen"),
            ("/logs/sa-bench/result.json", "/home/me/outputs/7/logs/sa-bench/result.json"),
            ("--model-path=/model", "--model-path=/data/qwen"),
            ("PATH=/configs:$PATH", "PATH=/home/me/srt-slurm/configs:$PATH"),
            (
                "export LOG='/logs/x' && cat \"/logs/y\"",
                "export LOG='/home/me/outputs/7/logs/x' && cat \"/home/me/outputs/7/logs/y\"",
            ),
            ("tee $(ls /logs)", "tee $(ls /home/me/outputs/7/logs)"),
        ],
    )
    def test_container_paths_become_host_paths(self, text, expected):
        assert ContainerPaths(HOST_MOUNTS).to_host(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "/models/other",  # a longer path segment
            "/data/logs/old",  # inside another path
            "http://127.0.0.1:8000/model",  # a URL
            "/v1/models",
            "/logs.bak",
            "/scratch/stage/qwen",  # mounted at its own path
        ],
    )
    def test_lookalikes_are_left_alone(self, text):
        assert ContainerPaths(HOST_MOUNTS).to_host(text) == text

    def test_a_rewritten_path_is_not_rewritten_again(self):
        paths = ContainerPaths({Path("/srv/model"): Path("/model"), Path("/x/model"): Path("/srv")})
        assert paths.to_host("/model /srv") == "/srv/model /x/model"

    def test_no_mounts_leave_text_unchanged(self):
        assert ContainerPaths(None).to_host("/logs/x") == "/logs/x"


class TestHostCommand:
    def test_command_runs_on_the_host_with_container_paths_rewritten(self, host, cluster):
        cluster(launcher="host", default_bash_preamble="mkdir -p /logs/cache")
        spec = LaunchSpec(
            command=["vllm", "serve", "/model", "--log-file", "/logs/worker.log"],
            container_image="vllm/vllm-openai:v0.11.0",
            container_mounts={Path("/data/qwen"): Path("/model"), Path("/home/me/outputs/7/logs"): Path("/logs")},
            env_to_set={"CUDA_VISIBLE_DEVICES": "0", "HF_HOME": "/model/.hf"},
            env_to_unset=["NCCL_DEBUG"],
            bash_preamble="cd /logs",
            step_name="worker_agg_0",
        )
        cmd, _, cwd = host.host_command(spec)
        assert cmd[:2] == ["bash", "-c"]
        assert "docker" not in cmd[2] and "vllm-openai" not in cmd[2]
        assert cmd[2] == (
            "mkdir -p /home/me/outputs/7/logs/cache && export CUDA_VISIBLE_DEVICES=0 && export HF_HOME=/data/qwen/.hf"
            " && unset -- NCCL_DEBUG && cd /home/me/outputs/7/logs"
            " && exec vllm serve /data/qwen --log-file /home/me/outputs/7/logs/worker.log"
        )
        assert cwd is None

    def test_unwrapped_command_is_rewritten_argument_by_argument(self, host, cluster):
        cluster(launcher="host")
        spec = LaunchSpec(
            command=["/configs/nats-server", "-c", "/configs/nats.conf"],
            container_mounts={Path("/repo/configs"): Path("/configs")},
            use_bash_wrapper=False,
        )
        assert host.host_command(spec)[0] == ["/repo/configs/nats-server", "-c", "/repo/configs/nats.conf"]

    def test_exported_env_and_workdir_are_rewritten(self, host, cluster, caplog):
        cluster(launcher="host")
        spec = LaunchSpec(
            command=["true"],
            container_mounts={Path("/gha/workspace"): Path("/infmax-workspace")},
            srun_export_env={"RESULTS": "/infmax-workspace/results"},
            srun_options={"container-workdir": "/infmax-workspace", "mem": "0", "gres": "gpu:1"},
        )
        with caplog.at_level("WARNING"):
            _, env, cwd = host.host_command(spec)
        assert env["RESULTS"] == "/gha/workspace/results"
        assert cwd == "/gha/workspace"
        warnings = [r.getMessage() for r in caplog.records if "ignores srun_options" in r.getMessage()]
        assert warnings == ["launcher: host ignores srun_options gres"]

    def test_bash_wrapper_matches_slurm_without_mounts(self, cluster):
        cluster(default_bash_preamble="ulimit -n 4096")
        spec = LaunchSpec(command=["python", "-m", "x"], env_to_set={"A": "a b"}, bash_preamble="echo hi")
        assert HostLauncher().host_command(spec)[0] == SlurmLauncher().build_task_command(spec)

    def test_container_image_is_ignored_and_logged_once(self, host, cluster, caplog):
        cluster(launcher="host")
        with (
            patch("srtctl.core.local.subprocess.Popen", return_value=MagicMock()) as popen,
            caplog.at_level("INFO"),
        ):
            for step in ("a", "b"):
                host.launch(LaunchSpec(command=["true"], container_image="img:1", step_name=step))
        assert all(call.args[0][0] == "bash" for call in popen.call_args_list)
        ignored = [r for r in caplog.records if "ignores container image img:1" in r.getMessage()]
        assert len(ignored) == 1
        assert any("host launcher command: bash -c" in r.getMessage() for r in caplog.records)

    def test_multi_task_launch_is_rejected(self, host):
        with pytest.raises(ValueError, match="launcher: host runs one task per launch"):
            host.launch(LaunchSpec(command=["true"], ntasks=2))

    def test_other_hosts_are_rejected(self, host):
        with pytest.raises(ValueError, match="other-node"):
            host.launch(LaunchSpec(command=["true"], nodelist=["other-node"]))


class TestHostEnvironment:
    """A host process gets this machine's Python, never srtctl's, and pip cannot install outside a virtualenv."""

    def test_pip_cannot_install_outside_a_virtualenv(self):
        assert host_environment({"PATH": "/usr/bin"}, None)["PIP_REQUIRE_VIRTUALENV"] == "1"

    def test_the_users_own_pip_setting_wins(self):
        assert host_environment({"PIP_REQUIRE_VIRTUALENV": "0"}, None)["PIP_REQUIRE_VIRTUALENV"] == "0"

    def test_srtctls_own_virtualenv_steps_aside(self, tmp_path):
        own = tmp_path / "srt-slurm" / ".venv"
        engine = tmp_path / "vllm-env" / "bin"
        base = {
            "PATH": os.pathsep.join([str(own / "bin"), str(engine), "/usr/bin"]),
            "VIRTUAL_ENV": str(own),
            "UV_PROJECT_ENVIRONMENT": str(own),
            "HOME": "/home/me",
        }
        env = host_environment(base, str(own))
        assert env["PATH"].split(os.pathsep) == [str(engine), "/usr/bin", str(own / "bin")]
        assert "VIRTUAL_ENV" not in env and "UV_PROJECT_ENVIRONMENT" not in env
        assert env["HOME"] == "/home/me"
        assert base["VIRTUAL_ENV"] == str(own)  # the caller's mapping is not modified

    def test_another_virtualenv_is_left_alone(self, tmp_path):
        engine = tmp_path / "vllm-env"
        base = {"PATH": os.pathsep.join([str(engine / "bin"), "/usr/bin"]), "VIRTUAL_ENV": str(engine)}
        env = host_environment(base, str(tmp_path / "srt-slurm" / ".venv"))
        assert env["PATH"] == base["PATH"] and env["VIRTUAL_ENV"] == str(engine)

    def test_every_launch_starts_from_it(self, host, cluster, monkeypatch):
        cluster(launcher="host")
        monkeypatch.delenv("PIP_REQUIRE_VIRTUALENV", raising=False)
        _, env, _ = host.host_command(LaunchSpec(command=["true"]))
        assert env["PIP_REQUIRE_VIRTUALENV"] == "1"


class TestHostProcesses:
    def test_process_runs_with_rewritten_paths_and_writes_output(self, host, cluster, tmp_path):
        cluster(launcher="host")
        logs = tmp_path / "logs"
        logs.mkdir()
        out = logs / "worker.out"
        proc = host.launch(
            LaunchSpec(
                command=["bash", "-c", 'echo "$GREETING $EXPORTED" > /logs/marker && echo done'],
                container_image="ignored:1",
                container_mounts={logs: Path("/logs")},
                env_to_set={"GREETING": "hello"},
                srun_export_env={"EXPORTED": "world"},
                output=str(out),
            )
        )
        assert proc.wait(timeout=10) == 0
        assert (logs / "marker").read_text().strip() == "hello world"
        assert out.read_text().strip() == "done"

    def test_the_pip_guard_reaches_the_process(self, host, cluster, tmp_path, monkeypatch):
        cluster(launcher="host")
        monkeypatch.delenv("PIP_REQUIRE_VIRTUALENV", raising=False)
        out = tmp_path / "guard.out"
        proc = host.launch(LaunchSpec(command=["bash", "-c", 'echo "$PIP_REQUIRE_VIRTUALENV"'], output=str(out)))
        assert proc.wait(timeout=10) == 0
        assert out.read_text().strip() == "1"

    def test_named_step_is_signalled_through_its_process_group(self, host, cluster, tmp_path):
        cluster(launcher="host")
        proc = host.launch(LaunchSpec(command=["sleep", "60"], step_name="worker", output=str(tmp_path / "w.out")))
        try:
            assert host.list_step_ids() == {"worker": str(proc.pid)}
            assert host.signal_step("worker", "TERM")
            assert proc.wait(timeout=10) == -15
            assert host.list_step_ids() == {}
        finally:
            if proc.poll() is None:
                proc.kill()


class TestHostJob:
    def test_every_role_is_this_host(self, host):
        nodes = host.nodes()
        this = socket.gethostname()
        assert (nodes.head, nodes.bench, nodes.infra, nodes.worker) == (this, this, this, (this,))

    def test_job_id_and_submitted_ids(self, host, tmp_path):
        assert host.job_id() == "host-test"
        assert host.submit(tmp_path / "s.sh").startswith("host-")

    def test_start_runs_the_staged_script_without_slurm(self, host, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "999")
        (tmp_path / "sbatch_script.sh").write_text('echo "job=${SLURM_JOB_ID:-$SRTCTL_JOB_ID}"\n')
        assert host.start("host-1", tmp_path) == 0
        assert (tmp_path / "logs" / "sweep_host-1.log").read_text() == "job=host-1\n"

    def test_status_hint_finds_the_job_by_id(self, host):
        assert "grep -F -- host-1" in host.status_hint("host-1")


class TestHostValidation:
    def test_single_node_recipe_is_accepted(self, tmp_path, cluster):
        cluster(launcher="host")
        assert HostLauncher().validate(_config(tmp_path, SINGLE_NODE_DISAGG)) == []

    def test_an_enroot_image_file_is_not_an_error(self, tmp_path, cluster):
        """The image is never used, so a .sqsh path (a Slurm recipe as is) does not block the run."""
        cluster(launcher="host")
        image = tmp_path / "vllm.sqsh"
        image.write_text("")
        data = {**SINGLE_NODE_DISAGG, "model": {**SINGLE_NODE_DISAGG["model"], "container": str(image)}}
        assert HostLauncher().validate(_config(tmp_path, data)) == []

    def test_multi_node_and_mpi_step_engines_are_rejected(self, tmp_path, cluster):
        cluster(launcher="host")
        two_nodes = {**SINGLE_NODE_DISAGG, "roles": {"agg": {"nodes": 2, "workers": 1}}}
        assert any(
            "launcher: host runs on one machine" in p for p in HostLauncher().validate(_config(tmp_path, two_nodes))
        )
        trtllm = {
            **SINGLE_NODE_DISAGG,
            "frontend": {"type": "dynamo"},
            "engine": "trtllm",
            "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 4}},
        }
        problems = HostLauncher().validate(_config(tmp_path, trtllm))
        assert any("which launcher: host does not support" in p for p in problems)

    def test_installing_dynamo_is_rejected(self, tmp_path, cluster):
        """The install and its done-marker would land in the user's own Python and outlive the job."""
        cluster(launcher="host")
        dynamo = {**SINGLE_NODE_DISAGG, "frontend": {"type": "dynamo"}}
        assert any("set dynamo.install: false" in p for p in HostLauncher().validate(_config(tmp_path, dynamo)))
        preinstalled = {**dynamo, "dynamo": {"install": False}}
        assert HostLauncher().validate(_config(tmp_path, preinstalled)) == []

    def test_ai_analysis_is_rejected(self, tmp_path, cluster):
        """It curl-installs Claude Code and gh into the home directory."""
        cluster(launcher="host", reporting={"ai_analysis": {"enabled": True}})
        problems = HostLauncher().validate(_config(tmp_path, SINGLE_NODE_DISAGG))
        assert any("set reporting.ai_analysis.enabled: false" in p for p in problems)
        cluster(launcher="host", reporting={"ai_analysis": {"enabled": False}})
        assert HostLauncher().validate(_config(tmp_path, SINGLE_NODE_DISAGG)) == []


def test_mock_sweep_under_host_launcher_places_everything_on_this_host(tmp_path, cluster):
    """The real orchestrator, with only process launches faked, carves its nodes from this machine."""
    cluster(launcher="host")
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(SINGLE_NODE_DISAGG))
    launches: list[dict] = []
    exit_code = run_mock_sweep(
        config_path=path,
        output_dir=tmp_path / "out",
        job_id="host-mock",
        options=MockOptions(phase_pause_s=0, on_srun=launches.append),
    )
    assert exit_code == 0
    placed = {node for launch in launches for node in launch.get("nodelist") or ()}
    assert placed == {socket.gethostname()}


def test_subprocess_is_not_left_running(docker, cluster, tmp_path):
    cluster(launcher="docker")
    proc = docker.launch(LaunchSpec(command=["true"], step_name="quick", output=str(tmp_path / "q.out")))
    assert proc.wait(timeout=10) == 0
    assert isinstance(proc, subprocess.Popen)
