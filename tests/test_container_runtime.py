# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`container_runtime: host` in srtslurm.yaml: the job already runs inside one sandbox
built from the recipe's image, so no step may carry a pyxis option."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.core import config as config_module
from srtctl.core.config import CONTAINER_RUNTIME_ENV, container_runtime
from srtctl.core.runtime import resolve_container_image
from srtctl.core.schema import ClusterConfig
from srtctl.core.slurm import start_srun_process


class TestClusterConfigContainerRuntime:
    def test_defaults_to_pyxis(self):
        assert ClusterConfig.Schema().load({}).container_runtime == "pyxis"

    def test_accepts_host(self):
        assert ClusterConfig.Schema().load({"container_runtime": "host"}).container_runtime == "host"

    def test_refuses_anything_else(self):
        with pytest.raises(ValidationError):
            ClusterConfig.Schema().load({"container_runtime": "podman"})


class TestContainerRuntimeResolution:
    def test_pyxis_without_cluster_file_or_variable(self, monkeypatch):
        monkeypatch.delenv(CONTAINER_RUNTIME_ENV, raising=False)
        monkeypatch.setattr(config_module, "load_cluster_config", lambda: None)
        assert container_runtime() == "pyxis"

    def test_cluster_file_decides(self, monkeypatch):
        monkeypatch.delenv(CONTAINER_RUNTIME_ENV, raising=False)
        monkeypatch.setattr(config_module, "load_cluster_config", lambda: {"container_runtime": "host"})
        assert container_runtime() == "host"

    def test_exported_variable_wins_over_the_cluster_file(self, monkeypatch):
        # The sbatch script exports the submit-time choice; the compute node's cwd
        # may find a different srtslurm.yaml, which must not flip the mode.
        monkeypatch.setenv(CONTAINER_RUNTIME_ENV, "host")
        monkeypatch.setattr(config_module, "load_cluster_config", lambda: {"container_runtime": "pyxis"})
        assert container_runtime() == "host"

    def test_unknown_value_is_refused(self, monkeypatch):
        monkeypatch.setenv(CONTAINER_RUNTIME_ENV, "docker")
        monkeypatch.setattr(config_module, "load_cluster_config", lambda: None)
        with pytest.raises(ValueError, match="container_runtime"):
            container_runtime()

    def test_a_cluster_config_in_hand_is_read_instead_of_the_active_file(self, monkeypatch):
        # The preflight judges the cluster file it was given, not whichever one cwd finds.
        monkeypatch.delenv(CONTAINER_RUNTIME_ENV, raising=False)
        monkeypatch.setattr(config_module, "load_cluster_config", lambda: {"container_runtime": "pyxis"})
        assert container_runtime({"container_runtime": "host"}) == "host"
        assert container_runtime({}) == "pyxis"


_LAUNCH_KWARGS = {
    "container_image": "/images/vllm.sqsh",
    "container_mounts": {Path("/models/qwen"): Path("/model")},
    "reuse_container": True,
    "container_prepare_nodes": ("node1",),
    "srun_options": {"container-remap-root": "", "mem": "16G"},
    "env_to_set": {"HEAD_NODE_IP": "127.0.0.1"},
}


class TestEffectiveLaunch:
    """What the mock orchestrator records is what the launcher would honour."""

    def test_pyxis_mode_returns_the_arguments_as_given(self, monkeypatch):
        from srtctl.core.slurm import effective_launch

        monkeypatch.delenv(CONTAINER_RUNTIME_ENV, raising=False)
        monkeypatch.setattr(config_module, "load_cluster_config", lambda: None)
        assert effective_launch(dict(_LAUNCH_KWARGS)) == _LAUNCH_KWARGS

    def test_host_mode_drops_only_the_container_arguments(self, monkeypatch):
        from srtctl.core.slurm import effective_launch

        monkeypatch.setenv(CONTAINER_RUNTIME_ENV, "host")
        launch = effective_launch(dict(_LAUNCH_KWARGS))
        assert launch == {
            "container_image": None,
            "container_mounts": None,
            "reuse_container": False,
            "container_prepare_nodes": None,
            "srun_options": {"mem": "16G"},
            "env_to_set": {"HEAD_NODE_IP": "127.0.0.1"},
        }

    def test_host_mode_touches_only_keys_that_are_present(self, monkeypatch):
        from srtctl.core.slurm import effective_launch

        monkeypatch.setenv(CONTAINER_RUNTIME_ENV, "host")
        assert effective_launch({"command": ["true"], "nodes": 1}) == {"command": ["true"], "nodes": 1}


def _launch(monkeypatch, **kwargs) -> tuple[list[str], MagicMock]:
    """Run start_srun_process with Popen and the rootfs preparation replaced.

    Returns the srun argv it built and the preparation mock, so a test can say
    whether the shared rootfs would have been prepared.
    """
    monkeypatch.setattr(config_module, "load_cluster_config", lambda: None)
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    popen = MagicMock()
    with (
        patch("srtctl.core.slurm.subprocess.Popen", popen),
        patch("srtctl.core.slurm._prepare_shared_container") as prepare,
    ):
        start_srun_process(["python3", "-m", "worker"], nodelist=["node1"], use_bash_wrapper=False, **kwargs)
    return list(popen.call_args.args[0]), prepare


def _launch_argv(monkeypatch, **kwargs) -> list[str]:
    return _launch(monkeypatch, **kwargs)[0]


_CONTAINER_KWARGS = {
    "container_image": "/images/vllm.sqsh",
    "container_mounts": {Path("/models/qwen"): Path("/model"), Path("/out/logs"): Path("/logs")},
    "reuse_container": True,
    "srun_options": {"container-remap-root": "", "mem": "16G"},
}


class TestStartSrunProcessHostMode:
    def test_pyxis_mode_is_unchanged(self, monkeypatch):
        monkeypatch.delenv(CONTAINER_RUNTIME_ENV, raising=False)
        argv = _launch_argv(monkeypatch, **_CONTAINER_KWARGS)
        assert "--container-image" in argv
        assert argv[argv.index("--container-image") + 1] == "/images/vllm.sqsh"
        assert "--no-container-entrypoint" in argv
        assert "--no-container-mount-home" in argv
        assert "--container-mounts" in argv
        assert any(arg.startswith("--container-name=srtctl_") for arg in argv)
        assert "--container-remap-root" in argv
        assert "--mem=16G" in argv

    def test_host_mode_carries_no_pyxis_option(self, monkeypatch):
        monkeypatch.setenv(CONTAINER_RUNTIME_ENV, "host")
        argv = _launch_argv(monkeypatch, **_CONTAINER_KWARGS)
        assert not [arg for arg in argv if "container" in arg], argv
        assert "--mem=16G" in argv
        assert argv[-3:] == ["python3", "-m", "worker"]
        assert "--jobid" in argv

    def test_pyxis_mode_prepares_the_shared_rootfs(self, monkeypatch):
        monkeypatch.delenv(CONTAINER_RUNTIME_ENV, raising=False)
        _, prepare = _launch(monkeypatch, **_CONTAINER_KWARGS)
        prepare.assert_called_once()

    def test_host_mode_never_prepares_a_shared_container(self, monkeypatch):
        monkeypatch.setenv(CONTAINER_RUNTIME_ENV, "host")
        _, prepare = _launch(monkeypatch, **_CONTAINER_KWARGS)
        prepare.assert_not_called()


class TestSaBenchCacheHostMode:
    @staticmethod
    def _plan():
        from srtctl.benchmarks.sa_bench_cache import CacheInputsPlan

        return CacheInputsPlan(
            recipe_name="r",
            cache_dir=Path("/cache"),
            container_image=Path("/images/vllm.sqsh"),
            mounts={Path("/models/qwen"): Path("/model")},
            isl=128,
            osl=128,
            concurrencies=(4,),
            prompt_counts=(40,),
            command=["python3", "build.py"],
            time_limit="00:10:00",
        )

    def test_pyxis_mode_mounts_the_container(self, monkeypatch):
        monkeypatch.delenv(CONTAINER_RUNTIME_ENV, raising=False)
        monkeypatch.setattr(config_module, "load_cluster_config", lambda: None)
        monkeypatch.setenv("SLURM_JOB_ID", "4242")
        argv = self._plan().srun_command()
        assert "--container-image=/images/vllm.sqsh" in argv
        assert "--container-mounts=/models/qwen:/model" in argv

    def test_host_mode_carries_no_pyxis_option(self, monkeypatch):
        monkeypatch.setenv(CONTAINER_RUNTIME_ENV, "host")
        monkeypatch.setattr(config_module, "load_cluster_config", lambda: None)
        monkeypatch.setenv("SLURM_JOB_ID", "4242")
        argv = self._plan().srun_command()
        assert not [arg for arg in argv if "container" in arg], argv
        assert argv[-2:] == ["python3", "build.py"]


class TestSbatchScriptExportsTheMode:
    @staticmethod
    def _config():
        from srtctl.core.schema import ModelConfig, ResourceConfig, RoleConfig, SrtConfig

        return SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8),
            roles={"agg": RoleConfig(nodes=1)},
        )

    def test_pyxis_cluster_exports_nothing(self, monkeypatch):
        from srtctl.cli.submit import generate_minimal_sbatch_script

        monkeypatch.setattr(config_module, "load_cluster_config", lambda: None)
        script = generate_minimal_sbatch_script(self._config(), Path("/tmp/test.yaml"))
        assert "SRTCTL_CONTAINER_RUNTIME" not in script

    def test_host_cluster_exports_the_mode(self, monkeypatch):
        from srtctl.cli.submit import generate_minimal_sbatch_script

        monkeypatch.setattr(config_module, "load_cluster_config", lambda: {"container_runtime": "host"})
        script = generate_minimal_sbatch_script(self._config(), Path("/tmp/test.yaml"))
        assert 'export SRTCTL_CONTAINER_RUNTIME="host"' in script


class TestPreflightHostMode:
    def test_container_alias_is_not_checked(self):
        from srtctl.core.validation import _preflight_container

        raw = {"model": {"container": "dynamo-vllm"}}
        resolved = {"model": {"container": "/images/absent.sqsh"}}
        cluster = {"containers": {"dynamo-vllm": "/images/absent.sqsh"}, "container_runtime": "host"}
        resolution, issues = _preflight_container(raw, resolved, cluster)
        assert issues == []
        assert resolution.ok is True
        assert "not checked" in resolution.message

    def test_pyxis_cluster_still_wants_the_file(self):
        from srtctl.core.validation import _preflight_container

        raw = {"model": {"container": "dynamo-vllm"}}
        resolved = {"model": {"container": "/images/absent.sqsh"}}
        cluster = {"containers": {"dynamo-vllm": "/images/absent.sqsh"}}
        resolution, issues = _preflight_container(raw, resolved, cluster)
        assert resolution.ok is False
        assert issues and issues[0].code == "container-not-available"


def test_example_cluster_file_documents_the_key():
    text = Path(__file__).resolve().parent.parent.joinpath("srtslurm.yaml.example").read_text()
    assert "# container_runtime: host" in text
    assert yaml.safe_load(text.replace("# container_runtime: host", "container_runtime: host")) is not None


class TestContainerImageUnderHostRuntime:
    def test_the_pyxis_image_path_need_not_exist_where_no_step_opens_it(self, tmp_path):
        missing = tmp_path / "vllm-runtime.sqsh"
        with pytest.raises(FileNotFoundError):
            resolve_container_image(str(missing))
        assert resolve_container_image(str(missing), must_exist=False) == missing

    def test_a_registry_image_is_returned_as_is_either_way(self):
        image = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2"
        assert resolve_container_image(image) == Path(image)
        assert resolve_container_image(image, must_exist=False) == Path(image)
