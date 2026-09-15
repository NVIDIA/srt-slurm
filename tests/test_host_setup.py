# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for host_setup: commands run on each node's bare host, outside the container."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import HostSetupConfig, ResourceConfig, SrtConfig

LOCK_CLOCKS = "sudo -n nvidia-smi -lmc <min>,<max>"
RESET_CLOCKS = "sudo -n nvidia-smi -rmc"

BASE_RECIPE = {
    "name": "host-setup-test",
    "model": {"path": "/models/test", "container": "test.sqsh", "precision": "fp8"},
    "resources": {
        "gpu_type": "gb200",
        "gpus_per_node": 4,
        "prefill_nodes": 1,
        "decode_nodes": 1,
        "prefill_workers": 1,
        "decode_workers": 1,
    },
}


def _load_recipe(overrides: dict | None = None) -> SrtConfig:
    data = {**BASE_RECIPE, **(overrides or {})}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        return SrtConfig.from_yaml(Path(f.name))


def _config(**host_setup_kwargs) -> SrtConfig:
    return SrtConfig(
        name="host-setup-test",
        model={"path": "/models/test", "container": "test.sqsh", "precision": "fp8"},
        resources=ResourceConfig(gpu_type="gb200", gpus_per_node=4, prefill_nodes=1, decode_nodes=1),
        host_setup=HostSetupConfig(**host_setup_kwargs),
    )


def _runtime(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(
        job_id="12345",
        run_name="test-run",
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node1", "node2")),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=tmp_path,
        model_path=Path("/models/test"),
        container_image=Path("/img.sqsh"),
        gpus_per_node=4,
        network_interface=None,
        container_mounts={},
        environment={},
    )


def _ok_proc() -> MagicMock:
    proc = MagicMock()
    proc.wait.return_value = 0
    return proc


class TestSchema:
    def test_defaults_to_disabled(self):
        config = _load_recipe()
        assert config.host_setup.commands == []
        assert config.host_setup.teardown == []
        assert config.host_setup.enabled is False

    def test_loads_from_recipe(self):
        config = _load_recipe(
            {
                "host_setup": {
                    "commands": [LOCK_CLOCKS],
                    "teardown": [RESET_CLOCKS],
                    "nodes": "workers",
                    "ignore_failure": True,
                    "timeout_seconds": 60,
                }
            }
        )
        assert config.host_setup.commands == [LOCK_CLOCKS]
        assert config.host_setup.teardown == [RESET_CLOCKS]
        assert config.host_setup.nodes == "workers"
        assert config.host_setup.ignore_failure is True
        assert config.host_setup.timeout_seconds == 60
        assert config.host_setup.enabled is True

    def test_rejects_unknown_node_scope(self):
        with pytest.raises(ValidationError):
            _load_recipe({"host_setup": {"commands": [LOCK_CLOCKS], "nodes": "prefill"}})

    def test_rejects_empty_command(self):
        with pytest.raises(ValidationError, match="host_setup.commands"):
            _load_recipe({"host_setup": {"commands": [LOCK_CLOCKS, "   "]}})

    def test_rejects_empty_teardown_command(self):
        with pytest.raises(ValidationError, match="host_setup.teardown"):
            _load_recipe({"host_setup": {"commands": [LOCK_CLOCKS], "teardown": [""]}})

    def test_rejects_nonpositive_timeout(self):
        with pytest.raises(ValidationError, match="timeout_seconds"):
            _load_recipe({"host_setup": {"commands": [LOCK_CLOCKS], "timeout_seconds": 0}})


class TestClusterDefault:
    """default_host_setup in srtslurm.yaml applies cluster-wide."""

    CLUSTER = {"default_host_setup": {"commands": [LOCK_CLOCKS], "teardown": [RESET_CLOCKS]}}

    def test_applied_when_recipe_omits_block(self):
        resolved = resolve_config_with_defaults({**BASE_RECIPE}, self.CLUSTER)
        assert resolved["host_setup"]["commands"] == [LOCK_CLOCKS]
        assert resolved["host_setup"]["teardown"] == [RESET_CLOCKS]

    def test_recipe_block_wins(self):
        recipe = {**BASE_RECIPE, "host_setup": {"commands": ["sudo -n nvidia-smi -lgc 1980,1980"]}}
        resolved = resolve_config_with_defaults(recipe, self.CLUSTER)
        assert resolved["host_setup"]["commands"] == ["sudo -n nvidia-smi -lgc 1980,1980"]
        # Whole-block replace: the cluster teardown does not leak into a recipe
        # that took ownership of the block.
        assert not resolved["host_setup"].get("teardown")

    def test_empty_commands_opts_out(self):
        """`host_setup: {commands: []}` is the documented way to skip the cluster default."""
        recipe = {**BASE_RECIPE, "host_setup": {"commands": []}}
        resolved = resolve_config_with_defaults(recipe, self.CLUSTER)
        assert resolved["host_setup"]["commands"] == []

    def test_no_cluster_config_is_a_noop(self):
        resolved = resolve_config_with_defaults({**BASE_RECIPE}, None)
        assert "host_setup" not in resolved

    def test_cluster_schema_accepts_the_key(self):
        """srtslurm.yaml is schema-validated, and a failure there silently drops
        every cluster default -- so the key has to be declared, not just read."""
        from srtctl.core.schema import ClusterConfig

        loaded = ClusterConfig.Schema().load({"cluster": "bruh", **self.CLUSTER})
        assert loaded.default_host_setup.commands == [LOCK_CLOCKS]


class TestOrchestrator:
    def test_runs_one_container_less_srun_per_node(self, tmp_path):
        orchestrator = SweepOrchestrator(config=_config(commands=[LOCK_CLOCKS]), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=_ok_proc()) as srun:
            orchestrator._run_host_setup()

        assert srun.call_count == 3  # head/infra (node0) + node1 + node2
        for call in srun.call_args_list:
            assert call.kwargs["container_image"] is None, "host_setup must not run inside the job container"
            assert call.kwargs["command"] == ["bash", "-c", LOCK_CLOCKS]
        assert [call.kwargs["nodelist"] for call in srun.call_args_list] == [["node0"], ["node1"], ["node2"]]

    def test_workers_scope_skips_head(self, tmp_path):
        config = _config(commands=[LOCK_CLOCKS], nodes="workers")
        orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=_ok_proc()) as srun:
            orchestrator._run_host_setup()

        assert [call.kwargs["nodelist"] for call in srun.call_args_list] == [["node1"], ["node2"]]

    def test_head_node_not_run_twice_when_it_also_hosts_workers(self, tmp_path):
        runtime = RuntimeContext(
            job_id="12345",
            run_name="test-run",
            nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node0", "node1")),
            head_node_ip="10.0.0.1",
            infra_node_ip="10.0.0.1",
            log_dir=tmp_path,
            model_path=Path("/models/test"),
            container_image=Path("/img.sqsh"),
            gpus_per_node=4,
            network_interface=None,
            container_mounts={},
            environment={},
        )
        orchestrator = SweepOrchestrator(config=_config(commands=[LOCK_CLOCKS]), runtime=runtime)

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=_ok_proc()) as srun:
            orchestrator._run_host_setup()

        assert [call.kwargs["nodelist"] for call in srun.call_args_list] == [["node0"], ["node1"]]

    def test_multiple_commands_are_chained(self, tmp_path):
        config = _config(commands=[LOCK_CLOCKS, "sudo -n nvidia-smi -pm 1"])
        orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=_ok_proc()) as srun:
            orchestrator._run_host_setup()

        assert srun.call_args_list[0].kwargs["command"] == [
            "bash",
            "-c",
            f"{LOCK_CLOCKS} && sudo -n nvidia-smi -pm 1",
        ]

    def test_disabled_block_launches_nothing(self, tmp_path):
        orchestrator = SweepOrchestrator(config=_config(), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process") as srun:
            orchestrator._run_host_setup()

        srun.assert_not_called()

    def test_failure_fails_the_job(self, tmp_path):
        proc = MagicMock()
        proc.wait.return_value = 1
        orchestrator = SweepOrchestrator(config=_config(commands=[LOCK_CLOCKS]), runtime=_runtime(tmp_path))

        with (
            patch("srtctl.cli.do_sweep.start_srun_process", return_value=proc),
            pytest.raises(RuntimeError, match="host_setup failed on"),
        ):
            orchestrator._run_host_setup()

    def test_ignore_failure_continues(self, tmp_path):
        proc = MagicMock()
        proc.wait.return_value = 1
        config = _config(commands=[LOCK_CLOCKS], ignore_failure=True)
        orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=proc):
            orchestrator._run_host_setup()  # does not raise

    def test_timeout_is_killed_and_fails(self, tmp_path):
        """A sudo that prompts for a password hangs rather than exiting nonzero."""

        def hang(timeout=None):
            # The timed wait never returns; the reaping wait after kill() does.
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="srun", timeout=timeout)
            return -9

        proc = MagicMock()
        proc.wait.side_effect = hang
        config = _config(commands=[LOCK_CLOCKS], timeout_seconds=5)
        orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))

        with (
            patch("srtctl.cli.do_sweep.start_srun_process", return_value=proc),
            pytest.raises(RuntimeError, match="host_setup failed on"),
        ):
            orchestrator._run_host_setup()

        proc.kill.assert_called()

    def test_timeout_uses_configured_budget(self, tmp_path):
        config = _config(commands=[LOCK_CLOCKS], timeout_seconds=42)
        orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))
        proc = _ok_proc()

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=proc):
            orchestrator._run_host_setup()

        proc.wait.assert_called_with(timeout=42)


class TestTeardown:
    def test_runs_after_setup(self, tmp_path):
        config = _config(commands=[LOCK_CLOCKS], teardown=[RESET_CLOCKS])
        orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=_ok_proc()) as srun:
            orchestrator._run_host_setup()
            srun.reset_mock()
            orchestrator._run_host_teardown()

        assert srun.call_count == 3
        for call in srun.call_args_list:
            assert call.kwargs["command"] == ["bash", "-c", RESET_CLOCKS]
            assert call.kwargs["container_image"] is None

    def test_skipped_when_setup_never_ran(self, tmp_path):
        """A job that dies before Stage 0 has not touched the nodes."""
        config = _config(commands=[LOCK_CLOCKS], teardown=[RESET_CLOCKS])
        orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process") as srun:
            orchestrator._run_host_teardown()

        srun.assert_not_called()

    def test_runs_after_an_ignored_setup_failure(self, tmp_path):
        """Setup may have half-applied, so the nodes still need reverting."""
        setup_proc = MagicMock()
        setup_proc.wait.return_value = 1
        config = _config(commands=[LOCK_CLOCKS], teardown=[RESET_CLOCKS], ignore_failure=True)
        orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=setup_proc):
            orchestrator._run_host_setup()
        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=_ok_proc()) as srun:
            orchestrator._run_host_teardown()

        assert srun.call_count == 3

    def test_failure_never_raises(self, tmp_path):
        """Teardown runs in the finally block; raising would mask the job's result."""
        config = _config(commands=[LOCK_CLOCKS], teardown=[RESET_CLOCKS])
        orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=_ok_proc()):
            orchestrator._run_host_setup()
        with patch("srtctl.cli.do_sweep.start_srun_process", side_effect=OSError("srun gone")):
            orchestrator._run_host_teardown()  # does not raise

    def test_teardown_only_block_still_runs(self, tmp_path):
        config = _config(teardown=[RESET_CLOCKS])
        orchestrator = SweepOrchestrator(config=config, runtime=_runtime(tmp_path))

        with patch("srtctl.cli.do_sweep.start_srun_process", return_value=_ok_proc()) as srun:
            orchestrator._run_host_setup()
            assert srun.call_count == 0
            orchestrator._run_host_teardown()

        assert srun.call_count == 3


class TestDryRun:
    """host_setup must be visible in `srtctl dry-run` before an allocation is spent."""

    def test_commands_are_shown(self, capsys):
        from srtctl.cli.submit import show_config_details

        show_config_details(_load_recipe({"host_setup": {"commands": [LOCK_CLOCKS], "teardown": [RESET_CLOCKS]}}))
        output = capsys.readouterr().out

        assert "Host Setup" in output
        assert "nvidia-smi -lmc <min>,<max>" in output
        assert "nvidia-smi -rmc" in output
        assert "nodes=all" in output

    def test_sudo_note_is_shown(self, capsys):
        from srtctl.cli.submit import show_config_details

        show_config_details(_load_recipe({"host_setup": {"commands": [LOCK_CLOCKS]}}))
        assert "passwordless" in capsys.readouterr().out

    def test_missing_teardown_is_flagged(self, capsys):
        from srtctl.cli.submit import show_config_details

        show_config_details(_load_recipe({"host_setup": {"commands": [LOCK_CLOCKS]}}))
        assert "outlives" in capsys.readouterr().out

    def test_teardown_present_is_not_flagged(self, capsys):
        from srtctl.cli.submit import show_config_details

        show_config_details(_load_recipe({"host_setup": {"commands": [LOCK_CLOCKS], "teardown": [RESET_CLOCKS]}}))
        assert "outlives" not in capsys.readouterr().out

    def test_hidden_when_unconfigured(self, capsys):
        from srtctl.cli.submit import show_config_details

        show_config_details(_load_recipe())
        assert "Host Setup" not in capsys.readouterr().out

    def test_cluster_default_is_attributed_to_srtslurm_yaml(self, capsys):
        """An unexpected `sudo` in dry-run should name the file it came from."""
        from srtctl.cli import submit

        config = _load_recipe({"host_setup": {"commands": [LOCK_CLOCKS]}})
        with patch.object(
            submit,
            "get_srtslurm_setting",
            side_effect=lambda key, default=None: (
                {"commands": [LOCK_CLOCKS], "teardown": []} if key == "default_host_setup" else default
            ),
        ):
            submit.show_config_details(config)

        assert "srtslurm.yaml (default_host_setup)" in capsys.readouterr().out

    def test_recipe_block_is_attributed_to_the_recipe(self, capsys):
        from srtctl.cli import submit

        config = _load_recipe({"host_setup": {"commands": ["sudo -n nvidia-smi -pm 1"]}})
        with patch.object(
            submit,
            "get_srtslurm_setting",
            side_effect=lambda key, default=None: (
                {"commands": [LOCK_CLOCKS], "teardown": []} if key == "default_host_setup" else default
            ),
        ):
            submit.show_config_details(config)

        assert "source: recipe" in capsys.readouterr().out
