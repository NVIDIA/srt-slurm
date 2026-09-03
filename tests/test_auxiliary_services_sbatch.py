# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auxiliary_services on the real multi-node sbatch/SLURM path.

Counterpart to tests/test_auxiliary_services.py, which covers the --bash dev-mode
DirectRunner stage; this covers AuxiliaryServiceStageMixin / SweepOrchestrator, mocking
srun the same way tests/test_host_setup.py does for the sibling real-path stage.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import AuxiliaryServiceConfig, AuxiliaryServiceSourceConfig, ResourceConfig, SrtConfig

ROUTER = AuxiliaryServiceConfig(
    name="thunderagent-router",
    command=["python3", "-m", "dynamo.thunderagent_router"],
)


def _config(*, auxiliary_services: list[AuxiliaryServiceConfig] | None = None) -> SrtConfig:
    return SrtConfig(
        name="aux-sbatch-test",
        model={"path": "/models/test", "container": "test.sqsh", "precision": "fp8"},
        resources=ResourceConfig(gpu_type="gb200", gpus_per_node=4, prefill_nodes=1, decode_nodes=1),
        auxiliary_services=auxiliary_services or [],
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


class TestOrchestrator:
    def test_no_services_is_a_noop(self, tmp_path):
        orchestrator = SweepOrchestrator(config=_config(), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.mixins.auxiliary_stage.start_srun_process") as srun:
            processes = orchestrator.start_auxiliary_services()

        srun.assert_not_called()
        assert processes == []

    def test_launches_on_head_node_and_injects_discovery_env(self, tmp_path):
        orchestrator = SweepOrchestrator(config=_config(auxiliary_services=[ROUTER]), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.mixins.auxiliary_stage.start_srun_process", return_value=_ok_proc()) as srun:
            processes = orchestrator.start_auxiliary_services()

        srun.assert_called_once()
        call = srun.call_args
        assert call.kwargs["nodelist"] == ["node0"]
        assert call.kwargs["command"] == ROUTER.command
        assert call.kwargs["container_image"] == "/img.sqsh"
        assert call.kwargs["env_to_set"]["ETCD_ENDPOINTS"] == "http://node0:2379"
        assert call.kwargs["env_to_set"]["NATS_SERVER"] == "nats://node0:4222"

        assert len(processes) == 1
        proc = processes[0]
        assert proc.name == "auxiliary_thunderagent-router"
        assert proc.node == "node0"
        assert proc.critical is False

    def test_own_container_image_overrides_job_container(self, tmp_path):
        service = AuxiliaryServiceConfig(
            name="svc", command=["true"], container_image="my-image.sqsh", inherit_discovery_env=False
        )
        orchestrator = SweepOrchestrator(config=_config(auxiliary_services=[service]), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.mixins.auxiliary_stage.start_srun_process", return_value=_ok_proc()) as srun:
            orchestrator.start_auxiliary_services()

        assert srun.call_args.kwargs["container_image"] == "my-image.sqsh"
        assert "ETCD_ENDPOINTS" not in srun.call_args.kwargs["env_to_set"]
        assert "NATS_SERVER" not in srun.call_args.kwargs["env_to_set"]

    def test_own_env_merges_on_top_of_discovery_env(self, tmp_path):
        service = AuxiliaryServiceConfig(name="svc", command=["true"], env={"FOO": "bar"})
        orchestrator = SweepOrchestrator(config=_config(auxiliary_services=[service]), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.mixins.auxiliary_stage.start_srun_process", return_value=_ok_proc()) as srun:
            orchestrator.start_auxiliary_services()

        env = srun.call_args.kwargs["env_to_set"]
        assert env["FOO"] == "bar"
        assert env["ETCD_ENDPOINTS"] == "http://node0:2379"

    def test_launches_in_declared_order(self, tmp_path):
        svc_b = AuxiliaryServiceConfig(name="b", command=["echo", "b"])
        svc_a = AuxiliaryServiceConfig(name="a", command=["echo", "a"])
        svc_c = AuxiliaryServiceConfig(name="c", command=["echo", "c"])
        orchestrator = SweepOrchestrator(
            config=_config(auxiliary_services=[svc_b, svc_a, svc_c]), runtime=_runtime(tmp_path)
        )

        with patch("srtctl.cli.mixins.auxiliary_stage.start_srun_process", return_value=_ok_proc()) as srun:
            orchestrator.start_auxiliary_services()

        launched_names = [call.kwargs["command"][1] for call in srun.call_args_list]
        assert launched_names == ["b", "a", "c"]

    def test_source_is_cloned_and_built_once_before_launch(self, tmp_path):
        service = AuxiliaryServiceConfig(
            name="router",
            command=["python3", "-m", "router"],
            source=AuxiliaryServiceSourceConfig(git="https://example.com/repo", rev="refs/pull/1/head"),
            build_command=["bash", "-lc", "pip install -e ."],
        )
        orchestrator = SweepOrchestrator(config=_config(auxiliary_services=[service]), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.mixins.auxiliary_stage.start_srun_process", return_value=_ok_proc()) as srun:
            orchestrator.start_auxiliary_services()

        # clone (bare host) -> build (containerized) -> launch (containerized)
        assert srun.call_count == 3
        clone_call, build_call, launch_call = srun.call_args_list
        assert clone_call.kwargs["container_image"] is None
        assert "git -c http.version=HTTP/1.1 clone" in clone_call.kwargs["command"][-1]
        assert build_call.kwargs["container_image"] == "/img.sqsh"
        assert build_call.kwargs["command"] == service.build_command
        assert "cd " in build_call.kwargs["bash_preamble"]
        assert launch_call.kwargs["command"] == service.command
        assert "cd " in launch_call.kwargs["bash_preamble"]

        # Regression: build/launch run inside the container, where log_dir is
        # mounted at /logs (RuntimeContext.from_config), not at its host path.
        # cd'ing to the host path there is a silent, 100%-reproducible
        # "No such file or directory" -- not a timing issue a longer poll fixes.
        assert "/logs/auxiliary_services/router/src" in build_call.kwargs["bash_preamble"]
        assert str(tmp_path) not in build_call.kwargs["bash_preamble"]
        assert "/logs/auxiliary_services/router/src" in launch_call.kwargs["bash_preamble"]
        assert str(tmp_path) not in launch_call.kwargs["bash_preamble"]

    def test_clone_skipped_when_checkout_already_exists(self, tmp_path):
        service = AuxiliaryServiceConfig(
            name="router",
            command=["python3", "-m", "router"],
            source=AuxiliaryServiceSourceConfig(git="https://example.com/repo", rev="refs/pull/1/head"),
        )
        checkout = tmp_path / "auxiliary_services" / "router" / "src"
        checkout.mkdir(parents=True)
        orchestrator = SweepOrchestrator(config=_config(auxiliary_services=[service]), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.mixins.auxiliary_stage.start_srun_process", return_value=_ok_proc()) as srun:
            orchestrator.start_auxiliary_services()

        # The bash script no-ops on an existing checkout (an `if [ ! -d ... ]` guard) but
        # the clone srun call is still made; the launch call follows.
        assert srun.call_count == 2

    def test_source_clone_failure_raises(self, tmp_path):
        service = AuxiliaryServiceConfig(
            name="router",
            command=["true"],
            source=AuxiliaryServiceSourceConfig(git="https://example.com/repo", rev="refs/pull/1/head"),
        )
        proc = MagicMock()
        proc.wait.return_value = 1
        orchestrator = SweepOrchestrator(config=_config(auxiliary_services=[service]), runtime=_runtime(tmp_path))

        with patch("srtctl.cli.mixins.auxiliary_stage.start_srun_process", return_value=proc):
            try:
                orchestrator.start_auxiliary_services()
                raise AssertionError("expected RuntimeError")
            except RuntimeError as exc:
                assert "router" in str(exc)
                assert "clone" in str(exc)

    def test_build_command_failure_raises(self, tmp_path):
        # build_command only runs when source is set (same as the --bash dev-mode stage:
        # it's "meaningful only when source is set" per AuxiliaryServiceConfig's docstring).
        service = AuxiliaryServiceConfig(
            name="router",
            command=["true"],
            source=AuxiliaryServiceSourceConfig(git="https://example.com/repo", rev="refs/pull/1/head"),
            build_command=["bash", "-lc", "false"],
        )
        clone_proc = _ok_proc()
        build_proc = MagicMock()
        build_proc.wait.return_value = 1
        orchestrator = SweepOrchestrator(config=_config(auxiliary_services=[service]), runtime=_runtime(tmp_path))

        with patch(
            "srtctl.cli.mixins.auxiliary_stage.start_srun_process", side_effect=[clone_proc, build_proc]
        ):
            try:
                orchestrator.start_auxiliary_services()
                raise AssertionError("expected RuntimeError")
            except RuntimeError as exc:
                assert "router" in str(exc)
                assert "build_command" in str(exc)
