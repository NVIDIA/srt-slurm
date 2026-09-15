# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SLURM command construction."""

import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from srtctl.cli.mixins.worker_stage import WorkerStageMixin, worker_step_name
from srtctl.core.processes import ManagedProcess, ProcessRegistry
from srtctl.core.schema import ObservabilityConfig, ResourceConfig, RestartPolicy
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
    assert bash_cmd.startswith("ulimit -n 1048576 && exec python3 -m server")


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


def test_srun_export_env_renders_export_with_all_prefix() -> None:
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="12345"),
        patch("srtctl.core.slurm._get_cluster_bash_preamble", return_value=None),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value = MagicMock()
        start_srun_process(
            ["python3", "-m", "server"],
            srun_export_env={"ENROOT_REMAP_ROOT": "yes"},
        )
    srun_cmd = mock_popen.call_args.args[0]
    # ALL prefix preserves srun's normal full-env propagation; the var is added on top.
    assert "--export=ALL,ENROOT_REMAP_ROOT=yes" in srun_cmd


def test_srun_export_env_omitted_adds_no_export_flag() -> None:
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="12345"),
        patch("srtctl.core.slurm._get_cluster_bash_preamble", return_value=None),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value = MagicMock()
        start_srun_process(["python3", "-m", "server"])
    srun_cmd = mock_popen.call_args.args[0]
    assert not any(str(arg).startswith("--export") for arg in srun_cmd)


def test_start_srun_unsets_env_after_exports_before_preamble() -> None:
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="12345"),
        patch("srtctl.core.slurm._get_cluster_bash_preamble", return_value=None),
        patch("subprocess.Popen") as mock_popen,
    ):
        mock_popen.return_value = MagicMock()
        start_srun_process(
            ["python3", "-m", "server"],
            env_to_set={"VLLM_PORT": "20000"},
            env_to_unset=["VLLM_PORT"],
            bash_preamble="echo preamble",
        )

    bash_cmd = _built_bash_command(mock_popen)
    assert bash_cmd.index("export VLLM_PORT=20000") < bash_cmd.index("unset -- VLLM_PORT")
    assert bash_cmd.index("unset -- VLLM_PORT") < bash_cmd.index("echo preamble")
    assert bash_cmd.index("echo preamble") < bash_cmd.index("python3 -m server")


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
    backend.type = "vllm"

    mixin = WorkerStageMixin()
    mixin.config = SimpleNamespace(
        setup_script="setup.sh",
        frontend=SimpleNamespace(type="sglang"),
        dynamo=SimpleNamespace(install=False, request_plane="nats", event_plane="zmq"),
        observability=ObservabilityConfig(),
        profiling=SimpleNamespace(enabled=False, is_nsys=False),
        resources=ResourceConfig(),
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
    assert mock_srun.call_args.kwargs["env_to_unset"] is None
    # Named step, so cleanup can SIGTERM the engine through scancel instead of killing srun.
    assert mock_srun.call_args.kwargs["step_name"] == "prefill_0_node-a"


def _remap_worker_mixin(tmp_path: Path, *, frontend_type: str, dynamo_install: bool):
    """Build a WorkerStageMixin with a minimal config for remap-root injection tests."""
    backend = MagicMock()
    backend.type = "sglang"
    backend.build_worker_command.return_value = ["python3", "-m", "worker"]
    backend.get_environment_for_mode.return_value = {}
    backend.get_process_environment.return_value = {}

    mixin = WorkerStageMixin()
    mixin.config = SimpleNamespace(
        setup_script=None,
        frontend=SimpleNamespace(type=frontend_type),
        dynamo=SimpleNamespace(
            install=dynamo_install,
            get_install_commands=lambda: "echo install-dynamo",
            request_plane="nats",
            event_plane="zmq",
        ),
        observability=ObservabilityConfig(),
        profiling=SimpleNamespace(enabled=False, is_nsys=False),
        resources=ResourceConfig(),
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
    return mixin, process


def test_worker_stage_injects_remap_root_for_dynamo_install(tmp_path: Path) -> None:
    mixin, process = _remap_worker_mixin(tmp_path, frontend_type="dynamo", dynamo_install=True)
    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_worker(process, [process])

    assert mock_srun.call_args.kwargs["srun_export_env"] == {"ENROOT_REMAP_ROOT": "yes"}


def test_worker_stage_no_remap_root_for_sglang_frontend(tmp_path: Path) -> None:
    mixin, process = _remap_worker_mixin(tmp_path, frontend_type="sglang-router", dynamo_install=False)
    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_worker(process, [process])

    assert mock_srun.call_args.kwargs["srun_export_env"] is None


def test_worker_step_name_suffixes_relaunches_only() -> None:
    assert worker_step_name("decode", 1, "node-b") == "decode_1_node-b"
    assert worker_step_name("decode", 1, "node-b", 0) == "decode_1_node-b"
    assert worker_step_name("decode", 1, "node-b", 2) == "decode_1_node-b_r2"


def test_worker_stage_relaunch_suffixes_the_step_and_reuses_the_log_path(tmp_path: Path) -> None:
    """A supervisor relaunch is the same process spec under a new step name (the old log was rotated away)."""
    mixin, process = _remap_worker_mixin(tmp_path, frontend_type="sglang-router", dynamo_install=False)
    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        managed = mixin.start_worker(process, [process], attempt=2)

    assert mock_srun.call_args.kwargs["step_name"] == "prefill_0_node-a_r2"
    assert mock_srun.call_args.kwargs["output"] == str(tmp_path / "node-a_prefill_w0.out")
    assert managed.name == managed.step_name == "prefill_0_node-a_r2"
    assert managed.log_file == tmp_path / "node-a_prefill_w0.out"


def test_relaunch_endpoint_follows_the_backend_launch_strategy(tmp_path: Path) -> None:
    mixin, leader = _remap_worker_mixin(tmp_path, frontend_type="sglang-router", dynamo_install=False)
    follower = SimpleNamespace(**{**vars(leader), "node": "node-b"})
    mixin.runtime.nodes.worker.append("node-b")
    patches = (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process", return_value=MagicMock()),
    )

    # Per-process launching (SGLang): one step per rank of the endpoint.
    mixin.backend.get_srun_config.return_value = SimpleNamespace(
        launch_per_endpoint=False, mpi=None, oversubscribe=False, cpu_bind=None
    )
    with patches[0], patches[1] as mock_srun:
        procs = mixin.relaunch_endpoint([leader, follower], attempt=1)
    assert [p.name for p in procs] == ["prefill_0_node-a_r1", "prefill_0_node-b_r1"]
    assert [call.kwargs["nodelist"] for call in mock_srun.call_args_list] == [["node-a"], ["node-b"]]

    # Per-endpoint launching (TRT-LLM): one MPI step spanning every node.
    mixin.backend.get_srun_config.return_value = SimpleNamespace(
        launch_per_endpoint=True, mpi="pmix", oversubscribe=False, cpu_bind=None
    )
    with patches[0], patches[1] as mock_srun:
        procs = mixin.relaunch_endpoint([leader, follower], attempt=3)
    assert [p.name for p in procs] == ["prefill_0_node-a_r3"]
    assert mock_srun.call_args.kwargs["nodelist"] == ["node-a", "node-b"]
    assert mock_srun.call_args.kwargs["step_name"] == "prefill_0_node-a_r3"


def test_worker_ready_probe_targets_the_port_the_frontend_health_checks(tmp_path: Path) -> None:
    mixin, leader = _remap_worker_mixin(tmp_path, frontend_type="dynamo", dynamo_install=False)
    leader.http_port = 8100
    with patch("srtctl.cli.mixins.worker_stage.get_hostname_ip", return_value="10.0.0.7"):
        assert mixin.worker_ready_probe([leader]) == ("10.0.0.7", 5000)  # DYN_SYSTEM_PORT under dynamo
        mixin.config.frontend.type = "sglang-router"
        assert mixin.worker_ready_probe([leader]) == ("10.0.0.7", 8100)  # the engine's own HTTP port
        leader.http_port = 0
        assert mixin.worker_ready_probe([leader]) is None


def test_track_workers_supervises_only_roles_with_a_restart_policy(tmp_path: Path) -> None:
    base, prefill = _remap_worker_mixin(tmp_path, frontend_type="dynamo", dynamo_install=False)
    decode = SimpleNamespace(**{**vars(prefill), "endpoint_mode": "decode"})

    class Stage(WorkerStageMixin):
        backend_processes: ClassVar[list] = [prefill, decode]  # shadows the abstract property

    stage = Stage()
    stage.config = base.config
    stage.runtime = base.runtime
    stage.config.resources = ResourceConfig(decode_restart=RestartPolicy(policy="on-failure"))
    stage.config.health_check = SimpleNamespace(max_attempts=3, interval_seconds=10)

    registry = ProcessRegistry(job_id="1")
    supervisor = stage.build_worker_supervisor(registry, threading.Event())
    assert supervisor.ready_timeout == 30.0  # the same budget as the initial health gate

    worker_procs = {}
    for name in ("prefill_0_node-a", "decode_0_node-a"):
        popen = MagicMock(spec=subprocess.Popen)
        popen.poll.return_value = None
        popen.pid = 1
        worker_procs[name] = ManagedProcess(name=name, popen=popen, step_name=name)
    registry.add_processes(worker_procs)

    stage.track_workers(supervisor, worker_procs)

    assert worker_procs["decode_0_node-a"].supervised is True
    assert worker_procs["prefill_0_node-a"].supervised is False
    assert stage.worker_endpoint_groups() == {("prefill", 0): [prefill], ("decode", 0): [decode]}


def test_sglang_workers_skip_the_post_sigterm_crash_diagnostics_by_default(tmp_path: Path) -> None:
    """SGLang waits 60s for CUDA coredumps after a SIGTERM drain; nothing is collected without opting in."""
    mixin, process = _remap_worker_mixin(tmp_path, frontend_type="sglang-router", dynamo_install=False)
    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_worker(process, [process])
    env = mock_srun.call_args.kwargs["env_to_set"]
    assert env["SGLANG_CUDA_COREDUMP_BEFORE_CRASH"] == "0"
    assert env["SGLANG_PYSPY_DUMP_BEFORE_CRASH"] == "0"


def test_sglang_workers_keep_the_coredump_wait_when_the_recipe_opts_in(tmp_path: Path) -> None:
    mixin, process = _remap_worker_mixin(tmp_path, frontend_type="sglang-router", dynamo_install=False)
    mixin.runtime.environment = {"SGLANG_CUDA_COREDUMP": "1"}
    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_worker(process, [process])
    env = mock_srun.call_args.kwargs["env_to_set"]
    assert "SGLANG_CUDA_COREDUMP_BEFORE_CRASH" not in env
    assert env["SGLANG_CUDA_COREDUMP"] == "1"


def test_worker_stage_no_remap_root_when_dynamo_install_false(tmp_path: Path) -> None:
    # Dynamo frontend but container already has dynamo (install=False) → no install, no remap.
    mixin, process = _remap_worker_mixin(tmp_path, frontend_type="dynamo", dynamo_install=False)
    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_worker(process, [process])

    assert mock_srun.call_args.kwargs["srun_export_env"] is None


# ---- Event-plane propagation (DYN_EVENT_PLANE) ----


def _start_worker_env(tmp_path: Path, *, event_plane: str | None) -> dict[str, str]:
    mixin, process = _remap_worker_mixin(tmp_path, frontend_type="sglang-router", dynamo_install=False)
    mixin.config.dynamo.event_plane = event_plane
    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_worker(process, [process])
    return mock_srun.call_args.kwargs["env_to_set"]


def _start_endpoint_worker_env(tmp_path: Path, *, event_plane: str | None) -> dict[str, str]:
    mixin, process = _remap_worker_mixin(tmp_path, frontend_type="sglang-router", dynamo_install=False)
    mixin.config.dynamo.event_plane = event_plane
    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_endpoint_worker([process])
    return mock_srun.call_args.kwargs["env_to_set"]


def test_start_worker_event_plane_default_not_injected(tmp_path: Path) -> None:
    env = _start_worker_env(tmp_path, event_plane=None)
    assert "DYN_EVENT_PLANE" not in env


@pytest.mark.parametrize("event_plane", ["zmq", "nats"])
def test_start_worker_event_plane_injected(tmp_path: Path, event_plane: str) -> None:
    env = _start_worker_env(tmp_path, event_plane=event_plane)
    assert env["DYN_EVENT_PLANE"] == event_plane


def test_start_endpoint_worker_event_plane_default_not_injected(tmp_path: Path) -> None:
    env = _start_endpoint_worker_env(tmp_path, event_plane=None)
    assert "DYN_EVENT_PLANE" not in env


def test_start_endpoint_worker_request_plane_injected(tmp_path: Path) -> None:
    env = _start_endpoint_worker_env(tmp_path, event_plane=None)
    assert env["DYN_REQUEST_PLANE"] == "nats"


def test_trtllm_native_kv_events_receive_endpoint_hosts(tmp_path: Path) -> None:
    mixin, process = _remap_worker_mixin(tmp_path, frontend_type="dynamo", dynamo_install=False)
    mixin.config.backend.type = "trtllm"
    mixin.runtime.environment = {"DYN_TRTLLM_PUBLISH_KV_EVENTS": "true"}
    second_process = SimpleNamespace(**{**process.__dict__, "node": "node-b"})

    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_endpoint_worker([process, second_process])

    assert mock_srun.call_args.kwargs["env_to_set"]["DYN_TRTLLM_KV_EVENT_HOSTS"] == "node-a,node-b"


def test_trtllm_native_kv_event_host_override_is_preserved(tmp_path: Path) -> None:
    mixin, process = _remap_worker_mixin(tmp_path, frontend_type="dynamo", dynamo_install=False)
    mixin.config.backend.type = "trtllm"
    mixin.runtime.environment = {
        "DYN_TRTLLM_PUBLISH_KV_EVENTS": "true",
        "DYN_TRTLLM_KV_EVENT_HOSTS": "override-a,override-b",
    }
    second_process = SimpleNamespace(**{**process.__dict__, "node": "node-b"})

    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_endpoint_worker([process, second_process])

    assert mock_srun.call_args.kwargs["env_to_set"]["DYN_TRTLLM_KV_EVENT_HOSTS"] == "override-a,override-b"


def test_trtllm_sidecar_endpoint_kills_step_on_rank_failure(tmp_path: Path) -> None:
    mixin, process = _remap_worker_mixin(tmp_path, frontend_type="dynamo", dynamo_install=False)
    mixin.config.backend.type = "trtllm"
    mixin.config.dynamo.sidecar = True
    mixin.runtime.srun_options = {"exclusive": "", "kill-on-bad-exit": "0"}

    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_endpoint_worker([process])

    assert mock_srun.call_args.kwargs["srun_options"] == {
        "exclusive": "",
        "kill-on-bad-exit": "1",
    }


def test_sglang_sidecar_trusts_the_bundled_rust_extension(tmp_path: Path) -> None:
    mixin, process = _remap_worker_mixin(tmp_path, frontend_type="dynamo", dynamo_install=False)
    mixin.config.backend.type = "sglang"
    mixin.config.dynamo.sidecar = True

    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_worker(process, [process])

    assert mock_srun.call_args.kwargs["env_to_set"]["SGLANG_RUST_BUILD_MODE"] == "never"


def test_sglang_sidecar_rust_build_mode_respects_the_recipe(tmp_path: Path) -> None:
    mixin, process = _remap_worker_mixin(tmp_path, frontend_type="dynamo", dynamo_install=False)
    mixin.config.backend.type = "sglang"
    mixin.config.dynamo.sidecar = True
    mixin.runtime.environment = {"SGLANG_RUST_BUILD_MODE": "auto"}

    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_worker(process, [process])

    assert mock_srun.call_args.kwargs["env_to_set"]["SGLANG_RUST_BUILD_MODE"] == "auto"

    mixin_off, process_off = _remap_worker_mixin(tmp_path, frontend_type="dynamo", dynamo_install=False)
    mixin_off.config.backend.type = "sglang"
    mixin_off.config.dynamo.sidecar = False
    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin_off.start_worker(process_off, [process_off])
    assert "SGLANG_RUST_BUILD_MODE" not in mock_srun.call_args.kwargs["env_to_set"]


def test_vllm_sidecar_disables_plugins_by_default(tmp_path: Path) -> None:
    mixin, process = _remap_worker_mixin(tmp_path, frontend_type="dynamo", dynamo_install=False)
    mixin.config.backend.type = "vllm"
    mixin.config.dynamo.sidecar = True

    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_worker(process, [process])

    assert mock_srun.call_args.kwargs["env_to_set"]["VLLM_PLUGINS"] == ""


@pytest.mark.parametrize("event_plane", ["zmq", "nats"])
def test_start_endpoint_worker_event_plane_injected(tmp_path: Path, event_plane: str) -> None:
    env = _start_endpoint_worker_env(tmp_path, event_plane=event_plane)
    assert env["DYN_EVENT_PLANE"] == event_plane


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


def test_worker_stage_unsets_vllm_port_for_multinode_endpoint(tmp_path: Path) -> None:
    backend = MagicMock()
    backend.type = "vllm"
    backend.build_worker_command.return_value = ["python3", "-m", "worker"]
    backend.get_environment_for_mode.return_value = {}
    backend.get_process_environment.return_value = {}

    mixin = WorkerStageMixin()
    mixin.config = SimpleNamespace(
        setup_script=None,
        frontend=SimpleNamespace(type="sglang"),
        dynamo=SimpleNamespace(install=False, request_plane="nats", event_plane=None),
        observability=ObservabilityConfig(),
        profiling=SimpleNamespace(enabled=False, is_nsys=False),
        resources=ResourceConfig(),
        backend=backend,
    )
    mixin.runtime = SimpleNamespace(
        log_dir=tmp_path,
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        network_interface=None,
        nodes=SimpleNamespace(infra="infra-node", worker=["node-a", "node-b"]),
        gpus_per_node=8,
        environment={},
        container_image=Path("/container.sqsh"),
        container_mounts={},
        srun_options=[],
    )
    process = SimpleNamespace(
        endpoint_mode="decode",
        endpoint_index=0,
        node="node-a",
        sys_port=5000,
        gpu_indices=list(range(8)),
        cuda_visible_devices="0,1,2,3,4,5,6,7",
        het_group=None,
    )
    peer_process = SimpleNamespace(node="node-b")

    with (
        patch("srtctl.cli.mixins.worker_stage.generate_capture_script", return_value="fingerprint || true"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as mock_srun,
    ):
        mock_srun.return_value = MagicMock()
        mixin.start_worker(process, [process, peer_process])

    assert mock_srun.call_args.kwargs["env_to_unset"] == ["VLLM_PORT"]
