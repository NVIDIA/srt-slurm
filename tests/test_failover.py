# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shadow engine recovery (``engine.failover``): schema, topology, commands, launch, and dry-run.

The acceptance recipe is two TP1 vLLM workers on one node with one shadow each.
Per worker that is three steps on the node: the GMS sidecar, engine 0, and the
shadow, all pinned to the same GPU, with the engines told the same socket
directory and lock file.
"""

from __future__ import annotations

import shlex
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends.vllm import (
    FAILOVER_LOCK_FILENAME,
    GMS_READY_MARKER,
    RESPAWN_COMMAND_ENV,
    VLLMFailoverConfig,
    VLLMProtocol,
    build_gms_sidecar_command,
    build_respawn_command,
    failover_worker_dir,
)
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.submit import show_config_details
from srtctl.core.readiness import ProcessDied
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Endpoint, NodePortAllocator, Process, endpoints_to_processes
from srtctl.mock import MockOptions, run_mock_sweep
from srtctl.ports import VLLM_MASTER_PORT_BASE, VLLM_MASTER_PORT_STRIDE

SRUN = "srtctl.cli.mixins.worker_stage.start_srun_process"
WAIT = "srtctl.cli.mixins.worker_stage.wait_until_ready"
HOST_IP = "srtctl.core.slurm.get_hostname_ip"

TOY = {
    "schema": 2,
    "name": "failover-toy",
    "model": {"path": "/models/qwen3-0.6b", "container": "/vllm-runtime.sqsh", "precision": "bf16"},
    "resources": {"gpu_type": "b200", "gpus_per_node": 8},
    "dynamo": {"install": False},
    "frontend": {"type": "dynamo", "enable_multiple_frontends": False},
    "engine": {"type": "vllm", "failover": {}},
    "roles": {
        "agg": {
            "nodes": 1,
            "workers": 2,
            "gpus": 1,
            "args": {"tensor-parallel-size": 1, "gpu-memory-utilization": 0.4, "max-model-len": 4096},
        }
    },
    "benchmark": {"type": "manual"},
    "observability": {"tachometer": {"enabled": False}},
}


def _data(**overrides) -> dict:
    data = yaml.safe_load(yaml.dump(TOY))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return data


def _load_data(data: dict) -> SrtConfig:
    """The 2.0 layout (``engine:``, ``roles:``) is normalized by the YAML loader, so go through it."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        yaml.dump(data, handle)
        path = Path(handle.name)
    try:
        return SrtConfig.from_yaml(path)
    finally:
        path.unlink(missing_ok=True)


def _load(**overrides) -> SrtConfig:
    return _load_data(_data(**overrides))


def _runtime(tmp_path: Path, workers: tuple[str, ...] = ("node1",)) -> RuntimeContext:
    return RuntimeContext(
        job_id="15600",
        run_name="failover-toy",
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=workers),
        head_node_ip="10.0.0.10",
        infra_node_ip="10.0.0.10",
        log_dir=tmp_path,
        model_path=Path("/models/qwen3-0.6b"),
        container_image=Path("/vllm-runtime.sqsh"),
        gpus_per_node=8,
        network_interface="eth0",
        container_mounts={},
        environment={},
    )


def _proc() -> MagicMock:
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    return proc


# --- schema -------------------------------------------------------------------


def test_defaults_and_engines_per_worker() -> None:
    config = _load()
    assert isinstance(config.backend, VLLMProtocol)
    failover = config.backend.failover
    assert failover == VLLMFailoverConfig()
    assert failover.shadow_engines == 1
    assert failover.restart == "always"
    assert failover.restart_backoff_seconds == 5
    assert failover.shared_dir == "/dev/shm"
    assert failover.engines_per_worker == 2
    assert config.backend.engines_per_process == 2


def test_without_failover_nothing_changes() -> None:
    config = _load(engine="vllm")
    assert config.backend.failover is None
    assert config.backend.engines_per_process == 1


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"shadow_engines": 0}, "shadow_engines must be at least 1"),
        ({"restart": "on-failure"}, "restart"),
        ({"restart_backoff_seconds": 0}, "restart_backoff_seconds must be at least 1"),
        ({"shared_dir": "shm"}, "shared_dir must be an absolute directory"),
        ({"shared_dir": "/"}, "shared_dir must be an absolute directory"),
        ({"gms_startup_timeout_seconds": 0}, "gms_startup_timeout_seconds must be at least 1"),
    ],
)
def test_block_validation(block: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _load(engine={"type": "vllm", "failover": block})


def test_requires_dynamo_frontend() -> None:
    with pytest.raises(ValidationError, match="requires frontend.type: dynamo"):
        _load(frontend={"type": "vllm-router"})


def test_rejects_sidecar_mode() -> None:
    with pytest.raises(ValidationError, match="dynamo.sidecar"):
        _load(dynamo={"install": False, "sidecar": True})


def test_rejects_data_parallel() -> None:
    data = _data()
    data["roles"]["agg"]["gpus"] = 2
    data["roles"]["agg"]["args"]["data-parallel-size"] = 2
    with pytest.raises(ValidationError, match="does not support data-parallel-size"):
        _load_data(data)


def test_rejects_other_load_format_but_accepts_gms() -> None:
    data = _data()
    data["roles"]["agg"]["args"]["load-format"] = "safetensors"
    with pytest.raises(ValidationError, match="load-format must be gms or unset"):
        _load_data(data)
    data["roles"]["agg"]["args"]["load_format"] = data["roles"]["agg"]["args"].pop("load-format")
    data["roles"]["agg"]["args"]["load_format"] = "gms"
    _load_data(data)


def test_pip_installed_dynamo_only_warns(caplog) -> None:
    with caplog.at_level("WARNING"):
        config = _load(dynamo={"install": True, "source": {"pypi": "1.4.2"}})
    assert config.backend.failover is not None
    assert "gpu_memory_service package" in caplog.text


# --- topology -----------------------------------------------------------------


def _endpoints() -> list[Endpoint]:
    return [
        Endpoint(mode="agg", index=0, nodes=("node1",), gpu_indices=frozenset({0}), gpus_per_node=8),
        Endpoint(mode="agg", index=1, nodes=("node1",), gpu_indices=frozenset({1}), gpus_per_node=8),
    ]


def test_engines_per_process_emits_one_process_per_engine_with_distinct_ports() -> None:
    processes = endpoints_to_processes(_endpoints(), engines_per_process=2)
    assert [(p.endpoint_index, p.engine_id) for p in processes] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    e0, e1 = processes[0], processes[1]
    assert e0.gpu_indices == e1.gpu_indices == frozenset({0})
    assert e0.node_rank == e1.node_rank == 0
    assert e0.engine_suffix == "" and e1.engine_suffix == "_e1"
    for port in ("sys_port", "http_port", "kv_events_port", "nixl_port"):
        values = [getattr(p, port) for p in processes]
        assert len(set(values)) == len(values), f"{port} collides: {values}"


def test_prefill_engines_get_their_own_bootstrap_port() -> None:
    endpoints = [Endpoint(mode="prefill", index=0, nodes=("node1",), gpu_indices=frozenset({0}), gpus_per_node=8)]
    e0, e1 = endpoints_to_processes(endpoints, engines_per_process=2)
    assert e0.bootstrap_port is not None and e1.bootstrap_port is not None
    assert e0.bootstrap_port != e1.bootstrap_port


def test_engines_per_process_default_is_the_old_layout() -> None:
    old = endpoints_to_processes(_endpoints(), port_allocator=NodePortAllocator())
    new = endpoints_to_processes(_endpoints(), port_allocator=NodePortAllocator(), engines_per_process=1)
    assert old == new
    with pytest.raises(ValueError, match="at least 1"):
        endpoints_to_processes(_endpoints(), engines_per_process=0)


def test_backend_doubles_processes_under_failover() -> None:
    config = _load()
    processes = config.backend.endpoints_to_processes(_endpoints(), frontend_type="dynamo")
    assert len(processes) == 4
    assert sum(p.engine_id == 0 for p in processes) == 2


# --- commands and environment -----------------------------------------------------


def _process(engine_id: int = 0, gpus: frozenset[int] = frozenset({3}), node_rank: int = 0) -> Process:
    return Process(
        node="node1",
        gpu_indices=gpus,
        sys_port=7500 + engine_id,
        http_port=6100,
        endpoint_mode="agg",
        endpoint_index=0,
        node_rank=node_rank,
        kv_events_port=5200 + engine_id,
        nixl_port=5400 + engine_id,
        engine_id=engine_id,
    )


def test_worker_command_loads_through_gms_and_drops_device_ids(tmp_path: Path) -> None:
    config = _load()
    process = _process()
    with patch(HOST_IP, return_value="10.0.0.11"):
        cmd = config.backend.build_worker_command(
            process=process, endpoint_processes=[process, _process(1)], runtime=_runtime(tmp_path)
        )
    text = shlex.join(cmd)
    assert "--load-format gms --gms-shadow-mode" in text
    assert "--device-ids" not in text
    assert "--master-port" not in text  # single node: no torch.distributed rendezvous to stagger


def test_multi_node_engines_stagger_master_port(tmp_path: Path) -> None:
    data = _data()
    data["roles"]["agg"] = {"nodes": 2, "workers": 1, "gpus": 8, "args": {"tensor-parallel-size": 16}}
    config = _load_data(data)
    leader0 = Process("node1", frozenset(range(8)), 7500, 6100, "agg", 0, 0, engine_id=0)
    leader1 = Process("node1", frozenset(range(8)), 7501, 6132, "agg", 0, 0, engine_id=1)
    follower1 = Process("node2", frozenset(range(8)), 7503, 0, "agg", 0, 1, engine_id=1)
    endpoint = [leader0, leader1, Process("node2", frozenset(range(8)), 7502, 0, "agg", 0, 1), follower1]
    with patch(HOST_IP, return_value="10.0.0.11"):
        cmd0 = shlex.join(config.backend.build_worker_command(leader0, endpoint, _runtime(tmp_path)))
        cmd1 = shlex.join(config.backend.build_worker_command(follower1, endpoint, _runtime(tmp_path)))
    assert f"--master-port {VLLM_MASTER_PORT_BASE}" in cmd0
    assert f"--master-port {VLLM_MASTER_PORT_BASE + VLLM_MASTER_PORT_STRIDE}" in cmd1
    assert "--headless" in cmd1 and "--headless" not in cmd0


def test_recipe_load_format_gms_is_not_duplicated(tmp_path: Path) -> None:
    data = _data()
    data["roles"]["agg"]["args"]["load-format"] = "gms"
    config = _load_data(data)
    process = _process()
    with patch(HOST_IP, return_value="10.0.0.11"):
        cmd = config.backend.build_worker_command(process, [process], _runtime(tmp_path))
    assert cmd.count("--load-format") == 1


def test_failover_environment_names_the_worker_directory() -> None:
    config = _load()
    env0 = config.backend.get_failover_environment(_process(0), "15600")
    env1 = config.backend.get_failover_environment(_process(1), "15600")
    worker_dir = "/dev/shm/srtctl-15600/agg_0"
    assert failover_worker_dir("/dev/shm", "15600", _process()) == worker_dir
    assert env0 == {
        "ENGINE_ID": "0",
        "GMS_SOCKET_DIR": worker_dir,
        "FAILOVER_LOCK_PATH": f"{worker_dir}/{FAILOVER_LOCK_FILENAME}",
        "DYN_VLLM_GMS_SHADOW_MODE": "true",
        "DYN_SYSTEM_STARTING_HEALTH_STATUS": "notready",
    }
    assert env1["ENGINE_ID"] == "1"
    # Both engines of a worker share the lock; a different worker gets its own.
    assert env1["FAILOVER_LOCK_PATH"] == env0["FAILOVER_LOCK_PATH"]
    other = Process("node1", frozenset({4}), 7502, 6164, "agg", 1, 0)
    assert config.backend.get_failover_environment(other, "15600")["GMS_SOCKET_DIR"] == "/dev/shm/srtctl-15600/agg_1"


def test_gms_sidecar_script_starts_one_server_per_gpu_and_reports_ready() -> None:
    cmd = build_gms_sidecar_command("/dev/shm/srtctl-15600/agg_0", device_count=2, startup_timeout_seconds=90)
    assert cmd[:2] == ["bash", "-c"]
    script = cmd[2]
    assert "python3 -m gpu_memory_service --device" in script
    assert "seq 0 1" in script  # devices 0 and 1 as the worker sees them
    assert "-ge 4" in script  # 2 sockets per device
    assert "seq 1 90" in script
    assert GMS_READY_MARKER in script
    assert 'export GMS_SOCKET_DIR="$dir"' in script
    assert "trap stop TERM INT" in script


def test_respawn_wrapper_relaunches_and_stops_on_term() -> None:
    cmd = build_respawn_command(backoff_seconds=7, label="agg_0 engine 1")
    assert cmd[:2] == ["bash", "-c"]
    script = cmd[2]
    # The engine argv is not in the loop's command line: `pkill -f dynamo.vllm` must not hit the loop.
    assert "dynamo.vllm" not in script
    assert f'eval "${RESPAWN_COMMAND_ENV} &"' in script
    assert "relaunching in 7s" in script
    assert "trap on_term TERM INT" in script
    assert "while :; do" in script


# --- launch -----------------------------------------------------------------------


def _orchestrator(config: SrtConfig, tmp_path: Path) -> SweepOrchestrator:
    return SweepOrchestrator(config=config, runtime=_runtime(tmp_path))


def test_worker_stage_launches_sidecar_then_engines_per_worker(tmp_path: Path) -> None:
    orchestrator = _orchestrator(_load(), tmp_path)
    with (
        patch(SRUN, return_value=_proc()) as srun,
        patch(WAIT, return_value=True) as wait,
        patch("srtctl.cli.mixins.worker_stage.get_hostname_ip", return_value="10.0.0.11"),
        patch(HOST_IP, return_value="10.0.0.11"),
    ):
        procs = orchestrator.start_all_workers()

    names = list(procs)
    assert names == [
        "gms_agg_0_node1",
        "agg_0_node1",
        "agg_0_node1_e1",
        "gms_agg_1_node1",
        "agg_1_node1",
        "agg_1_node1_e1",
    ]
    assert wait.call_count == 2
    calls = {call.kwargs["step_name"]: call.kwargs for call in srun.call_args_list}

    gms = calls["gms_agg_0_node1"]
    assert gms["env_to_set"] == {"CUDA_VISIBLE_DEVICES": "0"}
    assert gms["command"][:2] == ["bash", "-c"] and "gpu_memory_service --device" in gms["command"][2]
    assert gms["container_image"] == "/vllm-runtime.sqsh"
    assert gms["output"] == str(tmp_path / "node1_agg_w0_gms.out")

    e0, e1 = calls["agg_0_node1"], calls["agg_0_node1_e1"]
    for step in (e0, e1):
        assert step["env_to_set"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert step["env_to_set"]["GMS_SOCKET_DIR"] == "/dev/shm/srtctl-15600/agg_0"
        assert step["env_to_set"]["FAILOVER_LOCK_PATH"] == "/dev/shm/srtctl-15600/agg_0/failover.lock"
        assert step["env_to_set"]["DYN_VLLM_GMS_SHADOW_MODE"] == "true"
        assert "mkdir -p /dev/shm/srtctl-15600/agg_0" in step["bash_preamble"]
        # restart: always wraps the engine in the relaunch loop; the engine argv rides in the env
        assert step["command"][:2] == ["bash", "-c"]
        assert "relaunching in 5s" in step["command"][2]
        assert "dynamo.vllm" not in step["command"][2]
        assert step["env_to_set"][RESPAWN_COMMAND_ENV].startswith("python3 -m dynamo.vllm ")
        assert "--load-format gms --gms-shadow-mode" in step["env_to_set"][RESPAWN_COMMAND_ENV]
    assert e0["env_to_set"]["ENGINE_ID"] == "0" and e1["env_to_set"]["ENGINE_ID"] == "1"
    assert e0["env_to_set"]["DYN_SYSTEM_PORT"] != e1["env_to_set"]["DYN_SYSTEM_PORT"]
    assert e0["env_to_set"]["VLLM_NIXL_SIDE_CHANNEL_PORT"] != e1["env_to_set"]["VLLM_NIXL_SIDE_CHANNEL_PORT"]
    assert e1["output"] == str(tmp_path / "node1_agg_w0_e1.out")
    # the second worker pins its own GPU and its own directory
    assert calls["gms_agg_1_node1"]["env_to_set"] == {"CUDA_VISIBLE_DEVICES": "1"}
    assert calls["agg_1_node1"]["env_to_set"]["GMS_SOCKET_DIR"] == "/dev/shm/srtctl-15600/agg_1"

    sidecar, engine = procs["gms_agg_0_node1"], procs["agg_0_node1"]
    assert sidecar.critical is True and sidecar.shutdown_tier == 1 and sidecar.step_name == "gms_agg_0_node1"
    assert engine.shutdown_tier == 0 and engine.step_name == "agg_0_node1"
    assert procs["agg_0_node1_e1"].step_name == "agg_0_node1_e1"


def test_restart_never_launches_the_bare_engine(tmp_path: Path) -> None:
    orchestrator = _orchestrator(_load(engine={"type": "vllm", "failover": {"restart": "never"}}), tmp_path)
    with (
        patch(SRUN, return_value=_proc()) as srun,
        patch(WAIT, return_value=True),
        patch("srtctl.cli.mixins.worker_stage.get_hostname_ip", return_value="10.0.0.11"),
        patch(HOST_IP, return_value="10.0.0.11"),
    ):
        orchestrator.start_all_workers()
    engine = next(call.kwargs for call in srun.call_args_list if call.kwargs["step_name"] == "agg_0_node1")
    assert engine["command"][:3] == ["python3", "-m", "dynamo.vllm"]


def test_sidecar_that_dies_or_never_reports_fails_the_stage(tmp_path: Path) -> None:
    orchestrator = _orchestrator(_load(), tmp_path)
    with (
        patch(SRUN, return_value=_proc()),
        patch(WAIT, side_effect=ProcessDied()),
        patch(HOST_IP, return_value="10.0.0.11"),
        pytest.raises(RuntimeError, match="exited .* before its sockets came up"),
    ):
        orchestrator.start_all_workers()
    proc = _proc()
    with (
        patch(SRUN, return_value=proc),
        patch(WAIT, return_value=False),
        patch(HOST_IP, return_value="10.0.0.11"),
        pytest.raises(RuntimeError, match="did not report ready"),
    ):
        orchestrator.start_all_workers()
    proc.terminate.assert_called()


def test_full_node_worker_pins_nothing(tmp_path: Path) -> None:
    data = _data()
    data["roles"]["agg"] = {"nodes": 1, "workers": 1, "gpus": 8, "args": {"tensor-parallel-size": 8}}
    orchestrator = _orchestrator(_load_data(data), tmp_path)
    with (
        patch(SRUN, return_value=_proc()) as srun,
        patch(WAIT, return_value=True),
        patch("srtctl.cli.mixins.worker_stage.get_hostname_ip", return_value="10.0.0.11"),
        patch(HOST_IP, return_value="10.0.0.11"),
    ):
        orchestrator.start_all_workers()
    calls = {call.kwargs["step_name"]: call.kwargs for call in srun.call_args_list}
    assert calls["gms_agg_0_node1"]["env_to_set"] == {}
    assert "CUDA_VISIBLE_DEVICES" not in calls["agg_0_node1"]["env_to_set"]
    assert "seq 0 7" in calls["gms_agg_0_node1"]["command"][2]


def test_mock_sweep_runs_the_failover_recipe_end_to_end(tmp_path: Path) -> None:
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        yaml.dump(
            _data(
                model={"path": "hf:fake/model", "container": "nvcr.io/fake:latest"},
                benchmark={"type": "custom", "command": "echo failover"},
            )
        )
    )
    output_dir = tmp_path / "outputs" / "15600"
    exit_code = run_mock_sweep(
        config_path=config_path,
        output_dir=output_dir,
        job_id="15600",
        options=MockOptions(child_duration_s=0.2, phase_pause_s=0.05),
    )
    assert exit_code == 0
    logs = output_dir / "logs"
    # Per worker: the sidecar's log, engine 0's log, and the shadow's log, all through the fake srun.
    assert (logs / "mock-node-01_agg_w0_gms.out").is_file()
    assert (logs / "mock-node-01_agg_w0.out").is_file()
    assert (logs / "mock-node-01_agg_w0_e1.out").is_file()
    assert (logs / "mock-node-01_agg_w1_e1.out").is_file()
    assert (output_dir / "recipe.lock.yaml").is_file()


# --- dry-run --------------------------------------------------------------------------


def test_dry_run_shows_the_layout(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.dump(_data()))
    show_config_details(SrtConfig.from_yaml(config_path))
    out = capsys.readouterr().out
    assert "Shadow Engine Recovery" in out
    assert "engines per worker: 2" in out
    assert "/dev/shm/srtctl-<job_id>/<role>_<index>/" in out
    assert "--load-format gms --gms-shadow-mode" in out
    assert "always (relaunch in place after 5s)" in out


def test_dry_run_is_silent_without_failover(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.dump(_data(engine="vllm")))
    show_config_details(SrtConfig.from_yaml(config_path))
    assert "Shadow Engine Recovery" not in capsys.readouterr().out
