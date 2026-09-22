# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shadow engine recovery (``engine.failover``): schema, the implied gms service, topology, launch, dry-run.

The acceptance recipe is two TP1 vLLM workers on one node with one shadow each.
Per worker that is a gms service instance (``before_workers``, pinned to the
worker's GPU) and two engine steps on the same GPU, all told the same socket
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

from srtctl.backends.vllm import FAILOVER_LOCK_FILENAME, VLLMFailoverConfig, VLLMProtocol, failover_worker_dir
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.submit import show_config_details
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Endpoint, NodePortAllocator, Process, endpoints_to_processes
from srtctl.mock import MockOptions, run_mock_sweep
from srtctl.ports import VLLM_MASTER_PORT_BASE, VLLM_MASTER_PORT_STRIDE
from srtctl.services import ServiceConfig, ServicePlacementConfig
from srtctl.services.gms import GMS_READY_MARKER, PREVIEW_COMMAND, GMSService, build_gms_sidecar_command
from srtctl.services.implicit import effective_services
from srtctl.services.registry import ServiceLaunchContext

SRUN_WORKER = "srtctl.cli.mixins.worker_stage.start_srun_process"
HOST_IP_WORKER = "srtctl.cli.mixins.worker_stage.get_hostname_ip"
SRUN_SERVICE = "srtctl.cli.mixins.service_stage.start_srun_process"
WAIT_SERVICE = "srtctl.cli.mixins.service_stage.wait_until_ready"
HOST_IP_SERVICE = "srtctl.cli.mixins.service_stage.get_hostname_ip"
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
        container_mounts={tmp_path: Path("/logs")},
        environment={},
    )


def _proc() -> MagicMock:
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    return proc


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


# --- schema -------------------------------------------------------------------


def test_defaults_and_engines_per_worker() -> None:
    config = _load()
    assert isinstance(config.backend, VLLMProtocol)
    failover = config.backend.failover
    assert failover == VLLMFailoverConfig()
    assert failover.shadow_engines == 1
    assert failover.shared_dir == "/dev/shm"
    assert failover.engines_per_worker == 2
    assert config.backend.engines_per_process == 2


def test_without_failover_nothing_changes() -> None:
    config = _load(engine="vllm")
    assert config.backend.failover is None
    assert config.backend.engines_per_process == 1
    assert all(entry.service.type != "gms" for entry in effective_services(config))


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"shadow_engines": 0}, "shadow_engines must be at least 1"),
        ({"shared_dir": "shm"}, "shared_dir must be an absolute directory"),
        ({"shared_dir": "/"}, "shared_dir must be an absolute directory"),
        ({"restart": "always"}, "Unknown field"),  # relaunch is roles.<role>.restart, not a failover knob
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


# --- the implied gms service -----------------------------------------------------


def test_failover_implies_a_gms_service_per_worker() -> None:
    config = _load()
    entry = next(entry for entry in effective_services(config) if entry.service.type == "gms")
    assert entry.implicit and entry.reason == "engine.failover"
    service = entry.service
    assert service.name == "gms"
    assert service.effective_placement == "workers"
    assert service.effective_per == "worker"
    assert service.effective_start == "before_workers"
    assert service.effective_critical is True
    assert service.preview_command() == PREVIEW_COMMAND


def test_declared_gms_takes_over_and_keeps_the_rules() -> None:
    config = _load(
        services=[
            {"name": "gms", "type": "gms", "container": "/other.sqsh", "placement": {"node": "agg", "per": "worker"}}
        ]
    )
    entry = next(entry for entry in effective_services(config) if entry.service.type == "gms")
    assert not entry.implicit
    assert entry.service.container == "/other.sqsh"
    with pytest.raises(ValidationError, match="requires engine.failover"):
        _load(engine="vllm", services=[{"name": "gms", "type": "gms"}])
    with pytest.raises(ValidationError, match="set placement.per: worker"):
        _load(services=[{"name": "gms", "type": "gms", "placement": {"node": "agg"}}])


def test_gms_script_starts_one_server_per_gpu_and_reports_ready() -> None:
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


def test_gms_kind_sizes_the_script_for_the_attached_worker(tmp_path: Path) -> None:
    config = _load()
    kind = GMSService()
    service = ServiceConfig(name="gms", type="gms", placement=ServicePlacementConfig(node="workers", per="worker"))
    process = _process(gpus=frozenset({4, 5}))
    ctx = ServiceLaunchContext(
        runtime=_runtime(tmp_path),
        node="node1",
        node_ip="ip",
        node_id=0,
        index=0,
        role="workers",
        process=process,
        config=config,
    )
    script = kind.build_command(service, ctx)[2]
    assert "seq 0 1" in script and "-ge 4" in script and "seq 1 120" in script
    assert "dir=/dev/shm/srtctl-15600/agg_0" in script
    assert kind.forced_environment(service, ctx) == {"GMS_SOCKET_DIR": "/dev/shm/srtctl-15600/agg_0"}
    probe = kind.readiness(service, ctx)
    assert probe is not None and probe.log is not None and "GMS" in probe.log.pattern
    # Without a worker (dry-run) there is nothing to size: a placeholder argv and no env.
    assert kind.build_command(service, ServiceLaunchContext.preview()) == PREVIEW_COMMAND
    assert kind.forced_environment(service, ServiceLaunchContext.preview()) == {}


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


# --- engine command and environment -------------------------------------------------


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


# --- launch: the gms service, then the engines -------------------------------------------


def _orchestrator(config: SrtConfig, tmp_path: Path) -> SweepOrchestrator:
    return SweepOrchestrator(config=config, runtime=_runtime(tmp_path))


def test_gms_service_launches_one_pinned_instance_per_worker(tmp_path: Path) -> None:
    orchestrator = _orchestrator(_load(), tmp_path)
    with (
        patch(SRUN_SERVICE, return_value=_proc()) as srun,
        patch(WAIT_SERVICE, return_value=True) as wait,
        patch(HOST_IP_SERVICE, return_value="10.0.0.11"),
    ):
        procs = orchestrator.start_services("before_workers")

    assert [p.name for p in procs] == ["service_gms_agg_0_node1", "service_gms_agg_1_node1"]
    calls = {call.kwargs["step_name"]: call.kwargs for call in srun.call_args_list}
    first = calls["service_gms_agg_0_node1"]
    assert first["env_to_set"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert first["env_to_set"]["GMS_SOCKET_DIR"] == "/dev/shm/srtctl-15600/agg_0"
    assert first["command"][:2] == ["bash", "-c"] and "gpu_memory_service --device" in first["command"][2]
    assert "seq 0 0" in first["command"][2]  # one GPU
    assert first["container_image"] == "/vllm-runtime.sqsh"
    assert first["output"] == str(tmp_path / "service_gms_agg_0_node1.out")
    second = calls["service_gms_agg_1_node1"]
    assert second["env_to_set"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert second["env_to_set"]["GMS_SOCKET_DIR"] == "/dev/shm/srtctl-15600/agg_1"
    # Gated on the ready line of each instance; critical; stopped after the engines.
    assert wait.call_count == 2
    assert "GMS" in wait.call_args.args[0].pattern
    assert all(p.critical and p.shutdown_tier == 1 for p in procs)


def test_worker_stage_launches_only_engines(tmp_path: Path) -> None:
    orchestrator = _orchestrator(_load(), tmp_path)
    with (
        patch(SRUN_WORKER, return_value=_proc()) as srun,
        patch(HOST_IP_WORKER, return_value="10.0.0.11"),
        patch(HOST_IP, return_value="10.0.0.11"),
    ):
        procs = orchestrator.start_all_workers()

    assert list(procs) == ["agg_0_node1", "agg_0_node1_e1", "agg_1_node1", "agg_1_node1_e1"]
    calls = {call.kwargs["step_name"]: call.kwargs for call in srun.call_args_list}
    e0, e1 = calls["agg_0_node1"], calls["agg_0_node1_e1"]
    for step in (e0, e1):
        assert step["env_to_set"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert step["env_to_set"]["GMS_SOCKET_DIR"] == "/dev/shm/srtctl-15600/agg_0"
        assert step["env_to_set"]["FAILOVER_LOCK_PATH"] == "/dev/shm/srtctl-15600/agg_0/failover.lock"
        assert step["env_to_set"]["DYN_VLLM_GMS_SHADOW_MODE"] == "true"
        assert "mkdir -p /dev/shm/srtctl-15600/agg_0" in step["bash_preamble"]
        # The engine is the step's task; relaunch is roles.<role>.restart's job, not a wrapper's.
        assert step["command"][:3] == ["python3", "-m", "dynamo.vllm"]
        assert "--load-format gms --gms-shadow-mode" in shlex.join(step["command"])
    assert e0["env_to_set"]["ENGINE_ID"] == "0" and e1["env_to_set"]["ENGINE_ID"] == "1"
    assert e0["env_to_set"]["DYN_SYSTEM_PORT"] != e1["env_to_set"]["DYN_SYSTEM_PORT"]
    assert e0["env_to_set"]["VLLM_NIXL_SIDE_CHANNEL_PORT"] != e1["env_to_set"]["VLLM_NIXL_SIDE_CHANNEL_PORT"]
    assert e1["output"] == str(tmp_path / "node1_agg_w0_e1.out")
    assert e0["command"][e0["command"].index("--dump-config-to") + 1] == "/logs/node1_config.json"
    assert e1["command"][e1["command"].index("--dump-config-to") + 1] == "/logs/node1_config_e1.json"
    assert calls["agg_1_node1"]["env_to_set"]["GMS_SOCKET_DIR"] == "/dev/shm/srtctl-15600/agg_1"
    assert procs["agg_0_node1_e1"].step_name == "agg_0_node1_e1"
    assert procs["agg_0_node1"].shutdown_tier == 0


def test_full_node_worker_pins_nothing(tmp_path: Path) -> None:
    data = _data()
    data["roles"]["agg"] = {"nodes": 1, "workers": 1, "gpus": 8, "args": {"tensor-parallel-size": 8}}
    orchestrator = _orchestrator(_load_data(data), tmp_path)
    with (
        patch(SRUN_SERVICE, return_value=_proc()) as srun_service,
        patch(WAIT_SERVICE, return_value=True),
        patch(HOST_IP_SERVICE, return_value="10.0.0.11"),
    ):
        (gms,) = orchestrator.start_services("before_workers")
    assert gms.name == "service_gms_agg_0_node1"
    assert "CUDA_VISIBLE_DEVICES" not in srun_service.call_args.kwargs["env_to_set"]
    assert "seq 0 7" in srun_service.call_args.kwargs["command"][2]
    with (
        patch(SRUN_WORKER, return_value=_proc()) as srun_worker,
        patch(HOST_IP_WORKER, return_value="10.0.0.11"),
        patch(HOST_IP, return_value="10.0.0.11"),
    ):
        orchestrator.start_all_workers()
    assert "CUDA_VISIBLE_DEVICES" not in srun_worker.call_args.kwargs["env_to_set"]


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
    # Per worker: the gms service instance, engine 0, and the shadow, all through the fake srun.
    assert (logs / "service_gms_agg_0_mock-node-01.out").is_file()
    assert (logs / "service_gms_agg_1_mock-node-01.out").is_file()
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
    assert "type=gms" in out and "per=worker" in out
    assert "implied by: engine.failover" in out
    assert "gpu_memory_service" in out


def test_dry_run_is_silent_without_failover(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.dump(_data(engine="vllm")))
    show_config_details(SrtConfig.from_yaml(config_path))
    out = capsys.readouterr().out
    assert "Shadow Engine Recovery" not in out
    assert "type=gms" not in out
