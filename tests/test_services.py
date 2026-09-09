# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the top-level ``services:`` block: schema, kinds, and the launch stage."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.readiness import ProcessDied
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Endpoint
from srtctl.ports import MOONCAKE_HTTP_METADATA_PORT, MOONCAKE_MASTER_PORT
from srtctl.services import ServiceConfig, ServiceSourceConfig, list_service_types
from srtctl.services.implicit import discovery_env, effective_services, uses_discovery_plane

SRUN = "srtctl.cli.mixins.service_stage.start_srun_process"
WAIT = "srtctl.cli.mixins.service_stage.wait_until_ready"
HOST_IP = "srtctl.cli.mixins.service_stage.get_hostname_ip"

DISAGG_HEAD = """
name: services-test
model:
  path: /model
  container: /job.sqsh
  precision: bf16
resources:
  gpu_type: b200
  gpus_per_node: 8
  prefill_nodes: 1
  decode_nodes: 2
  prefill_workers: 1
  decode_workers: 2
  gpus_per_prefill: 8
  gpus_per_decode: 8
benchmark:
  type: manual
observability:
  tachometer:
    enabled: false
"""

# Same cluster, tachometer left at its default (on): the exporters are implied.
TACHOMETER_HEAD = DISAGG_HEAD.replace("observability:\n  tachometer:\n    enabled: false\n", "")
assert TACHOMETER_HEAD != DISAGG_HEAD


def _load(services_yaml: str, head: str = DISAGG_HEAD, backend: str = "backend:\n  type: sglang\n") -> SrtConfig:
    return SrtConfig.Schema().load(yaml.safe_load(head + backend + services_yaml))


def _runtime(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(
        job_id="12345",
        run_name="test-run",
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node1", "node2", "node3")),
        head_node_ip="10.0.0.10",
        infra_node_ip="10.0.0.10",
        log_dir=tmp_path,
        model_path=Path("/model"),
        container_image=Path("/job.sqsh"),
        gpus_per_node=8,
        network_interface="eth0",
        container_mounts={},
        environment={},
    )


def _proc(returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.wait.return_value = returncode
    proc.returncode = returncode
    proc.poll.return_value = None
    return proc


# --- schema -------------------------------------------------------------------


def test_registered_kinds() -> None:
    assert list_service_types() == [
        "dcgm-exporter",
        "etcd",
        "generic",
        "mooncake-master",
        "mooncake-store",
        "nats",
        "node-exporter",
    ]


def test_generic_service_loads_block_yaml_with_defaults() -> None:
    config = _load(
        """
services:
  - name: files
    command:
      - python3
      - -m
      - http.server
    args:
      - "9911"
    readiness:
      port: 9911
      timeout_seconds: 30
"""
    )
    (svc,) = config.services
    assert svc.type == "generic"
    assert svc.effective_command == ["python3", "-m", "http.server", "9911"]
    assert svc.effective_placement == "head"
    assert svc.effective_start == "after_frontend"
    assert svc.effective_critical is False
    assert svc.inherit_discovery_env is True
    assert svc.readiness is not None and svc.readiness.port == 9911


def test_generic_requires_command() -> None:
    with pytest.raises(ValidationError, match="command is required"):
        _load("services:\n  - name: nothing\n")


def test_unknown_type_rejected() -> None:
    with pytest.raises(ValidationError, match="not a known service type"):
        _load("services:\n  - name: x\n    type: sidecar\n    command: [/bin/true]\n")


def test_duplicate_names_rejected() -> None:
    with pytest.raises(ValidationError, match="must be unique"):
        _load("services:\n  - name: a\n    command: [/bin/true]\n  - name: a\n    command: [/bin/true]\n")


def test_invalid_placement_and_start_rejected() -> None:
    with pytest.raises(ValidationError, match="placement.node must be one of"):
        _load("services:\n  - name: a\n    command: [/bin/true]\n    placement:\n      node: everywhere\n")
    with pytest.raises(ValidationError, match="start must be one of"):
        _load("services:\n  - name: a\n    command: [/bin/true]\n    start: eventually\n")


def test_source_rules() -> None:
    with pytest.raises(ValidationError, match="immutable ref"):
        ServiceSourceConfig(git="https://example.com/r", rev="main")
    with pytest.raises(ValidationError, match="single-node placement"):
        _load(
            """
services:
  - name: router
    command: [python3, -m, router]
    placement:
      node: workers
    source:
      git: https://example.com/r
      rev: refs/pull/1/head
"""
        )


def test_mooncake_store_defaults_and_requires_master() -> None:
    with pytest.raises(ValidationError, match="requires backend.mooncake_kv_store"):
        _load("services:\n  - name: store\n    type: mooncake-store\n    placement:\n      node: workers\n")

    config = _load(
        """
services:
  - name: store
    type: mooncake-store
    placement:
      node: workers
""",
        backend="backend:\n  type: sglang\n  mooncake_kv_store:\n    container: /mooncake.sqsh\n"
        "  sglang_config:\n    prefill:\n      disaggregation-transfer-backend: mooncake\n"
        "    decode:\n      disaggregation-transfer-backend: mooncake\n",
    )
    (svc,) = config.services
    assert svc.effective_command == ["python", "-m", "mooncake.mooncake_store_service"]
    assert svc.effective_start == "before_workers"
    assert svc.effective_critical is True


# --- stage ---------------------------------------------------------------------


def _orchestrator(config: SrtConfig, tmp_path: Path) -> SweepOrchestrator:
    return SweepOrchestrator(config=config, runtime=_runtime(tmp_path))


def test_no_matching_services_is_a_noop(tmp_path: Path) -> None:
    orchestrator = _orchestrator(_load("services:\n  - name: a\n    command: [/bin/true]\n"), tmp_path)
    with patch(SRUN) as srun:
        assert orchestrator.start_services("before_workers") == []
    srun.assert_not_called()


def test_generic_launches_on_head_with_discovery_env(tmp_path: Path) -> None:
    config = _load(
        """
services:
  - name: router
    command: [python3, -m, router, --node, "{node}", --infra, "{infra_ip}"]
    env:
      LOG_LEVEL: debug
"""
    )
    orchestrator = _orchestrator(config, tmp_path)
    with patch(SRUN, return_value=_proc()) as srun, patch(HOST_IP, return_value="10.0.0.10"):
        procs = orchestrator.start_services("after_frontend")

    srun.assert_called_once()
    kw = srun.call_args.kwargs
    assert kw["nodelist"] == ["node0"]
    assert kw["command"] == ["python3", "-m", "router", "--node", "node0", "--infra", "10.0.0.10"]
    assert kw["container_image"] == "/job.sqsh"
    assert kw["env_to_set"]["ETCD_ENDPOINTS"] == "http://node0:2379"
    assert kw["env_to_set"]["NATS_SERVER"] == "nats://node0:4222"
    assert kw["env_to_set"]["LOG_LEVEL"] == "debug"
    assert kw["bash_preamble"] is None
    (proc,) = procs
    assert proc.name == "service_router"
    assert proc.node == "node0"
    assert proc.critical is False
    assert proc.log_file == tmp_path / "service_router.out"


def test_container_alias_and_no_discovery_env(tmp_path: Path) -> None:
    config = _load(
        "services:\n  - name: s\n    command: [/bin/true]\n    container: /mine.sqsh\n    inherit_discovery_env: false\n"
        "    critical: true\n"
    )
    with patch(SRUN, return_value=_proc()) as srun, patch(HOST_IP, return_value="10.0.0.10"):
        (proc,) = _orchestrator(config, tmp_path).start_services("after_frontend")
    assert srun.call_args.kwargs["container_image"] == "/mine.sqsh"
    assert "ETCD_ENDPOINTS" not in srun.call_args.kwargs["env_to_set"]
    assert proc.critical is True


def test_declared_order_is_launch_order(tmp_path: Path) -> None:
    config = _load(
        "services:\n  - name: b\n    command: [echo, b]\n  - name: a\n    command: [echo, a]\n"
        "  - name: c\n    command: [echo, c]\n"
    )
    with patch(SRUN, return_value=_proc()) as srun, patch(HOST_IP, return_value="10.0.0.10"):
        _orchestrator(config, tmp_path).start_services("after_frontend")
    assert [call.kwargs["command"][1] for call in srun.call_args_list] == ["b", "a", "c"]


def test_readiness_gate_blocks_and_failure_terminates_started(tmp_path: Path) -> None:
    config = _load(
        "services:\n  - name: a\n    command: [/bin/true]\n    readiness:\n      port: 9000\n      timeout_seconds: 5\n"
    )
    orchestrator = _orchestrator(config, tmp_path)
    with (
        patch(SRUN, return_value=_proc()),
        patch(HOST_IP, return_value="10.0.0.10"),
        patch(WAIT, return_value=True) as wait,
    ):
        orchestrator.start_services("after_frontend")
    wait.assert_called_once()
    assert wait.call_args.kwargs["host"] == "node0"
    assert wait.call_args.kwargs["timeout"] == 5
    assert wait.call_args.kwargs["log_file"] == tmp_path / "service_a.out"
    assert wait.call_args.args[0].port == 9000

    popen = _proc()
    registry = MagicMock()
    with (
        patch(SRUN, return_value=popen),
        patch(HOST_IP, return_value="10.0.0.10"),
        patch(WAIT, return_value=False),
        pytest.raises(RuntimeError, match="was not ready within 5s"),
    ):
        orchestrator.start_services("after_frontend", registry)
    popen.terminate.assert_called_once()
    # Registered before the readiness wait, so a signal during the wait still finds it.
    (registered,) = [call.args[0] for call in registry.add_process.call_args_list]
    assert registered.popen is popen


def test_readiness_fails_fast_when_the_process_dies(tmp_path: Path) -> None:
    config = _load(
        "services:\n  - name: a\n    command: [/bin/true]\n    readiness:\n      port: 9000\n      timeout_seconds: 600\n"
    )
    dead = _proc()
    dead.poll.return_value = 127
    with (
        patch(SRUN, return_value=dead),
        patch(HOST_IP, return_value="10.0.0.10"),
        patch(WAIT, side_effect=ProcessDied("gone")),
        pytest.raises(RuntimeError, match="exited with code 127 .* before its readiness probe passed"),
    ):
        _orchestrator(config, tmp_path).start_services("after_frontend")


def test_signal_during_readiness_wait_terminates_started(tmp_path: Path) -> None:
    # The SIGTERM handler raises SystemExit inside whatever the orchestrator is doing;
    # the stage must still tear down what it launched.
    config = _load("services:\n  - name: a\n    command: [/bin/true]\n    readiness:\n      port: 9000\n")
    popen = _proc()
    with (
        patch(SRUN, return_value=popen),
        patch(HOST_IP, return_value="10.0.0.10"),
        patch(WAIT, side_effect=SystemExit(1)),
        pytest.raises(SystemExit),
    ):
        _orchestrator(config, tmp_path).start_services("after_frontend")
    popen.terminate.assert_called_once()


def test_source_is_cloned_on_bare_host_and_built_in_container(tmp_path: Path) -> None:
    config = _load(
        """
services:
  - name: router
    command: [python3, -m, router]
    source:
      git: https://example.com/repo
      rev: refs/pull/1/head
      path: lib/router
    build_command: [bash, -lc, "pip install -e ."]
"""
    )
    with patch(SRUN, return_value=_proc()) as srun, patch(HOST_IP, return_value="10.0.0.10"):
        _orchestrator(config, tmp_path).start_services("after_frontend")

    clone, build, launch = srun.call_args_list
    assert clone.kwargs["container_image"] is None
    assert "git -c http.version=HTTP/1.1 clone" in clone.kwargs["command"][-1]
    assert "refs/pull/1/head" in clone.kwargs["command"][-1]
    assert build.kwargs["container_image"] == "/job.sqsh"
    assert build.kwargs["command"] == ["bash", "-lc", "pip install -e ."]
    # Build and launch run inside the container, where log_dir is mounted at /logs.
    assert "/logs/services/router/src/lib/router" in build.kwargs["bash_preamble"]
    assert str(tmp_path) not in build.kwargs["bash_preamble"]
    assert launch.kwargs["command"] == ["python3", "-m", "router"]
    assert "/logs/services/router/src/lib/router" in launch.kwargs["bash_preamble"]


def test_clone_and_build_failures_raise(tmp_path: Path) -> None:
    config = _load(
        """
services:
  - name: router
    command: [/bin/true]
    source:
      git: https://example.com/repo
      rev: abc123
    build_command: [/bin/false]
"""
    )
    orchestrator = _orchestrator(config, tmp_path)
    with patch(SRUN, return_value=_proc(1)), pytest.raises(RuntimeError, match="source clone failed"):
        orchestrator.start_services("after_frontend")
    with (
        patch(SRUN, side_effect=[_proc(0), _proc(2)]),
        pytest.raises(RuntimeError, match="build_command failed"),
    ):
        orchestrator.start_services("after_frontend")


def test_clone_and_build_steps_are_registered_and_bounded(tmp_path: Path) -> None:
    config = _load(
        """
services:
  - name: router
    command: [/bin/true]
    source:
      git: https://example.com/repo
      rev: abc123
    build_command: [make]
    build_timeout_seconds: 7
"""
    )
    registry = MagicMock()
    hung = _proc()
    hung.wait.side_effect = subprocess.TimeoutExpired(cmd="make", timeout=7)
    hung.poll.return_value = None
    with (
        patch(SRUN, side_effect=[_proc(0), hung]),
        patch("srtctl.cli.mixins.service_stage.terminate_and_reap") as reap,
        pytest.raises(RuntimeError, match="build_command timed out after 7s"),
    ):
        _orchestrator(config, tmp_path).start_services("after_frontend", registry)

    hung.wait.assert_called_once_with(timeout=7)
    reap.assert_called_once_with(hung)
    names = [call.args[0].name for call in registry.add_process.call_args_list]
    assert names == ["service_router.clone", "service_router.build"]
    assert all(not call.args[0].critical for call in registry.add_process.call_args_list)


MOONCAKE_BACKEND = """backend:
  type: sglang
  mooncake_kv_store:
    container: /mooncake-master.sqsh
  sglang_config:
    prefill:
      disaggregation-transfer-backend: mooncake
    decode:
      disaggregation-transfer-backend: mooncake
"""

STORES = """
services:
  - name: store-prefill
    type: mooncake-store
    placement:
      node: prefill
    args: [--port, "8800", --label, "{role}-{node_id}"]
    env:
      MOONCAKE_PROTOCOL: rdma
      MOONCAKE_MASTER: ignored:9999
      MOONCAKE_EXTRA_CONFIG: '{"prefetch_timeout_base": 4}'
      MOONCAKE_GLOBAL_SEGMENT_SIZE: 100gb
    preamble: |
      ulimit -n 1048576
      echo starting-{role}-on-{node}
    cpus_per_task: 8
    cpu_bind: none
    srun_options:
      exclusive: ""
    readiness:
      port: 8800
      timeout_seconds: 90
  - name: store-decode
    type: mooncake-store
    placement:
      node: decode
    container: /mooncake-store.sqsh
    args: [--port, "8800"]
    env:
      MOONCAKE_GLOBAL_SEGMENT_SIZE: 400gb
    readiness:
      port: 8800
"""


def test_mooncake_stores_launch_once_per_role_node_with_master_env(tmp_path: Path) -> None:
    orchestrator = _orchestrator(_load(STORES, backend=MOONCAKE_BACKEND), tmp_path)
    ips = {"node0": "10.0.0.10", "node1": "10.0.0.11", "node2": "10.0.0.12", "node3": "10.0.0.13"}
    with (
        patch(SRUN, side_effect=lambda **_: _proc()) as srun,
        patch(HOST_IP, side_effect=lambda node, _iface: ips[node]),
        patch(WAIT, return_value=True) as wait,
    ):
        procs = orchestrator.start_services("before_workers")

    # The implied Mooncake master on the infra node first, then 1 prefill node +
    # 2 decode nodes of stores; no store launches for the head.
    assert [p.node for p in procs] == ["node0", "node1", "node2", "node3"]
    assert [p.name for p in procs] == [
        "service_mooncake-master",
        "service_store-prefill",
        "service_store-decode_node2",
        "service_store-decode_node3",
    ]
    assert all(p.critical for p in procs)
    assert all(p.step_name == p.name for p in procs)
    # Master: three default ports gated in turn; stores: one declared probe each.
    assert wait.call_count == 3 + 3

    master = srun.call_args_list[0].kwargs
    assert master["container_image"] == "/mooncake-master.sqsh"
    assert master["command"][0] == "mooncake_master"
    assert master["step_name"] == "service_mooncake-master"

    prefill = srun.call_args_list[1].kwargs
    assert prefill["container_image"] == "/mooncake-master.sqsh"  # falls back to mooncake_kv_store.container
    assert prefill["command"] == [
        "python",
        "-m",
        "mooncake.mooncake_store_service",
        "--port",
        "8800",
        "--label",
        "prefill-0",
    ]
    env = prefill["env_to_set"]
    assert env["MOONCAKE_LOCAL_HOSTNAME"] == "10.0.0.11"
    assert env["MOONCAKE_GLOBAL_SEGMENT_SIZE"] == "100gb"
    assert env["MOONCAKE_EXTRA_CONFIG"] == '{"prefetch_timeout_base": 4}'
    assert env["MOONCAKE_MASTER"] == f"10.0.0.10:{MOONCAKE_MASTER_PORT}"  # srtctl always wins
    assert env["MOONCAKE_TE_META_DATA_SERVER"] == f"http://10.0.0.10:{MOONCAKE_HTTP_METADATA_PORT}/metadata"
    assert prefill["bash_preamble"] == "ulimit -n 1048576\necho starting-prefill-on-node1"
    assert prefill["cpus_per_task"] == 8
    assert prefill["cpu_bind"] == "none"
    assert prefill["srun_options"] == {"exclusive": ""}

    decode = srun.call_args_list[2].kwargs
    assert decode["container_image"] == "/mooncake-store.sqsh"
    assert decode["env_to_set"]["MOONCAKE_GLOBAL_SEGMENT_SIZE"] == "400gb"
    assert decode["env_to_set"]["MOONCAKE_LOCAL_HOSTNAME"] == "10.0.0.12"


def test_colocated_roles_with_same_port_rejected_before_launch(tmp_path: Path) -> None:
    orchestrator = _orchestrator(_load(STORES, backend=MOONCAKE_BACKEND), tmp_path)
    orchestrator.__dict__["endpoints"] = [
        Endpoint(mode="prefill", index=0, nodes=("node1",)),
        Endpoint(mode="decode", index=0, nodes=("node1",)),
    ]
    with patch(SRUN) as srun, pytest.raises(ValueError, match="both listen on port 8800 on node node1"):
        orchestrator.start_services("before_workers")
    srun.assert_not_called()


def test_workers_placement_deduplicates_shared_nodes(tmp_path: Path) -> None:
    config = _load(
        "services:\n  - name: store\n    type: mooncake-store\n    placement:\n      node: workers\n"
        "    readiness:\n      port: 8800\n",
        backend=MOONCAKE_BACKEND,
    )
    orchestrator = _orchestrator(config, tmp_path)
    with (
        patch(SRUN, side_effect=lambda **_: _proc()) as srun,
        patch(HOST_IP, return_value="10.0.0.11"),
        patch(WAIT, return_value=True),
    ):
        procs = orchestrator.start_services("before_workers")
    # Implied Mooncake master on node0, then one store per worker node.
    assert srun.call_count == 4
    assert {p.node for p in procs} == {"node0", "node1", "node2", "node3"}


def test_service_config_direct_construction() -> None:
    svc = ServiceConfig(name="x", command=["true"], start="before_workers")
    assert svc.effective_start == "before_workers"
    with pytest.raises(ValidationError, match="must not contain empty arguments"):
        ServiceConfig(name="x", command=["python", ""])


# --- implicit services ------------------------------------------------------------


def _from_yaml(tmp_path: Path, text: str) -> SrtConfig:
    """Through the real loader (normalizers included), unlike ``_load``."""
    path = tmp_path / "recipe.yaml"
    path.write_text(text)
    return SrtConfig.from_yaml(path)


def _names(config: SrtConfig) -> list[tuple[str, bool]]:
    return [(entry.service.name, entry.implicit) for entry in effective_services(config)]


def test_dynamo_frontend_implies_etcd_and_nats_on_the_infra_node(tmp_path: Path) -> None:
    config = _load("")  # frontend defaults to dynamo
    assert _names(config) == [("etcd", True), ("nats", True)]
    assert uses_discovery_plane(config)

    orchestrator = _orchestrator(config, tmp_path)
    with (
        patch(SRUN, return_value=_proc()) as srun,
        patch(HOST_IP, return_value="10.0.0.10"),
        patch(WAIT, return_value=True) as wait,
    ):
        procs = orchestrator.start_services("infra")

    assert [p.name for p in procs] == ["service_etcd", "service_nats"]
    assert [p.node for p in procs] == ["node0", "node0"]
    assert all(p.critical for p in procs)
    assert [p.step_name for p in procs] == ["service_etcd", "service_nats"]
    etcd, nats = (call.kwargs for call in srun.call_args_list)
    assert etcd["command"] == [
        "/configs/etcd",
        "--data-dir",
        "/tmp/etcd",
        "--listen-client-urls",
        "http://0.0.0.0:2379",
        "--advertise-client-urls",
        "http://10.0.0.10:2379",
    ]
    assert etcd["bash_preamble"] == "rm -rf /tmp/etcd && mkdir -p /tmp/etcd"
    assert etcd["container_image"] == "/job.sqsh"
    assert "ETCD_ENDPOINTS" not in etcd["env_to_set"]  # the plane does not point at itself
    assert nats["command"] == ["/configs/nats-server", "-js", "-sd", "/tmp/nats"]
    # One tcp probe per kind default port, against the node the service runs on.
    assert [call.args[0].port for call in wait.call_args_list] == [2379, 4222]
    assert all(call.kwargs["host"] == "node0" for call in wait.call_args_list)
    assert all(call.kwargs["timeout"] == 300 for call in wait.call_args_list)


def test_static_frontend_implies_no_discovery_plane(tmp_path: Path) -> None:
    config = _load("frontend:\n  type: sglang\n")
    assert _names(config) == []
    assert not uses_discovery_plane(config)
    with patch(SRUN) as srun:
        assert _orchestrator(config, tmp_path).start_services("infra") == []
    srun.assert_not_called()


def test_declared_etcd_takes_over_and_external_is_not_launched(tmp_path: Path) -> None:
    config = _load("services:\n  - name: etcd\n    type: etcd\n    external: http://etcd.shared:2379\n")
    assert _names(config) == [("etcd", False), ("nats", True)]
    orchestrator = _orchestrator(config, tmp_path)
    with (
        patch(SRUN, return_value=_proc()) as srun,
        patch(HOST_IP, return_value="10.0.0.10"),
        patch(WAIT, return_value=True),
    ):
        procs = orchestrator.start_services("infra")
    assert [p.name for p in procs] == ["service_nats"]
    assert srun.call_count == 1
    assert discovery_env(config, orchestrator.runtime) == {
        "ETCD_ENDPOINTS": "http://etcd.shared:2379",
        "NATS_SERVER": "nats://node0:4222",
    }


def test_enabled_false_drops_an_implicit_service() -> None:
    config = _load("services:\n  - name: nats\n    type: nats\n    enabled: false\n")
    assert _names(config) == [("etcd", True)]


def test_nats_max_payload_renders_a_config_file(tmp_path: Path) -> None:
    config = _load("infra:\n  nats_max_payload_mb: 24\n")
    orchestrator = _orchestrator(config, tmp_path)
    with (
        patch(SRUN, return_value=_proc()) as srun,
        patch(HOST_IP, return_value="10.0.0.10"),
        patch(WAIT, return_value=True),
    ):
        orchestrator.start_services("infra")
    nats = srun.call_args_list[1].kwargs
    assert nats["command"] == ["/configs/nats-server", "-c", "/tmp/nats.conf"]
    assert f"max_payload: {24 * 1024 * 1024}" in nats["bash_preamble"]
    assert "> /tmp/nats.conf" in nats["bash_preamble"]


def test_declared_nats_options_flow_back_into_infra(tmp_path: Path) -> None:
    config = _from_yaml(
        tmp_path,
        DISAGG_HEAD
        + "backend:\n  type: sglang\n"
        + "services:\n  - name: etcd\n    type: etcd\n    placement:\n      node: dedicated\n"
        + "  - name: nats\n    type: nats\n    placement:\n      node: dedicated\n    options:\n"
        + "      max_payload_mb: 24\n",
    )
    assert config.infra.etcd_nats_dedicated_node is True
    assert config.infra.nats_max_payload_mb == 24
    assert [entry.service.effective_placement for entry in effective_services(config)] == ["dedicated", "dedicated"]


def test_declared_etcd_dedicated_must_agree_with_nats(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="dedicated"):
        _from_yaml(
            tmp_path,
            DISAGG_HEAD
            + "backend:\n  type: sglang\n"
            + "services:\n  - name: etcd\n    type: etcd\n    placement:\n      node: dedicated\n"
            + "  - name: nats\n    type: nats\n    placement:\n      node: infra\n",
        )


def test_declared_mooncake_master_maps_onto_the_backend(tmp_path: Path) -> None:
    config = _from_yaml(
        tmp_path,
        DISAGG_HEAD
        + """backend:
  type: sglang
  sglang_config:
    prefill:
      disaggregation-transfer-backend: mooncake
    decode:
      disaggregation-transfer-backend: mooncake
services:
  - name: mooncake-master
    type: mooncake-master
    container: /mm.sqsh
    args: [--nof_eviction_high_watermark_ratio=0.9]
""",
    )
    assert config.backend.mooncake_kv_store is not None
    assert config.backend.mooncake_kv_store.container == "/mm.sqsh"
    assert list(config.backend.mooncake_kv_store.master_extra_args) == ["--nof_eviction_high_watermark_ratio=0.9"]
    assert ("mooncake-master", False) in _names(config)


def test_tachometer_exporters_are_implied_on_every_worker_node(tmp_path: Path) -> None:
    config = _load("frontend:\n  type: sglang\n", head=TACHOMETER_HEAD)
    assert _names(config) == [("dcgm-exporter", True), ("node-exporter", True)]
    orchestrator = _orchestrator(config, tmp_path)
    with patch(SRUN, return_value=_proc()) as srun, patch(HOST_IP, return_value="10.0.0.11"):
        procs = orchestrator.start_services("after_frontend")

    assert [p.name for p in procs] == [
        "service_dcgm-exporter_node1",
        "service_dcgm-exporter_node2",
        "service_dcgm-exporter_node3",
        "service_node-exporter_node1",
        "service_node-exporter_node2",
        "service_node-exporter_node3",
    ]
    # A dead exporter costs its metrics, never the run.
    assert not any(p.critical for p in procs)
    dcgm = srun.call_args_list[0].kwargs
    assert dcgm["container_image"] == "nvcr.io#nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04"
    assert dcgm["command"] == ["dcgm-exporter", "--collect-interval=1000", "--address", ":9401"]
    # Distroless image: no bash wrapper, env rides on srun --export instead.
    assert dcgm["use_bash_wrapper"] is False
    assert dcgm["bash_preamble"] is None
    assert dcgm["env_to_set"] is None
    assert "ETCD_ENDPOINTS" in dcgm["srun_export_env"]
    node = srun.call_args_list[3].kwargs
    assert node["container_image"] == "quay.io#prometheus/node-exporter:v1.8.2"
    assert node["command"][:2] == ["/bin/node_exporter", "--web.listen-address=:9101"]


def test_declared_exporter_overrides_the_container(tmp_path: Path) -> None:
    config = _load(
        "frontend:\n  type: sglang\nservices:\n  - name: dcgm-exporter\n    type: dcgm-exporter\n"
        "    container: /mirror/dcgm.sqsh\n  - name: node-exporter\n    type: node-exporter\n    enabled: false\n",
        head=TACHOMETER_HEAD,
    )
    assert _names(config) == [("dcgm-exporter", False)]
    with patch(SRUN, return_value=_proc()) as srun, patch(HOST_IP, return_value="10.0.0.11"):
        _orchestrator(config, tmp_path).start_services("after_frontend")
    assert srun.call_count == 3
    assert srun.call_args.kwargs["container_image"] == "/mirror/dcgm.sqsh"


def test_power_telemetry_owns_the_dcgm_exporter() -> None:
    config = _load(
        "frontend:\n  type: sglang\nbenchmark:\n  type: sa-bench\n  concurrencies: [4]\ntelemetry:\n  enabled: true\n"
        "  dcgm_exporter:\n    container_image: dcgm-exporter\n    port: 9401\n",
        head=TACHOMETER_HEAD.replace("benchmark:\n  type: manual\n", ""),
    )
    assert _names(config) == [("node-exporter", True)]
