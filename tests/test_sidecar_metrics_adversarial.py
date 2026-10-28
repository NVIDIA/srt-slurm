# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edge cases of the Dynamo sidecar metrics contract (docs/sidecars.md#metrics).

``tests/test_sidecar_metrics.py`` covers one shape per engine. These tests drive
whole recipes through the real ``SweepOrchestrator`` (``run_mock_sweep`` with
``run`` replaced by a probe) across the layouts the allocator has to keep apart,
and check invariants rather than literal port numbers:

* every Tachometer worker target is a port the allocator handed that process
  (``sys_port`` or ``http_port``), unique per host, under a unique name;
* the built-in AIPerf URL list equals the Tachometer worker URLs (modulo the
  documented TRT-LLM ``publish_events_and_metrics`` gate), and the custom list keeps
  the docs/benchmarks.md contract: one live URL per logical leader, aligned with
  ``SRT_*_ENDPOINTS``, preferring the native engine's endpoint;
* ``runs_dynamo_sidecar`` is true exactly for the processes whose built command
  launches a sidecar, and the engine target's port is the ``--port`` the engine
  binds, on every interface, with ``--enable-metrics`` where SGLang needs it;
* no two listeners a process binds share a port on the same node.
"""

from __future__ import annotations

import ipaddress
import os
import re
import shlex
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from srtctl.backends import SGLangBackend, TRTLLMBackend, VLLMBackend
from srtctl.core.schema import RoleConfig
from srtctl.core.telemetry import sidecar_worker_metrics_targets
from srtctl.core.topology import Endpoint, Process
from srtctl.frontends import get_frontend
from srtctl.ports import VLLM_PORT_STRIDE
from tests.launch_snapshots import EXAMPLES_DIR, _isolated_cluster, example_recipes

DYNAMO = get_frontend("dynamo")
NODES = tuple(f"node-{i:02d}" for i in range(1, 41))


# --- Harness ----------------------------------------------------------------------------------


@dataclass
class Probe:
    """What a sidecar job advertises, captured from the real orchestrator under the mock."""

    processes: list[Process]
    backend: Any
    tachometer: str
    aiperf_builtin: list[str]
    aiperf_custom: list[str]
    logical_endpoints: list[tuple[str, str, int]]
    commands: dict[tuple[str, int, int], list[str]] = field(default_factory=dict)

    def tachometer_workers(self) -> list[dict[str, Any]]:
        """Every ``backend_*`` endpoint block of the Tachometer config, as a dict."""
        blocks = []
        for block in self.tachometer.split("[[endpoints]]")[1:]:
            name = re.search(r'^name = "(.*)"$', block, re.MULTILINE)
            url = re.search(r'^url = "(.*)"$', block, re.MULTILINE)
            assert name and url
            if not name[1].startswith("backend_"):
                continue
            metadata = dict(
                re.findall(r'^"([^"]+)" = "(.*)"$', block.split("[endpoints.node_metadata]")[-1], re.MULTILINE)
            )
            blocks.append({"name": name[1], "url": url[1], "metadata": metadata})
        return blocks


def _node_ip(node: str, *_args: Any, ipv6: bool = False) -> str:
    index = int(node.rsplit("-", 1)[1])
    return f"fd00::{index}" if ipv6 else f"10.0.0.{index}"


def _recipe(engine: Any, roles: dict[str, Any], *, gpus_per_node: int = 8, **extra: Any) -> dict[str, Any]:
    recipe: dict[str, Any] = {
        "schema": 2,
        "name": "sidecar-metrics-adversarial",
        "slurm": {"time_limit": "00:30:00"},
        "model": {"path": "/models/m", "container": "/containers/c.sqsh", "precision": "bf16"},
        "resources": {"gpu_type": "h100", "gpus_per_node": gpus_per_node},
        "dynamo": {"install": False},
        "frontend": {"type": "dynamo", "enable_multiple_frontends": False},
        "engine": engine,
        "roles": {name: {"sidecar": True, **role} for name, role in roles.items()},
        "benchmark": {"type": "sa-bench", "isl": 128, "osl": 64, "concurrencies": "4"},
    }
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(recipe.get(key), dict):
            recipe[key] = {**recipe[key], **value}
        else:
            recipe[key] = value
    return recipe


def with_orchestrator(recipe_path: Path, nodes: tuple[str, ...], inspect: Any) -> Any:
    """Run ``recipe_path`` up to the orchestrator under the mock and return ``inspect(orchestrator)``."""
    from srtctl.cli.do_sweep import SweepOrchestrator
    from srtctl.mock import MockOptions, run_mock_sweep

    captured: list[Any] = []

    def capture(orchestrator: SweepOrchestrator) -> int:
        captured.append(inspect(orchestrator))
        return 0

    with tempfile.TemporaryDirectory(prefix="srt-sidecar-adv-") as tmp:
        tmp_path = Path(tmp).resolve()
        (tmp_path / "model").mkdir()
        (tmp_path / "c.sqsh").touch()
        prior_cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            with (
                _isolated_cluster(tmp_path / "model", tmp_path / "c.sqsh"),
                patch.object(SweepOrchestrator, "run", capture),
            ):
                exit_code = run_mock_sweep(
                    config_path=recipe_path,
                    output_dir=tmp_path / "output",
                    job_id="4242",
                    options=MockOptions(child_duration_s=0.0, phase_pause_s=0.0, nodelist=nodes),
                )
        finally:
            os.chdir(prior_cwd)
    assert exit_code == 0 and captured, f"{recipe_path} did not reach the orchestrator"
    return captured[0]


def probe(recipe: dict[str, Any], *, ipv6: bool = False, build_commands: bool = True) -> Probe:
    """Load ``recipe`` into the real orchestrator under the mock and record what it advertises."""

    def inspect(orchestrator: Any) -> Probe:
        def ip(node: str, *args: Any, **kwargs: Any) -> str:
            return _node_ip(node, ipv6=ipv6)

        with (
            patch("srtctl.core.telemetry.get_hostname_ip", ip),
            patch("srtctl.cli.mixins.benchmark_stage.get_hostname_ip", ip),
        ):
            builtin = orchestrator._get_aiperf_server_metrics_env().get("AIPERF_SERVER_METRICS_URLS", "")
            custom = orchestrator._get_aiperf_server_metrics_env(logical_workers_only=True).get(
                "AIPERF_SERVER_METRICS_URLS", ""
            )
            result = Probe(
                processes=list(orchestrator.backend_processes),
                backend=orchestrator.config.backend,
                tachometer=orchestrator.tachometer_config_text(),
                aiperf_builtin=[url for url in builtin.split(",") if url],
                aiperf_custom=[url for url in custom.split(",") if url],
                logical_endpoints=orchestrator._logical_worker_endpoints(),
            )
        if build_commands:
            backend = orchestrator.config.backend
            launch_per_endpoint = backend.get_srun_config().launch_per_endpoint
            for group in orchestrator.worker_endpoint_groups().values():
                for process in group[:1] if launch_per_endpoint else group:
                    result.commands[(process.endpoint_mode, process.endpoint_index, process.node_rank)] = (
                        backend.build_worker_command(
                            process=process,
                            endpoint_processes=group,
                            runtime=orchestrator.runtime,
                            frontend_type=orchestrator.config.frontend.type,
                        )
                    )
        return result

    # Exactly the nodes the job reserves, as SLURM grants them: extra nodes would let a
    # `nodes: colocate` decode spill onto a node of its own.
    node_count = sum(role["nodes"] for role in recipe["roles"].values() if isinstance(role["nodes"], int))
    with tempfile.TemporaryDirectory(prefix="srt-sidecar-recipe-") as tmp:
        recipe_path = Path(tmp) / "recipe.yaml"
        recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
        return with_orchestrator(recipe_path, NODES[:node_count], inspect)


def _endpoint_groups(processes: list[Process]) -> dict[tuple[str, int], list[Process]]:
    groups: dict[tuple[str, int], list[Process]] = {}
    for process in processes:
        groups.setdefault((process.endpoint_mode, process.endpoint_index), []).append(process)
    return groups


def _engine_argv(command: list[str], marker: str) -> list[str]:
    """The engine's argv: the line of the supervising bash script that starts it, or the command itself."""
    if command[:2] != ["bash", "-lc"]:
        return command
    line = next(line for line in command[2].splitlines() if marker in line and line.rstrip().endswith("&"))
    return shlex.split(line.removesuffix(" &"))


def _flag(argv: list[str], flag: str) -> str | None:
    """The value of the last occurrence of ``flag`` (argparse and clap both keep the last one)."""
    values = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == flag]
    return values[-1] if values else None


_SIDECAR_MARKERS = ("dynamo.vllm.sidecar", "dynamo.sglang.sidecar", "dynamo.trtllm.sidecar", "/opt/bin/my-sidecar")
_ENGINE_MARKERS = {
    VLLMBackend: ("vllm-rs serve", "vllm.entrypoints.cli.main serve"),
    SGLangBackend: ("sglang.launch_server",),
    TRTLLMBackend: ("tensorrt_llm.commands.serve",),
}


def _launches_sidecar(command: list[str]) -> bool:
    script = command[2] if command[:2] == ["bash", "-lc"] else shlex.join(command)
    return any(f"{marker} " in script for marker in _SIDECAR_MARKERS)


def _engine_line(backend: Any, command: list[str]) -> list[str]:
    markers = next(markers for kind, markers in _ENGINE_MARKERS.items() if isinstance(backend, kind))
    return _engine_argv(command, next(m for m in markers if m in (command[2] if len(command) > 2 else "")))


def _listeners(processes: list[Process]) -> list[tuple[str, int, str]]:
    """``(node, port, what)`` for every port a process binds on its own node.

    Endpoint-wide ports (bootstrap, dist-init, DP RPC, TRT-LLM dist-init) are bound
    once, by the leader; ranges (vLLM's private scan block, the KVBM pair) count
    every port in them.
    """
    listeners: list[tuple[str, int, str]] = []
    for process in processes:
        node = process.node
        listeners.append((node, process.sys_port, "sys"))
        if process.http_port > 0:
            listeners.append((node, process.http_port, "http"))
        for name in ("sidecar_grpc_port", "grpc_http_port", "nccl_port", "kv_events_port"):
            if (port := getattr(process, name)) is not None:
                listeners.append((node, port, name))
        if process.kvbm_zmq_port is not None:
            listeners += [(node, process.kvbm_zmq_port, "kvbm"), (node, process.kvbm_zmq_port + 1, "kvbm_ack")]
        if process.vllm_scan_port is not None:
            listeners += [(node, process.vllm_scan_port + i, "vllm_scan") for i in range(VLLM_PORT_STRIDE)]
        if process.is_leader:
            for name in ("bootstrap_port", "dist_init_port", "dp_rpc_port", "trtllm_dist_init_port"):
                if (port := getattr(process, name)) is not None:
                    listeners.append((node, port, name))
    return listeners


# --- Shapes -----------------------------------------------------------------------------------

_VLLM_ARGS = {"served-model-name": "m", "max-model-len": 4096}
_SGLANG_ARGS = {"served-model-name": "m", "context-length": 2048}

SHAPES: dict[str, dict[str, Any]] = {
    # Two TP1 workers share one node: every per-node listener must differ.
    "vllm-two-workers-one-node": _recipe(
        "vllm", {"agg": {"nodes": 1, "workers": 2, "gpus": 1, "args": {**_VLLM_ARGS, "tensor-parallel-size": 1}}}
    ),
    # Decode packs onto the prefill node (`nodes: colocate`; vLLM packs only with its own opt-in).
    "vllm-colocated-pd": _recipe(
        {"type": "vllm", "allow_prefill_decode_colocation": True},
        {
            "prefill": {"nodes": 1, "workers": 1, "gpus": 2, "args": {**_VLLM_ARGS, "tensor-parallel-size": 2}},
            "decode": {"nodes": "colocate", "workers": 1, "gpus": 2, "args": {**_VLLM_ARGS, "tensor-parallel-size": 2}},
        },
    ),
    # Hybrid DP: one Rust frontend + sidecar per node, named by the node's first DP rank.
    "vllm-hybrid-dp": _recipe(
        "vllm",
        {
            "decode": {
                "nodes": 2,
                "workers": 1,
                "gpus": 8,
                "args": {**_VLLM_ARGS, "data-parallel-size": 8, "enable-expert-parallel": True},
            },
            "prefill": {"nodes": 1, "workers": 1, "gpus": 4, "args": {**_VLLM_ARGS, "tensor-parallel-size": 4}},
        },
        gpus_per_node=4,
    ),
    # One TP8 replica across two nodes: the follower is headless and serves nothing.
    "vllm-cross-node-replica": _recipe(
        "vllm",
        {"agg": {"nodes": 2, "workers": 1, "gpus": 8, "args": {**_VLLM_ARGS, "tensor-parallel-size": 8}}},
        gpus_per_node=4,
    ),
    "sglang-two-workers-one-node": _recipe(
        "sglang", {"agg": {"nodes": 1, "workers": 2, "gpus": 1, "args": {**_SGLANG_ARGS, "tensor-parallel-size": 1}}}
    ),
    "sglang-colocated-pd": _recipe(
        "sglang",
        {
            "prefill": {"nodes": 1, "workers": 1, "gpus": 2, "args": {**_SGLANG_ARGS, "tensor-parallel-size": 2}},
            "decode": {
                "nodes": "colocate",
                "workers": 1,
                "gpus": 2,
                "args": {**_SGLANG_ARGS, "tensor-parallel-size": 2},
            },
        },
    ),
    # TP8 / attention-DP 2 over four 2-GPU nodes with KV events: nodes 0 and 2 publish.
    "sglang-dp-attention-kv-relay": _recipe(
        "sglang",
        {
            "agg": {
                "nodes": 4,
                "workers": 1,
                "gpus": 8,
                "kv_events": True,
                "args": {**_SGLANG_ARGS, "tp-size": 8, "dp-size": 2, "enable-dp-attention": True},
            }
        },
        gpus_per_node=2,
    ),
    # Two multi-node TP workers per role; prefill relays KV events, decode does not.
    "sglang-disagg-multinode": _recipe(
        "sglang",
        {
            "prefill": {
                "nodes": 4,
                "workers": 2,
                "gpus": 4,
                "kv_events": True,
                "args": {**_SGLANG_ARGS, "tensor-parallel-size": 4, "dp-size": 2, "enable-dp-attention": True},
            },
            "decode": {"nodes": 2, "workers": 1, "gpus": 4, "args": {**_SGLANG_ARGS, "tensor-parallel-size": 4}},
        },
        gpus_per_node=2,
    ),
    "trtllm-multinode": _recipe(
        {"type": "trtllm", "publish_events_and_metrics": True},
        {
            "prefill": {"nodes": 1, "workers": 2, "gpus": 2, "args": {"tensor_parallel_size": 2}},
            "decode": {"nodes": 2, "workers": 1, "gpus": 8, "args": {"tensor_parallel_size": 8}},
        },
        gpus_per_node=4,
    ),
}


@pytest.fixture(scope="module", params=sorted(SHAPES))
def shape(request) -> tuple[str, Probe]:
    return request.param, probe(SHAPES[request.param])


# --- Invariants over every shape ----------------------------------------------------------------


@pytest.mark.parametrize("name", [name for name in sorted(SHAPES) if "one-node" in name or "colocated" in name])
def test_single_node_shapes_really_share_a_node(name: str) -> None:
    """Guard the collision tests below: these shapes must put every worker on one node."""
    assert len({process.node for process in probe(SHAPES[name], build_commands=False).processes}) == 1


def test_no_two_listeners_share_a_port_on_a_node(shape) -> None:
    _, result = shape
    listeners = _listeners(result.processes)
    taken: dict[tuple[str, int], list[str]] = {}
    for node, port, what in listeners:
        taken.setdefault((node, port), []).append(what)
    clashes = {key: whats for key, whats in taken.items() if len(whats) > 1}
    assert not clashes


def test_every_worker_target_is_an_allocated_port_unique_per_host(shape) -> None:
    _, result = shape
    targets = sidecar_worker_metrics_targets(result.processes, result.backend, DYNAMO)
    assert targets
    for target in targets:
        allocated = target.process.http_port if target.engine else target.process.sys_port
        assert target.port == allocated > 0, target.endpoint_name
    assert len({(t.process.node, t.port) for t in targets}) == len(targets)
    assert len({t.endpoint_name for t in targets}) == len(targets)

    rendered = result.tachometer_workers()
    assert [block["name"] for block in rendered] == [t.endpoint_name for t in targets]
    for block, target in zip(rendered, targets, strict=True):
        assert block["url"] == f"http://{_node_ip(target.process.node)}:{target.port}{target.path}"
        assert block["metadata"] == {
            "hostname": target.process.node,
            "worker_index": str(target.process.endpoint_index),
            "worker_process": str(target.process.node_rank),
            "worker_role": target.process.endpoint_mode,
        }


def test_aiperf_runners_advertise_live_tachometer_worker_urls(shape) -> None:
    _, result = shape
    blocks = {block["url"]: block for block in result.tachometer_workers()}
    assert len(blocks) == len(result.tachometer_workers())
    # Built-in runners get the whole Tachometer list, sorted.
    assert result.aiperf_builtin == sorted(blocks)
    # Custom runners keep one URL per logical leader, position by position with
    # SRT_*_ENDPOINTS (consumers drain on it), and every URL is a live scrape target.
    assert len(result.aiperf_custom) == len(result.logical_endpoints) > 0
    engine_workers = {
        (b["metadata"]["worker_role"], b["metadata"]["worker_index"], b["metadata"]["worker_process"])
        for b in blocks.values()
        if b["name"].endswith("_engine")
    }
    for url, (mode, host, _port) in zip(result.aiperf_custom, result.logical_endpoints, strict=True):
        block = blocks[url]
        metadata = block["metadata"]
        assert metadata["worker_role"] == mode and _node_ip(metadata["hostname"]) == host
        worker = (metadata["worker_role"], metadata["worker_index"], metadata["worker_process"])
        # The native engine's endpoint wins wherever the leader's engine serves one.
        assert block["name"].endswith("_engine") or worker not in engine_workers


def test_the_process_that_runs_a_sidecar_is_the_one_whose_command_starts_it(shape) -> None:
    name, result = shape
    backend = result.backend
    groups = _endpoint_groups(result.processes)
    assert result.commands
    for (mode, index, node_rank), command in result.commands.items():
        group = groups[(mode, index)]
        process = next(p for p in group if p.node_rank == node_rank)
        assert _launches_sidecar(command) is backend.runs_dynamo_sidecar(process, group), (name, process.node)
        if isinstance(backend, TRTLLMBackend):
            # One sidecar per endpoint, gated to global MPI rank zero.
            assert 'if [[ "${SLURM_PROCID:-0}" != "0" ]]' in command[2]
            assert not any(backend.runs_dynamo_sidecar(p, group) for p in group[1:])
            continue
        engine = _engine_line(backend, command)
        port = backend.sidecar_engine_metrics_port(process, group)
        if port is None:
            # Only a headless vLLM follower serves no engine metrics in these shapes.
            assert isinstance(backend, VLLMBackend) and "--headless" in engine
            assert _flag(engine, "--port") is None
            continue
        assert _flag(engine, "--port") == str(port)
        assert _flag(engine, "--host") == "0.0.0.0"
        if isinstance(backend, SGLangBackend):
            assert engine.count("--enable-metrics") == 1


def test_trtllm_publication_gate_hides_urls_from_builtin_aiperf_only() -> None:
    recipe = _recipe(
        {"type": "trtllm", "publish_events_and_metrics": False},
        {"agg": {"nodes": 2, "workers": 1, "gpus": 8, "args": {"tensor_parallel_size": 8}}},
        gpus_per_node=4,
    )
    result = probe(recipe, build_commands=False)

    workers = result.tachometer_workers()
    assert [block["name"] for block in workers] == ["backend_agg0_rank0"]
    assert result.aiperf_builtin == []
    # Custom runners are not gated (docs/benchmarks.md): drain detection keeps a live URL.
    assert result.aiperf_custom == [workers[0]["url"]]


# --- Rendering: IPv6, metadata, ordering -------------------------------------------------------


def test_ipv6_hosts_are_bracketed_in_every_sidecar_url() -> None:
    recipe = _recipe(
        "sglang",
        {"agg": {"nodes": 2, "workers": 1, "gpus": 4, "args": {**_SGLANG_ARGS, "tensor-parallel-size": 4}}},
        gpus_per_node=2,
        observability={"tachometer": {"extra_metadata": {"cluster": "lab", "hostname": "spoofed"}}},
    )
    result = probe(recipe, ipv6=True, build_commands=False)

    urls = [block["url"] for block in result.tachometer_workers()]
    assert {block["name"] for block in result.tachometer_workers()} == {
        "backend_agg0_rank0",
        "backend_agg0_rank0_engine",
        "backend_agg0_rank1_engine",
    }
    for url in [*urls, *result.aiperf_builtin, *result.aiperf_custom]:
        host = re.fullmatch(r"http://\[([0-9a-f:]+)\]:\d+/metrics", url)
        assert host, url
        ipaddress.IPv6Address(host[1])
    # extra_metadata reaches both endpoint kinds the same way (and, as for every target, wins over a key it repeats).
    for block in result.tachometer_workers():
        assert block["metadata"]["cluster"] == "lab"
        assert block["metadata"]["hostname"] == "spoofed"


def test_targets_order_by_worker_index_numerically_not_lexically() -> None:
    backend = SGLangBackend(roles={"agg": RoleConfig(args={"tensor-parallel-size": 1})})
    endpoints = [
        Endpoint(mode="agg", index=i, nodes=(f"n{i // 8}",), gpu_indices=frozenset({i % 8}), gpus_per_node=8)
        for i in range(12)
    ]
    processes = backend.endpoints_to_processes(endpoints, dynamo_sidecar=True)
    names = [t.endpoint_name for t in sidecar_worker_metrics_targets(list(reversed(processes)), backend, DYNAMO)]

    assert names[:4] == [
        "backend_agg0_rank0",
        "backend_agg0_rank0_engine",
        "backend_agg1_rank0",
        "backend_agg1_rank0_engine",
    ]
    assert names[-2:] == ["backend_agg11_rank0", "backend_agg11_rank0_engine"]
    # The input order does not matter.
    assert names == [t.endpoint_name for t in sidecar_worker_metrics_targets(processes, backend, DYNAMO)]


# --- User overrides ---------------------------------------------------------------------------


def _sglang_sidecar(args: dict[str, Any], *, mode: str = "agg", nodes: int = 2) -> tuple[SGLangBackend, list[Process]]:
    backend = SGLangBackend(roles={mode: RoleConfig(args={"tensor-parallel-size": 2 * nodes, **args})})
    endpoint = Endpoint(
        mode=mode,
        index=0,
        nodes=tuple(f"node{r}" for r in range(nodes)),
        gpu_indices=frozenset({0, 1}),
        gpus_per_node=2,
    )
    return backend, backend.endpoints_to_processes([endpoint], dynamo_sidecar=True)


def _sglang_runtime():
    from unittest.mock import MagicMock

    from srtctl.core.schema import DynamoConfig

    runtime = MagicMock()
    runtime.model_path = Path("/models/m")
    runtime.worker_model_arg = "/model"
    runtime.network_interface = None
    runtime.dynamo = DynamoConfig(sidecar=True)
    return runtime


@pytest.mark.parametrize(
    "args",
    [
        pytest.param({}, id="default"),
        pytest.param({"enable-metrics": True}, id="dashed-true"),
        pytest.param({"enable_metrics": True}, id="underscored-true"),
        pytest.param({"enable-metrics": False}, id="dashed-false"),
        pytest.param({"enable_metrics": False}, id="underscored-false"),
        pytest.param({"enable-metrics": None}, id="null"),
        pytest.param({"enable-metrics": False, "enable_metrics": True}, id="conflicting-aliases"),
    ],
)
def test_sglang_engine_target_exists_iff_the_engine_gets_the_metrics_switch(args: dict[str, Any]) -> None:
    backend, processes = _sglang_sidecar(args)
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        for process in processes:
            engine = _engine_argv(
                backend.build_worker_command(process, processes, _sglang_runtime()), "sglang.launch_server"
            )
            switched_on = "--enable-metrics" in engine
            assert (backend.sidecar_engine_metrics_port(process, processes) is not None) is switched_on
            assert engine.count("--enable-metrics") <= 1


def test_sglang_metrics_override_is_per_role() -> None:
    backend = SGLangBackend(
        roles={
            "prefill": RoleConfig(args={"tensor-parallel-size": 4, "enable-metrics": False}),
            "decode": RoleConfig(args={"tensor-parallel-size": 4}),
        }
    )
    endpoints = [
        Endpoint(mode=mode, index=0, nodes=nodes, gpu_indices=frozenset({0, 1}), gpus_per_node=2)
        for mode, nodes in (("prefill", ("p0", "p1")), ("decode", ("d0", "d1")))
    ]
    processes = backend.endpoints_to_processes(endpoints, dynamo_sidecar=True)

    names = [t.endpoint_name for t in sidecar_worker_metrics_targets(processes, backend, DYNAMO)]
    assert names == [
        "backend_decode0_rank0",
        "backend_decode0_rank0_engine",
        "backend_decode0_rank1_engine",
        "backend_prefill0_rank0",
    ]


@pytest.mark.parametrize("kv_events", [False, True])
def test_sglang_follower_runtime_target_follows_the_kv_relay(kv_events: bool) -> None:
    """A follower's system port is scraped only when it runs a relay sidecar; its engine always is."""
    backend = SGLangBackend(
        roles={"agg": RoleConfig(args={"tp-size": 4, "dp-size": 2, "enable-dp-attention": True}, kv_events=kv_events)}
    )
    endpoint = Endpoint(mode="agg", index=0, nodes=("n0", "n1"), gpu_indices=frozenset({0, 1}), gpus_per_node=2)
    processes = backend.endpoints_to_processes([endpoint], dynamo_sidecar=True)

    names = [t.endpoint_name for t in sidecar_worker_metrics_targets(processes, backend, DYNAMO)]
    relay = ["backend_agg0_rank1"] if kv_events else []
    assert names == ["backend_agg0_rank0", "backend_agg0_rank0_engine", *relay, "backend_agg0_rank1_engine"]
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        follower_command = backend.build_worker_command(processes[1], processes, _sglang_runtime())
    assert _launches_sidecar(follower_command) is kv_events


def test_vllm_disable_log_stats_keeps_the_engine_endpoint() -> None:
    """``--disable-log-stats`` turns vllm:* series off, not the HTTP server: /metrics still answers."""
    recipe = _recipe(
        "vllm",
        {"agg": {"nodes": 1, "workers": 1, "gpus": 1, "args": {**_VLLM_ARGS, "disable-log-stats": True}}},
    )
    result = probe(recipe)

    ((_, command),) = result.commands.items()
    engine = _engine_argv(command, "vllm-rs serve")
    assert "--disable-log-stats" in engine
    assert [block["name"] for block in result.tachometer_workers()] == [
        "backend_agg0_rank0",
        "backend_agg0_rank0_engine",
    ]


def test_sidecar_binary_does_not_change_targets() -> None:
    base = _recipe("vllm", {"agg": {"nodes": 1, "workers": 2, "gpus": 1, "args": _VLLM_ARGS}})
    custom = _recipe(
        "vllm",
        {"agg": {"nodes": 1, "workers": 2, "gpus": 1, "args": _VLLM_ARGS}},
        dynamo={"install": False, "sidecar_binary": "/opt/bin/my-sidecar"},
    )
    plain, overridden = probe(base), probe(custom)

    assert overridden.tachometer_workers() == plain.tachometer_workers()
    for command in overridden.commands.values():
        assert "/opt/bin/my-sidecar --grpc-endpoint" in command[2]


@pytest.mark.parametrize(
    "recipe",
    [
        "vllm/dynamo-sidecar-agg.yaml",
        "features/sglang-sidecar-multinode-dp.yaml",
        "features/sglang-sidecar-multinode-disagg-dp.yaml",
        "trtllm/dynamo-sidecar-disagg.yaml",
    ],
)
def test_custom_runner_urls_keep_the_pre_sidecar_metrics_contract(recipe: str) -> None:
    """Custom benchmarks drain on one URL per logical worker, positionally aligned with SRT_*_ENDPOINTS."""

    def inspect(orchestrator: Any) -> tuple[list[str], list[tuple[str, str, int]], list[Process], Any]:
        def ip(node: str, *args: Any, **kwargs: Any) -> str:
            return _node_ip(node)

        with patch("srtctl.cli.mixins.benchmark_stage.get_hostname_ip", ip):
            env = orchestrator._get_aiperf_server_metrics_env(logical_workers_only=True)
            logical = orchestrator._logical_worker_endpoints()
        leaders = [
            process
            for process in orchestrator.backend_processes
            if orchestrator.frontend.worker_endpoint_port(process, orchestrator.config, orchestrator.runtime)
        ]
        return env["AIPERF_SERVER_METRICS_URLS"].split(","), logical, leaders, orchestrator.config.backend

    custom, logical, leaders, backend = with_orchestrator(EXAMPLES_DIR / recipe, NODES, inspect)

    assert len(custom) == len(logical) == len(leaders) > 0
    if isinstance(backend, TRTLLMBackend):
        # The pre-feature URL named the endpoint port, where TRT-LLM's gRPC server serves nothing;
        # the same position now names the leader's live sidecar runtime endpoint.
        assert custom == [f"http://{_node_ip(p.node)}:{p.sys_port}/metrics" for p in leaders]
    else:
        # Exactly what 0324b429 advertised: http://<leader>:<SRT_*_ENDPOINTS port>/metrics.
        assert custom == [f"http://{host}:{port}/metrics" for _, host, port in logical]


def test_tachometer_disabled_still_advertises_sidecar_urls_to_aiperf() -> None:
    recipe = _recipe(
        "vllm",
        {"agg": {"nodes": 1, "workers": 1, "gpus": 1, "args": _VLLM_ARGS}},
        observability={"tachometer": {"enabled": False}},
    )
    result = probe(recipe, build_commands=False)

    assert len(result.aiperf_builtin) == 2
    assert result.aiperf_custom == [
        block["url"] for block in result.tachometer_workers() if block["name"] == "backend_agg0_rank0_engine"
    ]


@pytest.mark.parametrize("frontend_type", ["sglang", "vllm", "trtllm_serve"])
def test_sidecar_with_a_non_dynamo_frontend_is_rejected(tmp_path, frontend_type: str) -> None:
    from marshmallow import ValidationError

    from srtctl.core.schema import SrtConfig

    engine = {"sglang": "sglang", "vllm": "vllm", "trtllm_serve": "trtllm"}[frontend_type]
    recipe = _recipe(engine, {"agg": {"nodes": 1, "workers": 1, "gpus": 1}}, frontend={"type": frontend_type})
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))

    with pytest.raises((ValidationError, ValueError), match="sidecar"):
        SrtConfig.from_yaml(path)


@pytest.mark.parametrize("override", [{"port": 30000}, {"host": "127.0.0.1"}], ids=["port", "host"])
def test_sglang_recipe_host_or_port_cannot_move_the_scraped_listener(override: dict[str, Any]) -> None:
    backend, processes = _sglang_sidecar(override)
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        for process in processes:
            engine = _engine_argv(
                backend.build_worker_command(process, processes, _sglang_runtime()), "sglang.launch_server"
            )
            # sglang.launch_server parses with argparse: the last occurrence of a flag wins.
            assert _flag(engine, "--port") == str(backend.sidecar_engine_metrics_port(process, processes))
            assert _flag(engine, "--host") == "0.0.0.0"


@pytest.mark.parametrize("override", [{"port": 30000}, {"host": "127.0.0.1"}], ids=["port", "host"])
def test_vllm_hybrid_dp_recipe_host_or_port_cannot_move_the_scraped_listener(override: dict[str, Any]) -> None:
    """The hybrid launcher is Python ``vllm serve`` (argparse, last wins), so a repeated flag is not an error."""
    recipe = _recipe(
        "vllm",
        {
            "agg": {
                "nodes": 2,
                "workers": 1,
                "gpus": 8,
                "args": {**_VLLM_ARGS, "data-parallel-size": 8, "enable-expert-parallel": True, **override},
            }
        },
        gpus_per_node=4,
    )
    result = probe(recipe)
    groups = _endpoint_groups(result.processes)
    for (mode, index, node_rank), command in result.commands.items():
        group = groups[(mode, index)]
        process = next(p for p in group if p.node_rank == node_rank)
        engine = _engine_argv(command, "vllm.entrypoints.cli.main serve")
        assert _flag(engine, "--port") == str(result.backend.sidecar_engine_metrics_port(process, group))
        assert _flag(engine, "--host") == "0.0.0.0"


# --- Non-sidecar regression ---------------------------------------------------------------------


def _is_sidecar_recipe(recipe: Path) -> bool:
    data = yaml.safe_load(recipe.read_text())
    roles = (data.get("roles") or {}).values()
    return bool((data.get("dynamo") or {}).get("sidecar")) or any((role or {}).get("sidecar") for role in roles)


NON_SIDECAR_EXAMPLES = [recipe for recipe in example_recipes() if not _is_sidecar_recipe(recipe)]


@pytest.mark.parametrize("recipe", NON_SIDECAR_EXAMPLES, ids=lambda r: r.relative_to(EXAMPLES_DIR).as_posix())
def test_non_sidecar_examples_scrape_exactly_the_frontends_per_rank_answer(recipe: Path) -> None:
    """The pre-sidecar-metrics worker targets, reimplemented inline: the frontend names every port and path."""
    from srtctl.frontends import FRONTEND_NONE

    def inspect(orchestrator: Any) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        config, runtime = orchestrator.config, orchestrator.runtime
        expected: list[tuple[str, str]] = []
        if config.frontend.type != FRONTEND_NONE:
            frontend = get_frontend(config.frontend.type)
            key = lambda p: (p.endpoint_mode, p.endpoint_index, p.node_rank, p.node)  # noqa: E731
            for process in sorted(orchestrator.backend_processes, key=key):
                port = frontend.worker_metrics_port(process, runtime)
                path = frontend.worker_metrics_path(config.backend, process.endpoint_mode)
                if port is not None and path is not None:
                    name = f"backend_{process.endpoint_mode}{process.endpoint_index}_rank{process.node_rank}"
                    expected.append((name, f"http://{_node_ip(process.node)}:{port}{path}"))
        with patch("srtctl.core.telemetry.get_hostname_ip", lambda node, *_a, **_k: _node_ip(node)):
            text = orchestrator.tachometer_config_text()
        actual = re.findall(r'^name = "(backend_[^"]*)"\nurl = "([^"]*)"$', text, re.MULTILINE)
        return expected, actual

    from srtctl.core.config import load_config

    with _isolated_cluster(Path("/nonexistent-model"), Path("/nonexistent.sqsh")):
        node_count = load_config(recipe).total_nodes
    expected, actual = with_orchestrator(recipe, NODES[: node_count + 2], inspect)
    assert actual == expected
    assert not any(name.endswith("_engine") for name, _ in actual)


def test_non_sidecar_sglang_followers_keep_their_unallocated_http_port_and_targets() -> None:
    """Without sidecars nothing new is allocated or scraped: the per-rank frontend answer is unchanged."""
    from srtctl.core.telemetry import generate_tachometer_config

    backend = SGLangBackend(roles={"agg": RoleConfig(args={"tensor-parallel-size": 4})})
    endpoint = Endpoint(mode="agg", index=0, nodes=("n0", "n1"), gpu_indices=frozenset({0, 1}), gpus_per_node=2)
    processes = backend.endpoints_to_processes([endpoint])
    assert [p.http_port for p in processes] == [processes[0].http_port, 0]

    runtime = _sglang_runtime()
    runtime.log_dir = Path("/logs")
    from srtctl.core.schema import TachometerConfig

    with patch("srtctl.core.telemetry.get_hostname_ip", lambda node, *_: f"ip-{node}"):
        text = generate_tachometer_config(
            processes=processes,
            frontend_topology=type("T", (), {"frontend_nodes": [], "frontend_port": 8000})(),
            runtime=runtime,
            tachometer=TachometerConfig(),
        )
    assert re.findall(r'name = "(backend_[^"]+)"', text) == ["backend_agg0_rank0", "backend_agg0_rank1"]
    assert "_engine" not in text
    assert Counter(re.findall(r'url = "(http://ip-n\d:\d+)/metrics"', text)) == Counter(
        [f"http://ip-n0:{processes[0].sys_port}", f"http://ip-n1:{processes[1].sys_port}"]
    )
