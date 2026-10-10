# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM DWDP prefill groups: per-worker rendezvous rank and private caches.

A role whose engine args carry ``dwdp_config`` is N single-GPU workers forming one
group. Each worker step must see its own ``TRTLLM_DWDP_RANK``, the group's shared
master (worker 0's node), and a private HOME / module cache / autotuner cache.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from srtctl.backends.trtllm import DWDP_MASTER_PORT, DWDP_RENDEZVOUS_TIMEOUT_S, TRTLLMBackend
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import RoleConfig, SrtConfig
from srtctl.core.topology import Process

SRUN_WORKER = "srtctl.cli.mixins.worker_stage.start_srun_process"
HOST_IP_WORKER = "srtctl.cli.mixins.worker_stage.get_hostname_ip"
HOST_IP = "srtctl.core.slurm.get_hostname_ip"

DWDP = {"dwdp_size": 4, "num_groups": 1, "num_experts_per_worker": 224, "num_prefetch_experts": 224}

RECIPE = {
    "schema": 2,
    "name": "dwdp-toy",
    "model": {"path": "/models/kimi-k3", "container": "/trtllm.sqsh", "precision": "fp4"},
    "resources": {"gpu_type": "gb300", "gpus_per_node": 4},
    "frontend": {"type": "trtllm_serve", "enable_multiple_frontends": False},
    "engine": {"type": "trtllm"},
    "roles": {
        "prefill": {
            "nodes": 1,
            "workers": 4,
            "gpus": 1,
            "args": {"tensor_parallel_size": 1, "enable_attention_dp": True, "dwdp_config": DWDP},
            "extra_args": ["--server_role", "CONTEXT"],
        },
        "decode": {
            "nodes": 1,
            "workers": 1,
            "gpus": 4,
            "args": {"tensor_parallel_size": 4},
        },
    },
    "benchmark": {"type": "manual"},
    "observability": {"tachometer": {"enabled": False}},
}


def _process(mode: str, index: int, node: str = "node1", gpu: int = 0) -> Process:
    return Process(
        node=node,
        gpu_indices=frozenset({gpu}),
        sys_port=8000 + index,
        http_port=9000 + index,
        endpoint_mode=mode,
        endpoint_index=index,
    )


def _backend(prefill_env: dict[str, str] | None = None) -> TRTLLMBackend:
    return TRTLLMBackend(
        roles={
            "prefill": RoleConfig(args={"tensor_parallel_size": 1, "dwdp_config": DWDP}, env=prefill_env or {}),
            "decode": RoleConfig(args={"tensor_parallel_size": 4}),
        }
    )


def test_dwdp_size_reads_the_role_args() -> None:
    backend = _backend()
    assert backend.dwdp_size("prefill") == 4
    assert backend.dwdp_size("decode") == 0
    assert TRTLLMBackend().dwdp_size("prefill") == 0


def test_process_environment_gives_each_dwdp_worker_its_rank_and_private_caches() -> None:
    backend = _backend()
    env = backend.get_process_environment(_process("prefill", 3, gpu=3))
    assert env["TRTLLM_DWDP_RANK"] == "3"
    assert env["TRTLLM_DWDP_MASTER_PORT"] == str(DWDP_MASTER_PORT)
    assert env["TRTLLM_DWDP_RENDEZVOUS_TIMEOUT_S"] == str(DWDP_RENDEZVOUS_TIMEOUT_S)
    assert env["HOME"] == "/logs/worker_priv/prefill_w3"
    assert env["HF_MODULES_CACHE"] == "/logs/worker_priv/prefill_w3/hf_modules"
    assert env["TLLM_AUTOTUNER_CACHE_PATH"] == "/logs/worker_priv/prefill_w3/autotune_cache.json"
    # the master's address is the worker stage's job (another endpoint's node)
    assert "TRTLLM_DWDP_MASTER_ADDR" not in env


def test_process_environment_is_empty_off_a_dwdp_role() -> None:
    assert _backend().get_process_environment(_process("decode", 0)) == {}
    assert TRTLLMBackend().get_process_environment(_process("prefill", 0)) == {}


def test_role_env_pins_win_over_the_private_cache_defaults() -> None:
    backend = _backend({"HOME": "/shared/home", "TRTLLM_DWDP_MASTER_PORT": "31000"})
    env = backend.get_process_environment(_process("prefill", 1, gpu=1))
    assert "HOME" not in env  # the role env's /shared/home stays
    assert "TRTLLM_DWDP_MASTER_PORT" not in env
    assert env["HF_MODULES_CACHE"] == "/logs/worker_priv/prefill_w1/hf_modules"
    assert env["TRTLLM_DWDP_RANK"] == "1"


def _config(data: dict) -> SrtConfig:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        yaml.dump(data, handle)
        path = Path(handle.name)
    try:
        return SrtConfig.from_yaml(path)
    finally:
        path.unlink(missing_ok=True)


def _runtime(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(
        job_id="4242",
        run_name="dwdp-toy",
        nodes=Nodes(head="node1", bench="node1", infra="node1", worker=("node1", "node2")),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=tmp_path,
        model_path=Path("/models/kimi-k3"),
        container_image=Path("/trtllm.sqsh"),
        gpus_per_node=4,
        network_interface="eth0",
        container_mounts={tmp_path: Path("/logs")},
        environment={},
    )


def _proc() -> MagicMock:
    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.return_value = 0
    return proc


def test_worker_stage_launches_a_dwdp_group_with_distinct_ranks_and_one_master(tmp_path: Path) -> None:
    orchestrator = SweepOrchestrator(config=_config(RECIPE), runtime=_runtime(tmp_path))
    ips = {"node1": "10.0.0.11", "node2": "10.0.0.12"}
    with (
        patch(SRUN_WORKER, return_value=_proc()) as srun,
        patch(HOST_IP_WORKER, side_effect=lambda node, _iface: ips[node]),
        patch(HOST_IP, side_effect=lambda node, _iface: ips[node]),
    ):
        orchestrator.start_all_workers()

    steps = {call.kwargs["step_name"]: call.kwargs for call in srun.call_args_list}
    prefill = sorted(name for name in steps if name.startswith("prefill_"))
    assert len(prefill) == 4
    ranks = []
    for name in prefill:
        env = steps[name]["env_to_set"]
        ranks.append(env["TRTLLM_DWDP_RANK"])
        assert env["TRTLLM_DWDP_MASTER_ADDR"] == "10.0.0.11"  # prefill worker 0's node, for every member
        assert env["TRTLLM_DWDP_MASTER_PORT"] == str(DWDP_MASTER_PORT)
        assert env["HOME"] == f"/logs/worker_priv/prefill_w{env['TRTLLM_DWDP_RANK']}"
        assert f"mkdir -p /logs/worker_priv/prefill_w{env['TRTLLM_DWDP_RANK']}" in steps[name]["bash_preamble"]
        assert "--server_role" in steps[name]["command"] and "CONTEXT" in steps[name]["command"]
    assert sorted(ranks) == ["0", "1", "2", "3"]

    (decode,) = [kwargs for name, kwargs in steps.items() if name.startswith("decode_")]
    assert "TRTLLM_DWDP_RANK" not in decode["env_to_set"]
    assert "TRTLLM_DWDP_MASTER_ADDR" not in decode["env_to_set"]
    assert "worker_priv" not in (decode["bash_preamble"] or "")
