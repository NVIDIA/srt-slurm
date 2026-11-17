# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for health-count expectations (Dynamo generate-instance counting)."""

from types import SimpleNamespace

from srtctl.backends import SGLangBackend, VLLMBackend
from srtctl.cli.mixins.benchmark_stage import _get_health_expectations
from srtctl.frontends.dynamo import vllm_data_parallel_size
from srtctl.core.schema import RoleConfig
from srtctl.core.topology import Endpoint


def _config(
    frontend_type,
    backend_type,
    *,
    num_prefill=0,
    num_decode=0,
    num_agg=0,
    roles=None,
    dp_launch_mode="per_node",
):
    """Build a stand-in for SrtConfig around a real backend dataclass, with only the fields the helpers read."""
    if backend_type == "vllm":
        backend = VLLMBackend(roles=roles or {}, dp_launch_mode=dp_launch_mode)
    else:
        backend = SGLangBackend()
    return SimpleNamespace(
        frontend=SimpleNamespace(type=frontend_type),
        backend=backend,
        topology=SimpleNamespace(num_prefill=num_prefill, num_decode=num_decode, num_agg=num_agg),
    )


def _processes(*, prefill=0, decode=0, agg=0, gpu_count=8):
    """Build backend-process stand-ins grouped by endpoint mode, each owning gpu_count GPUs of its node."""

    def process(mode):
        return SimpleNamespace(endpoint_mode=mode, gpu_indices=frozenset(range(gpu_count)))

    return [
        *(process("prefill") for _ in range(prefill)),
        *(process("decode") for _ in range(decode)),
        *(process("agg") for _ in range(agg)),
    ]


def test_dynamo_vllm_per_gpu_disagg_multiplies_by_data_parallel_size():
    """Legacy per-GPU DP registers one generate instance per DP rank."""
    roles = {
        "prefill": RoleConfig(args={"data-parallel-size": 2}),
        "decode": RoleConfig(args={"data-parallel-size": 8}),
    }
    config = _config(
        "dynamo",
        "vllm",
        num_prefill=6,
        num_decode=1,
        roles=roles,
        dp_launch_mode="per_gpu",
    )

    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config)

    assert (n_prefill, n_decode, num_workers) == (12, 8, 20)
    assert count_desc == "12P + 8D Dynamo generate instances; logical workers: 6P + 1D"


def test_dynamo_vllm_per_gpu_aggregated_multiplies_by_data_parallel_size():
    """Legacy per-GPU aggregate workers register once per DP rank."""
    roles = {"agg": RoleConfig(args={"data-parallel-size": 8})}
    config = _config(
        "dynamo",
        "vllm",
        num_agg=1,
        roles=roles,
        dp_launch_mode="per_gpu",
    )

    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config)

    assert (n_prefill, n_decode, num_workers) == (0, 8, 8)
    assert count_desc == "0P + 8D Dynamo generate instances; logical workers: 1 agg"


def test_dynamo_vllm_per_node_disagg_counts_node_processes():
    """Per-node DEP8/DEP16 registers once per node-local process, not per DP rank."""
    roles = {
        "prefill": RoleConfig(args={"data-parallel-size": 8}),
        "decode": RoleConfig(args={"data-parallel-size": 16}),
    }
    config = _config(
        "dynamo",
        "vllm",
        num_prefill=3,
        num_decode=1,
        roles=roles,
    )

    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config, _processes(prefill=6, decode=4))

    assert (n_prefill, n_decode, num_workers) == (6, 4, 10)
    assert count_desc == "6P + 4D Dynamo generate instances; logical workers: 3P + 1D"


def test_dynamo_vllm_per_node_aggregated_counts_node_processes():
    """Per-node aggregated DP workers register as one decode per process."""
    roles = {"agg": RoleConfig(args={"data-parallel-size": 8})}
    config = _config(
        "dynamo",
        "vllm",
        num_agg=1,
        roles=roles,
        dp_launch_mode="per_node",
    )

    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config, _processes(agg=2))

    assert (n_prefill, n_decode, num_workers) == (0, 2, 2)
    assert count_desc == "0P + 2D Dynamo generate instances; logical workers: 1 agg"


def test_dynamo_vllm_per_node_cross_node_tp_counts_dplb_endpoints():
    """Topology-aware per_node counts the global DPLB endpoint when TP spans nodes."""
    roles = {"agg": RoleConfig(args={"data-parallel-size": 2, "tensor-parallel-size": 8})}
    config = _config(
        "dynamo",
        "vllm",
        num_agg=1,
        roles=roles,
        dp_launch_mode="per_node",
    )

    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config, _processes(agg=4, gpu_count=4))

    assert (n_prefill, n_decode, num_workers) == (0, 1, 1)
    assert count_desc == "0P + 1D Dynamo generate instances; logical workers: 1 agg"


def test_dynamo_vllm_without_dp_config_defaults_to_logical_counts():
    """No data-parallel-size configured -> DP size 1 -> counts equal logical workers."""
    roles = {"prefill": RoleConfig(args={}), "decode": RoleConfig(args={})}
    config = _config("dynamo", "vllm", num_prefill=6, num_decode=1, roles=roles)

    n_prefill, n_decode, _count_desc, num_workers = _get_health_expectations(config)

    assert (n_prefill, n_decode, num_workers) == (6, 1, 7)


def test_non_dynamo_frontend_uses_logical_worker_counts():
    """Only Dynamo reports per-DP-rank generate instances; others stay logical."""
    roles = {
        "prefill": RoleConfig(args={"data-parallel-size": 2}),
        "decode": RoleConfig(args={"data-parallel-size": 8}),
    }
    config = _config("sglang-router", "vllm", num_prefill=6, num_decode=1, roles=roles)

    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config)

    assert (n_prefill, n_decode, num_workers) == (6, 1, 7)
    assert count_desc == "6P + 1D"


def test_dynamo_non_vllm_backend_uses_logical_worker_counts():
    """Dynamo + sglang has no DP-rank fan-out in these units; stay logical."""
    config = _config("dynamo", "sglang", num_prefill=6, num_decode=1)

    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config)

    assert (n_prefill, n_decode, num_workers) == (6, 1, 7)
    assert count_desc == "6P + 1D"


def _sglang_sidecar_config(roles, *, sidecar=True, num_prefill=0, num_decode=0, num_agg=0):
    """A Dynamo + SGLang stand-in whose role engine is the one SGLang backend."""
    backend = SGLangBackend(roles=roles)
    return SimpleNamespace(
        frontend=SimpleNamespace(type="dynamo"),
        backend=backend,
        backend_for_role=lambda mode: backend,
        dynamo=SimpleNamespace(sidecar=sidecar),
        topology=SimpleNamespace(num_prefill=num_prefill, num_decode=num_decode, num_agg=num_agg),
    )


def _endpoint(mode, nodes, index=0):
    """One endpoint over ``nodes`` with one GPU each."""
    return Endpoint(mode=mode, index=index, nodes=tuple(nodes), gpu_indices=frozenset({0}), gpus_per_node=1)


def test_dynamo_sglang_sidecar_counts_one_instance_per_listener_node():
    """Every node that hosts SGLang Rust listeners runs a sidecar, which registers its own generate instance."""
    attention_dp2 = RoleConfig(args={"tp-size": 2, "attn-dp-size": 2})
    config = _sglang_sidecar_config({"agg": attention_dp2}, num_agg=1)
    processes = config.backend.endpoints_to_processes([_endpoint("agg", ["n0", "n1"])], dynamo_sidecar=True)
    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config, processes)
    assert (n_prefill, n_decode, num_workers) == (0, 2, 2)
    assert count_desc == "0P + 2D Dynamo sidecars; logical workers: 1 agg"

    config = _sglang_sidecar_config({"prefill": attention_dp2, "decode": attention_dp2}, num_prefill=1, num_decode=1)
    endpoints = [_endpoint("prefill", ["n0", "n1"]), _endpoint("decode", ["n2", "n3"])]
    processes = config.backend.endpoints_to_processes(endpoints, dynamo_sidecar=True)
    n_prefill, n_decode, _count_desc, num_workers = _get_health_expectations(config, processes)
    assert (n_prefill, n_decode, num_workers) == (2, 2, 4)


def test_dynamo_sglang_sidecar_tp_follower_adds_no_instance():
    """TP over two nodes leads one listener, on node 0; the follower runs the engine only."""
    config = _sglang_sidecar_config({"agg": RoleConfig(args={"tp-size": 2})}, num_agg=1)
    processes = config.backend.endpoints_to_processes([_endpoint("agg", ["n0", "n1"])], dynamo_sidecar=True)
    n_prefill, n_decode, _count_desc, num_workers = _get_health_expectations(config, processes)
    assert (n_prefill, n_decode, num_workers) == (0, 1, 1)


def test_dynamo_sglang_without_sidecar_uses_logical_worker_counts():
    """Without sidecars the Python worker registers once per logical worker, whatever its DP layout."""
    config = _sglang_sidecar_config(
        {"agg": RoleConfig(args={"tp-size": 2, "attn-dp-size": 2})}, sidecar=False, num_agg=1
    )
    processes = config.backend.endpoints_to_processes([_endpoint("agg", ["n0", "n1"])])
    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config, processes)
    assert (n_prefill, n_decode, num_workers) == (0, 1, 1)
    assert count_desc == "1 agg"


def test_vllm_data_parallel_size_reads_both_key_styles_and_defaults():
    dashed = _config("dynamo", "vllm", roles={"prefill": RoleConfig(args={"data-parallel-size": 4})})
    assert vllm_data_parallel_size(dashed, "prefill") == 4

    underscored = _config("dynamo", "vllm", roles={"decode": RoleConfig(args={"data_parallel_size": 3})})
    assert vllm_data_parallel_size(underscored, "decode") == 3

    no_vllm_config = _config("dynamo", "vllm", roles=None)
    assert vllm_data_parallel_size(no_vllm_config, "prefill") == 1

    non_vllm = _config("dynamo", "sglang")
    assert vllm_data_parallel_size(non_vllm, "prefill") == 1
