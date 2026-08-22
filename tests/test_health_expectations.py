# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for health-count expectations (Dynamo generate-instance counting)."""

from types import SimpleNamespace

from srtctl.cli.mixins.benchmark_stage import _get_health_expectations, _vllm_data_parallel_size


def _config(
    frontend_type,
    backend_type,
    *,
    num_prefill=0,
    num_decode=0,
    num_agg=0,
    vllm_config=None,
    dp_launch_mode="per_node",
):
    """Build a duck-typed stand-in for SrtConfig with only the fields the helpers read."""
    backend = SimpleNamespace(type=backend_type, vllm_config=vllm_config, dp_launch_mode=dp_launch_mode)
    return SimpleNamespace(
        frontend=SimpleNamespace(type=frontend_type),
        backend=backend,
        resources=SimpleNamespace(num_prefill=num_prefill, num_decode=num_decode, num_agg=num_agg),
    )


def _processes(*, prefill=0, decode=0, agg=0, gpu_count=None):
    """Build backend-process stand-ins grouped by endpoint mode."""

    def process(mode):
        fields = {"endpoint_mode": mode}
        if gpu_count is not None:
            fields["gpu_indices"] = frozenset(range(gpu_count))
        return SimpleNamespace(**fields)

    return [
        *(process("prefill") for _ in range(prefill)),
        *(process("decode") for _ in range(decode)),
        *(process("agg") for _ in range(agg)),
    ]


def test_dynamo_vllm_per_gpu_disagg_multiplies_by_data_parallel_size():
    """Legacy per-GPU DP registers one generate instance per DP rank."""
    vllm_config = SimpleNamespace(
        prefill={"data-parallel-size": 2},
        decode={"data-parallel-size": 8},
        aggregated=None,
    )
    config = _config(
        "dynamo",
        "vllm",
        num_prefill=6,
        num_decode=1,
        vllm_config=vllm_config,
        dp_launch_mode="per_gpu",
    )

    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config)

    assert (n_prefill, n_decode, num_workers) == (12, 8, 20)
    assert count_desc == "12P + 8D Dynamo generate instances; logical workers: 6P + 1D"


def test_dynamo_vllm_per_gpu_aggregated_multiplies_by_data_parallel_size():
    """Legacy per-GPU aggregate workers register once per DP rank."""
    vllm_config = SimpleNamespace(prefill=None, decode=None, aggregated={"data-parallel-size": 8})
    config = _config(
        "dynamo",
        "vllm",
        num_agg=1,
        vllm_config=vllm_config,
        dp_launch_mode="per_gpu",
    )

    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config)

    assert (n_prefill, n_decode, num_workers) == (0, 8, 8)
    assert count_desc == "0P + 8D Dynamo generate instances; logical workers: 1 agg"


def test_dynamo_vllm_per_node_disagg_counts_node_processes():
    """Per-node DEP8/DEP16 registers once per node-local process, not per DP rank."""
    vllm_config = SimpleNamespace(
        prefill={"data-parallel-size": 8},
        decode={"data-parallel-size": 16},
        aggregated=None,
    )
    config = _config(
        "dynamo",
        "vllm",
        num_prefill=3,
        num_decode=1,
        vllm_config=vllm_config,
    )

    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config, _processes(prefill=6, decode=4))

    assert (n_prefill, n_decode, num_workers) == (6, 4, 10)
    assert count_desc == "6P + 4D Dynamo generate instances; logical workers: 3P + 1D"


def test_dynamo_vllm_per_node_aggregated_counts_node_processes():
    """Per-node aggregated DP workers register as one decode per process."""
    vllm_config = SimpleNamespace(prefill=None, decode=None, aggregated={"data-parallel-size": 8})
    config = _config(
        "dynamo",
        "vllm",
        num_agg=1,
        vllm_config=vllm_config,
        dp_launch_mode="per_node",
    )

    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config, _processes(agg=2))

    assert (n_prefill, n_decode, num_workers) == (0, 2, 2)
    assert count_desc == "0P + 2D Dynamo generate instances; logical workers: 1 agg"


def test_dynamo_vllm_per_node_cross_node_tp_counts_dplb_endpoints():
    """Topology-aware per_node counts the global DPLB endpoint when TP spans nodes."""
    vllm_config = SimpleNamespace(
        prefill=None,
        decode=None,
        aggregated={"data-parallel-size": 2, "tensor-parallel-size": 8},
    )
    config = _config(
        "dynamo",
        "vllm",
        num_agg=1,
        vllm_config=vllm_config,
        dp_launch_mode="per_node",
    )

    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config, _processes(agg=4, gpu_count=4))

    assert (n_prefill, n_decode, num_workers) == (0, 1, 1)
    assert count_desc == "0P + 1D Dynamo generate instances; logical workers: 1 agg"


def test_dynamo_vllm_without_dp_config_defaults_to_logical_counts():
    """No data-parallel-size configured -> DP size 1 -> counts equal logical workers."""
    vllm_config = SimpleNamespace(prefill={}, decode={}, aggregated=None)
    config = _config("dynamo", "vllm", num_prefill=6, num_decode=1, vllm_config=vllm_config)

    n_prefill, n_decode, _count_desc, num_workers = _get_health_expectations(config)

    assert (n_prefill, n_decode, num_workers) == (6, 1, 7)


def test_non_dynamo_frontend_uses_logical_worker_counts():
    """Only Dynamo reports per-DP-rank generate instances; others stay logical."""
    vllm_config = SimpleNamespace(prefill={"data-parallel-size": 2}, decode={"data-parallel-size": 8}, aggregated=None)
    config = _config("none", "vllm", num_prefill=6, num_decode=1, vllm_config=vllm_config)

    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config)

    assert (n_prefill, n_decode, num_workers) == (6, 1, 7)
    assert count_desc == "6P + 1D"


def test_dynamo_non_vllm_backend_uses_logical_worker_counts():
    """Dynamo + sglang has no DP-rank fan-out in these units; stay logical."""
    config = _config("dynamo", "sglang", num_prefill=6, num_decode=1)

    n_prefill, n_decode, count_desc, num_workers = _get_health_expectations(config)

    assert (n_prefill, n_decode, num_workers) == (6, 1, 7)
    assert count_desc == "6P + 1D"


def test_vllm_data_parallel_size_reads_both_key_styles_and_defaults():
    dashed = _config("dynamo", "vllm", vllm_config=SimpleNamespace(prefill={"data-parallel-size": 4}))
    assert _vllm_data_parallel_size(dashed, "prefill") == 4

    underscored = _config("dynamo", "vllm", vllm_config=SimpleNamespace(decode={"data_parallel_size": 3}))
    assert _vllm_data_parallel_size(underscored, "decode") == 3

    no_vllm_config = _config("dynamo", "vllm", vllm_config=None)
    assert _vllm_data_parallel_size(no_vllm_config, "prefill") == 1

    non_vllm = _config("dynamo", "sglang")
    assert _vllm_data_parallel_size(non_vllm, "prefill") == 1
