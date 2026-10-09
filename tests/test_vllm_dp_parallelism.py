# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DP rank placement includes every model-parallel dimension."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.backends import VLLMBackend
from srtctl.core.schema import RoleConfig
from srtctl.core.topology import Endpoint


@pytest.mark.parametrize("frontend", ["llm-d", "sglang-router", "dynamo"])
@pytest.mark.parametrize("node_count", [1, 2])
@pytest.mark.parametrize("tp,pp,pcp", [(1, 1, 2), (2, 1, 2), (1, 2, 2)])
def test_dp_ranks_own_complete_model_parallel_groups(frontend, node_count, tp, pp, pcp):
    replica_size = tp * pp * pcp
    backend = VLLMBackend(
        dp_launch_mode="per_gpu",
        roles={
            "agg": RoleConfig(
                args={
                    "data-parallel-size": 2 * node_count,
                    "tensor-parallel-size": tp,
                    "pipeline-parallel-size": pp,
                    "prefill-context-parallel-size": pcp,
                    "data-parallel-external-lb": True,
                }
            )
        },
    )
    endpoint = Endpoint("agg", 0, tuple(f"node{i}" for i in range(node_count)), frozenset(range(2 * replica_size)))
    processes = backend.endpoints_to_processes([endpoint], frontend_type=frontend)

    assert len(processes) == 2 * node_count
    assert [p.dp_rank for p in processes] == list(range(2 * node_count))
    for node in endpoint.nodes:
        ranks = [p for p in processes if p.node == node]
        assert [p.gpu_indices for p in ranks] == [
            frozenset(range(replica_size)),
            frozenset(range(replica_size, 2 * replica_size)),
        ]
    if frontend != "dynamo":
        assert all(p.http_port > 0 for p in processes)
        assert len({(p.node, p.http_port) for p in processes}) == len(processes)
        runtime = MagicMock(model_path=Path("/model"), is_hf_model=False, frontend_port=8000)
        with patch("srtctl.core.slurm.get_hostname_ip", return_value="127.0.0.1"):
            for process in processes:
                command = backend.build_worker_command(
                    process=process, endpoint_processes=processes, runtime=runtime, frontend_type=frontend
                )
                assert command[command.index("--data-parallel-rank") + 1] == str(process.dp_rank)
                assert command[command.index("--prefill-context-parallel-size") + 1] == str(pcp)
                assert "--headless" not in command


@pytest.mark.parametrize(
    "dp,pcp,gpu_count,node_count,error", [(2, 4, 2, 1, "require 8 GPUs"), (3, 2, 3, 2, "not divisible")]
)
def test_per_rank_launch_rejects_invalid_pcp_allocation(dp, pcp, gpu_count, node_count, error):
    backend = VLLMBackend(
        roles={
            "agg": RoleConfig(
                args={
                    "data-parallel-size": dp,
                    "prefill-context-parallel-size": pcp,
                    "data-parallel-external-lb": True,
                }
            )
        }
    )
    nodes = tuple(f"node{i}" for i in range(node_count))
    # The second case has the correct total world size but cannot fit a rank on each node.
    endpoint = Endpoint("agg", 0, nodes, frozenset(range(gpu_count)))
    with pytest.raises(ValueError, match=error):
        backend.endpoints_to_processes([endpoint], frontend_type="llm-d")
