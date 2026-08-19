from pathlib import Path
from unittest.mock import MagicMock, patch

from srtctl.backends import VLLMProtocol, VLLMServerConfig
from srtctl.core.topology import Process


def _render_tp2_numa_nodes(gpu_indices: list[int]) -> list[str]:
    backend = VLLMProtocol(
        dp_launch_mode="per_gpu",
        vllm_config=VLLMServerConfig(
            prefill={
                "tensor-parallel-size": 2,
                "numa-bind": True,
                "numa-bind-nodes": [0, 0, 1, 1],
            }
        ),
    )
    process = Process(
        node="node0",
        gpu_indices=frozenset(gpu_indices),
        sys_port=8081,
        http_port=30000,
        endpoint_mode="prefill",
        endpoint_index=0,
        node_rank=0,
    )
    runtime = MagicMock()
    runtime.model_path = Path("/models/minimax")
    runtime.is_hf_model = False
    runtime.worker_model_arg = "/model"

    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        command = backend.build_worker_command(
            process=process,
            endpoint_processes=[process],
            runtime=runtime,
        )

    flag = command.index("--numa-bind-nodes")
    return command[flag + 1 : flag + 3]


def test_tp2_services_slice_numa_map_by_physical_gpu_assignment():
    assert _render_tp2_numa_nodes([0, 1]) == ["0", "0"]
    assert _render_tp2_numa_nodes([2, 3]) == ["1", "1"]
