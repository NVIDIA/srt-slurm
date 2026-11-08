# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL workers get CUDA transports when the container's UCX_TLS has none."""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from srtctl.backends.sglang import SGLangBackend
from srtctl.backends.trtllm import TRTLLMBackend
from srtctl.backends.vllm import VLLMBackend
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.mixins.worker_stage import _NIXL_UCX_TLS_PREAMBLE
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import DynamoConfig, ModelConfig, ResourceConfig, RoleConfig, SrtConfig

_MULTI_NIXL = (
    '{"kv_connector": "MultiConnector", "kv_role": "kv_both", '
    '"kv_connector_extra_config": {"connectors": [{"kv_connector": "NixlConnector"}]}}'
)


@pytest.mark.parametrize(
    "engine,mode,args,expected",
    [
        (SGLangBackend, "prefill", {"disaggregation-transfer-backend": "nixl"}, True),
        (SGLangBackend, "decode", {"disaggregation_transfer_backend": "nixl"}, True),
        (SGLangBackend, "agg", {"disaggregation-transfer-backend": "nixl"}, False),
        (SGLangBackend, "prefill", {}, False),
        (VLLMBackend, "prefill", {}, True),
        (VLLMBackend, "agg", {}, False),
        (VLLMBackend, "prefill", {"connector": "lmcache"}, False),
        (VLLMBackend, "decode", {"kv-transfer-config": _MULTI_NIXL}, True),
        (TRTLLMBackend, "decode", {"cache_transceiver_config": {"backend": "NIXL"}}, True),
        (TRTLLMBackend, "decode", {"cache_transceiver_config": {"backend": "UCX"}}, False),
    ],
)
def test_uses_nixl(engine, mode, args, expected) -> None:
    assert engine(roles={mode: RoleConfig(args=args)}).uses_nixl(mode) is expected


@pytest.mark.parametrize(
    "ucx_tls,expected",
    [
        ("tcp", "cuda_copy,cuda_ipc,sm,self,tcp"),
        ("rc,tcp", "cuda_copy,cuda_ipc,sm,self,rc,tcp"),
        ("cuda_ipc,tcp", "cuda_copy,cuda_ipc,sm,self,cuda_ipc,tcp"),
        (None, None),
        ("all", "all"),
        ("cuda,tcp", "cuda,tcp"),
        ("tcp,cuda_copy", "tcp,cuda_copy"),
        ("^cuda_ipc", "^cuda_ipc"),
    ],
)
def test_preamble_adds_cuda_transports_only_where_nixl_would_refuse(ucx_tls: str | None, expected: str | None) -> None:
    env = {k: v for k, v in os.environ.items() if k != "UCX_TLS"}
    if ucx_tls is not None:
        env["UCX_TLS"] = ucx_tls
    script = f'{_NIXL_UCX_TLS_PREAMBLE} && printf %s "${{UCX_TLS-<unset>}}"'
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=True)
    assert result.stdout == (expected or "<unset>")
    assert ("srtctl: UCX_TLS=" in result.stderr) == (expected != ucx_tls)


def _launches(tmp_path: Path, transfer_backend: str, environment: dict[str, str]) -> list[dict]:
    args = {"disaggregation-transfer-backend": transfer_backend}
    config = SrtConfig(
        name="nixl-ucx-test",
        model=ModelConfig(path="/model", container="/container.sqsh", precision="bf16"),
        resources=ResourceConfig(gpus_per_node=4),
        roles={
            "prefill": RoleConfig(nodes=1, workers=1, gpus=1, args=args),
            "decode": RoleConfig(nodes="colocate", workers=1, gpus=1, args=args),
        },
        engine=SGLangBackend(),
        dynamo=DynamoConfig(install=False),
        environment=environment,
    )
    runtime = RuntimeContext(
        job_id="12345",
        run_name=config.name,
        nodes=Nodes(head="n0", bench="n0", infra="n0", worker=("n0",)),
        head_node_ip="192.0.2.1",
        infra_node_ip="192.0.2.1",
        log_dir=tmp_path,
        model_path=Path("/model"),
        container_image=Path("/container.sqsh"),
        container_mounts={tmp_path: Path("/logs")},
        gpus_per_node=4,
        network_interface=None,
        environment=environment,
    )
    orchestrator = SweepOrchestrator(config, runtime)
    with (
        patch("srtctl.core.slurm.get_hostname_ip", return_value="192.0.2.2"),
        patch("srtctl.cli.mixins.worker_stage.get_hostname_ip", return_value="192.0.2.2"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as srun,
    ):
        for worker in orchestrator.backend_processes:
            orchestrator.start_worker(worker, [worker])
    return [call.kwargs for call in srun.call_args_list]


@pytest.mark.parametrize(
    "transfer_backend,environment,patched",
    [
        ("nixl", {}, True),
        ("nixl", {"UCX_TLS": "tcp"}, False),
        ("mooncake", {}, False),
    ],
    ids=["nixl", "recipe-ucx-tls", "mooncake"],
)
def test_nixl_workers_check_ucx_tls_before_the_engine(
    tmp_path: Path, transfer_backend: str, environment: dict[str, str], patched: bool
) -> None:
    launches = _launches(tmp_path, transfer_backend, environment)
    assert [launch["command"][2] for launch in launches] == ["dynamo.sglang"] * 2
    for launch in launches:
        assert (_NIXL_UCX_TLS_PREAMBLE in launch["bash_preamble"]) == patched
        assert launch["env_to_set"].get("UCX_TLS") == environment.get("UCX_TLS")
