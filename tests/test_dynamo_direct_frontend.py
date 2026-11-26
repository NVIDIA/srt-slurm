# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo frontend dialing native SGLang gRPC engines itself (`frontend.type: dynamo-direct`)."""

import pytest
from marshmallow import ValidationError

from srtctl.backends import SGLangBackend, VLLMBackend
from srtctl.core.config import load_config
from srtctl.core.schema import DynamoConfig, FrontendConfig, ResourceConfig, RoleConfig, SrtConfig
from srtctl.core.worker_backends import allocate_worker_endpoints, worker_processes
from srtctl.frontends import DynamoDirectFrontend, DynamoFrontend, get_frontend
from tests.launch_snapshots import EXAMPLES_DIR, render_launch_plan

RECIPE = EXAMPLES_DIR / "sglang/dynamo-direct-disagg.yaml"


def _section(plan: str, name: str) -> str:
    return plan.split(f"## {name}")[1].split("\n## ")[0]


def _srt_config(engine=None, **overrides) -> SrtConfig:
    return SrtConfig(
        name="direct",
        model={"path": "model", "container": "image", "precision": "bf16"},
        resources=ResourceConfig(gpu_type="h100", gpus_per_node=8),
        roles={"agg": RoleConfig(nodes=1, workers=2, gpus=1)},
        engine=engine or SGLangBackend(),
        **{"frontend": FrontendConfig(type="dynamo-direct", enable_multiple_frontends=False), **overrides},
    )


def test_registry_resolves_the_dynamo_binary_with_direct_workers() -> None:
    frontend = get_frontend("dynamo-direct")
    assert isinstance(frontend, DynamoDirectFrontend)
    assert isinstance(frontend, DynamoFrontend)
    assert frontend.required_backend == "sglang"
    assert frontend.worker_launch == "direct"
    assert frontend.runs_dynamo and frontend.dials_native_grpc
    assert not get_frontend("dynamo").dials_native_grpc


def test_launch_plan_runs_bare_engines_and_a_direct_frontend() -> None:
    """Engines are the sidecar-mode engine command without a sidecar; the frontend lists their gRPC ports."""
    assert load_config(RECIPE).frontend.type == "dynamo-direct"
    plan = render_launch_plan(RECIPE)
    assert "# exit_code: 0" in plan
    # In-process discovery: no etcd or NATS step, and no plane handed to the frontend.
    assert "## service_etcd" not in plan and "## service_nats" not in plan
    frontend = _section(plan, "frontend_0")
    assert "ETCD_ENDPOINTS" not in frontend

    grpc_ports = []
    for step, mode in (("prefill_0_node-01", "prefill"), ("decode_0_node-02", "decode")):
        worker = _section(plan, step)
        command = next(line.strip() for line in worker.splitlines() if "sglang.launch_server" in line)
        assert command.startswith("python3 -m sglang.launch_server")
        assert "dynamo" not in command
        assert f"--disaggregation-mode {mode}" in command
        assert "--incremental-streaming-output" in command
        assert "--disaggregation-bootstrap-port" not in command
        grpc_ports.append(command.split("--grpc-port ")[1].split()[0])
        assert "SGLANG_RUST_SERVER: 1" in worker
        # Only the frontend's container installs Dynamo.
        assert "Installing dynamo" not in worker

    assert "Installing dynamo" in frontend
    launch = next(line.strip() for line in frontend.splitlines() if "dynamo.frontend" in line)
    assert launch == (
        "python3 -m dynamo.frontend --http-port=8000 --direct --direct-backend sglang "
        f"--engine-url grpc://node-01:{grpc_ports[0]} --engine-url grpc://node-02:{grpc_ports[1]} --router-mode kv"
    )


def test_readiness_counts_one_registration_per_dialed_engine() -> None:
    config = _srt_config()
    endpoints = allocate_worker_endpoints(config, ["node-a"])
    processes = worker_processes(config, endpoints)
    assert [process.sidecar_grpc_port is not None for process in processes] == [True, True]
    prefill, decode, description = DynamoDirectFrontend().health_expectations(config, processes)
    assert (prefill, decode) == (0, 2)
    assert "directly dialed engines" in description


def test_worker_ports_are_the_engines_own_http_ports() -> None:
    config = _srt_config()
    process = worker_processes(config, allocate_worker_endpoints(config, ["node-a"]))[0]
    frontend = DynamoDirectFrontend()
    assert frontend.worker_metrics_port(process, None) == process.http_port
    assert frontend.worker_endpoint_port(process, config, None) == process.http_port
    assert process.http_port != process.sys_port


@pytest.mark.parametrize("key", ["direct", "engine-url", "engine_url", "direct-backend"])
def test_managed_flags_cannot_be_set_in_frontend_args(key: str) -> None:
    with pytest.raises(ValidationError, match="manages frontend.args"):
        _srt_config(frontend=FrontendConfig(type="dynamo-direct", args={key: "x"}))


def test_only_sglang_engines_are_accepted() -> None:
    with pytest.raises(ValidationError, match="dynamo-direct"):
        _srt_config(engine=VLLMBackend())


def test_sidecar_mode_is_rejected() -> None:
    with pytest.raises(ValidationError, match="dynamo.sidecar: true requires frontend.type: dynamo"):
        _srt_config(dynamo=DynamoConfig(sidecar=True))
