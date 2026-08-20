# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess

import yaml

from srtctl.core.config import expand_observability
from srtctl.core.schema import SrtConfig
from srtctl.render.lifecycle import build_local_lifecycle_render_context, render_local_lifecycle


def _config(
    *,
    frontend_type: str = "sglang",
    frontend_env: dict[str, str] | None = None,
    environment: dict[str, str] | None = None,
) -> SrtConfig:
    raw = {
        "name": "direct-render",
        "model": {
            "path": "hf:fake/mock-model",
            "container": "unused-on-direct-host",
            "precision": "fp8",
        },
        "resources": {
            "gpu_type": "h100",
            "gpus_per_node": 8,
            "agg_nodes": 1,
            "agg_workers": 8,
            "gpus_per_agg": 1,
        },
        "backend": {
            "type": "sglang",
            "sglang_config": {
                "aggregated": {
                    "served-model-name": "fake/mock-model",
                    "tp": 1,
                    "enable-metrics": True,
                }
            },
        },
        "frontend": {
            "type": frontend_type,
            "enable_multiple_frontends": False,
            "args": {"policy": "cache_aware"} if frontend_type == "sglang" else {"router-mode": "kv"},
            "env": frontend_env,
        },
        "environment": environment or {},
        "benchmark": {"type": "custom", "command": "aiperf profile --ui none"},
        "observability": {
            "enabled": True,
            "tachometer": {"enabled": True},
        },
    }
    expand_observability(raw)
    return SrtConfig.Schema().load(yaml.safe_load(yaml.safe_dump(raw)))


def test_local_lifecycle_renders_eight_tp1_workers_with_separate_logs(tmp_path) -> None:
    context = build_local_lifecycle_render_context(
        _config(),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    script = render_local_lifecycle(context)

    assert len(context.worker_processes) == 8
    assert {worker.log_name for worker in context.worker_processes} == {f"worker-{index}.log" for index in range(8)}
    assert all(f"CUDA_VISIBLE_DEVICES={index}" in worker.command for index, worker in enumerate(context.worker_processes))
    assert "-m sglang_router.launch_router" in context.router_command
    assert "--policy cache_aware" in context.router_command
    assert 'name = "router"' in context.tachometer_config
    assert script.count('"${LOG_DIR}/worker-') == 8
    assert '"${LOG_DIR}/router.log"' in script
    assert '"${LOG_DIR}/tachometer.log"' in script
    assert "setsid" in script
    assert "kill -TERM 0" not in script
    for forbidden in ("#SBATCH", "SLURM_", "scontrol", "srun", "do_sweep", "run_benchmark"):
        assert forbidden not in script
    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr


def test_local_dynamo_lifecycle_starts_owned_infrastructure(tmp_path) -> None:
    context = build_local_lifecycle_render_context(
        _config(frontend_type="dynamo"),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    script = render_local_lifecycle(context)

    assert context.needs_dynamo_infra
    assert "-m dynamo.sglang" in context.worker_processes[0].command
    assert "-m dynamo.frontend" in context.router_command
    assert 'srt_launch "nats"' in script
    assert 'srt_launch "etcd"' in script
    assert "DYN_SYSTEM_PORT=7500" in context.worker_processes[0].command
    assert 'DYN_REQUEST_TRACE_FILE_PATH="${ARTIFACT_DIR}"/dynamo-request-trace' in context.router_command
    assert all(
        f"--nccl-port {17_500 + index}" in worker.command
        for index, worker in enumerate(context.worker_processes)
    )
    assert 'srt_wait_http_ready "http://127.0.0.1:6100/health"' not in script
    assert "srt_wait_router_ready" in script
    assert 'TACHOMETER_STORAGE="${ARTIFACT_DIR}/tachometer/raw/scrape"' in script
    assert context.ruter_enabled
    assert 'SRTCTL_RUTER_PYTHON="${SRTCTL_RUTER_PYTHON:-${SRTCTL_SOURCE}/.venv/bin/python}"' in script
    assert 'Observability enabled: Dynamo request tracing is on (DYN_REQUEST_TRACE=1; request-end gzip JSONL: ${ARTIFACT_DIR}/dynamo-request-trace.*.jsonl.gz)' in script
    assert '"${SRTCTL_RUTER_PYTHON}" -m srtctl.ruter init "${OUTPUT_DIR}" --output "${LOG_DIR}/.ruter"' in script
    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr


def test_local_lifecycle_expands_artifact_dir_in_frontend_environment(tmp_path) -> None:
    context = build_local_lifecycle_render_context(
        _config(
            frontend_type="dynamo",
            frontend_env={"DYN_REQUEST_TRACE_FILE_PATH": "{artifact_dir}/dynamo-request-trace.jsonl"},
        ),
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    script = render_local_lifecycle(context)

    assert 'DYN_REQUEST_TRACE_FILE_PATH="${ARTIFACT_DIR}"/dynamo-request-trace.jsonl' in context.router_command
    assert "DYN_REQUEST_TRACE_FILE_PATH=\"${ARTIFACT_DIR}\"/dynamo-request-trace.jsonl" in script
    assert 'export OUTPUT_DIR LOG_DIR ARTIFACT_DIR' in script
    syntax = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr


def test_local_dynamo_lifecycle_accepts_isolated_infra_ports(tmp_path) -> None:
    config = _config(
        frontend_type="dynamo",
        environment={
            "SRTCTL_ETCD_PORT": "22379",
            "SRTCTL_ETCD_PEER_PORT": "22380",
            "SRTCTL_NATS_PORT": "24222",
        },
    )
    context = build_local_lifecycle_render_context(
        config,
        source_dir=tmp_path / "srt-slurm",
        output_base=tmp_path / "outputs",
    )
    script = render_local_lifecycle(context)

    assert "ETCD_ENDPOINTS=http://127.0.0.1:22379" in context.router_command
    assert "NATS_SERVER=nats://127.0.0.1:24222" in context.router_command
    assert '"${SRTCTL_NATS_BINARY}" -js -a "127.0.0.1" -p 24222' in script
    assert "http://127.0.0.1:22379/health" in script
    assert "--listen-peer-urls \"http://127.0.0.1:22380\"" in script
