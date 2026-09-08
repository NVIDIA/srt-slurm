# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tachometer coverage for the SGLang Model Gateway frontend, and clean SIGTERM delivery for sruns.

The gateway only starts its Prometheus listener when ``--prometheus-port`` is
passed, and native ``sglang.launch_server`` workers only serve ``/metrics`` with
``--enable-metrics``; before this, every scrape of both failed for the whole
run. Separately, the srun bash wrapper held SIGTERM away from its child, so
tachometer was SIGKILLed at cleanup and never compacted its parquet.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from srtctl.backends import SGLangProtocol, SGLangServerConfig
from srtctl.cli.mixins.frontend_stage import FrontendTopology
from srtctl.cli.mixins.telemetry_stage import TACHOMETER_TERMINATE_TIMEOUT_SECONDS
from srtctl.core.processes import ManagedProcess, ProcessRegistry
from srtctl.core.schema import DynamoConfig, TachometerConfig
from srtctl.core.slurm import start_srun_process
from srtctl.core.telemetry import generate_tachometer_config
from srtctl.core.topology import Process
from srtctl.frontends.sglang import SGLangFrontend, router_metrics_port
from srtctl.ports import SGLANG_ROUTER_METRICS_PORT


def _process(node: str, *, mode: str, rank: int, http_port: int, sys_port: int) -> Process:
    return Process(
        node=node,
        gpu_indices=frozenset({0}),
        sys_port=sys_port,
        http_port=http_port,
        endpoint_mode=mode,
        endpoint_index=0,
        node_rank=rank,
    )


# --- tachometer targets -------------------------------------------------------------


@patch("srtctl.core.telemetry.get_hostname_ip", side_effect=lambda node, interface: f"ip-{node}")
def test_sglang_router_targets_worker_http_ports_and_gateway_prometheus_port(_ip) -> None:
    runtime = MagicMock(job_id="1", run_name="r_1", network_interface="eth0")
    runtime.log_dir = Path("/runs/1/logs")
    processes = [
        _process("node-a", mode="agg", rank=0, http_port=6100, sys_port=7500),
        _process("node-a", mode="agg", rank=0, http_port=6101, sys_port=7501),
        _process("node-b", mode="agg", rank=1, http_port=0, sys_port=7502),  # multi-node follower: serves nothing
    ]
    topology = FrontendTopology(nginx_node=None, frontend_nodes=["head"], frontend_port=8000, public_port=8000)

    text = generate_tachometer_config(
        processes=processes,
        frontend_topology=topology,
        runtime=runtime,
        tachometer=TachometerConfig(enabled=True, default_exporters=False),
        frontend_type="sglang",
        frontend_metrics_port=SGLANG_ROUTER_METRICS_PORT,
    )

    assert 'url = "http://ip-node-a:6100/metrics"' in text
    assert 'url = "http://ip-node-a:6101/metrics"' in text
    assert 'url = "http://ip-head:29000/metrics"' in text  # the gateway's Prometheus listener, not the routing port
    assert ":8000" not in text
    assert ":7500" not in text and ":7501" not in text and ":7502" not in text
    assert "ip-node-b" not in text


# --- gateway launch flags ---------------------------------------------------------------


def test_gateway_gets_prometheus_flags_unless_the_recipe_set_them() -> None:
    frontend = SGLangFrontend()
    config = SimpleNamespace(frontend=SimpleNamespace(args={"policy": "cache_aware"}))
    assert frontend.get_managed_frontend_args(config, None, []) == [
        "--prometheus-port",
        "29000",
        "--prometheus-host",
        "0.0.0.0",
    ]

    config = SimpleNamespace(frontend=SimpleNamespace(args={"prometheus_port": 31000, "prometheus-host": "::"}))
    assert frontend.get_managed_frontend_args(config, None, []) == []
    assert router_metrics_port(config.frontend.args) == 31000
    assert router_metrics_port(None) == SGLANG_ROUTER_METRICS_PORT


# --- worker --enable-metrics ---------------------------------------------------------------


def _runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.model_path = Path("/models/m")
    runtime.is_hf_model = False
    runtime.gpu_type = "h100"
    runtime.log_dir = Path("/tmp")
    runtime.network_interface = None
    runtime.dynamo = DynamoConfig()
    runtime.request_plane = "tcp"
    return runtime


def _worker_command(backend: SGLangProtocol, frontend_type: str) -> list[str]:
    process = _process("node0", mode="agg", rank=0, http_port=6100, sys_port=7500)
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"):
        return backend.build_worker_command(process, [process], _runtime(), frontend_type=frontend_type)


def test_native_sglang_workers_enable_metrics_only_for_the_sglang_frontend() -> None:
    backend = SGLangProtocol(sglang_config=SGLangServerConfig(aggregated={"tensor-parallel-size": 1}))
    assert _worker_command(backend, "sglang").count("--enable-metrics") == 1
    assert "--enable-metrics" not in _worker_command(backend, "dynamo")

    # A recipe that already sets the flag is not given it twice.
    explicit = SGLangProtocol(sglang_config=SGLangServerConfig(aggregated={"enable-metrics": True}))
    assert _worker_command(explicit, "sglang").count("--enable-metrics") == 1


# --- SIGTERM reaches the process ---------------------------------------------------------------


def test_srun_bash_wrapper_execs_the_command() -> None:
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="1"),
        patch("srtctl.core.slurm._get_cluster_bash_preamble", return_value=None),
        patch("subprocess.Popen") as popen,
    ):
        popen.return_value = MagicMock()
        start_srun_process(["tachometer-scraper", "--config", "c.toml"], env_to_set={"POLARS_MAX_THREADS": "4"})
    bash_cmd = popen.call_args.args[0][-1]
    assert bash_cmd.endswith("&& exec tachometer-scraper --config c.toml")


def test_tachometer_terminate_signals_the_step_then_waits() -> None:
    popen = MagicMock()
    popen.poll.return_value = None
    proc = ManagedProcess(name="tachometer", popen=popen, terminate_timeout=90.0, step_name="tachometer")
    squeue = SimpleNamespace(returncode=0, stdout="12440.16 tachometer\n12440.extern extern\n", stderr="")
    scancel = SimpleNamespace(returncode=0, stdout="", stderr="")
    with (
        patch.dict("os.environ", {"SLURM_JOB_ID": "12440"}),
        patch("srtctl.core.processes.subprocess.run", side_effect=[squeue, scancel]) as run,
        patch("srtctl.core.processes.terminate_and_reap") as reap,
    ):
        proc.terminate()
    assert run.call_args_list[0].args[0][:3] == ["squeue", "--steps", "--jobs=12440"]
    assert run.call_args_list[1].args[0] == ["scancel", "--signal=TERM", "--full", "12440.16"]
    popen.wait.assert_called_once_with(timeout=90.0)
    reap.assert_not_called()  # srun exited on its own once the task handled SIGTERM


def test_terminate_falls_back_to_srun_sigterm_when_the_step_is_not_found() -> None:
    popen = MagicMock()
    popen.poll.return_value = None
    proc = ManagedProcess(name="tachometer", popen=popen, step_name="tachometer")
    squeue = SimpleNamespace(returncode=0, stdout="12440.extern extern\n", stderr="")
    with (
        patch.dict("os.environ", {"SLURM_JOB_ID": "12440"}),
        patch("srtctl.core.processes.subprocess.run", return_value=squeue),
        patch("srtctl.core.processes.terminate_and_reap") as reap,
    ):
        reap.return_value = SimpleNamespace(reaped=True, force_killed=False)
        proc.terminate()
    reap.assert_called_once()


def test_srun_step_name_becomes_job_name() -> None:
    with (
        patch("srtctl.core.slurm.get_slurm_job_id", return_value="1"),
        patch("srtctl.core.slurm._get_cluster_bash_preamble", return_value=None),
        patch("subprocess.Popen") as popen,
    ):
        popen.return_value = MagicMock()
        start_srun_process(["tachometer-scraper"], step_name="tachometer")
    assert "--job-name=tachometer" in popen.call_args.args[0]


def test_registry_cleanup_uses_the_process_terminate_timeout() -> None:
    popen = MagicMock()
    popen.poll.return_value = None
    registry = ProcessRegistry(job_id="1")
    registry.add_process(ManagedProcess(name="tachometer", popen=popen, terminate_timeout=90.0))
    with patch("srtctl.core.processes.terminate_and_reap") as reap:
        reap.return_value = SimpleNamespace(reaped=True, force_killed=False)
        registry.cleanup()
    assert reap.call_args.kwargs["terminate_timeout"] == 90.0
    assert TACHOMETER_TERMINATE_TIMEOUT_SECONDS == 90.0
