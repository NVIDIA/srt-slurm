# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the role dispatch and TileRT wire-facing launch contract."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.benchmarks.sa_bench import SABenchRunner
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.frontends import TileRTRouterFrontend


def _recipe() -> dict:
    data = yaml.safe_load(Path("examples/tilert/glm5-rocm-mooncake-disagg.yaml").read_text())
    data["model"]["path"] = "/models/GLM-5.3"
    data["model"]["container"] = "decode-image"
    data["roles"]["prefill"]["container"] = "prefill-image"
    data["roles"]["decode"]["container"] = "decode-image"
    return data


def _load(data: dict) -> SrtConfig:
    return SrtConfig.Schema().load(
        resolve_config_with_defaults(
            data,
            {
                "containers": {"decode-image": "/images/decode.sqsh", "prefill-image": "/images/prefill.sqsh"},
            },
        )
    )


def _orchestrator(config: SrtConfig, tmp_path: Path, nodes: tuple[str, ...]) -> SweepOrchestrator:
    return SweepOrchestrator(
        config,
        RuntimeContext(
            job_id="7",
            run_name=config.name,
            nodes=Nodes(head=nodes[0], bench=nodes[0], infra=nodes[0], worker=nodes),
            head_node_ip="10.0.0.1",
            infra_node_ip="10.0.0.1",
            log_dir=tmp_path,
            model_path=Path("/models/GLM-5.3"),
            container_image=Path(config.model.container),
            container_mounts={tmp_path: Path("/logs")},
            gpus_per_node=8,
            network_interface=None,
            visible_devices_env="ROCR_VISIBLE_DEVICES",
        ),
    )


def _flag(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_mixed_workers_use_concrete_engines_images_and_gpu_masks(tmp_path: Path) -> None:
    data = _recipe()
    data["roles"]["prefill"]["gpus"] = 4
    data["roles"]["prefill"]["args"]["tensor-parallel-size"] = 4
    data["roles"]["prefill"]["engine"] = {"type": "vllm", "set_visible_devices": True}
    data["roles"]["decode"].update(nodes="colocate", gpus=4)
    config = _load(data)
    orchestrator = _orchestrator(config, tmp_path, ("n0",))
    with (
        patch("srtctl.core.slurm.get_hostname_ip", return_value="10.0.0.1"),
        patch("srtctl.cli.mixins.worker_stage.start_srun_process") as srun,
    ):
        orchestrator.start_all_workers()
    prefill, decode = [call.kwargs for call in srun.call_args_list]
    assert prefill["container_image"] == "/images/prefill.sqsh"
    assert decode["container_image"] == "/images/decode.sqsh"
    assert prefill["env_to_set"]["ROCR_VISIBLE_DEVICES"] == "0,1,2,3"
    assert decode["env_to_set"]["ROCR_VISIBLE_DEVICES"] == "4,5,6,7"
    assert prefill["env_to_set"]["VLLM_ROCM_USE_AITER"] == "1"
    assert "VLLM_ROCM_USE_AITER" not in decode["env_to_set"]
    assert decode["env_to_set"]["GLM5_AR_N"] == "2"
    assert "GLM5_AR_N" not in prefill["env_to_set"]
    for worker in (prefill, decode):
        assert not {
            "DYN_LOG",
            "DYN_REQUEST_PLANE",
            "DYN_SYSTEM_PORT",
            "DYN_EVENT_PLANE",
            "DYN_SKIP_SGLANG_LOG_FORMATTING",
            "ETCD_ENDPOINTS",
            "NATS_SERVER",
        }.intersection(worker["env_to_set"])
    assert not any(key.startswith("DYN_") for key in decode["env_to_set"])
    assert prefill["command"][:3] == ["vllm", "serve", "/model"]
    assert decode["command"][:3] == ["python", "-m", "tilert.pd_vllm.decode_server"]
    transfer = json.loads(_flag(prefill["command"], "--kv-transfer-config"))
    assert transfer["kv_connector"] == "TileRTConnector"
    assert transfer["kv_connector_extra_config"]["tilert_transport"] == "mooncake"
    assert _flag(decode["command"], "--transport") == "mooncake"
    assert _flag(decode["command"], "--model-weights-dir") == "/models/GLM-5.3-tilert-tp8"
    assert "--with-mtp" in decode["command"]
    assert _flag(prefill["command"], "--port") != _flag(decode["command"], "--http-port")
    assert _flag(decode["command"], "--ctrl-port") != _flag(decode["command"], "--http-port")


def test_router_advertises_all_decode_control_ports(tmp_path: Path) -> None:
    data = _recipe()
    data["roles"]["decode"].update(nodes=2, workers=2)
    config = _load(data)
    orchestrator = _orchestrator(config, tmp_path, ("p0", "d0", "d1"))
    frontend = TileRTRouterFrontend()
    hosts = {"p0": "10.0.0.10", "d0": "10.0.0.11", "d1": "10.0.0.12"}
    with patch("srtctl.frontends.static_router.get_hostname_ip", side_effect=lambda node, _: hosts[node]):
        workers = frontend.collect_workers(config.backend, orchestrator.backend_processes)
        health = frontend.get_backend_health_urls(config.backend, orchestrator.backend_processes)
    command = frontend.build_router_command(workers, "0.0.0.0", 8000, config.backend)
    command.extend(frontend.get_frontend_args_list(config.frontend.args))
    assert _flag(command, "--model-path") == "/model"
    prefill = next(p for p in orchestrator.backend_processes if p.endpoint_mode == "prefill")
    assert _flag(command, "--vllm-url") == f"http://10.0.0.10:{prefill.http_port}"
    assert command.count("--decode") == 1
    decode_specs = command[command.index("--decode") + 1 : command.index("--host")]
    assert decode_specs == [
        f"{hosts[p.node]}:{p.nixl_port}:{p.http_port}"
        for p in orchestrator.backend_processes
        if p.endpoint_mode == "decode"
    ]
    assert set(health) == {f"http://{hosts[p.node]}:{p.http_port}/health" for p in orchestrator.backend_processes}
    for process in orchestrator.backend_processes:
        expected = process.http_port if process.endpoint_mode == "prefill" else None
        assert frontend.worker_metrics_port(process, orchestrator.runtime) == expected


def test_benchmark_model_matches_prefill_without_a_decode_alias(tmp_path: Path) -> None:
    data = _recipe()
    data["roles"]["prefill"]["args"]["served-model-name"] = "org/model-alias"
    data["benchmark"] = {"type": "sa-bench", "isl": 128, "osl": 128, "concurrencies": "1"}
    config = _load(data)
    orchestrator = _orchestrator(config, tmp_path, ("p0", "d0"))
    command = SABenchRunner().build_command(config, orchestrator.runtime)
    assert command[8] == "org/model-alias"


def test_conflicting_decode_model_alias_is_rejected():
    data = _recipe()
    data["roles"]["decode"]["engine"] = {"type": "tilert", "served_model_name": "wrong-model"}
    with pytest.raises(ValidationError, match="must match the prefill"):
        _load(data)


@pytest.mark.parametrize(
    "prefill,decode,frontend,message",
    [
        ("sglang", "tilert", "tilert-router", "requires vLLM prefill"),
        ("vllm", "vllm", "tilert-router", "requires a TileRT decode engine"),
        ("vllm", "tilert", "vllm-router", "requires backend.type: vllm"),
        ("vllm", "tilert", "dynamo", "do not yet support the Dynamo frontend"),
    ],
)
def test_unsupported_engine_frontend_pairing_fails_at_load(prefill, decode, frontend, message):
    data = _recipe()
    data["roles"]["prefill"]["engine"] = prefill
    data["roles"]["decode"]["engine"] = decode
    data["frontend"]["type"] = frontend
    with pytest.raises(ValidationError, match=message):
        _load(data)


def test_decode_cannot_override_allocated_listener(tmp_path: Path) -> None:
    data = _recipe()
    data["roles"]["decode"]["args"]["ctrl-port"] = 8000
    config = _load(data)
    orchestrator = _orchestrator(config, tmp_path, ("p0", "d0"))
    process = orchestrator.backend_processes[-1]
    with pytest.raises(ValueError, match="owned by srtctl"):
        config.backend_for_role("decode").build_worker_command(process, [process], orchestrator.runtime)


def test_router_setup_runs_before_launch_and_failure_stops_launch(tmp_path: Path) -> None:
    setup = tmp_path / "router setup.sh"
    setup.write_text('printf "installed\\n"\nexit "${SETUP_RC:-0}"\n')
    data = _recipe()
    data["setup_script"] = str(setup)
    preamble = TileRTRouterFrontend().build_bash_preamble(_load(data))
    command = f'{preamble} && printf "launched\\n"'
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert result.stdout == "installed\nlaunched\n"
    failed = subprocess.run(
        ["bash", "-c", f"export SETUP_RC=7; {command}"], capture_output=True, text=True, check=False
    )
    assert failed.returncode == 7
    assert failed.stdout == "installed\n"
