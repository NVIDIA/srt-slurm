# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the optimized RapLab GLM-5.3 FP8 EGM load point without a cluster."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console

from srtctl.cli import submit
from srtctl.cli.submit import generate_minimal_sbatch_script, planned_total_nodes, show_config_details
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/trtllm/glm5.3/vrnvl72-lab-w2-fp8/glm5.3-disagg-5p1d-tp4-dep16-c620-egm-4x256.yaml"
IMAGE = "/home/rihuo/tensorrt-llm-arm64-feat_rubin-glm5-mega-30454-mooncake-egm-optimization.sqsh"
MOCK_IMAGE = "trtllm-egm-optimized.sqsh"
NATIVE_EGM_ENV = {
    "TRTLLM_M31_STORE_EGM": "1",
    "TRTLLM_M31_DISABLE_EGM_DIRECT": "0",
    "TRTLLM_M31_EGM_DIRECT_SAVE": "1",
    "TRTLLM_M31_EGM_GROUPED_LOADS": "1",
    "TRTLLM_M31_EGM_GROUPED_SAVES": "1",
    "TRTLLM_M31_EGM_LOAD_STREAMS": "2",
    "TRTLLM_M31_EGM_PRIVATE_LOOP_STREAM": "1",
}


@pytest.fixture
def recipe_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "absent-srtslurm.yaml"))
    monkeypatch.setenv("SRTCTL_SOURCE_DIR", str(ROOT))
    recipe = yaml.safe_load(RECIPE.read_text())
    assert recipe["model"]["container"] == IMAGE
    model = tmp_path / "model"
    model.mkdir()
    recipe["model"].update(path=str(model), container=MOCK_IMAGE)
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return path


def test_optimized_egm_transport_is_visible_without_changing_the_load_point(
    recipe_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SrtConfig.from_yaml(recipe_path)
    prefill, decode = config.roles["prefill"], config.roles["decode"]
    assert (prefill.nodes, prefill.workers, prefill.gpus) == (5, 5, 4)
    assert (decode.nodes, decode.workers, decode.gpus) == (4, 1, 16)
    assert prefill.args["tensor_parallel_size"] == 4
    assert decode.args["tensor_parallel_size"] == 16
    assert prefill.args["kv_cache_config"]["dtype"] == decode.args["kv_cache_config"]["dtype"] == "fp8"
    assert planned_total_nodes(config) == 10
    assert config.reuse_container
    recipe = yaml.safe_load(recipe_path.read_text())
    assert recipe["benchmark"]["env"]["CONC"] == "620"
    assert "--benchmark-duration 3600" in recipe["benchmark"]["command"]
    assert recipe["environment"]["ENROOT_MAX_PROCESSORS"] == "8"
    assert "host_setup" not in recipe
    for role in (prefill, decode):
        assert "store_host_donor" not in role.args
        assert "sparse_disable_index_value" not in role.args["sparse_attention_config"]
    script = generate_minimal_sbatch_script(config=config, config_path=recipe_path)
    assert "#SBATCH --nodes=10\n" in script
    assert "#SBATCH --segment=10\n" in script
    assert "#SBATCH --exclusive\n" in script
    assert "#SBATCH --mem=0\n" in script
    monkeypatch.setattr(submit, "console", Console(width=240, color_system=None))
    show_config_details(config)
    output = capsys.readouterr().out
    assert config.model.container == MOCK_IMAGE
    assert all(name in output for name in NATIVE_EGM_ENV)
    assert '"protocol": "nvlink"' in output
    assert '"stage_through_host": false' in output


def test_native_transport_and_sf4_master_reach_launches_with_fixed_provider_capacity(
    recipe_path: Path, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    output = tmp_path / "output"
    assert (
        run_mock_sweep(
            config_path=recipe_path,
            output_dir=output,
            job_id="6208",
            options=MockOptions(
                child_duration_s=0.1,
                phase_pause_s=0,
                nodelist=tuple(f"node-{index:02}" for index in range(10)),
                on_srun=calls.append,
            ),
        )
        == 0
    )
    master = next(call for call in calls if call.get("step_name") == "service_mooncake-master")
    assert master["command"][0] == "mooncake_master"
    assert any(arg.startswith("--rpc_port=") for arg in master["command"])
    assert "--default_kv_lease_ttl=300000" in master["command"]
    assert "--eviction_ratio=0.02" in master["command"]
    assert "--allocation_strategy=random" in master["command"]
    providers = [call for call in calls if "/configs/mooncake_egm_donor.py" in call["command"]]
    workers = [call for call in calls if call.get("step_name", "").startswith(("prefill_", "decode_"))]
    prefill_workers = [call for call in workers if call["step_name"].startswith("prefill_")]
    assert len(prefill_workers) == 5 and len(workers) == 6
    assert len(providers) == 24
    capacities = Counter(int(call["command"][call["command"].index("--segment-size") + 1][:-3]) for call in providers)
    assert capacities == {256: 20, 1050: 4}
    assert sum(size * count for size, count in capacities.items()) == 9320
    assert calls.index(master) < min(calls.index(call) for call in providers)
    assert max(calls.index(call) for call in providers) < min(calls.index(call) for call in workers)
    for call in prefill_workers:
        env = call["env_to_set"]
        assert all(env[name] == value for name, value in NATIVE_EGM_ENV.items())
        assert env["MOONCAKE_CONFIG_PATH"] == "/logs/mooncake_store_config_prefill.json"
    frontend = next(call for call in calls if call.get("step_name") == "frontend_0")
    assert all(call["container_image"] == MOCK_IMAGE for call in [master, *providers, *workers, frontend])
    for call in [master, *providers, *workers]:
        assert "setup_mooncake_egm.sh" not in (call.get("bash_preamble") or "")
        assert "SRT_MOONCAKE_EGM_WHEEL_DIR" not in call["env_to_set"]
    store_files = list(output.rglob("mooncake_store_config_*.json"))
    assert len(store_files) == 1 and store_files[0].name == "mooncake_store_config_prefill.json"
    store = json.loads(store_files[0].read_text())
    assert store["protocol"] == "nvlink"
    assert store["global_segment_size"] == 0
    assert store["stage_through_host"] is False
    assert store["transfer_batch_size"] == 512
    assert store["async_load"] is True and store["async_load_workers"] == 4
    engine_files = list(output.rglob("trtllm_config_*.yaml"))
    assert {path.name for path in engine_files} == {"trtllm_config_prefill.yaml", "trtllm_config_decode.yaml"}
    for path in engine_files:
        engine = yaml.safe_load(path.read_text())
        assert "store_host_donor" not in engine
        assert engine["kv_cache_config"]["dtype"] == "fp8"
        assert "sparse_disable_index_value" not in engine["sparse_attention_config"]
