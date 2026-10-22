# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the EGM example through config rendering and the real orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console

from srtctl.backends import TRTLLMBackend
from srtctl.cli import submit
from srtctl.cli.submit import generate_minimal_sbatch_script, show_config_details
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "examples/trtllm/dynamo-disagg-mooncake-egm.yaml"


@pytest.fixture
def recipe_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Replace unavailable model paths and ignore developer cluster defaults."""
    monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "absent-srtslurm.yaml"))
    monkeypatch.setenv("SRTCTL_SOURCE_DIR", str(ROOT))
    recipe = yaml.safe_load(RECIPE.read_text())
    model = tmp_path / "model"
    model.mkdir()
    recipe["model"].update(path=str(model), container="trtllm-egm.sqsh")
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return path


def test_client_capacity_and_fabric_allocation_are_visible_in_dry_run(
    recipe_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SrtConfig.from_yaml(recipe_path)
    assert config.resources.gpu_type == "vrnvl72"
    assert config.resources.gpus_per_node == 4
    backend = config.backend
    assert isinstance(backend, TRTLLMBackend)
    configs = backend.mooncake_store_configs("192.0.2.1", "Qwen/Qwen3-0.6B")
    assert len(configs) == 2
    for payload in configs.values():
        assert payload["master_server_address"] == "192.0.2.1:8700"
        assert payload["protocol"] == "nvlink"
        assert payload["global_segment_size"] == 0
        assert payload["local_buffer_size"] == "64MiB"
        assert payload["stage_through_host"] is True
        assert payload["staging_buffer_bytes"] == "256MiB"
    script = generate_minimal_sbatch_script(config=config, config_path=recipe_path)
    assert "#SBATCH --nodes=3\n" in script
    assert "#SBATCH --segment=3\n" in script
    assert "#SBATCH --exclusive\n" in script
    assert "#SBATCH --mem=0\n" in script
    assert "#SBATCH --time=00:30:00\n" in script
    monkeypatch.setattr(submit, "console", Console(width=180))
    show_config_details(config)
    output = capsys.readouterr().out
    assert '"protocol": "nvlink"' in output
    assert '"stage_through_host": true' in output
    assert "/mooncake_egm_wheels" in output
    assert "mooncake_egm_donor.py" in output
    assert "MC_USE_NVLINK_IPC" not in output


def test_mock_launch_prepares_pools_before_clients_and_renders_both_client_configs(
    recipe_path: Path, tmp_path: Path
) -> None:
    config = SrtConfig.from_yaml(recipe_path)
    services = {service.name: service for service in config.services}
    assert services["mooncake-master"].effective_start == "infra"
    assert services["mooncake-egm-donor"].effective_start == "before_workers"
    assert services["mooncake-egm-donor"].readiness.file.path == "mooncake_egm_{index}.ready"
    launches: list[dict[str, Any]] = []
    output_dir = tmp_path / "outputs" / "6201"
    assert (
        run_mock_sweep(
            config_path=recipe_path,
            output_dir=output_dir,
            job_id="6201",
            options=MockOptions(
                child_duration_s=0.1,
                phase_pause_s=0.01,
                nodelist=("node-00", "node-01", "node-02"),
                on_srun=launches.append,
            ),
        )
        == 0
    )
    master = next(launch for launch in launches if launch.get("step_name") == "service_mooncake-master")
    donors = [launch for launch in launches if launch.get("step_name", "").startswith("service_mooncake-egm-donor")]
    workers = [launch for launch in launches if launch.get("step_name", "").startswith(("prefill_", "decode_"))]
    assert len(donors) == 2
    assert len({launch["nodelist"][0] for launch in donors}) == 2
    assert len(workers) == 2
    assert launches.index(master) < min(launches.index(launch) for launch in donors)
    assert max(launches.index(launch) for launch in donors) < min(launches.index(launch) for launch in workers)
    assert "mooncake.cli" in " ".join(master["command"])
    for launch in [master, *donors]:
        assert "bash /configs/setup_mooncake_egm.sh --mooncake-only" in launch["bash_preamble"]
        assert launch["env_to_set"]["SRT_MOONCAKE_EGM_WHEEL_DIR"] == "/mooncake_egm_wheels"
    for launch in donors:
        command = " ".join(launch["command"])
        assert "/configs/mooncake_egm_donor.py serve" in command
        assert "--segment-size 4GiB" in command
        assert "--numa-nodes auto" in command
        assert "--config /logs/mooncake_store_config_prefill.json" in command
        assert "{index}" not in command
        assert launch["env_to_set"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    for launch in [*donors, *workers]:
        env = launch["env_to_set"]
        assert env["MC_MS_AUTO_DISC"] == "0"
        assert env["MC_USE_TENT"] == "0"
        assert env["MC_USE_TEV1"] == "0"
        assert env["MOONCAKE_STORE_BACKEND"] == "cpp"
        assert "MC_USE_NVLINK_IPC" not in env
        assert "MC_TE_FILTERS" not in env
    for launch in workers:
        assert "setup_mooncake_egm.sh" in launch["bash_preamble"]
        assert launch["env_to_set"]["MOONCAKE_CONFIG_PATH"].startswith("/logs/mooncake_store_config_")
    paths = list(output_dir.rglob("mooncake_store_config_*.json"))
    assert {path.name for path in paths} == {"mooncake_store_config_prefill.json", "mooncake_store_config_decode.json"}
    for path in paths:
        payload = json.loads(path.read_text())
        assert payload["global_segment_size"] == 0
        assert payload["stage_through_host"] is True
        assert payload["protocol"] == "nvlink"
