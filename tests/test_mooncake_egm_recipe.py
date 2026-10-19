# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the experimental EGM recipe's launch wiring without Slurm, CUDA or NVLink."""

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
RECIPE = ROOT / "recipes/trtllm/vrnvl72/agentic/raplab-dynamo/glm5.3-disagg-3p1d-tp4-dep8-c560-egm.yaml"


@pytest.fixture
def recipe_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep local cluster defaults and unavailable cluster paths out of launch validation."""
    monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "absent-srtslurm.yaml"))
    monkeypatch.setenv("SRTCTL_SOURCE_DIR", str(ROOT))
    recipe = yaml.safe_load(RECIPE.read_text())
    model = tmp_path / "model"
    model.mkdir()
    recipe["model"].update(path=str(model), container="trtllm-egm.sqsh")
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return path


def test_store_config_selects_direct_gpu_reads_without_an_ordinary_host_pool(recipe_path: Path) -> None:
    """The generated connector JSON must select NVLink and avoid allocating an RDMA donor segment."""
    config = SrtConfig.from_yaml(recipe_path)
    backend = config.backend
    assert isinstance(backend, TRTLLMBackend)
    assert backend.mooncake_store_modes() == ("prefill",)
    (payload,) = backend.mooncake_store_configs("10.0.0.1", "GLM-5.3").values()
    assert payload["protocol"] == "nvlink"
    assert payload["device_name"] == ""
    assert payload["global_segment_size"] == 0
    assert payload["local_buffer_size"] == "64MiB"
    assert payload["stage_through_host"] is False
    assert payload["transfer_batch_size"] == 512
    assert payload["master_server_address"] == "10.0.0.1:8700"
    assert config.roles["prefill"].args["kv_connector_config"]["connector"] == "mooncake-store"


def test_allocation_keeps_egm_participants_in_one_segment_with_all_memory(recipe_path: Path) -> None:
    config = SrtConfig.from_yaml(recipe_path)
    script = generate_minimal_sbatch_script(config=config, config_path=recipe_path)
    directives = [line for line in script.splitlines() if line.startswith("#SBATCH ")]
    assert "#SBATCH --nodes=6" in directives
    assert "#SBATCH --segment=6" in directives
    assert "#SBATCH --exclusive" in directives
    assert "#SBATCH --mem=0" in directives
    assert "#SBATCH --account=root" in directives
    assert "#SBATCH --partition=batch-r42" in directives
    assert "#SBATCH --time=02:00:00" in directives
    assert config.resources.het_jobs is False
    assert config.setup_script == "setup_mooncake_egm.sh"


def test_dry_run_exposes_nvlink_and_both_egm_donor_roles(
    recipe_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(submit, "console", Console(width=180))
    show_config_details(SrtConfig.from_yaml(recipe_path))
    output = capsys.readouterr().out
    assert '"protocol": "nvlink"' in output
    assert '"stage_through_host": false' in output
    assert "prefill-egm-donor" in output
    assert "decode-mooncake-donor" in output
    assert "mooncake_egm_donor.py" in output
    assert "MC_ENABLE_DEST_DEVICE_AFFINITY" not in output
    assert "MC_TE_FILTERS" not in output


def test_mock_launch_bootstraps_each_donor_and_renders_the_direct_read_config(
    recipe_path: Path, tmp_path: Path
) -> None:
    """Generic services need their own setup, CUDA visibility and transport environment."""
    config = SrtConfig.from_yaml(recipe_path)
    services = {service.name: service for service in config.services}
    # These phases also keep providers alive until GPU clients stop, and keep
    # the master alive until provider cleanup finishes during shutdown.
    assert services["mooncake-master"].effective_start == "infra"
    assert services["prefill-egm-donor"].effective_start == "before_workers"
    assert services["decode-mooncake-donor"].effective_start == "before_workers"
    launches: list[dict[str, Any]] = []
    output_dir = tmp_path / "outputs" / "6201"
    exit_code = run_mock_sweep(
        config_path=recipe_path,
        output_dir=output_dir,
        job_id="6201",
        options=MockOptions(
            child_duration_s=0.1,
            phase_pause_s=0.01,
            nodelist=tuple(f"node-{index:02d}" for index in range(6)),
            on_srun=launches.append,
        ),
    )
    assert exit_code == 0
    donor_launches = [
        launch
        for launch in launches
        if launch.get("step_name", "").startswith(("service_prefill-egm-donor", "service_decode-mooncake-donor"))
    ]
    assert len(donor_launches) == 5
    assert len({launch["nodelist"][0] for launch in donor_launches}) == 5
    prefill_donors = [
        launch for launch in donor_launches if launch["step_name"].startswith("service_prefill-egm-donor")
    ]
    decode_donors = [
        launch for launch in donor_launches if launch["step_name"].startswith("service_decode-mooncake-donor")
    ]
    # Four former 260GiB worker pools become one 1040GiB EGM pool per prefill node.
    assert len(prefill_donors) == 3
    assert len(decode_donors) == 2
    assert all("--segment-size 1040GiB" in " ".join(launch["command"]) for launch in prefill_donors)
    assert all("--segment-size 1100GiB" in " ".join(launch["command"]) for launch in decode_donors)
    for launch in donor_launches:
        command = " ".join(launch["command"])
        assert "/configs/mooncake_egm_donor.py" in command
        assert "--config /logs/mooncake_store_config_prefill.json" in command
        assert "--mooncake-only" in launch["bash_preamble"]
        assert "/configs/setup_mooncake_egm.sh" in launch["bash_preamble"]
        assert launch["env_to_set"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert launch["env_to_set"]["MC_MS_AUTO_DISC"] == "0"
        assert launch["env_to_set"]["MC_USE_TENT"] == "0"
        assert launch["env_to_set"]["MC_USE_TEV1"] == "0"
        assert launch["env_to_set"]["MOONCAKE_STORE_BACKEND"] == "cpp"
        assert "MC_ENABLE_DEST_DEVICE_AFFINITY" not in launch["env_to_set"]
        assert "MC_TE_FILTERS" not in launch["env_to_set"]
        assert Path("/configs") in launch["container_mounts"].values()
    master = next(launch for launch in launches if launch.get("step_name", "") == "service_mooncake-master")
    assert "/configs/setup_mooncake_egm.sh" in master["bash_preamble"]
    assert "--mooncake-only" in master["bash_preamble"]
    workers = [launch for launch in launches if launch.get("step_name", "").startswith(("prefill_", "decode_"))]
    assert workers
    assert launches.index(master) < min(launches.index(launch) for launch in donor_launches)
    assert max(launches.index(launch) for launch in donor_launches) < min(launches.index(launch) for launch in workers)
    assert all("setup_mooncake_egm.sh" in launch["bash_preamble"] for launch in workers)
    prefill_workers = [launch for launch in workers if launch["step_name"].startswith("prefill_")]
    assert prefill_workers
    for launch in prefill_workers:
        assert launch["env_to_set"]["MC_MS_AUTO_DISC"] == "0"
        assert launch["env_to_set"]["MC_USE_TENT"] == "0"
        assert launch["env_to_set"]["MC_USE_TEV1"] == "0"
        assert launch["env_to_set"]["MOONCAKE_STORE_BACKEND"] == "cpp"
        assert "MC_ENABLE_DEST_DEVICE_AFFINITY" not in launch["env_to_set"]
        assert "MC_TE_FILTERS" not in launch["env_to_set"]
    config_paths = list(output_dir.rglob("mooncake_store_config_prefill.json"))
    assert len(config_paths) == 1
    payload = json.loads(config_paths[0].read_text())
    assert payload["protocol"] == "nvlink"
    assert payload["stage_through_host"] is False
    assert payload["global_segment_size"] == 0
