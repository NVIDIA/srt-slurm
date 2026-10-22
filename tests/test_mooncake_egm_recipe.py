# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the experimental EGM recipes' launch wiring without Slurm, CUDA or NVLink."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console

from configs import mooncake_egm_donor as donor
from srtctl.backends import TRTLLMBackend
from srtctl.cli import submit
from srtctl.cli.submit import generate_minimal_sbatch_script, show_config_details
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep

ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = ROOT / "recipes/trtllm/vrnvl72/agentic"
RECIPE_NAME = "glm5.3-disagg-3p1d-tp4-dep8-c560-egm.yaml"


@pytest.fixture(params=["raplab-dynamo", "dynamo"])
def recipe_source(request: pytest.FixtureRequest) -> Path:
    return RECIPE_DIR / request.param / RECIPE_NAME


@pytest.fixture
def recipe_path(recipe_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep local cluster defaults and unavailable cluster paths out of launch validation."""
    monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "absent-srtslurm.yaml"))
    monkeypatch.setenv("SRTCTL_SOURCE_DIR", str(ROOT))
    recipe = yaml.safe_load(recipe_source.read_text())
    model = tmp_path / "model"
    model.mkdir()
    recipe["model"].update(path=str(model), container="trtllm-egm.sqsh")
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return path


def test_store_config_selects_host_staging_and_external_egm_pools(recipe_path: Path) -> None:
    """The connector must register CPU slots while providers own all remote cache capacity."""
    config = SrtConfig.from_yaml(recipe_path)
    backend = config.backend
    assert isinstance(backend, TRTLLMBackend)
    assert backend.mooncake_store_modes() == ("prefill",)
    (payload,) = backend.mooncake_store_configs("10.0.0.1", "GLM-5.3").values()
    assert payload["protocol"] == "nvlink"
    assert payload["device_name"] == ""
    assert payload["global_segment_size"] == 0
    assert payload["local_buffer_size"] == "64MiB"
    assert payload["stage_through_host"] is True
    assert payload["staging_buffer_bytes"] == "2GiB"
    assert payload["transfer_batch_size"] == 512
    assert payload["master_server_address"] == "10.0.0.1:8700"
    assert config.roles["prefill"].args["kv_connector_config"]["connector"] == "mooncake-store"


def test_allocation_keeps_egm_participants_in_one_segment_with_all_memory(
    recipe_source: Path, recipe_path: Path
) -> None:
    config = SrtConfig.from_yaml(recipe_path)
    script = generate_minimal_sbatch_script(config=config, config_path=recipe_path)
    directives = [line for line in script.splitlines() if line.startswith("#SBATCH ")]
    assert "#SBATCH --nodes=6" in directives
    assert "#SBATCH --segment=6" in directives
    assert "#SBATCH --exclusive" in directives
    assert "#SBATCH --mem=0" in directives
    if recipe_source.parent.name == "raplab-dynamo":
        assert "#SBATCH --account=root" in directives
        assert "#SBATCH --partition=batch-r42" in directives
    else:
        assert config.slurm.account is None
        assert config.slurm.partition is None
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
    assert '"stage_through_host": true' in output
    assert '"staging_buffer_bytes": "2GiB"' in output
    assert "prefill-egm-donor" in output
    assert "decode-mooncake-donor" in output
    assert "mooncake_egm_donor.py" in output
    assert "MC_ENABLE_DEST_DEVICE_AFFINITY" not in output
    assert "MC_TE_FILTERS" not in output


def test_mock_launch_bootstraps_each_donor_and_renders_the_host_staging_config(
    recipe_source: Path, recipe_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    decode_size = "1100GiB" if recipe_source.parent.name == "raplab-dynamo" else "1000GiB"
    assert all(f"--segment-size {decode_size}" in " ".join(launch["command"]) for launch in decode_donors)
    recipe = yaml.safe_load(recipe_source.read_text())
    wheel_dir = recipe["environment"]["SRT_MOONCAKE_EGM_WHEEL_DIR"]
    for launch in donor_launches:
        command = " ".join(launch["command"])
        assert "/configs/mooncake_egm_donor.py" in command
        assert "--config /logs/mooncake_store_config_prefill.json" in command
        assert "--numa-nodes auto" in command
        assert "--mooncake-only" in launch["bash_preamble"]
        assert "/configs/setup_mooncake_egm.sh" in launch["bash_preamble"]
        assert launch["env_to_set"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert launch["env_to_set"]["MC_MS_AUTO_DISC"] == "0"
        assert launch["env_to_set"]["MC_USE_TENT"] == "0"
        assert launch["env_to_set"]["MC_USE_TEV1"] == "0"
        assert launch["env_to_set"]["MOONCAKE_STORE_BACKEND"] == "cpp"
        assert launch["env_to_set"]["SRT_MOONCAKE_EGM_WHEEL_DIR"] == wheel_dir
        assert launch["reuse_container"] == recipe.get("reuse_container", False)
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
    assert payload["stage_through_host"] is True
    assert payload["staging_buffer_bytes"] == "2GiB"
    assert payload["global_segment_size"] == 0
    monkeypatch.setattr(donor.socket, "gethostbyname", lambda hostname: "192.0.2.2")
    provider = donor.provider_config(config_paths[0], 1040 * 1024**3, "auto")
    assert provider["protocol"] == "nvlink"
    assert provider["global_segment_size"] == str(1040 * 1024**3)
    assert provider["local_buffer_size"] == "0"
    assert provider["enable_egm_store_pool"] == "true"
    assert provider["egm_numa_nodes"] == "auto"


def test_egm_variant_preserves_the_requested_baseline(recipe_source: Path) -> None:
    """Changing Store transport must preserve the workload and this cluster's launch settings."""
    baseline_path = recipe_source.with_name("glm5.3-disagg-3p1d-tp4-dep8-c560.yaml")
    baseline = yaml.safe_load(baseline_path.read_text())
    variant = yaml.safe_load(recipe_source.read_text())
    for key in (
        "model",
        "resources",
        "host_setup",
        "reuse_container",
        "observability",
        "dynamo",
        "frontend",
        "extra_mount",
        "benchmark",
    ):
        assert variant.get(key) == baseline.get(key), key
    assert variant["engine"] == {**baseline["engine"], "numa_cpu_bind": True, "numa_memory_bind": "local"}
    baseline_store = next(service for service in baseline["services"] if service["type"] == "mooncake-master")
    variant_store = next(service for service in variant["services"] if service["type"] == "mooncake-master")
    for key in ("stage_through_host", "staging_buffer_bytes"):
        assert variant_store["options"]["store_config"][key] == baseline_store["options"]["store_config"][key]
    for role, settings in baseline["roles"].items():
        assert variant["roles"][role]["args"] == settings["args"]
        for key, value in settings["env"].items():
            # RDMA registration and NIC selection do not apply to the NVLink variant.
            if key not in ("MC_ENABLE_PARALLEL_REG_MR", "MC_TE_FILTERS", "MC_ENABLE_DEST_DEVICE_AFFINITY"):
                assert variant["roles"][role]["env"][key] == value, (role, key)
        for key in ("nodes", "workers", "gpus"):
            assert variant["roles"][role][key] == settings[key]
    baseline_disabled = [service for service in baseline["services"] if service.get("enabled") is False]
    assert [service for service in variant["services"] if service.get("enabled") is False] == baseline_disabled
    wheel_dir = variant["environment"]["SRT_MOONCAKE_EGM_WHEEL_DIR"]
    assert Path(wheel_dir).is_absolute()
    for service in variant["services"]:
        if service.get("enabled") is not False:
            assert service["env"]["SRT_MOONCAKE_EGM_WHEEL_DIR"] == wheel_dir
