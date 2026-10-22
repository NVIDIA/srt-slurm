# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the four-provider EGM comparison recipes without Slurm or CUDA."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console

from configs import mooncake_egm_donor as donor
from srtctl.cli import submit
from srtctl.cli.mixins.service_stage import ServiceStageMixin
from srtctl.cli.submit import generate_minimal_sbatch_script, show_config_details
from srtctl.core.processes import ManagedProcess
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep
from srtctl.services.config import ServiceConfig, ServiceReadinessConfig

ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = ROOT / "recipes/trtllm/vrnvl72/agentic"
BASE_NAME = "glm5.3-disagg-3p1d-tp4-dep8-c560-egm.yaml"
FOUR_POOL_NAME = "glm5.3-disagg-3p1d-tp4-dep8-c560-egm-4x260.yaml"
PREFILL_NAMES = tuple(f"prefill-egm-donor-{pool}" for pool in range(4))
PREFILL_NUMA_NODES = ("0", "0", "1", "1")


@pytest.fixture(params=["dynamo", "raplab-dynamo"])
def recipe_source(request: pytest.FixtureRequest) -> Path:
    return RECIPE_DIR / request.param / FOUR_POOL_NAME


@pytest.fixture
def prefill_size(recipe_source: Path) -> str:
    return {"dynamo": "224GiB", "raplab-dynamo": "250GiB"}[recipe_source.parent.name]


@pytest.fixture
def recipe_path(recipe_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Resolve the actual recipe against portable model and cluster paths."""
    monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "absent-srtslurm.yaml"))
    monkeypatch.setenv("SRTCTL_SOURCE_DIR", str(ROOT))
    monkeypatch.delenv("SLURM_ACCOUNT", raising=False)
    monkeypatch.delenv("SLURM_PARTITION", raising=False)
    recipe = yaml.safe_load(recipe_source.read_text())
    model = tmp_path / "model"
    model.mkdir()
    recipe["model"].update(path=str(model), container="trtllm-egm.sqsh")
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return path


def test_four_pool_variant_preserves_the_egm_comparison_baseline(recipe_source: Path, prefill_size: str) -> None:
    """Match worker settings while RapLab leaves scheduling overrides to the caller."""
    base = yaml.safe_load(recipe_source.with_name(BASE_NAME).read_text())
    variant = yaml.safe_load(recipe_source.read_text())
    if recipe_source.parent.name == "raplab-dynamo":
        assert "slurm" not in variant
        variant["sbatch_directives"].pop("exclude", None)
        assert variant["sbatch_directives"] == {"exclusive": "", "mem": "0"}
        base.pop("slurm", None)
        base["sbatch_directives"].pop("segment", None)
    assert variant["name"] == "glm5.3-disagg-3p1d-tp4-dep8-c560-egm-4x260"
    assert {key: value for key, value in variant.items() if key not in ("name", "services")} == {
        key: value for key, value in base.items() if key not in ("name", "services")
    }
    base_services = {service["name"]: service for service in base["services"]}
    variant_services = {service["name"]: service for service in variant["services"]}
    assert len(variant_services) == len(variant["services"]), "Every service needs its own name and log stem"
    assert set(variant_services) == (set(base_services) - {"prefill-egm-donor"}) | set(PREFILL_NAMES)
    for name, service in base_services.items():
        if name != "prefill-egm-donor":
            assert variant_services[name] == service, name
    for pool, name in enumerate(PREFILL_NAMES):
        service = variant_services[name]
        normalized = dict(service)
        normalized["name"] = "prefill-egm-donor"
        command = list(service["command"])
        assert command[command.index("--segment-size") + 1] == prefill_size
        command[command.index("--segment-size") + 1] = "1040GiB"
        assert command[command.index("--numa-nodes") + 1] == PREFILL_NUMA_NODES[pool]
        command[command.index("--numa-nodes") + 1] = "auto"
        assert command[command.index("--ready-file") + 1] == f"/logs/mooncake_egm_prefill_pool{pool}_{{index}}.ready"
        command[command.index("--ready-file") + 1] = "/logs/mooncake_egm_prefill_{index}.ready"
        normalized["command"] = command
        assert service["readiness"]["file"]["path"] == f"mooncake_egm_prefill_pool{pool}_{{index}}.ready"
        normalized["readiness"] = {
            **service["readiness"],
            "file": {"path": "mooncake_egm_prefill_{index}.ready"},
        }
        assert service["env"]["CUDA_VISIBLE_DEVICES"] == str(pool)
        normalized["env"] = {**service["env"], "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
        assert normalized == base_services["prefill-egm-donor"], name


def test_dry_run_exposes_all_four_prefill_pool_fleets(
    recipe_source: Path, recipe_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(submit, "console", Console(width=180))
    config = SrtConfig.from_yaml(recipe_path)
    show_config_details(config)
    output = capsys.readouterr().out
    for name in PREFILL_NAMES:
        assert name in output
    assert "decode-mooncake-donor" in output
    assert '"protocol": "nvlink"' in output
    assert '"global_segment_size": 0' in output
    assert '"stage_through_host": true' in output
    assert '"staging_buffer_bytes": "2GiB"' in output
    script = generate_minimal_sbatch_script(config=config, config_path=recipe_path)
    directives = [line for line in script.splitlines() if line.startswith("#SBATCH ")]
    assert "#SBATCH --nodes=6" in directives
    assert "#SBATCH --segment=6" in directives
    assert "#SBATCH --exclusive" in directives
    assert "#SBATCH --mem=0" in directives
    assert config.slurm.account is None
    assert config.slurm.partition is None
    assert "#SBATCH --account=default" in directives
    assert "#SBATCH --partition=default" in directives
    if recipe_source.parent.name == "raplab-dynamo":
        assert config.slurm.time_limit is None
        assert {key: config.sbatch_directives[key] for key in ("exclusive", "mem")} == {"exclusive": "", "mem": "0"}
        assert "#SBATCH --time=01:00:00" in directives
    else:
        assert "#SBATCH --time=02:00:00" in directives


def test_mock_launch_gates_workers_on_four_host_pools_per_prefill_node(
    recipe_source: Path, recipe_path: Path, prefill_size: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each provider maps its host pool through one GPU and has disjoint readiness/log files."""
    config = SrtConfig.from_yaml(recipe_path)
    services = {service.name: service for service in config.services}
    assert services["mooncake-master"].effective_start == "infra"
    for name in (*PREFILL_NAMES, "decode-mooncake-donor"):
        assert services[name].effective_start == "before_workers"
        assert services[name].effective_critical is True

    launches: list[dict[str, Any]] = []
    events: list[tuple[str, str]] = []
    probes: dict[str, ServiceReadinessConfig] = {}
    original_wait_ready = ServiceStageMixin._wait_ready

    def record_launch(launch: dict[str, Any]) -> None:
        launches.append(launch)
        events.append(("launch", launch.get("step_name", "")))

    def record_readiness(process: ManagedProcess, service: ServiceConfig, readiness: ServiceReadinessConfig) -> None:
        original_wait_ready(process, service, readiness)
        probes[process.name] = readiness
        events.append(("ready", process.name))

    monkeypatch.setattr(ServiceStageMixin, "_wait_ready", staticmethod(record_readiness))
    output_dir = tmp_path / "outputs" / "6202"
    assert (
        run_mock_sweep(
            config_path=recipe_path,
            output_dir=output_dir,
            job_id="6202",
            options=MockOptions(
                child_duration_s=0.1,
                phase_pause_s=0.01,
                nodelist=tuple(f"node-{index:02d}" for index in range(6)),
                on_srun=record_launch,
            ),
        )
        == 0
    )
    prefill = [launch for launch in launches if launch.get("step_name", "").startswith("service_prefill-egm-donor-")]
    decode = [launch for launch in launches if launch.get("step_name", "").startswith("service_decode-mooncake-donor")]
    assert len(prefill) == 12
    assert len(decode) == 2
    prefill_nodes = Counter(launch["nodelist"][0] for launch in prefill)
    assert sorted(prefill_nodes.values()) == [4, 4, 4]
    assert len({launch["nodelist"][0] for launch in decode}) == 2
    assert not set(prefill_nodes) & {launch["nodelist"][0] for launch in decode}
    for pool, name in enumerate(PREFILL_NAMES):
        fleet = [launch for launch in prefill if launch["step_name"].startswith(f"service_{name}_")]
        assert len(fleet) == 3
        assert {launch["nodelist"][0] for launch in fleet} == set(prefill_nodes)
        assert {launch["command"][launch["command"].index("--ready-file") + 1] for launch in fleet} == {
            f"/logs/mooncake_egm_prefill_pool{pool}_{index}.ready" for index in range(3)
        }
        assert {launch["command"][launch["command"].index("--numa-nodes") + 1] for launch in fleet} == {
            PREFILL_NUMA_NODES[pool]
        }
        assert {launch["env_to_set"]["CUDA_VISIBLE_DEVICES"] for launch in fleet} == {str(pool)}
    for node in prefill_nodes:
        assert Counter(
            launch["command"][launch["command"].index("--numa-nodes") + 1]
            for launch in prefill
            if launch["nodelist"][0] == node
        ) == {"0": 2, "1": 2}
    donors = prefill + decode
    assert len({launch["output"] for launch in donors}) == 14
    markers: set[str] = set()
    total_capacity = 0
    baseline = yaml.safe_load(recipe_source.with_name(BASE_NAME).read_text())
    decode_command = next(
        service["command"] for service in baseline["services"] if service["name"] == "decode-mooncake-donor"
    )
    decode_size = decode_command[decode_command.index("--segment-size") + 1]
    for launch in donors:
        command = launch["command"]
        expected_size = prefill_size if launch in prefill else decode_size
        assert command[command.index("--segment-size") + 1] == expected_size
        total_capacity += donor.parse_size(expected_size)
        assert "/configs/mooncake_egm_donor.py" in command
        assert command[command.index("--config") + 1] == "/logs/mooncake_store_config_prefill.json"
        if launch in decode:
            assert command[command.index("--numa-nodes") + 1] == "auto"
        assert "--mooncake-only" in launch["bash_preamble"]
        assert "/configs/setup_mooncake_egm.sh" in launch["bash_preamble"]
        env = launch["env_to_set"]
        if launch in decode:
            assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert env["MC_USE_TENT"] == "0"
        assert env["MC_USE_TEV1"] == "0"
        assert env["MC_MS_AUTO_DISC"] == "0"
        assert env["MOONCAKE_STORE_BACKEND"] == "cpp"
        assert env["SRT_MOONCAKE_EGM_WHEEL_DIR"] == config.environment["SRT_MOONCAKE_EGM_WHEEL_DIR"]
        assert "MC_USE_NVLINK_IPC" not in env
        assert "MC_TE_FILTERS" not in env
        readiness = probes[launch["step_name"]]
        assert readiness.file is not None
        marker = readiness.file.path
        assert marker not in markers
        markers.add(marker)
        assert command[command.index("--ready-file") + 1] == f"/logs/{marker}"
        assert "{" not in marker, "Every instance needs a fully rendered marker"
    assert total_capacity == 12 * donor.parse_size(prefill_size) + 2 * donor.parse_size(decode_size)

    master = next(launch for launch in launches if launch.get("step_name") == "service_mooncake-master")
    assert "/configs/setup_mooncake_egm.sh" in master["bash_preamble"]
    assert "--mooncake-only" in master["bash_preamble"]
    master_ready = events.index(("ready", master["step_name"]))
    workers = [launch for launch in launches if launch.get("step_name", "").startswith(("prefill_", "decode_"))]
    assert workers
    first_worker = min(events.index(("launch", launch["step_name"])) for launch in workers)
    for launch in donors:
        donor_launch = events.index(("launch", launch["step_name"]))
        donor_ready = events.index(("ready", launch["step_name"]))
        assert master_ready < donor_launch < donor_ready < first_worker

    (config_path,) = output_dir.rglob("mooncake_store_config_prefill.json")
    payload = json.loads(config_path.read_text())
    assert payload["protocol"] == "nvlink"
    assert payload["global_segment_size"] == 0
    assert payload["stage_through_host"] is True
    assert payload["staging_buffer_bytes"] == "2GiB"
    monkeypatch.setattr(donor.socket, "gethostbyname", lambda hostname: "192.0.2.2")
    prefill_size_bytes = donor.parse_size(prefill_size)
    for numa_node in set(PREFILL_NUMA_NODES):
        provider = donor.provider_config(config_path, prefill_size_bytes, numa_node)
        assert provider["protocol"] == "nvlink"
        assert provider["enable_egm_store_pool"] == "true"
        assert provider["global_segment_size"] == str(prefill_size_bytes)
        assert provider["local_buffer_size"] == "0"
        assert provider["egm_numa_nodes"] == numa_node
        # Let the native Store select separate RPC/transfer ports for colocated handles.
        assert provider["local_hostname"] == "192.0.2.2"
        assert not any("port" in key for key in provider)
