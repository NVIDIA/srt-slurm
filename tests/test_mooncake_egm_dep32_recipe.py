# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the DEP32 recipe's host-only EGM pools without Slurm or CUDA."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console

from configs import mooncake_egm_donor as donor
from srtctl.backends import TRTLLMBackend
from srtctl.cli import submit
from srtctl.cli.mixins.service_stage import ServiceStageMixin
from srtctl.cli.submit import generate_minimal_sbatch_script, show_config_details
from srtctl.core.processes import ManagedProcess
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep
from srtctl.services.config import ServiceConfig, ServiceReadinessConfig

ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = ROOT / "recipes/trtllm/vrnvl72/agentic/dynamo"
BASE_RECIPE = RECIPE_DIR / "glm5.3-disagg-3p1d-tp4-dep32-c560.yaml"
EGM_RECIPE = RECIPE_DIR / "glm5.3-disagg-3p1d-tp4-dep32-c560-egm-4x224.yaml"
PREFILL_NAMES = tuple(f"prefill-egm-donor-{pool}" for pool in range(4))
CLASSIC_NVLINK_ENV = {"MC_MS_AUTO_DISC": "0", "MC_USE_TENT": "0", "MC_USE_TEV1": "0", "MOONCAKE_STORE_BACKEND": "cpp"}


@pytest.fixture
def recipe_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Resolve the actual recipe independently of local cluster defaults."""
    monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "absent-srtslurm.yaml"))
    monkeypatch.setenv("SRTCTL_SOURCE_DIR", str(ROOT))
    recipe = yaml.safe_load(EGM_RECIPE.read_text())
    model = tmp_path / "model"
    model.mkdir()
    recipe["model"].update(path=str(model), container="trtllm-egm.sqsh")
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return path


def test_dep32_egm_preserves_the_requested_workload_and_capacity() -> None:
    baseline = yaml.safe_load(BASE_RECIPE.read_text())
    variant = yaml.safe_load(EGM_RECIPE.read_text())
    assert variant["name"] == "glm5.3-disagg-3p1d-tp4-dep32-c560-egm-4x224"
    for key in (
        "schema",
        "model",
        "resources",
        "host_setup",
        "reuse_container",
        "observability",
        "dynamo",
        "frontend",
        "engine",
        "extra_mount",
        "benchmark",
    ):
        assert variant[key] == baseline[key], key
    for role, settings in baseline["roles"].items():
        expected = {**settings, "env": {**settings["env"], **CLASSIC_NVLINK_ENV}} if role == "prefill" else settings
        assert variant["roles"][role] == expected, role
    assert variant["environment"] == {
        **baseline["environment"],
        "SRT_MOONCAKE_EGM_WHEEL_DIR": "/lustre/fsw/coreai_comparch_inferencex/rihuo/mooncake_egm_wheels",
    }
    disabled = [service for service in baseline["services"] if service.get("enabled") is False]
    assert [service for service in variant["services"] if service.get("enabled") is False] == disabled
    base_master = next(service for service in baseline["services"] if service["name"] == "mooncake-master")
    master = next(service for service in variant["services"] if service["name"] == "mooncake-master")
    expected_store = {
        **base_master["options"]["store_config"],
        "protocol": "nvlink",
        "device_name": "",
        "global_segment_size": 0,
        "local_buffer_size": "64MiB",
    }
    assert master["options"]["store_config"] == expected_store
    assert "--allocation_strategy=random" in master["command"]
    assert base_master["options"]["store_config"]["global_segment_size"] == "224GiB"
    baseline_donor = next(service for service in baseline["services"] if service["name"] == "decode-mooncake-donor")
    assert "--segment_size 768GiB" in " ".join(baseline_donor["command"])


def test_dep32_allocation_and_dry_run_expose_host_only_nvlink_pools(
    recipe_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SrtConfig.from_yaml(recipe_path)
    script = generate_minimal_sbatch_script(config=config, config_path=recipe_path)
    directives = {line for line in script.splitlines() if line.startswith("#SBATCH ")}
    assert {
        "#SBATCH --nodes=12",
        "#SBATCH --segment=12",
        "#SBATCH --exclusive",
        "#SBATCH --mem=0",
        "#SBATCH --time=02:00:00",
    } <= directives
    assert config.resources.het_jobs is False
    assert config.setup_script == "setup_mooncake_egm.sh"
    backend = config.backend
    assert isinstance(backend, TRTLLMBackend)
    assert backend.mooncake_store_modes() == ("prefill",)
    (payload,) = backend.mooncake_store_configs("192.0.2.1", "GLM-5.3").values()
    assert payload["protocol"] == "nvlink"
    assert payload["device_name"] == ""
    assert payload["global_segment_size"] == 0
    assert payload["local_buffer_size"] == "64MiB"
    assert payload["stage_through_host"] is True
    assert payload["staging_buffer_bytes"] == "2GiB"
    assert "kv_connector_config" not in config.roles["decode"].args
    monkeypatch.setattr(submit, "console", Console(width=180))
    show_config_details(config)
    output = capsys.readouterr().out
    for name in (*PREFILL_NAMES, "decode-mooncake-donor"):
        assert name in output
    assert '"protocol": "nvlink"' in output
    assert '"global_segment_size": 0' in output
    assert '"stage_through_host": true' in output
    assert '"staging_buffer_bytes": "2GiB"' in output


def test_dep32_mock_launch_places_twenty_host_donors_before_workers(
    recipe_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SrtConfig.from_yaml(recipe_path)
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
    output_dir = tmp_path / "outputs" / "6232"
    assert (
        run_mock_sweep(
            config_path=recipe_path,
            output_dir=output_dir,
            job_id="6232",
            options=MockOptions(
                child_duration_s=0.1,
                phase_pause_s=0.01,
                nodelist=tuple(f"node-{index:02d}" for index in range(12)),
                on_srun=record_launch,
            ),
        )
        == 0
    )
    prefill = [launch for launch in launches if launch.get("step_name", "").startswith("service_prefill-egm-donor-")]
    decode = [launch for launch in launches if launch.get("step_name", "").startswith("service_decode-mooncake-donor")]
    assert len(prefill) == 12
    assert len(decode) == 8
    prefill_nodes = Counter(launch["nodelist"][0] for launch in prefill)
    decode_nodes = Counter(launch["nodelist"][0] for launch in decode)
    assert sorted(prefill_nodes.values()) == [4, 4, 4]
    assert sorted(decode_nodes.values()) == [1] * 8
    assert not set(prefill_nodes) & set(decode_nodes)
    for pool, name in enumerate(PREFILL_NAMES):
        fleet = [launch for launch in prefill if launch["step_name"].startswith(f"service_{name}_")]
        assert len(fleet) == 3
        assert {launch["nodelist"][0] for launch in fleet} == set(prefill_nodes)
        assert {launch["env_to_set"]["CUDA_VISIBLE_DEVICES"] for launch in fleet} == {str(pool)}
        assert {launch["command"][launch["command"].index("--numa-nodes") + 1] for launch in fleet} == {str(pool // 2)}
    donors = prefill + decode
    assert len({launch["output"] for launch in donors}) == 20
    markers: set[str] = set()
    total_capacity = 0
    services = {service.name: service for service in config.services}
    for launch in donors:
        command = launch["command"]
        expected_size = "224GiB" if launch in prefill else "768GiB"
        assert command[command.index("--segment-size") + 1] == expected_size
        total_capacity += donor.parse_size(expected_size)
        assert "/configs/mooncake_egm_donor.py" in command
        assert command[command.index("--config") + 1] == "/logs/mooncake_store_config_prefill.json"
        env = launch["env_to_set"]
        assert CLASSIC_NVLINK_ENV.items() <= env.items()
        assert env["SRT_MOONCAKE_EGM_WHEEL_DIR"] == config.environment["SRT_MOONCAKE_EGM_WHEEL_DIR"]
        assert "MC_USE_NVLINK_IPC" not in env
        assert "MC_TE_FILTERS" not in env
        if launch in decode:
            assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
            assert command[command.index("--numa-nodes") + 1] == "auto"
        assert "--mooncake-only" in launch["bash_preamble"]
        assert "/configs/setup_mooncake_egm.sh" in launch["bash_preamble"]
        readiness = probes[launch["step_name"]]
        assert readiness.file is not None
        marker = readiness.file.path
        assert marker not in markers
        markers.add(marker)
        assert "{" not in marker
        assert command[command.index("--ready-file") + 1] == f"/logs/{marker}"
    assert total_capacity == 8832 * 1024**3
    assert services["mooncake-master"].effective_start == "infra"
    for name in (*PREFILL_NAMES, "decode-mooncake-donor"):
        assert services[name].effective_start == "before_workers"
        assert services[name].effective_critical is True
    master_ready = events.index(("ready", "service_mooncake-master"))
    workers = [launch for launch in launches if launch.get("step_name", "").startswith(("prefill_", "decode_"))]
    assert workers
    first_worker = min(events.index(("launch", launch["step_name"])) for launch in workers)
    for launch in donors:
        donor_launch = events.index(("launch", launch["step_name"]))
        donor_ready = events.index(("ready", launch["step_name"]))
        assert master_ready < donor_launch < donor_ready < first_worker
    for launch in workers:
        assert "/configs/numa_cpu_bind.sh" in launch["command"]
        assert "setup_mooncake_egm.sh" in launch["bash_preamble"]
    (config_path,) = output_dir.rglob("mooncake_store_config_prefill.json")
    payload = json.loads(config_path.read_text())
    assert payload["global_segment_size"] == 0
    assert payload["stage_through_host"] is True
    monkeypatch.setattr(donor.socket, "gethostbyname", lambda hostname: "192.0.2.2")
    for size, numa_node in ((224, "0"), (224, "1"), (768, "auto")):
        provider = donor.provider_config(config_path, size * 1024**3, numa_node)
        assert provider["enable_egm_store_pool"] == "true"
        assert provider["global_segment_size"] == str(size * 1024**3)
        assert provider["local_buffer_size"] == "0"
        assert provider["egm_numa_nodes"] == numa_node
