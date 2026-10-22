# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the DEP16 and DEP32 recipes' host-only EGM pools without Slurm or CUDA."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
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
from srtctl.core.config import expand_engine_config_defaults, resolve_config_with_defaults
from srtctl.core.processes import ManagedProcess
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep
from srtctl.services.config import ServiceConfig, ServiceReadinessConfig

ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = ROOT / "recipes/trtllm/vrnvl72/agentic"
PREFILL_NAMES = tuple(f"prefill-egm-donor-{pool}" for pool in range(4))
CLASSIC_NVLINK_ENV = {"MC_MS_AUTO_DISC": "0", "MC_USE_TENT": "0", "MC_USE_TEV1": "0", "MOONCAKE_STORE_BACKEND": "cpp"}
RDMA_ENV = {"MC_TE_FILTERS", "MC_ENABLE_DEST_DEVICE_AFFINITY"}
WHEEL_DIRS = {
    "dynamo": "/lustre/fsw/coreai_comparch_inferencex/rihuo/mooncake_egm_wheels",
    "raplab-dynamo": "/home/rihuo/mooncake_egm_wheels",
}


@dataclass(frozen=True)
class RecipeCase:
    directory: str
    decode_gpus: int
    concurrency: int
    prefill_gib: int
    decode_gib: int
    time_limit: str | None = None

    @property
    def stem(self) -> str:
        return f"glm5.3-disagg-3p1d-tp4-dep{self.decode_gpus}-c{self.concurrency}"

    @property
    def baseline_path(self) -> Path:
        return RECIPE_DIR / self.directory / f"{self.stem}.yaml"

    @property
    def variant_path(self) -> Path:
        return RECIPE_DIR / self.directory / f"{self.stem}-egm-4x{self.prefill_gib}.yaml"

    @property
    def decode_nodes(self) -> int:
        return self.decode_gpus // 4

    @property
    def total_nodes(self) -> int:
        return 3 + self.decode_nodes + 1


CASES = (
    RecipeCase("dynamo", 32, 560, 224, 768, time_limit="02:00:00"),
    RecipeCase("raplab-dynamo", 32, 560, 224, 768),
    *(RecipeCase(directory, 16, concurrency, 256, 1050) for directory in WHEEL_DIRS for concurrency in (576, 640)),
)


@pytest.fixture(params=CASES, ids=lambda case: f"{case.directory}-dep{case.decode_gpus}-c{case.concurrency}")
def recipe_case(request: pytest.FixtureRequest) -> RecipeCase:
    return request.param


@pytest.fixture
def recipe_path(recipe_case: RecipeCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Resolve the actual recipe independently of local cluster defaults."""
    monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "absent-srtslurm.yaml"))
    monkeypatch.setenv("SRTCTL_SOURCE_DIR", str(ROOT))
    recipe = yaml.safe_load(recipe_case.variant_path.read_text())
    model = tmp_path / "model"
    model.mkdir()
    recipe["model"].update(path=str(model), container="trtllm-egm.sqsh")
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return path


def test_egm_preserves_each_requested_workload_and_capacity(recipe_case: RecipeCase) -> None:
    baseline = yaml.safe_load(recipe_case.baseline_path.read_text())
    variant = yaml.safe_load(recipe_case.variant_path.read_text())
    assert variant["name"] == recipe_case.variant_path.stem
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
        assert variant.get(key) == baseline.get(key), key
    for role, settings in baseline["roles"].items():
        env = {key: value for key, value in settings["env"].items() if key not in RDMA_ENV}
        expected = {**settings, "env": {**env, **CLASSIC_NVLINK_ENV}} if role == "prefill" else settings
        assert variant["roles"][role] == expected, role
    expected_environment = {
        **baseline["environment"],
        "SRT_MOONCAKE_EGM_WHEEL_DIR": WHEEL_DIRS[recipe_case.directory],
    }
    if recipe_case.directory == "raplab-dynamo":
        expected_environment["ENROOT_MAX_PROCESSORS"] = "8"
    assert variant["environment"] == expected_environment
    if recipe_case.time_limit is None:
        assert "slurm" not in variant
        assert variant["sbatch_directives"] == {"exclusive": "", "mem": "0"}
    else:
        assert variant["slurm"] == {"time_limit": recipe_case.time_limit}
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
    assert master["command"][:3] == ["python3", "-m", "mooncake.cli"]
    assert master["command"][3:] == base_master["command"][1:]
    assert "--allocation_strategy=random" in master["command"]
    assert base_master["options"]["store_config"]["global_segment_size"] == f"{recipe_case.prefill_gib}GiB"
    baseline_donor = next(service for service in baseline["services"] if service["name"] == "decode-mooncake-donor")
    assert f"--segment_size {recipe_case.decode_gib}GiB" in " ".join(baseline_donor["command"])
    for service in variant["services"]:
        if service.get("enabled") is not False:
            assert service["env"]["SRT_MOONCAKE_EGM_WHEEL_DIR"] == expected_environment["SRT_MOONCAKE_EGM_WHEEL_DIR"]
            for key, value in expected_environment.items():
                if key.startswith("PIP_"):
                    assert service["env"][key] == value


def test_allocation_and_dry_run_expose_host_only_nvlink_pools(
    recipe_case: RecipeCase, recipe_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster_defaults = {
        "default_account": "egm-test-account",
        "default_partition": "egm-test-partition",
        "default_time_limit": "03:00:00",
    }
    resolved = resolve_config_with_defaults(yaml.safe_load(recipe_path.read_text()), cluster_defaults)
    expand_engine_config_defaults(resolved)
    config = SrtConfig.Schema().load(resolved)
    script = generate_minimal_sbatch_script(config=config, config_path=recipe_path)
    directives = {line for line in script.splitlines() if line.startswith("#SBATCH ")}
    expected_time = recipe_case.time_limit or cluster_defaults["default_time_limit"]
    assert {
        f"#SBATCH --nodes={recipe_case.total_nodes}",
        f"#SBATCH --segment={recipe_case.total_nodes}",
        "#SBATCH --exclusive",
        "#SBATCH --mem=0",
        f"#SBATCH --time={expected_time}",
        "#SBATCH --account=egm-test-account",
        "#SBATCH --partition=egm-test-partition",
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


def test_mock_launch_places_four_host_donors_per_prefill_before_workers(
    recipe_case: RecipeCase, recipe_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
                nodelist=tuple(f"node-{index:02d}" for index in range(recipe_case.total_nodes)),
                on_srun=record_launch,
            ),
        )
        == 0
    )
    prefill = [launch for launch in launches if launch.get("step_name", "").startswith("service_prefill-egm-donor-")]
    decode = [launch for launch in launches if launch.get("step_name", "").startswith("service_decode-mooncake-donor")]
    assert len(prefill) == 12
    assert len(decode) == recipe_case.decode_nodes
    prefill_nodes = Counter(launch["nodelist"][0] for launch in prefill)
    decode_nodes = Counter(launch["nodelist"][0] for launch in decode)
    assert sorted(prefill_nodes.values()) == [4, 4, 4]
    assert sorted(decode_nodes.values()) == [1] * recipe_case.decode_nodes
    assert not set(prefill_nodes) & set(decode_nodes)
    for pool, name in enumerate(PREFILL_NAMES):
        fleet = [launch for launch in prefill if launch["step_name"].startswith(f"service_{name}_")]
        assert len(fleet) == 3
        assert {launch["nodelist"][0] for launch in fleet} == set(prefill_nodes)
        assert {launch["env_to_set"]["CUDA_VISIBLE_DEVICES"] for launch in fleet} == {str(pool)}
        assert {launch["command"][launch["command"].index("--numa-nodes") + 1] for launch in fleet} == {str(pool // 2)}
    donors = prefill + decode
    assert len({launch["output"] for launch in donors}) == 12 + recipe_case.decode_nodes
    markers: set[str] = set()
    total_capacity = 0
    services = {service.name: service for service in config.services}
    for launch in donors:
        command = launch["command"]
        expected_size = f"{recipe_case.prefill_gib if launch in prefill else recipe_case.decode_gib}GiB"
        assert command[command.index("--segment-size") + 1] == expected_size
        total_capacity += donor.parse_size(expected_size)
        assert "/configs/mooncake_egm_donor.py" in command
        assert command[command.index("--config") + 1] == "/logs/mooncake_store_config_prefill.json"
        env = launch["env_to_set"]
        assert CLASSIC_NVLINK_ENV.items() <= env.items()
        assert env["SRT_MOONCAKE_EGM_WHEEL_DIR"] == config.environment["SRT_MOONCAKE_EGM_WHEEL_DIR"]
        assert "MC_USE_NVLINK_IPC" not in env
        assert not RDMA_ENV & env.keys()
        for key, value in config.environment.items():
            if key.startswith("PIP_"):
                assert env[key] == value
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
    expected_gib = 12 * recipe_case.prefill_gib + recipe_case.decode_nodes * recipe_case.decode_gib
    assert total_capacity == expected_gib * 1024**3
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
    for size, numa_node in (
        (recipe_case.prefill_gib, "0"),
        (recipe_case.prefill_gib, "1"),
        (recipe_case.decode_gib, "auto"),
    ):
        provider = donor.provider_config(config_path, size * 1024**3, numa_node)
        assert provider["enable_egm_store_pool"] == "true"
        assert provider["global_segment_size"] == str(size * 1024**3)
        assert provider["local_buffer_size"] == "0"
        assert provider["egm_numa_nodes"] == numa_node
