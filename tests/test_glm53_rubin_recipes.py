# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the imported GLM-5.3 load points without containers or Slurm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from srtctl.cli.submit import generate_minimal_sbatch_script, planned_total_nodes
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes/trtllm/glm5.3"
# Stem suffix, prefill workers, decode workers, decode GPUs per worker, concurrency,
# EGM prefill pools per node, EGM prefill pool size, EGM decode pool size.
CASES = (
    ("1p1d-tp4-tp16-c1", 1, 1, 16, 1, 0, "", ""),
    ("1p2d-tp4-tp4-c36", 1, 2, 4, 36, 0, "", ""),
    ("1p3d-tp4-tp8-c40", 1, 3, 8, 40, 0, "", ""),
    ("1p3d-tp4-tp4-c64", 1, 3, 4, 64, 0, "", ""),
    ("3p1d-tp4-dep32-c560", 3, 1, 32, 560, 0, "", ""),
    ("3p1d-tp4-dep16-c640", 3, 1, 16, 640, 0, "", ""),
    ("3p1d-tp4-dep16-c576", 3, 1, 16, 576, 0, "", ""),
    ("3p1d-tp4-dep8-c560", 3, 1, 8, 560, 0, "", ""),
    ("3p1d-tp4-dep8-c560-egm", 3, 1, 8, 560, 1, "1040GiB", "1100GiB"),
    ("3p1d-tp4-dep8-c560-egm-4x260", 3, 1, 8, 560, 4, "260GiB", "1100GiB"),
    ("3p1d-tp4-dep16-c576-egm-4x256", 3, 1, 16, 576, 4, "256GiB", "1050GiB"),
    ("3p1d-tp4-dep16-c640-egm-4x256", 3, 1, 16, 640, 4, "256GiB", "1050GiB"),
    ("3p1d-tp4-dep32-c560-egm-4x224", 3, 1, 32, 560, 4, "224GiB", "768GiB"),
)


# Per-directory DEP8 pool expectations.
POOL_OVERRIDES = {
    "vrnvl72": {
        "3p1d-tp4-dep8-c560-egm": ("1040GiB", "1050GiB"),
        "3p1d-tp4-dep8-c560-egm-4x260": ("260GiB", "1050GiB"),
    },
    "vrnvl72-lab": {
        "3p1d-tp4-dep8-c560-egm-4x260": ("260GiB", "1050GiB"),
    },
}
# The lab directory keeps TP baselines and EGM variants with four prefill pools.
CASES_BY_DIR = {
    "vrnvl72": CASES,
    "vrnvl72-lab": tuple(case for case in CASES if case[1] == 1 or case[5] == 4),
}


@pytest.mark.parametrize(
    "recipe_dir,variant,prefill_workers,decode_workers,decode_gpus,concurrency,pools,prefill_pool,decode_pool",
    [
        pytest.param(recipe_dir, *case, id=f"{recipe_dir}/{case[0]}")
        for recipe_dir, cases in CASES_BY_DIR.items()
        for case in cases
    ],
)
def test_recipe_config_and_mock_launch(
    recipe_dir: str,
    variant: str,
    prefill_workers: int,
    decode_workers: int,
    decode_gpus: int,
    concurrency: int,
    pools: int,
    prefill_pool: str,
    decode_pool: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SRTSLURM_CONFIG", str(tmp_path / "absent-srtslurm.yaml"))
    monkeypatch.setenv("SRTCTL_SOURCE_DIR", str(ROOT))
    prefill_pool, decode_pool = POOL_OVERRIDES.get(recipe_dir, {}).get(variant, (prefill_pool, decode_pool))
    source = RECIPES / recipe_dir / f"glm5.3-disagg-{variant}.yaml"
    recipe = yaml.safe_load(source.read_text())
    model = tmp_path / "model"
    model.mkdir()
    recipe["model"].update(path=str(model), container="trtllm-glm53.sqsh")
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    config = SrtConfig.from_yaml(path)

    assert config.resources.gpu_type == "vrnvl72"
    assert config.resources.gpus_per_node == 4
    prefill, decode = config.roles["prefill"], config.roles["decode"]
    assert (prefill.nodes, prefill.workers, prefill.gpus) == (prefill_workers, prefill_workers, 4)
    decode_nodes = decode_workers * decode_gpus // 4
    assert (decode.nodes, decode.workers, decode.gpus) == (decode_nodes, decode_workers, decode_gpus)
    assert prefill.args["tensor_parallel_size"] == 4
    assert decode.args["tensor_parallel_size"] == decode_gpus
    assert recipe["benchmark"]["env"]["CONC"] == str(concurrency)
    benchmark_command = recipe["benchmark"]["command"]
    for arg in ("--benchmark-duration 3600", "--warmup-requests-per-lane 10", "--server-metrics"):
        assert arg in benchmark_command
    assert config.setup_script == "install_dynamo_wheel.sh"
    assert (ROOT / "configs" / config.setup_script).is_file()
    assert not config.dynamo.install
    assert "setup_mooncake_egm.sh" not in source.read_text()
    assert "SRT_MOONCAKE_EGM_WHEEL_DIR" not in source.read_text()
    assert config.reuse_container is (recipe_dir == "vrnvl72")

    total_nodes = prefill_workers + decode_nodes + 1
    assert planned_total_nodes(config) == total_nodes
    script = generate_minimal_sbatch_script(config=config, config_path=path)
    assert f"#SBATCH --nodes={total_nodes}\n" in script
    calls: list[dict[str, Any]] = []
    output = tmp_path / "output"
    assert (
        run_mock_sweep(
            config_path=path,
            output_dir=output,
            job_id="6203",
            options=MockOptions(
                child_duration_s=0.1,
                phase_pause_s=0,
                nodelist=tuple(f"node-{i:02}" for i in range(total_nodes)),
                on_srun=calls.append,
            ),
        )
        == 0
    )
    workers = [call for call in calls if call.get("step_name", "").startswith(("prefill_", "decode_"))]
    assert len(workers) == prefill_workers + decode_workers
    for call in workers:
        assert "install_dynamo_wheel.sh" in call["bash_preamble"]
    assert all("setup_mooncake_egm.sh" not in (call.get("bash_preamble") or "") for call in calls)
    assert all(
        call.get("reuse_container", False) is config.reuse_container for call in calls if call.get("container_image")
    )

    if variant == "3p1d-tp4-dep8-c560-egm-4x260":
        assert recipe["frontend"]["args"]["router-mode"] == "least-loaded"
        assert "router-session-affinity-ttl-secs" not in recipe["frontend"]["args"]
        for call in workers:
            env = call["env_to_set"]
            assert env["DYN_ROUTER_MODE"] == "least-loaded"
            assert "DYN_ROUTER_SESSION_AFFINITY_TTL_SECS" not in env

    generated = list(output.rglob("mooncake_store_config_*.json"))
    if prefill_workers == 1:
        assert not generated
        frontend = recipe["frontend"]
        for call in workers:
            env = call["env_to_set"]
            if call["step_name"].startswith("prefill_"):
                assert env["DYN_ROUTER_MODE"] == frontend["args"]["router-mode"] == "kv"
                assert (
                    env["DYN_ROUTER_SESSION_AFFINITY_TTL_SECS"]
                    == str(frontend["args"]["router-session-affinity-ttl-secs"])
                    == "14400"
                )
                assert env["DYN_ROUTER_USE_KV_EVENTS"] == frontend["env"]["DYN_ROUTER_USE_KV_EVENTS"] == "0"
                assert env["DYN_ENGINE_CONV_AFFINITY"] == "1"
            else:
                assert env["DYN_ROUTER_MODE"] == "least-loaded"
        return
    assert len(generated) == 1
    payload = json.loads(generated[0].read_text())
    assert generated[0].name == "mooncake_store_config_prefill.json"
    assert payload["stage_through_host"] is True
    assert payload["staging_buffer_bytes"] == "2GiB"
    if not pools:
        assert payload["protocol"] == "rdma"
        donors = [call for call in calls if call.get("step_name", "").startswith("service_decode-mooncake-donor")]
        assert len(donors) == decode_nodes
        assert all("trtllm-serve mooncake_donor" in " ".join(call["command"]) for call in donors)
        if recipe_dir == "vrnvl72":
            assert max(calls.index(call) for call in donors) < min(calls.index(call) for call in workers)
        else:
            assert max(calls.index(call) for call in workers) < min(calls.index(call) for call in donors)
        return

    assert payload["protocol"] == "nvlink"
    assert payload["global_segment_size"] == 0
    assert payload["local_buffer_size"] == "64MiB"
    services = {service.name: service for service in config.services if service.enabled}
    assert services.pop("mooncake-master").effective_start == "infra"
    master = next(call for call in calls if call.get("step_name") == "service_mooncake-master")
    assert "mooncake.cli" in master["command"]
    donors = []
    for name, service in services.items():
        assert service.effective_start == "before_workers"
        assert service.readiness.file is not None
        launches = [call for call in calls if call.get("step_name", "").startswith(f"service_{name}")]
        is_prefill = service.placement.node == "prefill"
        assert len(launches) == (prefill_workers if is_prefill else decode_nodes)
        assert len({call["nodelist"][0] for call in launches}) == len(launches)
        for call in launches:
            command = call["command"]
            assert "/configs/mooncake_egm_donor.py" in command
            assert command[command.index("--segment-size") + 1] == (prefill_pool if is_prefill else decode_pool)
            env = call["env_to_set"]
            if is_prefill and pools == 4:
                index = int(name.rsplit("-", 1)[1])
                assert env["CUDA_VISIBLE_DEVICES"] == str(index)
                assert command[command.index("--numa-nodes") + 1] == str(index // 2)
            else:
                assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
                assert command[command.index("--numa-nodes") + 1] == "auto"
            assert env["MOONCAKE_STORE_BACKEND"] == "cpp"
            assert "MC_USE_NVLINK_IPC" not in env
            ready = command[command.index("--ready-file") + 1]
            assert "{index}" not in ready
            assert Path(ready).name.startswith(service.readiness.file.path.split("{index}")[0])
        donors.extend(launches)
    assert len(donors) == prefill_workers * pools + decode_nodes
    assert calls.index(master) < min(calls.index(call) for call in donors)
    assert max(calls.index(call) for call in donors) < min(calls.index(call) for call in workers)
