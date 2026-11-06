# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-role Mooncake store configs for vLLM (``roles.<role>.mooncake_store_config``), without Slurm or GPUs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends import VLLMBackend
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Process
from srtctl.mock import MockOptions, run_mock_sweep

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "vllm" / "dynamo-disagg-mooncake.yaml"
NIXL_ONLY = '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'


def _recipe() -> dict[str, Any]:
    """The example recipe, with the model and container resolvable without srtslurm.yaml."""
    recipe = yaml.safe_load(EXAMPLE.read_text())
    recipe["model"].update(path="hf:Qwen/Qwen3-0.6B", container="vllm.sqsh")
    return recipe


def _master(recipe: dict[str, Any]) -> dict[str, Any]:
    return next(service for service in recipe["services"] if service["type"] == "mooncake-master")


def _load(recipe: dict[str, Any], tmp_path: Path) -> SrtConfig:
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return SrtConfig.from_yaml(path)


def _backend(recipe: dict[str, Any], tmp_path: Path) -> VLLMBackend:
    backend = _load(recipe, tmp_path).backend
    assert isinstance(backend, VLLMBackend)
    return backend


def _sweep(recipe: dict[str, Any], tmp_path: Path) -> tuple[dict[str, dict[str, Any]], Path]:
    """Each role's first worker launch, and the outputs directory the orchestrator wrote."""
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    launches: list[dict[str, Any]] = []
    exit_code = run_mock_sweep(
        config_path=path,
        output_dir=tmp_path / "outputs" / "4302",
        job_id="4302",
        options=MockOptions(child_duration_s=0.05, phase_pause_s=0.01, nodelist=("node-01",), on_srun=launches.append),
    )
    assert exit_code == 0
    workers = {
        mode: next(launch for launch in launches if launch.get("step_name", "").startswith(f"{mode}_"))
        for mode in ("prefill", "decode")
    }
    return workers, tmp_path / "outputs"


def _written(outputs: Path, filename: str) -> dict[str, Any]:
    (path,) = outputs.rglob(filename)
    return json.loads(path.read_text())


def test_prefill_and_decode_mount_different_segments(tmp_path: Path) -> None:
    """Each role's workers read a store config with that role's global_segment_size."""
    recipe = _recipe()
    recipe["roles"]["prefill"]["mooncake_store_config"] = {"global_segment_size": "12GB"}
    workers, outputs = _sweep(recipe, tmp_path)

    for mode, size in (("prefill", "12GB"), ("decode", "16GB")):
        filename = f"mooncake_store_config_{mode}.json"
        assert workers[mode]["env_to_set"]["MOONCAKE_CONFIG_PATH"] == f"/logs/{filename}"
        written = _written(outputs, filename)
        assert written["global_segment_size"] == size
        # Everything else is the service's shared store_config.
        assert written["protocol"] == "rdma"
        assert written["metadata_server"] == "P2PHANDSHAKE"
        assert written["local_buffer_size"] == "4GB"
        assert written["master_server_address"] == workers[mode]["env_to_set"]["MOONCAKE_MASTER"]
    # The shared file is still written, unchanged.
    assert _written(outputs, "mooncake_store_config.json")["global_segment_size"] == "8GB"


def test_a_role_without_its_own_keys_keeps_the_shared_config(tmp_path: Path) -> None:
    """The example overrides decode only: prefill reads the shared file, as before."""
    workers, outputs = _sweep(_recipe(), tmp_path)
    assert workers["prefill"]["env_to_set"]["MOONCAKE_CONFIG_PATH"] == "/logs/mooncake_store_config.json"
    assert workers["decode"]["env_to_set"]["MOONCAKE_CONFIG_PATH"] == "/logs/mooncake_store_config_decode.json"
    assert _written(outputs, "mooncake_store_config.json")["global_segment_size"] == "8GB"
    assert _written(outputs, "mooncake_store_config_decode.json")["global_segment_size"] == "16GB"
    assert not list(outputs.rglob("mooncake_store_config_prefill.json"))


def test_without_role_keys_every_role_reads_the_shared_config(tmp_path: Path) -> None:
    recipe = _recipe()
    del recipe["roles"]["decode"]["mooncake_store_config"]
    workers, outputs = _sweep(recipe, tmp_path)
    for mode in ("prefill", "decode"):
        assert workers[mode]["env_to_set"]["MOONCAKE_CONFIG_PATH"] == "/logs/mooncake_store_config.json"
    assert sorted(p.name for p in outputs.rglob("mooncake_store_config*.json")) == ["mooncake_store_config.json"]


def test_render_layers_role_keys_over_the_shared_keys(tmp_path: Path) -> None:
    recipe = _recipe()
    _master(recipe)["options"]["store_config"]["master_server_address"] = "10.0.0.9:1234"
    recipe["roles"]["prefill"]["mooncake_store_config"] = {"global_segment_size": 4 << 30, "local_buffer_size": "1GB"}
    backend = _backend(recipe, tmp_path)
    assert backend.mooncake_store_role_modes() == ("prefill", "decode")
    assert backend.uses_mooncake_store("prefill") and backend.uses_mooncake_store("decode")
    prefill = backend.build_mooncake_store_config("10.1.1.1", "prefill")
    assert prefill["global_segment_size"] == 4 << 30
    assert prefill["local_buffer_size"] == "1GB"
    assert prefill["master_server_address"] == "10.1.1.1:8700"
    assert backend.build_mooncake_store_config("10.1.1.1")["global_segment_size"] == "8GB"
    assert set(backend.mooncake_store_configs("10.1.1.1", "served")) == {
        "mooncake_store_config.json",
        "mooncake_store_config_prefill.json",
        "mooncake_store_config_decode.json",
    }


def test_process_local_device_configs_keep_the_role_keys(tmp_path: Path) -> None:
    """With device_names_by_gpu, a role with its own keys gets a process config named after the role."""
    recipe = _recipe()
    _master(recipe)["options"]["device_names_by_gpu"] = [f"mlx5_{i}" for i in range(8)]
    backend = _backend(recipe, tmp_path)
    decode = Process("n0", frozenset({2, 3}), 7500, 6100, "decode", 0)
    prefill = Process("n0", frozenset({0, 1}), 7501, 6101, "prefill", 0)
    filename, payload = backend.build_mooncake_process_config(decode, "infra", 8)
    assert filename == "mooncake_store_config_decode_gpu2-3.json"
    assert payload["device_name"] == "mlx5_2,mlx5_3"
    assert payload["global_segment_size"] == "16GB"
    filename, payload = backend.build_mooncake_process_config(prefill, "infra", 8)
    assert filename == "mooncake_store_config_gpu0-1.json"
    assert payload["global_segment_size"] == "8GB"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(services=[]), "roles.decode.mooncake_store_config needs a mooncake-master service"),
        (
            lambda r: r["roles"]["decode"]["args"].update({"kv-transfer-config": NIXL_ONLY}),
            "roles.decode.args has no kv-transfer-config with MooncakeStoreConnector",
        ),
        (
            lambda r: r["roles"]["decode"]["mooncake_store_config"].update(protocol="tcp"),
            "protocol must be the same for every client of the pool",
        ),
        (
            lambda r: r["roles"]["decode"]["mooncake_store_config"].update(master_server_address="10.0.0.9:1234"),
            "master_server_address must be the same for every client of the pool",
        ),
        (
            lambda r: (
                _master(r)["options"].update(device_names_by_gpu=[f"mlx5_{i}" for i in range(8)]),
                r["roles"]["decode"]["mooncake_store_config"].update(device_name="mlx5_0"),
            ),
            "device_name conflicts with the mooncake-master service's device_names_by_gpu",
        ),
    ],
)
def test_invalid_role_mooncake_store_config_fails_at_load(tmp_path: Path, mutate: Any, message: str) -> None:
    recipe = _recipe()
    _load(recipe, tmp_path)
    mutate(recipe)
    with pytest.raises(ValidationError, match=message):
        _load(recipe, tmp_path)
