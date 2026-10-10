# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""roles.<role>.mooncake_store_config: each role's Mooncake client config, for vLLM, without Slurm or GPUs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends import VLLMBackend
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
KV_TRANSFER_CONFIG = json.dumps(
    {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "connectors": [
                {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
                {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"},
            ]
        },
    }
)


def _recipe() -> dict[str, Any]:
    """examples/vllm/dynamo-disagg.yaml on a Mooncake pool, each role carrying its own client config."""
    recipe = yaml.safe_load((EXAMPLES / "vllm" / "dynamo-disagg.yaml").read_text())
    recipe["model"].update(path="hf:Qwen/Qwen3-0.6B", container="vllm.sqsh")
    recipe["engine"] = {"type": "vllm"}
    for mode, size in (("prefill", "8GB"), ("decode", "16GB")):
        role = recipe["roles"][mode]
        role["args"]["kv-transfer-config"] = KV_TRANSFER_CONFIG
        role["mooncake_store_config"] = {"protocol": "rdma", "metadata_server": "P2PHANDSHAKE", "global_segment_size": size}
    recipe["services"] = [{"name": "mooncake-master", "type": "mooncake-master"}]
    return recipe


def _load(recipe: dict[str, Any], tmp_path: Path) -> SrtConfig:
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return SrtConfig.from_yaml(path)


def test_vllm_roles_carry_their_own_client_config(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Each role's workers read the file rendered from that role's keys; nothing is deprecated."""
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(_recipe()))
    launches: list[dict[str, Any]] = []
    with caplog.at_level(logging.WARNING):
        exit_code = run_mock_sweep(
            config_path=path,
            output_dir=tmp_path / "outputs" / "4302",
            job_id="4302",
            options=MockOptions(child_duration_s=0.05, phase_pause_s=0.01, nodelist=("node-01",), on_srun=launches.append),
        )
    assert exit_code == 0
    assert "options.store_config is deprecated" not in caplog.text
    for mode, size in (("prefill", "8GB"), ("decode", "16GB")):
        (written,) = (tmp_path / "outputs").rglob(f"mooncake_store_config_{mode}.json")
        config = json.loads(written.read_text())
        assert config["global_segment_size"] == size
        assert config["protocol"] == "rdma"
        assert config["master_server_address"].endswith(":8700")
        worker = next(launch for launch in launches if launch.get("step_name", "").startswith(f"{mode}_"))
        assert worker["env_to_set"]["MOONCAKE_CONFIG_PATH"] == f"/logs/mooncake_store_config_{mode}.json"
    assert not list((tmp_path / "outputs").rglob("mooncake_store_config.json"))


def test_master_store_config_still_applies_under_each_role_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The service's options.store_config is deprecated but keeps working: a role's own keys win over it."""
    recipe = _recipe()
    recipe["services"][0]["options"] = {"store_config": {"local_buffer_size": "4GB", "global_segment_size": "2GB"}}
    del recipe["roles"]["prefill"]["mooncake_store_config"]["global_segment_size"]
    with caplog.at_level(logging.WARNING):
        backend = _load(recipe, tmp_path).backend
    assert "options.store_config is deprecated" in caplog.text
    assert isinstance(backend, VLLMBackend)
    prefill = backend.build_mooncake_store_config("prefill", "10.1.1.1")
    decode = backend.build_mooncake_store_config("decode", "10.1.1.1")
    assert (prefill["global_segment_size"], decode["global_segment_size"]) == ("2GB", "16GB")
    assert prefill["local_buffer_size"] == decode["local_buffer_size"] == "4GB"


@pytest.mark.parametrize("key", ["protocol", "metadata_server"])
def test_keys_the_pool_shares_must_match_across_roles(tmp_path: Path, key: str) -> None:
    recipe = _recipe()
    recipe["roles"]["decode"]["mooncake_store_config"][key] = "tcp"
    with pytest.raises(ValidationError, match=f"{key} must be the same for every role in the Mooncake pool"):
        _load(recipe, tmp_path)


def test_a_role_without_a_mooncake_connector_takes_no_client_config(tmp_path: Path) -> None:
    recipe = _recipe()
    recipe["roles"]["decode"]["args"]["kv-transfer-config"] = json.dumps({"kv_connector": "NixlConnector"})
    with pytest.raises(ValidationError, match="roles.decode reads no Mooncake client config file"):
        _load(recipe, tmp_path)
    del recipe["roles"]["decode"]["mooncake_store_config"]
    backend = _load(recipe, tmp_path).backend
    assert backend.mooncake_store_modes() == ("prefill",)
