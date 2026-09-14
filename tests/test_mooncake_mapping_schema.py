# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load-time and migration coverage for process-local Mooncake configuration."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends.vllm import VLLMProtocol
from srtctl.cli.submit import show_config_details
from srtctl.core.migrate import migrate_recipe_text, verify_migration_text
from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Process


def recipe(style: str, mode: str, devices: list[str] | None) -> dict[str, Any]:
    """Build independent v1, v2-engine, and v2-services recipes."""
    store: dict[str, Any] = {"store_config": {"device_name": "shared", "global_segment_size": "100GB"}}
    if devices is not None:
        store["device_names_by_gpu"] = devices
    args = {"kv-transfer-config": '{"kv_connector":"MooncakeStoreConnector"}'}
    modes = ("prefill", "decode") if mode == "disagg" else ("aggregated",)
    data: dict[str, Any] = {
        "name": "mapping-test",
        "model": {"path": "/model", "container": "/container.sqsh", "precision": "bf16"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 4},
        "benchmark": {"type": "manual"},
    }
    if style == "v1":
        for role in modes:
            prefix = "agg" if role == "aggregated" else role
            data["resources"].update({f"{prefix}_nodes": 1, f"{prefix}_workers": 1})
        data["backend"] = {"type": "vllm", "mooncake_kv_store": store, "vllm_config": dict.fromkeys(modes, args)}
    else:
        data.update(
            schema=2,
            roles={"agg" if role == "aggregated" else role: {"nodes": 1, "workers": 1, "args": args} for role in modes},
        )
        if style == "v2-engine":
            data["engine"] = {"type": "vllm", "mooncake_kv_store": store}
        else:
            data["engine"] = "vllm"
            data["services"] = [{"name": "mooncake-master", "type": "mooncake-master", "options": store}]
    return data


def load(tmp_path: Path, data: dict[str, Any]) -> SrtConfig:
    """Exercise the same YAML loader used before submission."""
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(data))
    return SrtConfig.from_yaml(path)


@pytest.mark.parametrize("style", ["v1", "v2-engine", "v2-services"])
@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("devices", [None, [], ["h0", "h1", "h2", "h3"], ["h0", "h0", "h1", "h1"]])
def test_valid_mapping_reaches_runtime_and_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], style: str, mode: str, devices: list[str] | None
) -> None:
    """All authoring paths preserve the map, shared defaults, and process selection."""
    config = load(tmp_path, recipe(style, mode, devices))
    backend = config.backend
    assert isinstance(backend, VLLMProtocol)
    assert backend.mooncake_kv_store is not None
    assert backend.mooncake_kv_store.device_names_by_gpu == (devices or [])
    process = Process("node0", frozenset({2, 3}), 7500, 6100, "decode", 0)
    rendered = backend.build_mooncake_process_config(process, "infra", 4)
    if devices:
        assert rendered is not None
        assert rendered[1]["device_name"] == ",".join(dict.fromkeys(devices[2:]))
        assert rendered[1]["global_segment_size"] == "100GB"
    else:
        assert rendered is None
    assert backend.build_mooncake_store_config("infra")["device_name"] == "shared"
    show_config_details(config)
    output = capsys.readouterr().out
    if devices:
        assert "device_names_by_gpu" in output


@pytest.mark.parametrize("style", ["v1", "v2-engine", "v2-services"])
@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize(
    "devices",
    [
        ["h0"],
        ["h0", "h1", "h2", "h3", "h4"],
        ["h0", "", "h2", "h3"],
        ["h0", " h1", "h2", "h3"],
        ["h0", "h1 ", "h2", "h3"],
        ["h0", "h 1", "h2", "h3"],
        ["h0", "h\t1", "h2", "h3"],
        ["h0", "h1,h2", "h2", "h3"],
    ],
)
def test_invalid_mapping_rejected_before_submission(tmp_path: Path, style: str, mode: str, devices: list[str]) -> None:
    """Reject malformed mappings during load, including aggregated configurations."""
    with pytest.raises(ValidationError, match="device_names_by_gpu"):
        load(tmp_path, recipe(style, mode, devices))


@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("devices", [None, [], ["h0", "h0", "h1", "h1"]])
def test_migration_preserves_process_mapping(tmp_path: Path, mode: str, devices: list[str] | None) -> None:
    """Migrating a v1 map yields loadable, runtime-equivalent service options."""
    original = yaml.safe_dump(recipe("v1", mode, devices))
    migrated = migrate_recipe_text(original)
    doc = yaml.safe_load(migrated.text)
    assert doc["engine"] == "vllm"
    assert "backend" not in doc
    options = next(s for s in doc["services"] if s["type"] == "mooncake-master")["options"]
    if devices is None:
        assert "device_names_by_gpu" not in options
    else:
        assert options["device_names_by_gpu"] == devices
    assert load(tmp_path, doc).backend.mooncake_kv_store.device_names_by_gpu == (devices or [])
    verified = verify_migration_text(original)
    assert verified.status == "ok", verified.detail
