# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM's mooncake-store KV connector on the native mooncake-master service, without Slurm or GPUs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.backends import SGLangBackend, TRTLLMBackend
from srtctl.core.schema import SrtConfig
from srtctl.mock import MockOptions, run_mock_sweep

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "trtllm"
CONNECTOR = {"connector": "mooncake-store"}


def _recipe(frontend: str = "trtllm_serve") -> dict[str, Any]:
    """The example recipe for ``frontend``, with the model and container resolvable without srtslurm.yaml."""
    name = "dynamo-disagg-mooncake.yaml" if frontend == "dynamo" else "trtllm-serve-disagg-mooncake.yaml"
    recipe = yaml.safe_load((EXAMPLES / name).read_text())
    recipe["model"].update(path="hf:Qwen/Qwen3-0.6B", container="trtllm.sqsh")
    return recipe


def _master(recipe: dict[str, Any]) -> dict[str, Any]:
    return next(service for service in recipe["services"] if service["type"] == "mooncake-master")


def _load(recipe: dict[str, Any], tmp_path: Path) -> SrtConfig:
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    return SrtConfig.from_yaml(path)


def _sweep(recipe: dict[str, Any], tmp_path: Path) -> tuple[list[dict[str, Any]], Path]:
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    launches: list[dict[str, Any]] = []
    exit_code = run_mock_sweep(
        config_path=path,
        output_dir=tmp_path / "outputs" / "4301",
        job_id="4301",
        options=MockOptions(child_duration_s=0.05, phase_pause_s=0.01, nodelist=("node-01",), on_srun=launches.append),
    )
    assert exit_code == 0
    return launches, tmp_path / "outputs"


def _written(outputs: Path, filename: str) -> dict[str, Any]:
    (path,) = outputs.rglob(filename)
    return json.loads(path.read_text())


@pytest.mark.parametrize("frontend", ["trtllm_serve", "dynamo"])
def test_workers_join_the_native_master_through_their_role_config(tmp_path: Path, frontend: str) -> None:
    """The native master runs before workers; each role reads its own rendered client config."""
    launches, outputs = _sweep(_recipe(frontend), tmp_path)

    steps = [launch.get("step_name", "") for launch in launches]
    master = steps.index("service_mooncake-master")
    workers = {
        mode: next(launch for launch in launches if launch.get("step_name", "").startswith(f"{mode}_"))
        for mode in ("prefill", "decode")
    }
    assert master < min(steps.index(worker["step_name"]) for worker in workers.values())
    assert launches[master]["command"][:2] == ["mooncake_master", "--port=8700"]
    # Nothing goes through TRT-LLM's own pool provisioning or a donor.
    assert not any(
        "mooncake_donor" in launch["command"] or "mooncake_master" in launch["command"][1:] for launch in launches
    )

    for mode, role in (("prefill", "both"), ("decode", "capacity")):
        worker = workers[mode]
        assert worker["env_to_set"]["MOONCAKE_CONFIG_PATH"] == f"/logs/mooncake_store_config_{mode}.json"
        launcher = worker["command"].index("trtllm-llmapi-launch")
        entry = worker["command"][launcher + 1 : launcher + 4]
        assert entry[0] == "trtllm-serve" if frontend == "trtllm_serve" else entry == ["python3", "-m", "dynamo.trtllm"]
        written = _written(outputs, f"mooncake_store_config_{mode}.json")
        assert written["role"] == role
        assert written["model_key"] == "Qwen/Qwen3-0.6B"
        assert written["stage_through_host"] is True
        assert written["master_server_address"].endswith(":8700")
        assert written["master_server_address"] == worker["env_to_set"]["MOONCAKE_MASTER"]
        # The engine config selects the connector and carries no mooncake_store block.
        engine_yaml = yaml.safe_load(next(outputs.rglob(f"trtllm_config_{mode}.yaml")).read_text())
        assert engine_yaml["kv_connector_config"] == CONNECTOR


def test_role_mooncake_store_config_layers_over_the_shared_keys(tmp_path: Path) -> None:
    _, outputs = _sweep(_recipe(), tmp_path)
    assert _written(outputs, "mooncake_store_config_prefill.json")["global_segment_size"] == "8GiB"
    assert _written(outputs, "mooncake_store_config_decode.json")["global_segment_size"] == "16GiB"


def test_model_key_defaults_to_the_served_model_name(tmp_path: Path) -> None:
    """The pool belongs to this job alone, so the name the workers serve is a safe key; a recipe value wins."""
    recipe = _recipe()
    del recipe["engine"]["served_model_name"]
    launches, outputs = _sweep(recipe, tmp_path)
    prefill = next(launch for launch in launches if launch.get("step_name", "").startswith("prefill_"))
    model_arg = prefill["command"][prefill["command"].index("trtllm-serve") + 1]
    for mode in ("prefill", "decode"):
        assert _written(outputs, f"mooncake_store_config_{mode}.json")["model_key"] == Path(model_arg).name

    _master(recipe)["options"]["store_config"]["model_key"] = "qwen3-pool"
    backend = _load(recipe, tmp_path).backend
    assert isinstance(backend, TRTLLMBackend)
    assert backend.build_mooncake_store_config("decode", "10.1.1.1", "served")["model_key"] == "qwen3-pool"


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_role_mooncake_store_config_is_read_by_trtllm_only(tmp_path: Path, engine: str) -> None:
    recipe = yaml.safe_load((EXAMPLES.parent / engine / "dynamo-disagg.yaml").read_text())
    recipe["model"].update(path="hf:Qwen/Qwen3-0.6B", container=f"{engine}.sqsh")
    _load(recipe, tmp_path)
    recipe["roles"]["decode"]["mooncake_store_config"] = {"global_segment_size": "16GiB"}
    with pytest.raises(ValidationError, match="roles.decode.mooncake_store_config is read by TRT-LLM only"):
        _load(recipe, tmp_path)


def test_role_mooncake_store_config_needs_a_master(tmp_path: Path) -> None:
    """Without a mooncake-master service srtslurm renders no client config, so the per-role keys would be dropped."""
    recipe = _recipe()
    recipe["services"] = []
    for mode in ("prefill", "decode"):
        recipe["roles"][mode]["env"]["MOONCAKE_CONFIG_PATH"] = "/data/mooncake.json"
    with pytest.raises(ValidationError, match="roles.decode.mooncake_store_config needs a mooncake-master service"):
        _load(recipe, tmp_path)


def test_role_without_the_connector_gets_no_client_config(tmp_path: Path) -> None:
    recipe = _recipe()
    del recipe["roles"]["decode"]["args"]["kv_connector_config"]
    del recipe["roles"]["decode"]["mooncake_store_config"]
    launches, outputs = _sweep(recipe, tmp_path)
    decode = next(launch for launch in launches if launch.get("step_name", "").startswith("decode_"))
    assert "MOONCAKE_CONFIG_PATH" not in decode["env_to_set"]
    assert not list(outputs.rglob("mooncake_store_config_decode.json"))
    assert _written(outputs, "mooncake_store_config_prefill.json")["role"] == "both"


def test_render_defaults_role_by_mode_and_owns_the_master_address(tmp_path: Path) -> None:
    recipe = _recipe()
    _master(recipe)["options"]["store_config"]["master_server_address"] = "10.0.0.9:1234"
    recipe["roles"]["decode"]["mooncake_store_config"]["role"] = "both"
    backend = _load(recipe, tmp_path).backend
    assert isinstance(backend, TRTLLMBackend)
    assert backend.mooncake_store_modes() == ("prefill", "decode")
    prefill = backend.build_mooncake_store_config("prefill", "10.1.1.1", "served")
    assert prefill["role"] == "both"
    assert prefill["master_server_address"] == "10.1.1.1:8700"
    assert backend.build_mooncake_store_config("decode", "10.1.1.1", "served")["role"] == "both"
    assert set(backend.mooncake_store_configs("10.1.1.1", "served")) == {
        "mooncake_store_config_prefill.json",
        "mooncake_store_config_decode.json",
    }


def test_engines_without_a_client_config_file_write_none() -> None:
    assert SGLangBackend().mooncake_store_configs("10.1.1.1", "served") == {}
    assert TRTLLMBackend().mooncake_store_configs("10.1.1.1", "served") == {}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: [r["roles"][m]["args"].pop("kv_connector_config") for m in ("prefill", "decode")], "no roles"),
        (
            lambda r: r["roles"]["prefill"]["args"]["kv_connector_config"].update(
                mooncake_store={"pool": "file:///logs/pool.json", "model_key": "m"}
            ),
            "mooncake_store is set",
        ),
        (lambda r: _master(r)["options"]["store_config"].update(model_key=""), "model_key must be a non-empty"),
        (lambda r: _master(r)["options"]["store_config"].update(global_segment_size="16GB"), "'GB' is refused"),
        (lambda r: _master(r)["options"]["store_config"].update(local_buffer_size=0), "at least 1 bytes"),
        (lambda r: _master(r)["options"]["store_config"].update(global_segment_size=-1), "at least 0 bytes"),
        (lambda r: _master(r)["options"]["store_config"].update(transfer_batch_size=0), "positive integer"),
        (lambda r: _master(r)["options"]["store_config"].update(stage_through_host="false"), "not the string"),
        (lambda r: _master(r)["options"]["store_config"].update(segment_size="160GiB"), "set global_segment_size"),
        (lambda r: _master(r)["options"]["store_config"].update(pool="file:///logs/pool.json"), "drop it"),
        (lambda r: _master(r).update(external="10.0.0.5:8700"), "external mooncake-master"),
        (lambda r: r["roles"]["decode"]["mooncake_store_config"].update(role="donor"), "role must be"),
        (
            lambda r: r["roles"]["decode"]["args"].pop("kv_connector_config"),
            r"roles.decode.mooncake_store_config is set, but only \['prefill'\]",
        ),
        (
            lambda r: r["roles"]["decode"]["mooncake_store_config"].update(protocol="tcp"),
            "roles.decode.mooncake_store_config: protocol must be the same for every server of the pool",
        ),
        (
            lambda r: r["roles"]["decode"]["mooncake_store_config"].update(model_key="other"),
            "model_key must be the same for every server of the pool",
        ),
    ],
)
def test_invalid_recipes_fail_at_load(tmp_path: Path, mutate, message: str) -> None:
    recipe = _recipe()
    mutate(recipe)
    with pytest.raises(ValidationError, match=message):
        _load(recipe, tmp_path)


def _own_pool(frontend: str, pool: dict[str, Any]) -> dict[str, Any]:
    """The example without the mooncake-master service, each role naming its pool through ``pool``."""
    recipe = _recipe(frontend)
    recipe["services"] = []
    for mode in ("prefill", "decode"):
        recipe["roles"][mode].pop("mooncake_store_config", None)
        for key, value in pool.items():
            recipe["roles"][mode][key] = {**recipe["roles"][mode][key], **copy.deepcopy(value)}
    return recipe


BLOCK = {"pool": "10.0.0.5:50051", "model_key": "m"}


@pytest.mark.parametrize(
    ("frontend", "pool"),
    [
        ("trtllm_serve", {"env": {"MOONCAKE_CONFIG_PATH": "/data/mooncake.json"}}),
        ("dynamo", {"env": {"MOONCAKE_CONFIG_PATH": "/data/mooncake.json"}}),
        (
            "trtllm_serve",
            {"args": {"kv_connector_config": {**CONNECTOR, "mooncake_store": {**BLOCK, "run_dir": "/d"}}}},
        ),
    ],
)
def test_connector_with_a_pool_of_its_own_needs_no_master(tmp_path: Path, frontend: str, pool: dict[str, Any]) -> None:
    """A role whose ranks can find a client config of their own is left alone."""
    recipe = _own_pool(frontend, pool)
    if "args" in pool:  # one server owns a run_dir, so each role needs its own
        recipe["roles"]["decode"]["args"]["kv_connector_config"]["mooncake_store"]["run_dir"] = "/d-decode"
    assert _load(recipe, tmp_path).backend.mooncake_kv_store is None


@pytest.mark.parametrize(
    ("frontend", "pool"),
    [
        ("trtllm_serve", {}),
        # The ranks run under trtllm-llmapi-launch and never see the path trtllm-serve exports.
        ("trtllm_serve", {"args": {"kv_connector_config": {**CONNECTOR, "mooncake_store": BLOCK}}}),
        # dynamo.trtllm renders no client config from the block at all.
        ("dynamo", {"args": {"kv_connector_config": {**CONNECTOR, "mooncake_store": {**BLOCK, "run_dir": "/d"}}}}),
    ],
)
def test_connector_without_a_reachable_pool_fails_at_load(tmp_path: Path, frontend: str, pool: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="would not find a Mooncake client config"):
        _load(_own_pool(frontend, pool), tmp_path)


def test_connector_named_by_module_is_recognized(tmp_path: Path) -> None:
    """TRT-LLM resolves the connector by module, so the explicit module form gets a client config too."""
    recipe = _recipe()
    recipe["roles"]["decode"]["args"]["kv_connector_config"] = {
        "connector_module": "tensorrt_llm._torch.pyexecutor.connectors.mooncake_store",
        "connector_scheduler_class": "MooncakeStoreConnectorScheduler",
        "connector_worker_class": "MooncakeStoreConnectorWorker",
    }
    backend = _load(recipe, tmp_path).backend
    assert isinstance(backend, TRTLLMBackend)
    assert backend.mooncake_store_modes() == ("prefill", "decode")
    # A preset name with another module resolves to that module, as in TRT-LLM.
    recipe["roles"]["decode"]["args"]["kv_connector_config"] = {**CONNECTOR, "connector_module": "my.connector"}
    del recipe["roles"]["decode"]["mooncake_store_config"]
    assert _load(recipe, tmp_path).backend.mooncake_store_modes() == ("prefill",)


def test_role_and_flags_are_read_as_trtllm_reads_them(tmp_path: Path) -> None:
    recipe = _recipe()
    _master(recipe)["options"]["store_config"]["stage_through_host"] = 1
    recipe["roles"]["decode"]["mooncake_store_config"]["role"] = " Capacity "
    assert _load(recipe, tmp_path).backend.mooncake_kv_store is not None


def test_block_only_keys_are_reported_where_they_were_written(tmp_path: Path) -> None:
    for where, mutate in (
        ("options.store_config:", lambda r: _master(r)["options"]["store_config"].update(master_timeout=900)),
        (
            "roles.decode.mooncake_store_config:",
            lambda r: r["roles"]["decode"]["mooncake_store_config"].update(run_dir="/logs/mc"),
        ),
    ):
        recipe = _recipe()
        mutate(recipe)
        with pytest.raises(ValidationError, match=where):
            _load(recipe, tmp_path)


@pytest.mark.parametrize(("value", "ok"), [("64", True), (64.0, True), ("lots", False), (True, False)])
def test_transfer_batch_size_is_read_with_int(tmp_path: Path, value: Any, ok: bool) -> None:
    recipe = _recipe()
    _master(recipe)["options"]["store_config"]["transfer_batch_size"] = value
    if ok:
        _load(recipe, tmp_path)
    else:
        with pytest.raises(ValidationError, match="positive integer"):
            _load(recipe, tmp_path)


def test_non_finite_size_names_the_key(tmp_path: Path) -> None:
    recipe = _recipe()
    _master(recipe)["options"]["store_config"]["global_segment_size"] = float("inf")
    with pytest.raises(ValidationError, match="global_segment_size must be a finite size"):
        _load(recipe, tmp_path)


def test_disabled_external_master_entry_is_ignored(tmp_path: Path) -> None:
    recipe = _recipe()
    recipe["services"].append(
        {"name": "mooncake-master-ext", "type": "mooncake-master", "enabled": False, "external": "10.0.0.5:8700"}
    )
    assert _load(recipe, tmp_path).backend.mooncake_kv_store is not None


def test_run_dir_shared_by_several_servers_fails_at_load(tmp_path: Path) -> None:
    block = {**CONNECTOR, "mooncake_store": {**BLOCK, "run_dir": "/logs/mc"}}
    recipe = _own_pool("trtllm_serve", {"args": {"kv_connector_config": block}})
    with pytest.raises(ValidationError, match="would be shared by several servers"):
        _load(recipe, tmp_path)
    recipe = _own_pool("trtllm_serve", {"args": {"kv_connector_config": block}})
    del recipe["roles"]["decode"]["args"]["kv_connector_config"]
    recipe["roles"]["prefill"]["workers"] = 2
    with pytest.raises(ValidationError, match="would be shared by several servers"):
        _load(recipe, tmp_path)


def test_services_only_trtllm_recipe_still_loads(tmp_path: Path) -> None:
    """No connector role, no master, no frontend: nothing to check, and frontend.type none must not be looked up."""
    recipe = yaml.safe_load((EXAMPLES.parent / "features" / "torchrun-pool.yaml").read_text())
    recipe["engine"] = {"type": "trtllm"}
    _load(recipe, tmp_path)


def test_dynamo_sidecar_engine_provisions_its_own_pool(tmp_path: Path) -> None:
    """A Dynamo sidecar runs the engine as tensorrt_llm.commands.serve, which renders the block's config."""
    recipe = _recipe("dynamo")
    recipe["services"] = []
    agg = recipe["roles"].pop("prefill")
    del recipe["roles"]["decode"]
    agg["args"]["kv_connector_config"] = {**CONNECTOR, "mooncake_store": {**BLOCK, "run_dir": "/logs/mc"}}
    recipe["roles"] = {"agg": agg}
    with pytest.raises(ValidationError, match="would not find a Mooncake client config"):
        _load(recipe, tmp_path)
    recipe["dynamo"]["sidecar"] = True
    assert _load(recipe, tmp_path).backend.mooncake_store_modes() == ("agg",)
