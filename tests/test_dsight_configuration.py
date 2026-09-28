# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved configuration remains optional, source-backed metadata beside runtime data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pytest
from test_dsight import write_run
from test_dsight_log_metrics import batch, config

from srtctl.dsight.build import _browser_payload
from srtctl.dsight.cli import add_commands
from srtctl.dsight.configuration import ConfigComparison, MetricConfiguration
from srtctl.dsight.configuration.reader import attach_configuration
from srtctl.dsight.importer import Importer
from srtctl.dsight.log_metrics.tokenspeed import ACTIVE_DECODE, ACTIVE_PAGES, DECODE_LIMIT, POOL_PAGES
from srtctl.dsight.query import TraceDataset


def scheduler(second=30, maximum=8, dp=1):
    return config(second, str(maximum)).replace("global max_num_seqs=64, dp_size=8", f"global max_num_seqs={maximum}, dp_size={dp}")


def recipe(value="16"):
    return (
        "schema: 2\nengine:\n  type: tokenspeed\n  args:\n    max-num-seqs: 64\n"
        "roles:\n  prefill:\n    args:\n      max-num-seqs: 128\n"
        "  decode:\n    args:\n      max-num-seqs: " + value + "\n"
        "      max-total-tokens: 8192\n      prefix-granularity: 64\n"
        "environment:\n  UNUSED_PRIVATE_FIELD: SENSITIVE_VALUE\n"
    )


def configured_run(root: Path, content: str | None = None, lines: list[str] | None = None, **options):
    logs, _ = write_run(root)
    (logs / "decode-host_decode_w0.out").write_text("\n".join(lines or [scheduler(), batch(), batch(36)]) + "\n")
    path = logs.parent / "recipe.yaml"
    if content is not None:
        path.write_text(content)
    run = Importer(logs, iteration_timezone="UTC", **options)
    return run, run.run(), path


def metric(data, name=ACTIVE_DECODE):
    return next(series for series in data["metrics"] if series["name"] == name)


def test_recipe_metadata_preserves_runtime_samples_and_exact_lineage(tmp_path):
    run, original, path = configured_run(tmp_path)
    path.write_text(recipe())
    data = Importer(run.logs, iteration_timezone="UTC").run()
    assert [(s["name"], s["points"]) for s in data["metrics"]] == [(s["name"], s["points"]) for s in original["metrics"]]
    assert data["metric_catalog"] == original["metric_catalog"]
    active = metric(data)
    annotation = active["configuration"][0]
    assert annotation["source"]["value"] == 16
    assert annotation["source"]["field"] == "roles.decode.args.max-num-seqs"
    source = data["sources"][annotation["source"]["source_id"]]
    assert source["kind"] == "run_config" and Path(source["path"]) == path
    assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_text().splitlines()[annotation["source"]["line"] - 1].strip() == "max-num-seqs: 16"
    comparison = annotation["comparison"]
    assert comparison["value"] == 16 and comparison["unit"] == "requests" and comparison["start"] == -1
    evidence = comparison["evidence"][0]
    assert evidence["field"] == "dp_size" and evidence["value"] == 1 and evidence["line"] == 1
    assert data["sources"][evidence["source_id"]]["kind"] == "worker_log"
    assert metric(data, DECODE_LIMIT)["points"][0][1] == 8  # Logged value is not replaced by the recipe's 16.
    assert metric(data, DECODE_LIMIT)["configuration"][0]["source"]["value"] == 16
    assert "SENSITIVE_VALUE" not in json.dumps(data)
    item = TraceDataset(data).query("metrics", name=ACTIVE_DECODE, points=True)["items"][0]
    assert item["configuration"] == active["configuration"] and item["max"] == 4


def test_missing_config_and_optional_fields_add_no_empty_metadata(tmp_path):
    _, data, path = configured_run(tmp_path)
    assert all("configuration" not in series for series in data["metrics"])
    path.write_text("engine:\n  type: tokenspeed\n")
    data = Importer(path.parent / "logs", iteration_timezone="UTC").run()
    assert all("configuration" not in series for series in data["metrics"])


def test_kv_configuration_retains_units_without_inventing_a_page_limit(tmp_path):
    _, data, _ = configured_run(tmp_path, recipe())
    for name in (ACTIVE_PAGES, POOL_PAGES):
        fields = metric(data, name)["configuration"]
        assert [entry["source"]["value"] for entry in fields] == [8192, 64]
        assert all(entry["unit"] == "tokens" and entry["comparison"] is None for entry in fields)
    assert metric(data, POOL_PAGES)["points"][0][1] == 128


@pytest.mark.parametrize("value", ["null", "0", "-1", "true", '"auto"'])
def test_invalid_role_override_does_not_reuse_engine_default(tmp_path, value):
    _, data, _ = configured_run(tmp_path, recipe(value))
    annotation = metric(data)["configuration"][0]
    assert annotation["source"]["field"].startswith("roles.decode.")
    assert annotation["comparison"] is None
    assert annotation["source"]["value"] != 64


def test_engine_fallback_retains_actual_field_path(tmp_path):
    _, data, _ = configured_run(tmp_path, "engine:\n  type: tokenspeed\n  args:\n    max-num-seqs: 24\n")
    annotation = metric(data)["configuration"][0]
    assert annotation["source"]["field"] == "engine.args.max-num-seqs"
    assert annotation["source"]["value"] == 24


@pytest.mark.parametrize("lines", [
    [scheduler(dp=2), batch()],
    [config().replace(", dp_size=8", ""), batch()],
    [scheduler(), batch(), scheduler(35, dp=2), batch(36)],
    [scheduler(), scheduler(dp=2), batch()],
    [batch()],
])
def test_unknown_changed_or_multiple_dp_has_context_without_capacity_overlay(tmp_path, lines):
    _, data, _ = configured_run(tmp_path, recipe(), lines)
    annotation = metric(data)["configuration"][0]
    assert annotation["source"]["value"] == 16 and annotation["comparison"] is None
    assert "per-scheduler" in annotation["note"]


def test_late_scope_evidence_does_not_project_back_before_it(tmp_path):
    _, data, _ = configured_run(tmp_path, recipe(), [batch(), scheduler(35), batch(36)])
    assert metric(data)["configuration"][0]["comparison"]["start"] == 4


def test_comparison_does_not_borrow_another_rank_or_worker(tmp_path):
    _, data, _ = configured_run(tmp_path, recipe(), [scheduler().replace("RANK 0", "RANK 1"), batch()])
    assert metric(data)["configuration"][0]["comparison"] is None


@pytest.mark.parametrize("content", ["roles: [broken", "- not\n- a mapping\n", "engine:\n  type: tokenspeed\nroles:\n  decode:\n    args: []\n", recipe(".nan")])
def test_bad_optional_configuration_keeps_runtime_metrics(tmp_path, content):
    _, data, _ = configured_run(tmp_path, content)
    assert metric(data)["points"][0][1] == 4
    assert "configuration" not in metric(data)
    if "args: []" not in content:
        assert any("Configuration" in warning for warning in data["meta"]["warnings"])


def test_duplicate_keys_are_not_silently_accepted(tmp_path):
    _, data, _ = configured_run(tmp_path, recipe().replace("max-num-seqs: 16", "max-num-seqs: 16\n      max-num-seqs: 32"))
    assert "configuration" not in metric(data)
    assert any("duplicate" in warning for warning in data["meta"]["warnings"])


def test_explicit_config_selects_one_source_and_ambiguous_discovery_omits_it(tmp_path):
    run, _, path = configured_run(tmp_path, recipe())
    selected = run.logs / "recipe.yaml"
    selected.write_text(recipe("32"))
    data = Importer(run.logs, iteration_timezone="UTC").run()
    assert "configuration" not in metric(data)
    assert any("multiple recipe.yaml" in warning for warning in data["meta"]["warnings"])
    data = Importer(run.logs, iteration_timezone="UTC", config=selected).run()
    annotation = metric(data)["configuration"][0]
    assert annotation["source"]["value"] == 32
    assert data["sources"][annotation["source"]["source_id"]]["path"] == str(selected)
    data = Importer(run.logs, iteration_timezone="UTC", config=path.with_name("missing.yaml")).run()
    assert "configuration" not in metric(data)
    assert any("does not exist" in warning for warning in data["meta"]["warnings"])


def test_configuration_survives_lazy_html_payload_without_new_metric_families(tmp_path):
    _, data, _ = configured_run(tmp_path, recipe())
    payload, elements = _browser_payload(gzip.compress(json.dumps(data).encode()), data)
    core = json.loads(gzip.decompress(payload))
    assert metric(core)["configuration"] == json.loads(json.dumps(metric(data)["configuration"]))
    assert "points" not in metric(core)
    assert set(core["metric_payloads"]) == {series["name"] for series in data["metrics"]}
    assert len(elements.splitlines()) == len(core["metric_payloads"])
    assert "SENSITIVE_VALUE" not in json.dumps(core)


def test_second_adapter_attaches_metadata_to_native_tachometer_series(tmp_path):
    run, data, _path = configured_run(tmp_path, "engine:\n  type: example\ncapacity: 20\n")
    assert all("configuration" not in series for series in data["metrics"])

    class ExampleConfiguration:
        def read(self, document, series, role, metric_series, logs):
            field = document.field("capacity")
            if series["name"] != "trtllm_num_requests_running" or field is None:
                return ()
            return (MetricConfiguration("Configured capacity", "requests", "Worker", field,
                                        ConfigComparison(20, "requests", 0, "Direct worker setting")),)

    attach_configuration(run, {"example": ExampleConfiguration()})
    native = [s for s in run.metric_series if s["name"] == "trtllm_num_requests_running"]
    assert native and all(s["configuration"][0]["source"]["field"] == "capacity" for s in native)
    assert all(s["configuration"][0]["comparison"]["value"] == 20 for s in native)
    assert {s["name"] for s in run.metric_series} == {s["name"] for s in data["metrics"]}


def test_cli_exposes_optional_saved_config_path():
    parser = argparse.ArgumentParser()
    add_commands(parser)
    args = parser.parse_args(["build", "/run", "-o", "/output", "--config", "/saved/recipe.yaml"])
    assert args.config == Path("/saved/recipe.yaml")
