# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `observability.enabled` knob and its config expansion."""

import copy

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.core.config import (
    expand_observability,
    expand_trtllm_engine_defaults,
    expand_trtllm_serve_defaults,
    legacy_keys_present,
    load_config,
)
from srtctl.core.schema import SrtConfig

BASE_CONFIG = {
    "name": "test-job",
    "model": {"path": "/models/test-model", "container": "test.sqsh", "precision": "fp8"},
    "resources": {"gpu_type": "h100", "gpus_per_node": 8},
    "roles": {"prefill": {"nodes": 1, "workers": 1}, "decode": {"nodes": 1, "workers": 1}},
}


def _base_config() -> dict:
    return copy.deepcopy(BASE_CONFIG)


def _trtllm_config(**observability):
    cfg = _base_config()
    cfg["engine"] = {"type": "trtllm"}
    cfg["roles"]["prefill"].update({"env": {"TLLM_LOG_LEVEL": "INFO"}, "args": {"max_batch_size": 256}})
    cfg["roles"]["decode"].update({"env": {}, "args": {"max_batch_size": 64}})
    cfg["frontend"] = {"env": {"DYN_TOKENIZER": "fastokens"}}
    if observability:
        cfg["observability"] = observability
    return cfg


def _role_args(cfg: dict) -> dict[str, dict]:
    """``roles.<role>.args`` of every role that has args, by role."""
    return {role: spec["args"] for role, spec in cfg["roles"].items() if "args" in spec}


def _recipe(cfg):
    """``cfg`` as a recipe file: the fixtures are in the recipe layout already, so only ``schema`` is added."""
    recipe = {"schema": 2, **copy.deepcopy(cfg)}
    assert not legacy_keys_present(recipe), legacy_keys_present(recipe)
    return recipe


# --------------------------------------------------------------- expansion ---
class TestExpandObservability:
    @pytest.mark.parametrize("enabled", [None, False, True])
    def test_metrics_default_does_not_depend_on_observability(self, enabled):
        cfg = _trtllm_config() if enabled is None else _trtllm_config(enabled=enabled)
        loaded = SrtConfig.Schema().load(expand_observability(cfg))

        assert loaded.backend.publish_metrics is True
        assert loaded.backend.publish_events_and_metrics is None
        assert loaded.backend.dynamo_metrics_flags == ("--publish-metrics",)

    @pytest.mark.parametrize("loader_name", ["from_yaml", "load_config"])
    @pytest.mark.parametrize("enabled", [False, True])
    @pytest.mark.parametrize("publish_metrics", [False, True])
    @pytest.mark.parametrize("publish_events_and_metrics", [None, False, True])
    def test_publishing_yaml_loaders_and_roundtrip(
        self, tmp_path, monkeypatch, loader_name, enabled, publish_metrics, publish_events_and_metrics
    ):
        monkeypatch.setattr("srtctl.core.config.load_cluster_config", lambda: None)
        cfg = _trtllm_config(enabled=enabled)
        cfg["benchmark"] = {"type": "sa-bench", "concurrencies": [4]}
        cfg["engine"]["publish_metrics"] = publish_metrics
        cfg["engine"]["publish_events_and_metrics"] = publish_events_and_metrics
        loader = SrtConfig.from_yaml if loader_name == "from_yaml" else load_config
        expected_events = publish_events_and_metrics
        expected_flags = (
            ("--publish-events-and-metrics",)
            if expected_events is True
            else (("--publish-metrics",) if publish_metrics else ())
        )

        # Check the raw recipe and a recipe carrying the schema-dumped backend
        # through the same real loader: configured values and effective flags
        # must both survive.
        recipe = _recipe(cfg)
        for filename in ("recipe.yaml", "roundtrip.yaml"):
            path = tmp_path / filename
            path.write_text(yaml.safe_dump(recipe))
            loaded = loader(path)
            assert loaded.backend.publish_metrics is publish_metrics
            assert loaded.backend.publish_events_and_metrics is expected_events
            assert loaded.backend.dynamo_metrics_flags == expected_flags
            dumped = SrtConfig.Schema().dump(loaded)
            assert dumped["engine"]["publish_metrics"] is publish_metrics
            assert dumped["engine"]["publish_events_and_metrics"] is expected_events
            # A dump is a recipe: carry its engine and roles through the second pass.
            recipe = _recipe({**cfg, "engine": dumped["engine"], "roles": dumped["roles"]})

    @pytest.mark.parametrize("loader_name", ["from_yaml", "load_config"])
    @pytest.mark.parametrize("publish_metrics", [False, True])
    @pytest.mark.parametrize("explicit_false", [False, True])
    def test_schema_dump_preserves_omission_before_observability_is_enabled(
        self, tmp_path, monkeypatch, loader_name, publish_metrics, explicit_false
    ):
        """A schema dump (a saved or locked recipe) preserves the legacy setting without promoting it."""
        monkeypatch.setattr("srtctl.core.config.load_cluster_config", lambda: None)
        cfg = _trtllm_config()
        cfg["benchmark"] = {"type": "sa-bench", "concurrencies": [4]}
        cfg["engine"]["publish_metrics"] = publish_metrics
        if explicit_false:
            cfg["engine"]["publish_events_and_metrics"] = False
        else:
            assert "publish_events_and_metrics" not in cfg["engine"]
        schema = SrtConfig.Schema()
        dumped = schema.dump(schema.load(cfg))
        assert "publish_events_and_metrics" in dumped["engine"]
        assert dumped["engine"]["publish_events_and_metrics"] is (False if explicit_false else None)

        dumped["observability"]["enabled"] = True
        path = tmp_path / "enable-observability.yaml"
        path.write_text(
            yaml.safe_dump(
                _recipe(
                    {
                        **cfg,
                        "engine": dumped["engine"],
                        "roles": dumped["roles"],
                        "observability": dumped["observability"],
                    }
                )
            )
        )
        loader = SrtConfig.from_yaml if loader_name == "from_yaml" else load_config
        loaded = loader(path)

        assert loaded.backend.publish_metrics is publish_metrics
        assert loaded.backend.publish_events_and_metrics is (False if explicit_false else None)
        assert loaded.backend.dynamo_metrics_flags == (("--publish-metrics",) if publish_metrics else ())

    def test_disabled_is_a_noop(self):
        cfg = expand_observability(_trtllm_config(enabled=False))
        assert "publish_events_and_metrics" not in cfg["engine"]
        assert "DYN_LOGGING_SPAN_EVENTS" not in cfg["roles"]["prefill"]["env"]

    def test_absent_block_is_a_noop(self):
        cfg = expand_observability(_trtllm_config())
        assert "publish_events_and_metrics" not in cfg["engine"]

    def test_enabled_expands_span_env_on_all_three_roles(self):
        cfg = expand_observability(_trtllm_config(enabled=True))
        for env in (
            cfg["roles"]["prefill"]["env"],
            cfg["roles"]["decode"]["env"],
            cfg["frontend"]["env"],
        ):
            assert env["DYN_LOGGING_SPAN_EVENTS"] == "true"
            assert env["DYN_LOGGING_JSONL"] == "true"
            # SPAN_CLOSED is emitted at DEBUG; anything higher yields no traces.
            assert env["DYN_LOG"] == "debug"

    def test_enabled_expands_request_trace_env_on_the_frontend_only(self):
        """The request-trace leg is frontend-only and uses Dynamo defaults.

        Dynamo's master switch keeps its default file sink and rotated jsonl_gz
        format. The built-in file path is /tmp, which dies with the job, so the
        override is what makes the capture survivable.
        """
        cfg = expand_observability(_trtllm_config(enabled=True))
        fe = cfg["frontend"]["env"]
        assert fe["DYN_REQUEST_TRACE"] == "1"
        assert "DYN_REQUEST_TRACE_SINKS" not in fe
        assert fe["DYN_REQUEST_TRACE_FILE_PATH"].startswith("/logs/")

        # Workers have no RequestTracker; tracing them would write empty files.
        for mode in ("prefill", "decode"):
            assert "DYN_REQUEST_TRACE" not in cfg["roles"][mode]["env"]

    def test_enabled_turns_on_metrics_surface_and_iteration_stats(self):
        cfg = expand_observability(_trtllm_config(enabled=True))
        assert "telemetry" not in cfg
        assert "publish_events_and_metrics" not in cfg["engine"]
        assert SrtConfig.Schema().load(cfg).backend.publish_metrics is True
        for mode in ("prefill", "decode"):
            section = cfg["roles"][mode]["args"]
            # enable_iter_perf_stats is what produces trtllm_kv_cache_*_blocks.
            assert section["enable_iter_perf_stats"] is True
            assert section["return_perf_metrics"] is True

    def test_enabled_creates_args_for_every_role(self):
        """A recipe with no engine yaml still gets the iteration-level gauges:
        every declared role gets args; no role is invented."""
        cfg = _trtllm_config(enabled=True)
        for spec in cfg["roles"].values():
            spec.pop("args", None)
        out = expand_observability(cfg)
        assert _role_args(out) == {
            "prefill": {"enable_iter_perf_stats": True, "return_perf_metrics": True},
            "decode": {"enable_iter_perf_stats": True, "return_perf_metrics": True},
        }
        assert "agg" not in out["roles"]

    def test_explicit_recipe_values_win(self):
        """setdefault semantics: an explicit recipe value must never be clobbered."""
        cfg = _trtllm_config(enabled=True)
        cfg["roles"]["decode"]["env"]["DYN_LOG"] = "info"
        cfg["roles"]["decode"]["args"]["return_perf_metrics"] = False
        cfg["engine"]["publish_events_and_metrics"] = False
        out = expand_observability(cfg)
        assert out["roles"]["decode"]["env"]["DYN_LOG"] == "info"
        assert out["roles"]["decode"]["args"]["return_perf_metrics"] is False
        assert out["engine"]["publish_events_and_metrics"] is False

    @pytest.mark.parametrize("frontend_type", ["dynamo", "trtllm_serve"])
    @pytest.mark.parametrize("publish_metrics", [False, True])
    @pytest.mark.parametrize("publish_events_and_metrics", [False, True])
    def test_explicit_publishing_settings_are_preserved_without_legacy_warnings(
        self, caplog, frontend_type, publish_metrics, publish_events_and_metrics
    ):
        cfg = _trtllm_config(enabled=True)
        cfg["frontend"]["type"] = frontend_type
        cfg["engine"]["publish_metrics"] = publish_metrics
        cfg["engine"]["publish_events_and_metrics"] = publish_events_and_metrics

        with caplog.at_level("WARNING"):
            out = expand_observability(cfg)

        assert out["engine"]["publish_metrics"] is publish_metrics
        assert out["engine"]["publish_events_and_metrics"] is publish_events_and_metrics
        publishing_warnings = [record for record in caplog.records if "publish_events_and_metrics" in record.message]
        assert not publishing_warnings

    def test_observability_preserves_disabled_metrics_without_enabling_legacy_flag(self, caplog):
        cfg = _trtllm_config(enabled=True)
        cfg["engine"]["publish_metrics"] = False

        with caplog.at_level("WARNING"):
            out = expand_observability(cfg)

        assert out["engine"]["publish_metrics"] is False
        assert "publish_events_and_metrics" not in out["engine"]
        assert SrtConfig.Schema().load(out).backend.dynamo_metrics_flags == ()
        assert not [record for record in caplog.records if "publish_metrics" in record.message]

    def test_preexisting_env_is_preserved(self):
        cfg = expand_observability(_trtllm_config(enabled=True))
        assert cfg["roles"]["prefill"]["env"]["TLLM_LOG_LEVEL"] == "INFO"
        assert cfg["frontend"]["env"]["DYN_TOKENIZER"] == "fastokens"

    def test_non_trtllm_backend_keeps_span_env_but_no_engine_keys(self):
        cfg = _base_config()
        cfg["engine"] = "sglang"
        cfg["observability"] = {"enabled": True}
        out = expand_observability(cfg)
        assert out["roles"]["prefill"]["env"]["DYN_LOGGING_SPAN_EVENTS"] == "true"
        assert out["engine"] == "sglang"

    def test_missing_sections_are_created(self):
        out = expand_observability({**_base_config(), "observability": {"enabled": True}})
        assert out["roles"]["prefill"]["env"]["DYN_LOGGING_JSONL"] == "true"
        assert out["frontend"]["env"]["DYN_LOGGING_JSONL"] == "true"

    def test_nested_tachometer_settings_stay_under_observability(self):
        cfg = _trtllm_config(
            enabled=True,
            tachometer={
                "enabled": True,
                "collect_interval_ms": 500,
                "dcgm_exporter": {"container_image": "dcgm", "port": 9401},
            },
        )

        out = expand_observability(cfg)

        assert out["observability"]["tachometer"] == {
            "enabled": True,
            "collect_interval_ms": 500,
            "dcgm_exporter": {"container_image": "dcgm", "port": 9401},
        }
        assert "telemetry" not in out

    def test_tachometer_and_power_telemetry_can_be_enabled_together(self):
        cfg = _trtllm_config(enabled=True, tachometer={"enabled": True})
        cfg["telemetry"] = {
            "enabled": True,
            "collect_interval_ms": 1000,
            "dcgm_exporter": {"container_image": "dcgm", "port": 9401},
        }

        out = expand_observability(cfg)

        assert out["observability"]["tachometer"]["enabled"] is True
        assert out["telemetry"]["enabled"] is True

    def test_tachometer_reuses_the_power_dcgm_exporter(self):
        cfg = {
            **BASE_CONFIG,
            "benchmark": {"type": "sa-bench", "concurrencies": [4]},
            "observability": {"enabled": True, "tachometer": {"enabled": True}},
            "telemetry": {
                "enabled": True,
                "dcgm_exporter": {"container_image": "dcgm", "port": 9401},
            },
        }

        loaded = SrtConfig.Schema().load(cfg)

        assert loaded.observability.tachometer.enabled is True
        assert loaded.telemetry.dcgm_exporter.container_image == "dcgm"

    def test_tachometer_rejects_a_duplicate_power_dcgm_exporter(self):
        cfg = {
            **BASE_CONFIG,
            "benchmark": {"type": "sa-bench", "concurrencies": [4]},
            "observability": {
                "enabled": True,
                "tachometer": {
                    "enabled": True,
                    "dcgm_exporter": {"container_image": "tach-dcgm", "port": 9401},
                },
            },
            "telemetry": {
                "enabled": True,
                "dcgm_exporter": {"container_image": "power-dcgm", "port": 9401},
            },
        }

        with pytest.raises(ValidationError, match="shared DCGM exporter"):
            SrtConfig.Schema().load(cfg)

    def test_tachometer_rejects_the_power_artifact_directory(self):
        cfg = {
            **BASE_CONFIG,
            "benchmark": {"type": "sa-bench", "concurrencies": [4]},
            "observability": {
                "enabled": True,
                "tachometer": {"enabled": True, "storage_subdir": "power"},
            },
            "telemetry": {
                "enabled": True,
                "storage_subdir": "power",
                "dcgm_exporter": {"container_image": "dcgm", "port": 9401},
            },
        }

        with pytest.raises(ValidationError, match="storage_subdir"):
            SrtConfig.Schema().load(cfg)

    def test_retired_build_dashboard_knob_is_rejected(self):
        """Recipes carrying the removed dashboard option must fail at submit time."""
        cfg = _trtllm_config(enabled=True, build_dashboard=False)

        with pytest.raises(ValidationError, match="build_dashboard"):
            SrtConfig.Schema().load(cfg)

    def test_retired_raw_scraper_knobs_are_rejected(self):
        """The in-job RAW Prometheus scraper is gone; Tachometer is the capture.

        A recipe still carrying `scrape_metrics` (or the other scrape_* knobs)
        must fail loudly at submit time rather than silently promising a
        raw_prometheus.jsonl that will never be written. Tachometer's parquet is
        the only in-job metrics capture."""
        cfg = _trtllm_config(enabled=True, scrape_metrics=True)

        with pytest.raises(ValidationError, match="scrape_metrics"):
            SrtConfig.Schema().load(cfg)

    def test_retired_hz_frequency_knob_is_rejected_on_tachometer(self):
        """``default_frequency`` (Hz) is retired in favor of ``collect_interval_ms``.

        A recipe still carrying the Hz knob must fail loudly at submit time:
        silently accepting it would run at the 1000ms default while promising
        a different cadence."""
        cfg = _trtllm_config(enabled=True, tachometer={"default_frequency": 2.0})

        with pytest.raises(ValidationError, match="default_frequency"):
            SrtConfig.Schema().load(cfg)

    def test_retired_frequency_knob_is_rejected_on_power_telemetry(self):
        """Power telemetry's ``default_frequency`` was a period in seconds
        despite its name; it is retired in favor of ``collect_interval_ms``."""
        cfg = _base_config()
        cfg["benchmark"] = {"type": "sa-bench", "concurrencies": [4]}
        cfg["telemetry"] = {
            "enabled": True,
            "default_frequency": 1.0,
            "dcgm_exporter": {"container_image": "dcgm", "port": 9401},
        }

        with pytest.raises(ValidationError, match="default_frequency"):
            SrtConfig.Schema().load(cfg)

    def test_tachometer_no_longer_requires_master_observability_knob(self):
        """Tachometer is decoupled from observability.enabled: explicit true
        without the master knob is valid, and the default is on for every run."""
        cfg = _trtllm_config(enabled=False, tachometer={"enabled": True})
        loaded = SrtConfig.Schema().load(cfg)
        assert loaded.observability.tachometer_enabled is True

    def test_tachometer_is_on_by_default_without_observability(self):
        cfg = _trtllm_config(enabled=False)
        loaded = SrtConfig.Schema().load(cfg)
        assert loaded.observability.tachometer_enabled is True

    def test_tachometer_explicit_false_still_opts_out(self):
        cfg = _trtllm_config(enabled=False, tachometer={"enabled": False})
        loaded = SrtConfig.Schema().load(cfg)
        assert loaded.observability.tachometer_enabled is False

    def test_telemetry_rejects_tachometer_fields(self):
        cfg = {
            **BASE_CONFIG,
            "telemetry": {
                "enabled": True,
                "binary_path": "tachometer-scraper",
            },
        }

        with pytest.raises(ValidationError, match="binary_path"):
            SrtConfig.Schema().load(cfg)


def _trtllm_serve_config(**observability):
    cfg = _trtllm_config(**observability)
    cfg["frontend"] = {"type": "trtllm_serve", "enable_multiple_frontends": False}
    return cfg


class TestTrtllmServeDefaults:
    """trtllm-serve mounts a worker's /prometheus/metrics route only when the engine
    runs with return_perf_metrics: true, and TensorRT-LLM's own default is false.
    srtctl bakes that default in for every trtllm_serve recipe so Tachometer's
    backend_* endpoints are never a silent 404."""

    def test_return_perf_metrics_defaults_on_without_observability(self):
        out = expand_trtllm_serve_defaults(_trtllm_serve_config())
        for mode in ("prefill", "decode"):
            section = out["roles"][mode]["args"]
            assert section["return_perf_metrics"] is True
            # Only this key is touched; the recipe's own values survive.
            assert "enable_iter_perf_stats" not in section
            assert section["max_batch_size"] in (256, 64)

    def test_explicit_false_wins_and_warns(self, caplog):
        cfg = _trtllm_serve_config()
        cfg["roles"]["decode"]["args"]["return_perf_metrics"] = False
        with caplog.at_level("WARNING"):
            out = expand_trtllm_serve_defaults(cfg)
        assert out["roles"]["decode"]["args"]["return_perf_metrics"] is False
        assert out["roles"]["prefill"]["args"]["return_perf_metrics"] is True
        assert any("decode" in rec.message and "/prometheus/metrics" in rec.message for rec in caplog.records)

    def test_dynamo_frontend_is_untouched(self):
        cfg = _trtllm_config()
        out = expand_trtllm_serve_defaults(cfg)
        for mode in ("prefill", "decode"):
            assert "return_perf_metrics" not in out["roles"][mode]["args"]

    def test_non_trtllm_backend_is_untouched(self):
        cfg = _base_config()
        cfg["frontend"] = {"type": "trtllm_serve"}
        cfg["engine"] = {"type": "sglang"}
        cfg["roles"]["prefill"]["args"] = {}
        out = expand_trtllm_serve_defaults(cfg)
        assert out["engine"] == {"type": "sglang"}
        assert _role_args(out) == {"prefill": {}}

    def test_missing_args_are_created_for_every_role(self):
        """A disaggregated recipe with no engine yaml at all still gets the route:
        prefill and decode args are created; no agg role is invented."""
        cfg = _trtllm_serve_config()
        for spec in cfg["roles"].values():
            spec.pop("args", None)
        out = expand_trtllm_serve_defaults(cfg)
        assert _role_args(out) == {
            "prefill": {"return_perf_metrics": True},
            "decode": {"return_perf_metrics": True},
        }
        assert "agg" not in out["roles"]

    def test_partial_sections_are_completed(self):
        cfg = _trtllm_serve_config()
        del cfg["roles"]["decode"]["args"]
        out = expand_trtllm_serve_defaults(cfg)
        assert out["roles"]["prefill"]["args"]["return_perf_metrics"] is True
        assert out["roles"]["decode"]["args"] == {"return_perf_metrics": True}
        assert "agg" not in out["roles"]

    def test_aggregated_layout_gets_the_default_and_can_opt_out(self, caplog):
        cfg = _base_config()
        cfg["roles"] = {"agg": {"nodes": 1, "workers": 1, "args": {"max_batch_size": 8}}}
        cfg["frontend"] = {"type": "trtllm_serve", "enable_multiple_frontends": False}
        cfg["engine"] = "trtllm"
        out = expand_trtllm_serve_defaults(cfg)
        assert out["roles"]["agg"]["args"]["return_perf_metrics"] is True
        assert "prefill" not in out["roles"]

        cfg["roles"]["agg"]["args"]["return_perf_metrics"] = False
        with caplog.at_level("WARNING"):
            expand_trtllm_serve_defaults(cfg)
        assert any("roles.agg" in rec.message for rec in caplog.records)

    def test_from_yaml_applies_the_default(self, tmp_path):
        cfg = _trtllm_serve_config()
        cfg["benchmark"] = {"type": "sa-bench", "concurrencies": [4]}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(_recipe(cfg)))

        loaded = SrtConfig.from_yaml(config_path)

        for mode in ("prefill", "decode"):
            section = loaded.backend.get_config_for_mode(mode)
            assert section["return_perf_metrics"] is True

    def test_load_config_applies_the_default(self, tmp_path, monkeypatch):
        """load_config is the path every real entry point uses (srtctl apply,
        dry-run, the in-job orchestrator); from_yaml has no production callers."""
        from srtctl.core.config import load_config

        monkeypatch.delenv("SRTSLURM_CONFIG", raising=False)
        monkeypatch.setattr("srtctl.core.config.load_cluster_config", lambda: None)
        cfg = _trtllm_serve_config()
        cfg["benchmark"] = {"type": "sa-bench", "concurrencies": [4]}
        config_path = tmp_path / "recipe.yaml"
        config_path.write_text(yaml.safe_dump(_recipe(cfg)))

        loaded = load_config(config_path)

        for mode in ("prefill", "decode"):
            assert loaded.backend.get_config_for_mode(mode)["return_perf_metrics"] is True

    def test_observability_expansion_is_unchanged_and_composes(self):
        """observability.enabled still injects both engine keys; the trtllm-serve
        default only fills return_perf_metrics where nothing else set it."""
        cfg = _trtllm_serve_config(enabled=True)
        expand_observability(cfg)
        out = expand_trtllm_serve_defaults(cfg)
        for mode in ("prefill", "decode"):
            section = out["roles"][mode]["args"]
            assert section["return_perf_metrics"] is True
            assert section["enable_iter_perf_stats"] is True


def _expand_all(cfg):
    """Apply the three engine-config expansions in load_config / from_yaml order."""
    expand_observability(cfg)
    expand_trtllm_serve_defaults(cfg)
    return expand_trtllm_engine_defaults(cfg)


class TestTrtllmEngineDefaults:
    """dynamo.trtllm derives enable_iter_perf_stats from --publish-metrics, which
    backend.publish_metrics passes by default, and the engine yaml wins over that
    derived value. srtctl therefore bakes enable_iter_perf_stats: false into every
    TRT-LLM engine section a recipe uses, under both frontends, so the default
    publication costs the per-request perf metrics only. Explicit recipe values
    and the observability expansion (which runs first) win."""

    def test_iteration_stats_default_off_under_dynamo(self):
        out = expand_trtllm_engine_defaults(_trtllm_config())
        for mode in ("prefill", "decode"):
            section = out["roles"][mode]["args"]
            assert section["enable_iter_perf_stats"] is False
            # Only this key is touched; the recipe's own values survive and
            # the dynamo path gets no trtllm-serve route default.
            assert section["max_batch_size"] in (256, 64)
            assert "return_perf_metrics" not in section
        assert "agg" not in out["roles"]

    def test_iteration_stats_default_off_under_trtllm_serve(self):
        out = _expand_all(_trtllm_serve_config())
        for mode in ("prefill", "decode"):
            section = out["roles"][mode]["args"]
            assert section["enable_iter_perf_stats"] is False
            assert section["return_perf_metrics"] is True

    def test_frontend_omitted_is_the_dynamo_default(self):
        cfg = _trtllm_config()
        del cfg["frontend"]
        out = expand_trtllm_engine_defaults(cfg)
        assert out["roles"]["prefill"]["args"]["enable_iter_perf_stats"] is False

    def test_explicit_true_wins(self):
        cfg = _trtllm_config()
        cfg["roles"]["decode"]["args"]["enable_iter_perf_stats"] = True
        out = expand_trtllm_engine_defaults(cfg)
        assert out["roles"]["decode"]["args"]["enable_iter_perf_stats"] is True
        assert out["roles"]["prefill"]["args"]["enable_iter_perf_stats"] is False

    @pytest.mark.parametrize("frontend_type", ["dynamo", "trtllm_serve"])
    @pytest.mark.parametrize("with_sections", [True, False])
    def test_observability_keeps_iteration_stats_on(self, frontend_type, with_sections):
        """The analytics capture reads the trtllm_kv_cache_* gauges, which only
        exist with iteration statistics; observability's setdefault runs first,
        also for a recipe that carries no engine yaml at all."""
        cfg = _trtllm_serve_config(enabled=True) if frontend_type == "trtllm_serve" else _trtllm_config(enabled=True)
        if not with_sections:
            for spec in cfg["roles"].values():
                spec.pop("args", None)
        out = _expand_all(cfg)
        for mode in ("prefill", "decode"):
            section = out["roles"][mode]["args"]
            assert section["enable_iter_perf_stats"] is True
            assert section["return_perf_metrics"] is True

    def test_observability_does_not_override_an_explicit_false(self):
        cfg = _trtllm_config(enabled=True)
        cfg["roles"]["decode"]["args"]["enable_iter_perf_stats"] = False
        out = _expand_all(cfg)
        assert out["roles"]["decode"]["args"]["enable_iter_perf_stats"] is False
        assert out["roles"]["prefill"]["args"]["enable_iter_perf_stats"] is True

    def test_missing_args_are_created_for_every_role(self):
        disagg = _trtllm_config()
        for spec in disagg["roles"].values():
            spec.pop("args", None)
        out = expand_trtllm_engine_defaults(disagg)
        assert _role_args(out) == {
            "prefill": {"enable_iter_perf_stats": False},
            "decode": {"enable_iter_perf_stats": False},
        }

        agg = _trtllm_config()
        agg["roles"] = {"agg": {"nodes": 1, "workers": 1}}
        out = expand_trtllm_engine_defaults(agg)
        assert _role_args(out) == {"agg": {"enable_iter_perf_stats": False}}

    def test_non_trtllm_backend_is_untouched(self):
        cfg = _base_config()
        cfg["engine"] = {"type": "sglang"}
        cfg["roles"]["prefill"]["args"] = {}
        out = expand_trtllm_engine_defaults(cfg)
        assert out["engine"] == {"type": "sglang"}
        assert _role_args(out) == {"prefill": {}}

    @pytest.mark.parametrize("loader_name", ["from_yaml", "load_config"])
    @pytest.mark.parametrize("frontend_type", ["dynamo", "trtllm_serve"])
    @pytest.mark.parametrize("enabled", [False, True])
    def test_yaml_loaders_apply_the_default(self, tmp_path, monkeypatch, loader_name, frontend_type, enabled):
        """Both real loaders bake the key into the engine config the worker will
        receive, and observability flips it back on through the same path."""
        monkeypatch.setattr("srtctl.core.config.load_cluster_config", lambda: None)
        cfg = (
            _trtllm_serve_config(enabled=enabled)
            if frontend_type == "trtllm_serve"
            else _trtllm_config(enabled=enabled)
        )
        cfg["benchmark"] = {"type": "sa-bench", "concurrencies": [4]}
        path = tmp_path / "recipe.yaml"
        path.write_text(yaml.safe_dump(_recipe(cfg)))
        loader = SrtConfig.from_yaml if loader_name == "from_yaml" else load_config
        loaded = loader(path)
        for mode in ("prefill", "decode"):
            section = loaded.backend.get_config_for_mode(mode)
            assert section["enable_iter_perf_stats"] is enabled
            assert section["max_batch_size"] in (256, 64)

    @pytest.mark.parametrize("loader_name", ["from_yaml", "load_config"])
    def test_v2_roles_args_get_the_default(self, tmp_path, monkeypatch, loader_name):
        """The engine yaml is roles.<role>.args; the default lands in each role's args."""
        monkeypatch.setattr("srtctl.core.config.load_cluster_config", lambda: None)
        cfg = {
            "schema": 2,
            "name": "test-job",
            "model": {"path": "/models/test-model", "container": "test.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h100", "gpus_per_node": 8},
            "engine": {"type": "trtllm"},
            "roles": {
                "prefill": {"nodes": 1, "workers": 1, "gpus": 8, "args": {"max_batch_size": 256}},
                "decode": {"nodes": 1, "workers": 1, "gpus": 8, "args": {"max_batch_size": 64}},
            },
            "benchmark": {"type": "sa-bench", "concurrencies": [4]},
        }
        path = tmp_path / "recipe.yaml"
        path.write_text(yaml.safe_dump(cfg))
        loader = SrtConfig.from_yaml if loader_name == "from_yaml" else load_config
        loaded = loader(path)
        assert loaded.backend.dynamo_metrics_flags == ("--publish-metrics",)
        for mode in ("prefill", "decode"):
            section = loaded.backend.get_config_for_mode(mode)
            assert section["enable_iter_perf_stats"] is False
            assert section["max_batch_size"] in (256, 64)

    def test_dump_reload_carries_the_key_and_observability_warns(self, tmp_path, monkeypatch, caplog):
        """A schema dump (a saved or locked recipe) carries the baked key as an
        explicit false. A recipe that spells that value out and later sets
        observability.enabled: true keeps it, so the load step says so instead of
        failing silently."""
        monkeypatch.setattr("srtctl.core.config.load_cluster_config", lambda: None)
        cfg = _trtllm_config()
        cfg["benchmark"] = {"type": "sa-bench", "concurrencies": [4]}
        schema = SrtConfig.Schema()
        dumped = schema.dump(schema.load(_expand_all(cfg)))
        assert dumped["roles"]["prefill"]["args"]["enable_iter_perf_stats"] is False

        dumped["observability"]["enabled"] = True
        path = tmp_path / "enable-observability.yaml"
        path.write_text(
            yaml.safe_dump(
                _recipe(
                    {
                        **cfg,
                        "engine": dumped["engine"],
                        "roles": dumped["roles"],
                        "observability": dumped["observability"],
                    }
                )
            )
        )
        with caplog.at_level("WARNING"):
            loaded = load_config(path)
        for mode in ("prefill", "decode"):
            assert loaded.backend.get_config_for_mode(mode)["enable_iter_perf_stats"] is False
        assert any(
            "roles.prefill.args.enable_iter_perf_stats" in rec.message
            and "roles.decode.args.enable_iter_perf_stats" in rec.message
            for rec in caplog.records
        )

    def test_observability_warns_on_an_explicit_false(self, caplog):
        cfg = _trtllm_config(enabled=True)
        cfg["roles"]["decode"]["args"]["enable_iter_perf_stats"] = False
        with caplog.at_level("WARNING"):
            expand_observability(cfg)
        warnings = [rec.message for rec in caplog.records if "enable_iter_perf_stats" in rec.message]
        assert warnings and "roles.decode.args.enable_iter_perf_stats" in warnings[0]
        assert "prefill" not in warnings[0]

    def test_non_mapping_role_args_are_left_for_schema_validation(self):
        """Bogus role args must still fail validation instead of being silently
        replaced by the defaults."""
        cfg = _trtllm_config()
        cfg["roles"]["decode"]["args"] = "bogus"
        out = expand_trtllm_engine_defaults(cfg)
        assert out["roles"]["decode"]["args"] == "bogus"
        with pytest.raises(ValidationError):
            SrtConfig.Schema().load(out)

        cfg = _trtllm_config()
        cfg["roles"]["decode"]["args"] = ["not", "a", "mapping"]
        out = expand_trtllm_engine_defaults(cfg)
        assert out["roles"]["decode"]["args"] == ["not", "a", "mapping"]
        assert out["roles"]["prefill"]["args"]["enable_iter_perf_stats"] is False

    def test_null_args_are_treated_as_absent(self):
        cfg = _trtllm_config()
        for spec in cfg["roles"].values():
            spec["args"] = None
        out = expand_trtllm_engine_defaults(cfg)
        assert _role_args(out) == {
            "prefill": {"enable_iter_perf_stats": False},
            "decode": {"enable_iter_perf_stats": False},
        }

    @pytest.mark.parametrize("spelling", ["tensorrt", "TensorRT", "trt"])
    def test_legacy_tensorrt_backend_sections_are_skipped(self, spelling):
        """The legacy TensorRT engine's LlmArgs rejects the key on containers that
        predate the backend's removal, and always collected the statistics anyway."""
        cfg = _trtllm_config()
        cfg["roles"]["decode"]["args"]["backend"] = spelling
        out = expand_trtllm_engine_defaults(cfg)
        assert out["roles"]["decode"]["args"] == {"max_batch_size": 64, "backend": spelling}
        assert out["roles"]["prefill"]["args"]["enable_iter_perf_stats"] is False


class TestObservabilitySchema:
    def test_defaults_are_off(self):
        cfg = SrtConfig.Schema().load(BASE_CONFIG)
        assert cfg.observability.enabled is False
        assert cfg.observability.enable_otel is False

    def test_knob_exposes_no_benchmark_client_fields(self):
        """The knob's scope is server-side emission only.

        Client-side capture flags belong to whatever drives the client, not
        here -- the ``/metrics`` surface this turns on is captured by scraping
        the endpoints directly. Guards against the scope creeping back.
        """
        cfg = SrtConfig.Schema().load({**BASE_CONFIG, "observability": {"enabled": True}})
        assert not [f for f in vars(cfg.observability) if f.startswith("aiperf")]

    def test_from_yaml_loads_nested_tachometer(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                _recipe(
                    {
                        **BASE_CONFIG,
                        "observability": {
                            "enabled": True,
                            "tachometer": {"enabled": True, "collect_interval_ms": 500},
                        },
                    }
                )
            )
        )

        cfg = SrtConfig.from_yaml(config_path)

        assert cfg.observability.tachometer.enabled is True
        assert cfg.observability.tachometer_enabled is True
        assert cfg.observability.tachometer.collect_interval_ms == 500
