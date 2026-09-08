# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `observability.enabled` knob and its config expansion."""

import pytest
import yaml
from marshmallow import ValidationError

from srtctl.core.config import expand_observability, expand_trtllm_serve_defaults
from srtctl.core.schema import SrtConfig

BASE_CONFIG = {
    "name": "test-job",
    "model": {"path": "/models/test-model", "container": "test.sqsh", "precision": "fp8"},
    "resources": {
        "gpu_type": "h100",
        "gpus_per_node": 8,
        "prefill_nodes": 1,
        "decode_nodes": 1,
        "prefill_workers": 1,
        "decode_workers": 1,
    },
}


def _trtllm_config(**observability):
    cfg = dict(BASE_CONFIG)
    cfg["backend"] = {
        "type": "trtllm",
        "prefill_environment": {"TLLM_LOG_LEVEL": "INFO"},
        "decode_environment": {},
        "trtllm_config": {"prefill": {"max_batch_size": 256}, "decode": {"max_batch_size": 64}},
    }
    cfg["frontend"] = {"env": {"DYN_TOKENIZER": "fastokens"}}
    if observability:
        cfg["observability"] = observability
    return cfg


# --------------------------------------------------------------- expansion ---
class TestExpandObservability:
    def test_disabled_is_a_noop(self):
        cfg = expand_observability(_trtllm_config(enabled=False))
        assert "publish_events_and_metrics" not in cfg["backend"]
        assert "DYN_LOGGING_SPAN_EVENTS" not in cfg["backend"]["prefill_environment"]

    def test_absent_block_is_a_noop(self):
        cfg = expand_observability(_trtllm_config())
        assert "publish_events_and_metrics" not in cfg["backend"]

    def test_enabled_expands_span_env_on_all_three_roles(self):
        cfg = expand_observability(_trtllm_config(enabled=True))
        for env in (
            cfg["backend"]["prefill_environment"],
            cfg["backend"]["decode_environment"],
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
            assert "DYN_REQUEST_TRACE" not in cfg["backend"][f"{mode}_environment"]

    def test_enabled_turns_on_metrics_surface_and_iteration_stats(self):
        cfg = expand_observability(_trtllm_config(enabled=True))
        assert "telemetry" not in cfg
        assert cfg["backend"]["publish_events_and_metrics"] is True
        for mode in ("prefill", "decode"):
            section = cfg["backend"]["trtllm_config"][mode]
            # enable_iter_perf_stats is what produces trtllm_kv_cache_*_blocks.
            assert section["enable_iter_perf_stats"] is True
            assert section["return_perf_metrics"] is True

    def test_explicit_recipe_values_win(self):
        """setdefault semantics: an explicit recipe value must never be clobbered."""
        cfg = _trtllm_config(enabled=True)
        cfg["backend"]["decode_environment"]["DYN_LOG"] = "info"
        cfg["backend"]["trtllm_config"]["decode"]["return_perf_metrics"] = False
        cfg["backend"]["publish_events_and_metrics"] = False
        out = expand_observability(cfg)
        assert out["backend"]["decode_environment"]["DYN_LOG"] == "info"
        assert out["backend"]["trtllm_config"]["decode"]["return_perf_metrics"] is False
        assert out["backend"]["publish_events_and_metrics"] is False

    def test_preexisting_env_is_preserved(self):
        cfg = expand_observability(_trtllm_config(enabled=True))
        assert cfg["backend"]["prefill_environment"]["TLLM_LOG_LEVEL"] == "INFO"
        assert cfg["frontend"]["env"]["DYN_TOKENIZER"] == "fastokens"

    def test_non_trtllm_backend_keeps_span_env_but_no_engine_keys(self):
        cfg = dict(BASE_CONFIG)
        cfg["backend"] = {"type": "sglang"}
        cfg["observability"] = {"enabled": True}
        out = expand_observability(cfg)
        assert out["backend"]["prefill_environment"]["DYN_LOGGING_SPAN_EVENTS"] == "true"
        assert "publish_events_and_metrics" not in out["backend"]

    def test_missing_sections_are_created(self):
        out = expand_observability({**BASE_CONFIG, "observability": {"enabled": True}})
        assert out["backend"]["prefill_environment"]["DYN_LOGGING_JSONL"] == "true"
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
        """The perf dashboard is built on every run, so the knob that used to gate it
        is gone. A recipe still carrying it must fail at submit time: silently
        accepting `build_dashboard: false` would promise a capture-only run and then
        render one anyway."""
        cfg = _trtllm_config(enabled=True, build_dashboard=False)

        with pytest.raises(ValidationError, match="build_dashboard"):
            SrtConfig.Schema().load(cfg)

    def test_retired_raw_scraper_knobs_are_rejected(self):
        """The in-job RAW Prometheus scraper is gone; Tachometer is the capture.

        A recipe still carrying `scrape_metrics` (or the other scrape_* knobs)
        must fail loudly at submit time rather than silently promising a
        raw_prometheus.jsonl that will never be written. The ingest keeps
        reading historical raw_prometheus.jsonl artifacts regardless."""
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
        cfg = dict(BASE_CONFIG)
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
            section = out["backend"]["trtllm_config"][mode]
            assert section["return_perf_metrics"] is True
            # Only this key is touched; the recipe's own values survive.
            assert "enable_iter_perf_stats" not in section
            assert section["max_batch_size"] in (256, 64)

    def test_explicit_false_wins_and_warns(self, caplog):
        cfg = _trtllm_serve_config()
        cfg["backend"]["trtllm_config"]["decode"]["return_perf_metrics"] = False
        with caplog.at_level("WARNING"):
            out = expand_trtllm_serve_defaults(cfg)
        assert out["backend"]["trtllm_config"]["decode"]["return_perf_metrics"] is False
        assert out["backend"]["trtllm_config"]["prefill"]["return_perf_metrics"] is True
        assert any("decode" in rec.message and "/prometheus/metrics" in rec.message for rec in caplog.records)

    def test_dynamo_frontend_is_untouched(self):
        cfg = _trtllm_config()
        out = expand_trtllm_serve_defaults(cfg)
        for mode in ("prefill", "decode"):
            assert "return_perf_metrics" not in out["backend"]["trtllm_config"][mode]

    def test_non_trtllm_backend_is_untouched(self):
        cfg = dict(BASE_CONFIG)
        cfg["frontend"] = {"type": "trtllm_serve"}
        cfg["backend"] = {"type": "sglang", "sglang_config": {"prefill": {}}}
        out = expand_trtllm_serve_defaults(cfg)
        assert out["backend"] == {"type": "sglang", "sglang_config": {"prefill": {}}}

    def test_missing_sections_are_created_for_the_modes_in_use(self):
        """A disaggregated recipe with no engine yaml at all still gets the route:
        prefill and decode sections are created; aggregated is not (unused)."""
        cfg = _trtllm_serve_config()
        del cfg["backend"]["trtllm_config"]
        out = expand_trtllm_serve_defaults(cfg)
        assert out["backend"]["trtllm_config"] == {
            "prefill": {"return_perf_metrics": True},
            "decode": {"return_perf_metrics": True},
        }

    def test_partial_sections_are_completed(self):
        cfg = _trtllm_serve_config()
        del cfg["backend"]["trtllm_config"]["decode"]
        out = expand_trtllm_serve_defaults(cfg)
        assert out["backend"]["trtllm_config"]["prefill"]["return_perf_metrics"] is True
        assert out["backend"]["trtllm_config"]["decode"] == {"return_perf_metrics": True}
        assert "aggregated" not in out["backend"]["trtllm_config"]

    def test_aggregated_layout_gets_the_default_and_can_opt_out(self, caplog):
        cfg = dict(BASE_CONFIG)
        cfg["resources"] = {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1, "agg_workers": 1}
        cfg["frontend"] = {"type": "trtllm_serve", "enable_multiple_frontends": False}
        cfg["backend"] = {"type": "trtllm", "trtllm_config": {"aggregated": {"max_batch_size": 8}}}
        out = expand_trtllm_serve_defaults(cfg)
        assert out["backend"]["trtllm_config"]["aggregated"]["return_perf_metrics"] is True
        assert "prefill" not in out["backend"]["trtllm_config"]

        cfg["backend"]["trtllm_config"]["aggregated"]["return_perf_metrics"] = False
        with caplog.at_level("WARNING"):
            expand_trtllm_serve_defaults(cfg)
        assert any("aggregated" in rec.message for rec in caplog.records)

    def test_from_yaml_applies_the_default(self, tmp_path):
        cfg = _trtllm_serve_config()
        cfg["benchmark"] = {"type": "sa-bench", "concurrencies": [4]}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg))

        loaded = SrtConfig.from_yaml(config_path)

        for mode in ("prefill", "decode"):
            section = getattr(loaded.backend.trtllm_config, mode)
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
        config_path.write_text(yaml.safe_dump(cfg))

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
            section = out["backend"]["trtllm_config"][mode]
            assert section["return_perf_metrics"] is True
            assert section["enable_iter_perf_stats"] is True


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
                {
                    **BASE_CONFIG,
                    "observability": {
                        "enabled": True,
                        "tachometer": {"enabled": True, "collect_interval_ms": 500},
                    },
                }
            )
        )

        cfg = SrtConfig.from_yaml(config_path)

        assert cfg.observability.tachometer.enabled is True
        assert cfg.observability.tachometer_enabled is True
        assert cfg.observability.tachometer.collect_interval_ms == 500
