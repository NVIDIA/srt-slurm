# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MLPerf benchmark driver script."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from srtctl.benchmarks import list_benchmarks
from srtctl.benchmarks.base import SCRIPTS_DIR

BENCH = SCRIPTS_DIR / "mlperf" / "bench.sh"


def _load_script_module(name, filename):
    """The scripts run inside the client's container, so they are not importable
    as package modules; load them by path the way the tests need them."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / "mlperf" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resolve_config = _load_script_module("mlperf_resolve", "resolve_config.py")
_parse = _load_script_module("mlperf_parse", "parse_report.py")
parse_report, build_rollup = _parse.parse_report, _parse.build_rollup

# A real submission config, taken verbatim from endpoints-launch
# (GR100-NVL72 x72, DeepSeek-R1, point_offline_disagg_dynamo_x72). Kept as a
# fixture so the passthrough is tested against a config nobody here wrote.
FIXTURE = Path(__file__).parent / "fixtures" / "mlperf_client_submission.yaml"


class TestMLPerfScript:
    """MLPerf runs as a `custom` benchmark driving the MLPerf team's client.

    Nothing about it lives in srt-slurm's schema: the client's config has ~60
    nested settings whose shape moves with the client version, so the script
    passes it through and rewrites only what the cluster decides.
    """

    def test_no_mlperf_benchmark_type(self):
        """mlperf is deliberately not a registered benchmark type."""
        assert "mlperf" not in list_benchmarks()

    def test_script_ships_with_the_package(self):
        """scripts/ is mounted at /srtctl-benchmarks for every benchmark type."""
        assert BENCH.exists()

    def test_takes_a_client_config_rather_than_modelling_its_settings(self):
        script = BENCH.read_text()
        assert "MLPERF_CLIENT_CONFIG" in script
        # A knob for any individual client setting would be the wrapper this
        # design exists to avoid.
        for leaked in ("MLPERF_TARGET_QPS", "MLPERF_MAX_NEW_TOKENS", "MLPERF_NUM_WORKERS"):
            assert leaked not in script, f"{leaked} re-models a client config setting"

    def test_delegates_to_the_tested_resolver(self):
        """The rewrite must be the code the tests cover, not an inline copy of it."""
        assert "resolve_config.py" in BENCH.read_text()

    def test_uses_the_clients_interpreter_for_helpers(self):
        """The image keeps pyyaml in its venv; system python3 cannot import it."""
        assert "CLIENT_PYTHON" in BENCH.read_text()

    def test_requires_an_injected_frontend(self):
        """No localhost default: benchmarking nothing must fail, not pass."""
        assert "SRT_FRONTEND_HOST is not set" in BENCH.read_text()

    def test_supports_multiple_endpoints(self):
        """The client load-balances across the list itself."""
        assert "MLPERF_ENDPOINTS" in BENCH.read_text()

    def test_finds_the_client_outside_the_default_path(self):
        """The client image installs into /opt/venv, which is not on its own PATH."""
        assert "/opt/venv/bin" in BENCH.read_text()

    def test_writes_the_rollup_srtctl_already_consumes(self):
        """No new rollup concept: postprocess reads benchmark-rollup.json if present."""
        assert "benchmark-rollup.json" in BENCH.read_text()


class TestResolveRealSubmissionConfig:
    """Exercised against a real submission config, not one written for the test."""

    def _resolved(self, endpoints=("http://10.0.0.1:8000",), report_dir="/logs/mlperf"):
        config = yaml.safe_load(FIXTURE.read_text())
        return resolve_config.resolve(config, list(endpoints), report_dir)

    def test_fills_the_empty_endpoints_list(self):
        """The checked-in config ships `endpoints: []` for the launcher to fill."""
        assert yaml.safe_load(FIXTURE.read_text())["endpoint_config"]["endpoints"] == []
        out = self._resolved(["http://10.0.0.1:8000", "http://10.0.0.2:8000"])
        assert out["endpoint_config"]["endpoints"] == ["http://10.0.0.1:8000", "http://10.0.0.2:8000"]

    def test_keeps_the_rest_of_endpoint_config(self):
        """Only the endpoints key is ours; api_type and api_key are the client's."""
        out = self._resolved()
        assert out["endpoint_config"]["api_type"] == "openai_completions"
        assert "api_key" in out["endpoint_config"]

    def test_replaces_report_dir(self):
        """The checked-in value is an sflow variable that does not exist here."""
        assert yaml.safe_load(FIXTURE.read_text())["report_dir"] == "${SFLOW_TASK_OUTPUT_DIR}"
        assert self._resolved()["report_dir"] == "/logs/mlperf"

    def test_preserves_unresolved_placeholders(self):
        """`${MODEL_DIR}` must still be a placeholder when the client opens the file:
        the client expands it, not us."""
        assert self._resolved()["model_params"]["tokenizer_name"] == "${MODEL_DIR}"

    def test_changes_nothing_else(self):
        """Every other key, at any depth, survives byte-identically."""
        before = yaml.safe_load(FIXTURE.read_text())
        after = self._resolved()
        before.pop("report_dir"), after.pop("report_dir")
        before.pop("endpoint_config"), after.pop("endpoint_config")
        assert before == after

    def test_round_trips_through_yaml(self):
        """What we write must reload as what we intended."""
        out = self._resolved()
        assert yaml.safe_load(yaml.dump(out, sort_keys=False)) == out

    def test_rejects_an_empty_endpoint_list(self):
        """Benchmarking nothing must fail loudly."""
        with pytest.raises(ValueError):
            resolve_config.resolve(yaml.safe_load(FIXTURE.read_text()), [], "/logs")

    def test_normalizes_bare_host_port(self):
        assert resolve_config.normalize_endpoints("10.0.0.1:8000, 10.0.0.2:8001") == [
            "http://10.0.0.1:8000",
            "http://10.0.0.2:8001",
        ]

    def test_leaves_an_explicit_scheme_alone(self):
        assert resolve_config.normalize_endpoints("https://x:8000") == ["https://x:8000"]


class TestParseRealReport:
    """Against unedited output from real MLPerf runs under endpoints-launch.

    Expected values are cross-checked against the human-readable report.txt the
    client wrote beside each summary, so these assert agreement with the client's
    own rendering rather than with my reading of its JSON.
    """

    REPORT = Path(__file__).parent / "fixtures" / "mlperf_report"

    def test_headline_throughput_matches_report_txt(self):
        """report.txt for run 474440 says QPS: 81.57, TPS: 300727.22."""
        perf = parse_report(self.REPORT)["metrics"]["performance"]
        assert perf["qps"] == pytest.approx(81.57, abs=0.01)
        assert perf["tps"] == pytest.approx(300727.22, abs=0.01)
        assert perf["samples_issued"] == 90000
        assert perf["samples_completed"] == 87103
        assert perf["samples_failed"] == 0
        assert perf["duration_s"] == pytest.approx(1072.27, abs=0.01)
        assert perf["complete"] is True

    def test_durations_are_converted_from_nanoseconds(self):
        """The summary is ns; report.txt renders TTFT median as 275.64 ms."""
        ttft = parse_report(self.REPORT)["metrics"]["performance"]["ttft"]
        assert ttft["median_ms"] == pytest.approx(275.64, abs=0.01)
        assert ttft["p99_ms"] == pytest.approx(3445.34, abs=0.01)
        # A parser that skipped the conversion would land in the millions here.
        assert ttft["median_ms"] < 10_000

    def test_percentile_keys_are_named_correctly(self):
        """report.txt for 474440: TTFT p50 275.64, p90 409.28, p95 608.55.

        Naming these by trimming "50.0" with rstrip(".0") yields "p5"/"p9",
        because rstrip takes a character set rather than a suffix -- and it is
        invisible unless p50/p90 are asserted, since p95/p99 come out right.
        """
        ttft = parse_report(self.REPORT)["metrics"]["performance"]["ttft"]
        assert set(ttft) == {"min_ms", "max_ms", "mean_ms", "median_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms"}
        assert ttft["p50_ms"] == pytest.approx(275.64, abs=0.01)
        assert ttft["p90_ms"] == pytest.approx(409.28, abs=0.01)
        assert ttft["p95_ms"] == pytest.approx(608.55, abs=0.01)

    def test_token_counts_are_not_scaled(self):
        """Output lengths are tokens, not durations -- they must pass through."""
        osl = parse_report(self.REPORT)["metrics"]["performance"]["output_sequence_lengths"]
        assert osl["median"] == 2054.0
        assert osl["max"] == 20004

    def test_records_which_window_produced_qps(self):
        """Run 474440 is poisson with the legacy LoadGen window set."""
        assert parse_report(self.REPORT)["metrics"]["performance"]["qps_window"] == "legacy_loadgen"

    def test_isl_absent_in_this_run_is_not_an_error(self):
        """Documented field, but empty without a tokenizer for text-only input."""
        assert "input_sequence_lengths" not in parse_report(self.REPORT)["metrics"]["performance"]

    def test_accuracy_and_subset_breakdown(self):
        """Run 412798 scored 81.43 overall, across five subsets."""
        accuracy = parse_report(self.REPORT)["metrics"]["accuracy"]
        assert accuracy["average_accuracy"] == pytest.approx(81.4266, abs=0.001)
        dataset = accuracy["datasets"][0]
        assert dataset["name"] == "deepseek_r1_accuracy"
        assert dataset["subset_scores"]["math500"] == pytest.approx(94.99, abs=0.01)
        assert set(dataset["subset_scores"]) == {"aime1983", "gpqa", "livecodebench", "math500", "mmlu_pro"}

    def test_records_which_files_it_read(self):
        sources = parse_report(self.REPORT)["metrics_sources"]
        assert "performance/result_summary.json" in sources
        assert "accuracy/accuracy_results.json" in sources


class TestParseReportDegradesGracefully:
    """The layout already moved once (286161 vs 474440), so assume it may again."""

    def test_missing_report_is_not_an_error(self, tmp_path):
        """A run that produced nothing still gets a rollup, minus metrics."""
        assert parse_report(tmp_path) == {"metrics": {}, "metrics_sources": []}

    def test_truncated_json_is_not_an_error(self, tmp_path):
        """Client killed mid-write is when the surrounding rollup matters most."""
        (tmp_path / "performance").mkdir()
        (tmp_path / "performance" / "result_summary.json").write_text('{"qps": 12.5, "tp')
        assert parse_report(tmp_path)["metrics"] == {}

    def test_finds_the_older_top_level_layout(self, tmp_path):
        """Run 286161 (July) wrote result_summary.json at the top level."""
        (tmp_path / "result_summary.json").write_text(json.dumps({"qps": 12.5, "n_samples_issued": 7}))
        perf = parse_report(tmp_path)["metrics"]["performance"]
        assert perf["qps"] == 12.5
        assert perf["samples_issued"] == 7

    def test_prefers_the_newer_layout_when_both_exist(self, tmp_path):
        (tmp_path / "result_summary.json").write_text(json.dumps({"qps": 1.0}))
        (tmp_path / "performance").mkdir()
        (tmp_path / "performance" / "result_summary.json").write_text(json.dumps({"qps": 2.0}))
        assert parse_report(tmp_path)["metrics"]["performance"]["qps"] == 2.0

    def test_partial_summary_keeps_what_it_can(self, tmp_path):
        """A dropped latency block must not cost us the throughput numbers."""
        (tmp_path / "result_summary.json").write_text(json.dumps({"qps": 9.0, "tps": 900.0}))
        perf = parse_report(tmp_path)["metrics"]["performance"]
        assert perf["qps"] == 9.0
        assert "ttft" not in perf

    def test_native_window_is_labelled(self, tmp_path):
        """tps/qps from the native window must not be read as LoadGen's."""
        (tmp_path / "result_summary.json").write_text(
            json.dumps({"qps": 5.0, "legacy_loadgen_window_duration_ns": None})
        )
        assert parse_report(tmp_path)["metrics"]["performance"]["qps_window"] == "native"


class TestRollup:
    def test_rollup_carries_metrics_and_run_shape(self):
        rollup = build_rollup(
            TestParseRealReport.REPORT,
            mode="both",
            endpoints=["http://node-0:8000"],
            client_config="/mlperf-configs/client.yaml",
            exit_code=0,
        )
        assert rollup["benchmark_type"] == "mlperf"
        assert rollup["client"] == "inference-endpoint"
        run = rollup["runs"][0]
        assert run["exit_code"] == 0
        assert run["endpoints"] == ["http://node-0:8000"]
        assert run["metrics"]["performance"]["qps"] == pytest.approx(81.57, abs=0.01)

    def test_failed_run_still_produces_a_rollup(self, tmp_path):
        """A non-zero exit must be recorded, not swallowed."""
        rollup = build_rollup(
            tmp_path, mode="performance", endpoints=["http://h:8000"], client_config="/c.yaml", exit_code=1
        )
        assert rollup["runs"][0]["exit_code"] == 1
        assert rollup["runs"][0]["metrics"] == {}

    def test_monitor_renders_the_rollup(self, tmp_path):
        """The point of writing benchmark-rollup.json is that shared consumers read it.

        `srtctl monitor` pulls flat keys off each run record
        (cli/monitor.py::_format_run), so a rollup that only nested its numbers
        would be collected and then shown as a blank line. Exercised through
        monitor's own functions so this fails if that contract moves.
        """
        from srtctl.cli.monitor import _format_run, _rollup_runs

        rollup = build_rollup(
            TestParseRealReport.REPORT, mode="both", endpoints=["http://h:8000"], client_config="/c.yaml", exit_code=0
        )
        (tmp_path / "benchmark-rollup.json").write_text(json.dumps(rollup))

        runs = _rollup_runs(tmp_path)
        assert runs is not None, "monitor could not read the rollup at all"

        line = _format_run(runs[0])
        # report.txt for 474440: TPS 300727.22, TTFT avg 384.20 ms, TPOT avg 10.87 ms.
        assert "300,727 tok/s" in line
        assert "TTFT   384ms" in line
        assert "TPOT 10.9ms" in line

    def test_canonical_keys_match_the_other_benchmarks(self):
        """Unify with sa-bench's names rather than inventing a private shape."""
        run = build_rollup(
            TestParseRealReport.REPORT, mode="both", endpoints=["http://h:8000"], client_config="/c.yaml", exit_code=0
        )["runs"][0]
        for key in (
            "throughput_toks",
            "request_throughput",
            "ttft_mean_ms",
            "ttft_p99_ms",
            "tpot_mean_ms",
            "tpot_p99_ms",
            "e2el_mean_ms",
            "completed_requests",
        ):
            assert key in run, f"{key} missing; monitor and sa-bench both use it"
        assert run["throughput_toks"] == pytest.approx(300727.22, abs=0.01)
        assert run["ttft_mean_ms"] == pytest.approx(384.20, abs=0.01)

    def test_no_fabricated_zeros_when_metrics_are_absent(self, tmp_path):
        """A blank line is honest; '0 tok/s' would read as a measured result."""
        run = build_rollup(tmp_path, mode="both", endpoints=["http://h:8000"], client_config="/c.yaml", exit_code=1)[
            "runs"
        ][0]
        assert "throughput_toks" not in run
        assert "ttft_mean_ms" not in run

    def test_rollup_is_json_serializable(self):
        """It is written with json.dumps; a msgspec type here would fail late."""
        rollup = build_rollup(
            TestParseRealReport.REPORT, mode="both", endpoints=["http://h:8000"], client_config="/c.yaml", exit_code=0
        )
        assert json.loads(json.dumps(rollup)) == rollup
