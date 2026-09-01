# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MLPerf benchmark runner and its LoadGen rollup."""

import importlib.util
import json
from pathlib import Path

from srtctl.benchmarks import list_benchmarks
from srtctl.benchmarks.base import SCRIPTS_DIR
from srtctl.core.schema import (
    BenchmarkConfig,
    ModelConfig,
    ResourceConfig,
    SrtConfig,
)

_spec = importlib.util.spec_from_file_location("mlperf_rollup", SCRIPTS_DIR / "mlperf" / "rollup.py")
rollup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rollup)


OFFLINE_SUMMARY = """================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 12.3456
Tokens per second: 4567.89
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes

================================================
Additional Stats
================================================
Min latency (ns)                : 1234567
Max latency (ns)                : 9876543210
Mean latency (ns)               : 4567890123
50.00 percentile latency (ns)   : 4000000000
99.00 percentile latency (ns)   : 9000000000

================================================
Test Parameters Used
================================================
samples_per_query : 24576
target_qps : 10
min_duration (ms): 600000

No warnings encountered during test.

No errors encountered during test.
"""

SERVER_SUMMARY = """================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Server
Mode     : PerformanceOnly
Scheduled samples per second : 10.0500
Completed samples per second : 9.9800
Completed tokens per second: 3200.50
Result is : INVALID
  Performance constraints satisfied : NO
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
Recommendations:
 * Reduce target QPS to improve latency.

================================================
Additional Stats
================================================
Min First Token latency (ns) : 100000000
Max First Token latency (ns) : 900000000
Mean First Token latency (ns) : 300000000
99.00 percentile first token latency (ns) : 850000000
Min Time to Output Token (ns) : 10000000
99.00 percentile time to output token (ns) : 25000000

================================================
Test Parameters Used
================================================
target_qps : 10
"""


class TestMLPerfScript:
    """The MLPerf harness driver is a script, not a benchmark type.

    It is driven as `benchmark.type: custom` so that nv_mlpinf's contract -
    which moves on the harness's schedule, not srt-slurm's - stays editable by
    whoever hits a breakage, instead of being frozen into this repo's schema.
    """

    def test_no_mlperf_benchmark_type(self):
        """mlperf is deliberately not a registered benchmark type."""
        assert "mlperf" not in list_benchmarks()

    def test_script_ships_with_the_package(self):
        """bench.sh and rollup.py are mounted at /srtctl-benchmarks for every type."""
        assert (SCRIPTS_DIR / "mlperf" / "bench.sh").exists()
        assert (SCRIPTS_DIR / "mlperf" / "rollup.py").exists()

    def test_script_is_env_driven(self):
        """A custom benchmark passes env, not argv, so the script reads env."""
        script = (SCRIPTS_DIR / "mlperf" / "bench.sh").read_text()
        for var in ("MLPERF_HARNESS_DIR", "MLPERF_BENCHMARK", "MLPERF_SCENARIO", "MLPERF_TEST_MODE"):
            assert var in script, f"{var} not read by bench.sh"

    def test_script_requires_an_injected_frontend(self):
        """No mock or localhost default: a missing injection must be an error."""
        script = (SCRIPTS_DIR / "mlperf" / "bench.sh").read_text()
        assert "SRT_FRONTEND_HOST is not set" in script

    def test_rollup_field_selects_a_bundled_normalizer(self):
        """benchmark.rollup is how a custom benchmark opts into a rollup."""
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model/deepseek-r1", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="gb300"),
            benchmark=BenchmarkConfig(
                type="custom", command="bash /srtctl-benchmarks/mlperf/bench.sh", rollup="mlperf"
            ),
        )
        assert config.benchmark.rollup == "mlperf"
        assert (SCRIPTS_DIR / config.benchmark.rollup / "rollup.py").exists()

    def test_rollup_defaults_to_unset(self):
        """Most benchmarks have no rollup, and that stays silent."""
        assert BenchmarkConfig(type="custom").rollup is None


# Verbatim from an AccuracyOnly run: LoadGen writes no sections at all in this
# mode, only the closing warnings/errors lines.
ACCURACY_SUMMARY = """
No warnings encountered during test.

No errors encountered during test.
"""


def _write_run(log_dir: Path, scenario: str, mode: str, summary: str) -> Path:
    run_dir = log_dir / "mlperf" / scenario / mode
    run_dir.mkdir(parents=True)
    (run_dir / "mlperf_log_summary.txt").write_text(summary)
    return run_dir


class TestParseSummary:
    def test_offline_sections(self):
        parsed = rollup.parse_summary(OFFLINE_SUMMARY)
        assert parsed["summary"]["Scenario"] == "Offline"
        assert parsed["summary"]["Tokens per second"] == 4567.89
        assert parsed["additional_stats"]["Min latency (ns)"] == 1234567
        assert parsed["test_parameters"]["samples_per_query"] == 24576

    def test_validity_constraints_are_separated_from_metrics(self):
        """ "Result is" is a headline; the indented lines under it are constraints."""
        parsed = rollup.parse_summary(OFFLINE_SUMMARY)
        assert parsed["summary"]["Result is"] == "VALID"
        assert "Min duration satisfied" not in parsed["summary"]
        assert parsed["constraints"]["Min duration satisfied"] == "Yes"

    def test_unknown_sections_are_ignored(self):
        """Trailing prose after the last section must not land in a metric dict."""
        parsed = rollup.parse_summary(OFFLINE_SUMMARY)
        assert all("warning" not in key.lower() for key in parsed["test_parameters"])


class TestBuildRun:
    def test_offline_run(self, tmp_path):
        run_dir = _write_run(tmp_path, "offline", "performance", OFFLINE_SUMMARY)
        run = rollup.build_run(run_dir / "mlperf_log_summary.txt", tmp_path / "mlperf")

        assert run["path"] == "offline/performance"
        assert run["valid"] is True
        assert run["throughput_toks"] == 4567.89
        assert run["samples_per_second"] == 12.3456
        assert run["target_qps"] == 10
        # Offline reports no first-token or per-token percentiles.
        assert run["ttft_p99_ms"] is None
        assert run["tpot_p99_ms"] is None

    def test_server_run_prefers_completed_over_scheduled(self, tmp_path):
        """Scheduled QPS is what LoadGen aimed for; completed is what the SUT did."""
        run_dir = _write_run(tmp_path, "server", "performance", SERVER_SUMMARY)
        run = rollup.build_run(run_dir / "mlperf_log_summary.txt", tmp_path / "mlperf")

        assert run["samples_per_second"] == 9.98
        assert run["throughput_toks"] == 3200.50

    def test_server_run_reports_invalid(self, tmp_path):
        run_dir = _write_run(tmp_path, "server", "performance", SERVER_SUMMARY)
        run = rollup.build_run(run_dir / "mlperf_log_summary.txt", tmp_path / "mlperf")

        assert run["result"] == "INVALID"
        assert run["valid"] is False
        assert run["constraints"]["Performance constraints satisfied"] == "NO"

    def test_server_latency_percentiles_are_milliseconds(self, tmp_path):
        run_dir = _write_run(tmp_path, "server", "performance", SERVER_SUMMARY)
        run = rollup.build_run(run_dir / "mlperf_log_summary.txt", tmp_path / "mlperf")

        assert run["ttft_p99_ms"] == 850.0
        assert run["tpot_p99_ms"] == 25.0

    def test_accuracy_json_is_attached(self, tmp_path):
        run_dir = _write_run(tmp_path, "offline", "accuracy", OFFLINE_SUMMARY)
        (run_dir / "accuracy.json").write_text(json.dumps({"overall": {"accuracy": 0.913}}))
        run = rollup.build_run(run_dir / "mlperf_log_summary.txt", tmp_path / "mlperf")

        assert run["accuracy"]["overall"]["accuracy"] == 0.913

    def test_accuracy_json_absent_is_not_an_error(self, tmp_path):
        run_dir = _write_run(tmp_path, "offline", "accuracy", OFFLINE_SUMMARY)
        run = rollup.build_run(run_dir / "mlperf_log_summary.txt", tmp_path / "mlperf")

        assert "accuracy" not in run

    def test_srt_run_sidecar_supplies_concurrency(self, tmp_path):
        """LoadGen never echoes --max-concurrency; monitor keys its column on it."""
        run_dir = _write_run(tmp_path, "offline", "performance", OFFLINE_SUMMARY)
        (run_dir / "srt_run.json").write_text(json.dumps({"concurrency": "32", "backend": "sglang"}))
        run = rollup.build_run(run_dir / "mlperf_log_summary.txt", tmp_path / "mlperf")

        # Coerced, because monitor does int(concurrency).
        assert run["concurrency"] == 32
        assert run["srt_args"]["backend"] == "sglang"

    def test_missing_srt_run_sidecar_is_not_an_error(self, tmp_path):
        run_dir = _write_run(tmp_path, "offline", "performance", OFFLINE_SUMMARY)
        run = rollup.build_run(run_dir / "mlperf_log_summary.txt", tmp_path / "mlperf")

        assert run["concurrency"] is None
        assert run["srt_args"] == {}

    def test_accuracy_only_stub_still_identifies_the_run(self, tmp_path):
        """AccuracyOnly writes no sections, so scenario/mode come from the layout."""
        run_dir = _write_run(tmp_path, "offline", "accuracy", ACCURACY_SUMMARY)
        run = rollup.build_run(run_dir / "mlperf_log_summary.txt", tmp_path / "mlperf")

        assert run["scenario"] == "offline"
        assert run["mode"] == "accuracy"
        assert run["loadgen_errors"] == 0
        assert run["loadgen_warnings"] == 0
        # No verdict is reported in AccuracyOnly mode; that is not a failure.
        assert run["valid"] is None
        assert run["result"] is None

    def test_incidents_are_counted_when_loadgen_reports_them(self, tmp_path):
        summary = OFFLINE_SUMMARY.replace(
            "No warnings encountered during test.", "3 warnings encountered. See detailed log."
        ).replace("No errors encountered during test.", "2 errors encountered. See detailed log.")
        run_dir = _write_run(tmp_path, "offline", "performance", summary)
        run = rollup.build_run(run_dir / "mlperf_log_summary.txt", tmp_path / "mlperf")

        assert run["loadgen_warnings"] == 3
        assert run["loadgen_errors"] == 2

    def test_performance_run_keeps_its_reported_scenario(self, tmp_path):
        """The summary's own Scenario wins over the directory name."""
        run_dir = _write_run(tmp_path, "offline", "performance", OFFLINE_SUMMARY)
        run = rollup.build_run(run_dir / "mlperf_log_summary.txt", tmp_path / "mlperf")

        assert run["scenario"] == "Offline"
        assert run["mode"] == "performance"


class TestMain:
    def test_writes_rollup_for_every_run(self, tmp_path):
        _write_run(tmp_path, "offline", "performance", OFFLINE_SUMMARY)
        _write_run(tmp_path, "offline", "accuracy", OFFLINE_SUMMARY)

        assert rollup.main(str(tmp_path)) == 0
        data = json.loads((tmp_path / "benchmark-rollup.json").read_text())
        assert data["benchmark_type"] == "mlperf"
        assert sorted(run["path"] for run in data["runs"]) == ["offline/accuracy", "offline/performance"]

    def test_no_results_dir_is_not_a_failure(self, tmp_path):
        """A manual or failed run leaves no logs; postprocess must not break on it."""
        assert rollup.main(str(tmp_path)) == 0
        assert not (tmp_path / "benchmark-rollup.json").exists()

    def test_results_dir_without_summaries(self, tmp_path):
        (tmp_path / "mlperf").mkdir()
        assert rollup.main(str(tmp_path)) == 0
        assert not (tmp_path / "benchmark-rollup.json").exists()
