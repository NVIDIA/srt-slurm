# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MLPerf benchmark runner and its LoadGen rollup."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

from srtctl.benchmarks import get_runner, list_benchmarks
from srtctl.benchmarks.base import SCRIPTS_DIR
from srtctl.core.schema import (
    BenchmarkConfig,
    FrontendConfig,
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


def _config(frontend_type="sglang", **benchmark_kwargs):
    return SrtConfig(
        name="test",
        model=ModelConfig(path="/model/gpt-oss-120b", container="/image", precision="fp4"),
        resources=ResourceConfig(gpu_type="gb300"),
        frontend=FrontendConfig(type=frontend_type),
        benchmark=BenchmarkConfig(type="mlperf", **benchmark_kwargs),
    )


def _valid_kwargs(**overrides):
    kwargs = {
        "mlperf_harness_dir": "/mlperf-inference",
        "mlperf_benchmark": "gpt-oss-120b",
        "mlperf_dataset": "/datasets/gpt-oss.parquet",
    }
    kwargs.update(overrides)
    return kwargs


class TestMLPerfRunner:
    """Test the MLPerf LoadGen benchmark runner."""

    def test_in_registry(self):
        """mlperf is registered in the benchmark list."""
        assert "mlperf" in list_benchmarks()

    def test_get_runner(self):
        """Can get a runner for mlperf."""
        runner = get_runner("mlperf")
        assert runner.name == "MLPerf"
        assert "mlperf" in runner.script_path

    def test_validate_missing_harness_dir(self):
        """Validates that mlperf_harness_dir is required."""
        errors = get_runner("mlperf").validate_config(_config(**_valid_kwargs(mlperf_harness_dir=None)))
        assert any("mlperf_harness_dir" in e for e in errors)

    def test_validate_missing_benchmark(self):
        """Validates that mlperf_benchmark is required."""
        errors = get_runner("mlperf").validate_config(_config(**_valid_kwargs(mlperf_benchmark=None)))
        assert any("mlperf_benchmark" in e for e in errors)

    def test_validate_missing_dataset(self):
        """Validates that mlperf_dataset is required."""
        errors = get_runner("mlperf").validate_config(_config(**_valid_kwargs(mlperf_dataset=None)))
        assert any("mlperf_dataset" in e for e in errors)

    def test_validate_rejects_unknown_scenario(self):
        """LoadGen only knows the scenarios the harness implements."""
        errors = get_runner("mlperf").validate_config(_config(**_valid_kwargs(mlperf_scenario="interactive")))
        assert any("mlperf_scenario" in e for e in errors)

    def test_validate_rejects_unknown_mode(self):
        """Modes are performance/accuracy."""
        errors = get_runner("mlperf").validate_config(_config(**_valid_kwargs(mlperf_mode="compliance")))
        assert any("mlperf_mode" in e for e in errors)

    def test_validate_rejects_combined_mode(self):
        """One LoadGen mode per job: the two modes do not share a token budget."""
        errors = get_runner("mlperf").validate_config(_config(**_valid_kwargs(mlperf_mode="both")))
        assert any("mlperf_mode" in e for e in errors)

    def test_validate_server_requires_user_conf(self):
        """Server target_qps only comes from user.conf; the harness default is a placeholder."""
        errors = get_runner("mlperf").validate_config(_config(**_valid_kwargs(mlperf_scenario="server")))
        assert any("mlperf_user_conf" in e for e in errors)

    def test_validate_offline_does_not_require_user_conf(self):
        """Offline is still measurable off mlperf.conf alone."""
        errors = get_runner("mlperf").validate_config(_config(**_valid_kwargs(mlperf_scenario="offline")))
        assert errors == []

    def test_validate_rejects_nonpositive_max_new_tokens(self):
        """A zero token budget would make every response empty."""
        errors = get_runner("mlperf").validate_config(_config(**_valid_kwargs(mlperf_max_new_tokens=0)))
        assert any("mlperf_max_new_tokens" in e for e in errors)

    def test_validate_rejects_incompatible_frontend(self):
        """The sglang backend posts to /generate, which a Dynamo frontend does not serve."""
        errors = get_runner("mlperf").validate_config(_config(frontend_type="dynamo", **_valid_kwargs()))
        assert any("frontend.type" in e for e in errors)

    def test_validate_rejects_nonpositive_concurrency(self):
        """Zero or negative concurrency is rejected."""
        errors = get_runner("mlperf").validate_config(_config(**_valid_kwargs(concurrency=0)))
        assert any("positive" in e for e in errors)

    def test_validate_valid(self):
        """A fully specified config passes validation."""
        errors = get_runner("mlperf").validate_config(
            _config(
                **_valid_kwargs(
                    mlperf_scenario="server",
                    mlperf_mode="accuracy",
                    mlperf_user_conf="/configs/user.conf",
                    mlperf_max_new_tokens=32768,
                    concurrency=256,
                )
            )
        )
        assert errors == []

    def test_build_command(self):
        """Build command carries every harness input positionally."""
        runtime = MagicMock()
        runtime.frontend_port = 8000
        config = _config(
            **_valid_kwargs(
                mlperf_scenario="server",
                mlperf_mode="accuracy",
                mlperf_user_conf="/configs/user.conf",
                mlperf_max_new_tokens=32768,
                mlperf_reference_data="/datasets/gpt-oss-reference.parquet",
                concurrency=256,
            )
        )
        cmd = get_runner("mlperf").build_command(config, runtime)
        assert cmd[0] == "bash"
        assert cmd[1] == "/srtctl-benchmarks/mlperf/bench.sh"
        assert cmd[2] == "http://localhost:8000"
        assert cmd[3] == "/mlperf-inference"
        assert cmd[4] == "gpt-oss-120b"
        assert cmd[5] == "server"
        assert cmd[6] == "accuracy"
        assert cmd[7] == "sglang"
        assert cmd[8] == "/datasets/gpt-oss.parquet"
        assert cmd[9] == "/configs/user.conf"
        assert cmd[10] == "256"
        assert cmd[11] == "32768"
        assert cmd[12] == "/datasets/gpt-oss-reference.parquet"
        # The accuracy scorer would otherwise pull a tokenizer from HuggingFace.
        assert cmd[13] == "/model/gpt-oss-120b"

    def test_build_command_defaults(self):
        """Optional inputs become empty positionals, not missing arguments."""
        runtime = MagicMock()
        runtime.frontend_port = 8000
        cmd = get_runner("mlperf").build_command(_config(**_valid_kwargs()), runtime)
        assert cmd[5] == "offline"
        assert cmd[6] == "performance"
        assert cmd[9] == ""
        assert cmd[10] == ""
        assert cmd[11] == ""
        assert cmd[12] == ""

    def test_script_exists(self):
        """mlperf bench.sh and rollup.py ship with the package."""
        assert (SCRIPTS_DIR / "mlperf" / "bench.sh").exists()
        assert (SCRIPTS_DIR / "mlperf" / "rollup.py").exists()

    def test_environment_passthrough(self):
        """benchmark.env reaches the harness environment."""
        config = _config(**_valid_kwargs(env={"MLPERF_EXTRA_ARGS": "--max-samples 500"}))
        env = get_runner("mlperf").get_environment(config, MagicMock())
        assert env["MLPERF_EXTRA_ARGS"] == "--max-samples 500"


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
