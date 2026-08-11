# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parent.parent
    / "src/srtctl/benchmarks/scripts/aiperf-bench/format_results.py"
)


@pytest.fixture(scope="module")
def format_results():
    spec = importlib.util.spec_from_file_location("format_results", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.format_results


class TestFormatResults:
    def test_aiperf_json_structure_full_output(self, format_results, tmp_path):
        """All fields in actual aiperf nested stats structure produce complete formatted block."""
        data = {
            "input_config": {"loadgen": {"concurrency": 64, "request_rate": None}},
            "burstiness_factor": 1.0,
            "request_count": {"avg": 640.0},
            "benchmark_duration": {"avg": 12.34},
            "total_isl": {"avg": 1024.0},
            "total_output_tokens": {"avg": 512.0},
            "request_throughput": {"avg": 51.87},
            "output_token_throughput": {"avg": 41.50},
            "total_token_throughput": {"avg": 93.37},
            "time_to_first_token": {"avg": 100.0, "p50": 90.0, "p99": 300.0},
            "inter_token_latency": {"avg": 10.0, "p50": 9.5, "p99": 15.0},
            "request_latency": {"avg": 200.0, "p50": 180.0, "p99": 400.0},
        }
        (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(data))
        result = format_results(str(tmp_path))
        assert "============ Serving Benchmark Result ============" in result
        assert "==================================================" in result
        assert "Traffic request rate: inf" in result
        assert "Burstiness factor: 1.00 (Poisson process)" in result
        assert "Maximum request concurrency: 64" in result
        assert "640" in result
        assert "12.34" in result
        assert "100.00" in result
        assert "90.00" in result
        assert "300.00" in result

    def test_itl_mean_includes_ttft(self, format_results, tmp_path):
        """Mean ITL is a weighted average of TTFT and TPOT; median/p99 are N/A."""
        data = {
            "request_count": {"avg": 100.0},
            "total_output_tokens": {"avg": 1000.0},
            "time_to_first_token": {"avg": 500.0, "p50": 450.0, "p99": 900.0},
            "inter_token_latency": {"avg": 50.0, "p50": 45.0, "p99": 80.0},
        }
        (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(data))
        result = format_results(str(tmp_path))
        # mean_ITL = (100*500 + 900*50) / 1000 = 95000/1000 = 95.00
        assert "Mean ITL (ms):" in result
        lines = result.splitlines()
        itl_lines = [line for line in lines if "ITL (ms)" in line]
        assert len(itl_lines) == 3
        assert "95.00" in itl_lines[0]
        assert "N/A" in itl_lines[1]
        assert "N/A" in itl_lines[2]

    def test_itl_mean_na_when_missing_fields(self, format_results, tmp_path):
        """Mean ITL falls back to N/A when required fields are missing."""
        data = {
            "time_to_first_token": {"avg": 500.0},
            "inter_token_latency": {"avg": 50.0},
        }
        (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(data))
        result = format_results(str(tmp_path))
        lines = result.splitlines()
        itl_mean_line = [line for line in lines if "Mean ITL (ms)" in line][0]
        assert "N/A" in itl_mean_line

    def test_output_throughput_per_user(self, format_results, tmp_path):
        """Output throughput per user is 1000 / mean_TPOT(ms)."""
        data = {
            "inter_token_latency": {"avg": 50.0, "p50": 45.0, "p99": 80.0},
        }
        (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(data))
        result = format_results(str(tmp_path))
        # 1000 / 50 = 20.00 tok/s
        assert "Output throughput per user (tok/s):" in result
        lines = result.splitlines()
        tpu_line = [line for line in lines if "Output throughput per user" in line][0]
        assert "20.00" in tpu_line

    def test_output_throughput_per_user_na_when_missing(self, format_results, tmp_path):
        """Output throughput per user is N/A when TPOT is missing."""
        (tmp_path / "profile_export_aiperf.json").write_text("{}")
        result = format_results(str(tmp_path))
        lines = result.splitlines()
        tpu_line = [line for line in lines if "Output throughput per user" in line][0]
        assert "N/A" in tpu_line

    def test_power_summary_shown_when_present(self, format_results, tmp_path):
        """Power summary section appears when GPU telemetry fields are present."""
        data = {
            "total_gpu_power": {"unit": "W", "avg": 15879.77},
            "total_gpu_energy": {"unit": "J", "avg": 11906900.64},
            "output_tokens_per_joule": {"unit": "tokens/J", "avg": 5.10},
        }
        (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(data))
        result = format_results(str(tmp_path))
        assert "Power Summary" in result
        assert "15879.77" in result
        assert "11906900.64" in result
        assert "5.10" in result

    def test_power_summary_hidden_when_absent(self, format_results, tmp_path):
        """Power summary section is omitted when no GPU telemetry fields are present."""
        data = {
            "inter_token_latency": {"avg": 10.0, "p50": 9.0, "p99": 20.0},
        }
        (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(data))
        result = format_results(str(tmp_path))
        assert "Power Summary" not in result

    def test_nested_fields(self, format_results, tmp_path):
        """Nested JSON fields resolved via dot-notation using .avg subkey."""
        data = {
            "time_to_first_token": {"avg": 123.45, "p50": 100.0, "p99": 500.0},
            "inter_token_latency": {"avg": 10.0, "p50": 9.0, "p99": 20.0},
            "request_latency": {"avg": 300.0, "p50": 250.0, "p99": 600.0},
        }
        (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(data))
        result = format_results(str(tmp_path))
        assert "123.45" in result
        assert "10.00" in result
        assert "300.00" in result

    def test_null_request_rate_shows_inf(self, format_results, tmp_path):
        """JSON null for request_rate (aiperf's 'unlimited') displays as 'inf'."""
        data = {"input_config": {"loadgen": {"request_rate": None}}}
        (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(data))
        result = format_results(str(tmp_path))
        assert "Traffic request rate: inf" in result

    def test_missing_fields_show_na(self, format_results, tmp_path):
        """Fields absent from JSON display as N/A."""
        (tmp_path / "profile_export_aiperf.json").write_text("{}")
        result = format_results(str(tmp_path))
        assert "N/A" in result
        assert "============ Serving Benchmark Result ============" in result

    def test_missing_file_returns_warning(self, format_results, tmp_path):
        """Missing JSON file returns a warning string (does not raise)."""
        result = format_results(str(tmp_path))
        assert "Warning" in result
        assert "profile_export_aiperf.json" in result

    def test_invalid_json_returns_warning(self, format_results, tmp_path):
        """Corrupt JSON file returns a warning string (does not raise)."""
        (tmp_path / "profile_export_aiperf.json").write_text("not valid json {{")
        result = format_results(str(tmp_path))
        assert "Warning" in result

    def test_burstiness_1_annotated(self, format_results, tmp_path):
        """Burstiness factor 1.0 appends '(Poisson process)'."""
        (tmp_path / "profile_export_aiperf.json").write_text(
            json.dumps({"burstiness_factor": 1.0})
        )
        result = format_results(str(tmp_path))
        assert "(Poisson process)" in result

    def test_burstiness_non_1_not_annotated(self, format_results, tmp_path):
        """Burstiness factor != 1.0 does not append '(Poisson process)'."""
        (tmp_path / "profile_export_aiperf.json").write_text(
            json.dumps({"burstiness_factor": 2.0})
        )
        result = format_results(str(tmp_path))
        assert "2.00" in result
        assert "(Poisson process)" not in result

    def test_integer_fields_have_no_decimal(self, format_results, tmp_path):
        """Integer-class fields (counts, tokens) display without decimal point."""
        data = {
            "request_count": {"avg": 1000.0},
            "total_isl": {"avg": 8192.0},
            "total_output_tokens": {"avg": 4096.0},
            "input_config": {"loadgen": {"concurrency": 32}},
        }
        (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(data))
        result = format_results(str(tmp_path))
        assert "1000" in result
        assert "1000." not in result
        assert "8192" in result

    def test_non_dict_json_returns_warning(self, format_results, tmp_path):
        """Valid JSON that is not an object returns a warning string."""
        (tmp_path / "profile_export_aiperf.json").write_text("[1, 2, 3]")
        result = format_results(str(tmp_path))
        assert "Warning" in result

    def test_row_alignment(self, format_results, tmp_path):
        """Each data row uses 40-char label column + space + value."""
        data = {
            "request_count": {"avg": 999.0},
            "request_throughput": {"avg": 12.34},
        }
        (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(data))
        result = format_results(str(tmp_path))
        for line in result.splitlines():
            if "Successful requests:" in line:
                assert line == "Successful requests:                     999"
            if "Request throughput (req/s):" in line:
                assert line == "Request throughput (req/s):              12.34"
