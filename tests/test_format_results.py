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
    def test_flat_fields_full_output(self, format_results, tmp_path):
        """All fields present in flat JSON produce complete formatted block."""
        data = {
            "request_rate": 1e30,
            "burstiness_factor": 1.0,
            "max_concurrency": 64,
            "num_successful_requests": 640,
            "duration": 12.34,
            "total_input_tokens": 1024,
            "total_output_tokens": 512,
            "request_throughput": 51.87,
            "output_token_throughput": 41.50,
            "total_token_throughput": 93.37,
            "ttft_mean_ms": 100.0,
            "ttft_median_ms": 90.0,
            "ttft_p99_ms": 300.0,
            "tpot_mean_ms": 10.0,
            "tpot_median_ms": 9.5,
            "tpot_p99_ms": 15.0,
            "itl_mean_ms": 5.0,
            "itl_median_ms": 4.5,
            "itl_p99_ms": 8.0,
            "e2el_mean_ms": 200.0,
            "e2el_median_ms": 180.0,
            "e2el_p99_ms": 400.0,
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

    def test_nested_fields(self, format_results, tmp_path):
        """Nested JSON fields resolved via dot-notation."""
        data = {
            "time_to_first_token": {"mean": 123.45, "median": 100.0, "p99": 500.0},
            "time_per_output_token": {"mean": 5.0, "median": 4.5, "p99": 9.0},
            "inter_token_latency": {"mean": 10.0, "median": 9.0, "p99": 20.0},
            "end_to_end_latency": {"mean": 300.0, "median": 250.0, "p99": 600.0},
        }
        (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(data))
        result = format_results(str(tmp_path))
        assert "123.45" in result
        assert "10.00" in result
        assert "300.00" in result

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
            "num_successful_requests": 1000,
            "total_input_tokens": 8192,
            "total_output_tokens": 4096,
            "max_concurrency": 32,
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
        data = {"num_successful_requests": 999, "request_throughput": 12.34}
        (tmp_path / "profile_export_aiperf.json").write_text(json.dumps(data))
        result = format_results(str(tmp_path))
        for line in result.splitlines():
            if "Successful requests:" in line:
                assert line == "Successful requests:                     999"
            if "Request throughput (req/s):" in line:
                assert line == "Request throughput (req/s):              12.34"
