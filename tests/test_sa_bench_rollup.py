# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SA-Bench rollup generation."""

import csv
import importlib.util
import json
from pathlib import Path


def _load_rollup_module():
    rollup_path = Path(__file__).parent.parent / "src/srtctl/benchmarks/scripts/sa-bench/rollup.py"
    spec = importlib.util.spec_from_file_location("sa_bench_rollup", rollup_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_sa_bench_rollup_generates_json_and_csv_without_metadata(tmp_path):
    """Rollup keeps legacy JSON and emits CSV even when metadata is absent."""
    rollup = _load_rollup_module()

    result_dir = tmp_path / "sa-bench_isl_8192_osl_1024"
    result_dir.mkdir()

    valid_low_concurrency = {
        "model_id": "GLM-5-FP8",
        "random_input_len": 8192,
        "random_output_len": 1024,
        "max_concurrency": 16,
        "output_throughput": 1234.5,
        "total_token_throughput": 9876.0,
        "request_throughput": 1.25,
        "mean_ttft_ms": 100.0,
        "mean_tpot_ms": 10.0,
        "mean_itl_ms": 20.0,
        "mean_e2el_ms": 200.0,
        "median_ttft_ms": 95.0,
        "median_tpot_ms": 8.0,
        "median_itl_ms": 12.5,
        "completed": 160,
        "total_input": None,
        "total_output": None,
        "percentiles_ttft_ms": [(99.0, 150.0)],
        "percentiles_tpot_ms": [(99.0, 15.0)],
        "percentiles_itl_ms": [(99.0, 30.0)],
    }
    valid_high_concurrency = {
        "model_id": "GLM-5-FP8",
        "random_input_len": 8192,
        "random_output_len": 1024,
        "max_concurrency": 32,
        "output_throughput": 2222.25,
        "total_token_throughput": 11111.0,
        "request_throughput": 2.5,
        "mean_ttft_ms": 110.0,
        "mean_tpot_ms": 11.0,
        "mean_itl_ms": 21.0,
        "mean_e2el_ms": 210.0,
        "median_ttft_ms": 96.0,
        "median_tpot_ms": 9.5,
        "median_itl_ms": 13.0,
        "completed": 320,
        "total_input": 1,
        "total_output": 2,
        "percentiles_ttft_ms": [(99.0, 151.0)],
        "percentiles_tpot_ms": [(99.0, 16.0)],
        "percentiles_itl_ms": [(99.0, 31.0)],
    }

    (result_dir / "results_concurrency_32_gpus_48_ctx_8_gen_40.json").write_text(json.dumps(valid_high_concurrency))
    (result_dir / "results_concurrency_16_gpus_48_ctx_8_gen_40.json").write_text(json.dumps(valid_low_concurrency))
    (result_dir / "results_concurrency_99_gpus_48_ctx_8_gen_40.json").write_text("{invalid json")

    rollup.main(tmp_path)

    json_rollup = tmp_path / "benchmark-rollup.json"
    csv_rollup = tmp_path / "benchmark-rollup.csv"
    assert json_rollup.exists()
    assert csv_rollup.exists()

    json_data = json.loads(json_rollup.read_text())
    assert json_data["benchmark_type"] == "sa-bench"
    assert [run["concurrency"] for run in json_data["runs"]] == [16, 32]
    assert json_data["runs"][0]["ttft_mean_ms"] == 100.0
    assert json_data["runs"][1]["ttft_p99_ms"] == 151.0

    rows = _read_csv_rows(csv_rollup)
    assert [row["Concurrency"] for row in rows] == ["16", "32"]

    first = rows[0]
    assert first["Config"] == "GLM-5-FP8"
    assert first["Total GPU Count"] == ""
    assert first["Decode GPU Count"] == ""
    assert first["Total Token Throughput"] == "9876"
    assert first["Output Token Throughput"] == "1234.5"
    assert first["Median TTFT"] == "95"
    assert first["Median TPOT"] == "8"
    assert first["Median ITL"] == "12.5"
    assert first["P90 Decode Running Requests"] == ""
    assert first["Output Token Throughput per User"] == "125"
    assert first["Total Token Throughput per GPU"] == ""


def test_sa_bench_rollup_uses_metadata_for_name_gpu_counts_and_p90(tmp_path):
    """Metadata-driven runs should use job_name, resource-derived GPU counts, and decode-log P90."""
    rollup = _load_rollup_module()

    logs_dir = tmp_path / "logs"
    result_dir = logs_dir / "sa-bench_isl_8192_osl_1024"
    result_dir.mkdir(parents=True)

    (tmp_path / "4049.json").write_text(
        json.dumps(
            {
                "job_name": "job-metadata-name",
                "backend_type": "sglang",
                "resources": {
                    "gpus_per_node": 8,
                    "prefill_nodes": 1,
                    "decode_nodes": 2,
                    "decode_workers": 2,
                    "agg_workers": 0,
                },
            }
        )
    )
    (logs_dir / "b300-003_decode_w0.out").write_text(
        "\n".join(
            [
                "[x] Decode batch, #running-req: 4, #token: 100",
                "[x] Decode batch, #running-req: 8, #token: 100",
                "[x] Decode batch, #running-req: 8, #token: 100",
                "[x] Decode batch, #running-req: 16, #token: 100",
                "[x] Decode batch, #running-req: 32, #token: 100",
                "[x] unrelated line",
            ]
        )
    )

    result = {
        "model_id": "GLM-5-FP8",
        "max_concurrency": 512,
        "output_throughput": 100.0,
        "total_token_throughput": 800.0,
        "median_ttft_ms": 200.0,
        "median_tpot_ms": 25.0,
        "median_itl_ms": 10.0,
    }
    (result_dir / "results_concurrency_512_gpus_999_ctx_8_gen_56.json").write_text(json.dumps(result))

    rollup.main(logs_dir)

    rows = _read_csv_rows(logs_dir / "benchmark-rollup.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["Config"] == "job-metadata-name"
    assert row["Total GPU Count"] == "24"
    assert row["Decode GPU Count"] == "16"
    assert row["P90 Decode Running Requests"] == "32"
    assert row["Output Token Throughput per User"] == "40"
    assert row["Total Token Throughput per GPU"] == "33.333"


def test_sa_bench_rollup_aggregated_deployment_reports_all_gpus(tmp_path):
    """Aggregated runs should report total and decode GPU counts from agg_* fields."""
    rollup = _load_rollup_module()

    logs_dir = tmp_path / "logs"
    result_dir = logs_dir / "sa-bench_isl_1024_osl_1024"
    result_dir.mkdir(parents=True)

    (tmp_path / "4688.json").write_text(
        json.dumps(
            {
                "job_name": "glm47flash-agg-tp4-baseline",
                "backend_type": "sglang",
                "resources": {
                    "gpus_per_node": 4,
                    "prefill_nodes": None,
                    "decode_nodes": None,
                    "agg_nodes": 1,
                    "prefill_workers": 0,
                    "decode_workers": 0,
                    "agg_workers": 1,
                    "gpus_per_agg": 4,
                },
            }
        )
    )

    result = {
        "model_id": "zai-org/GLM-4.7-Flash",
        "max_concurrency": 512,
        "output_throughput": 12617.0,
        "total_token_throughput": 25230.0,
        "median_ttft_ms": 1780.0,
        "median_tpot_ms": 38.5,
        "median_itl_ms": 38.5,
    }
    (result_dir / "results_concurrency_512_gpus_4.json").write_text(json.dumps(result))

    rollup.main(logs_dir)

    rows = _read_csv_rows(logs_dir / "benchmark-rollup.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["Config"] == "glm47flash-agg-tp4-baseline"
    assert row["Total GPU Count"] == "4"
    assert row["Decode GPU Count"] == "4"
    assert row["Total Token Throughput per GPU"] == "6307.5"


def test_sa_bench_rollup_tolerates_null_resource_fields(tmp_path):
    """Legacy metadata with null optional fields must not crash _compute_gpu_counts."""
    rollup = _load_rollup_module()

    logs_dir = tmp_path / "logs"
    result_dir = logs_dir / "sa-bench_isl_1024_osl_1024"
    result_dir.mkdir(parents=True)

    # Pre-fix metadata shape: prefill/decode_nodes are JSON null, no agg_nodes
    # key. Rollup used to crash with TypeError: int() argument must be ... not
    # 'NoneType'.
    (tmp_path / "4688.json").write_text(
        json.dumps(
            {
                "job_name": "legacy-agg-run",
                "backend_type": "sglang",
                "resources": {
                    "gpus_per_node": 4,
                    "prefill_nodes": None,
                    "decode_nodes": None,
                    "prefill_workers": 0,
                    "decode_workers": 0,
                    "agg_workers": 1,
                },
            }
        )
    )

    result = {
        "model_id": "zai-org/GLM-4.7-Flash",
        "max_concurrency": 128,
        "output_throughput": 7130.0,
        "total_token_throughput": 14278.0,
        "median_ttft_ms": 138.0,
        "median_tpot_ms": 17.4,
        "median_itl_ms": 17.4,
    }
    (result_dir / "results_concurrency_128_gpus_4.json").write_text(json.dumps(result))

    rollup.main(logs_dir)

    rows = _read_csv_rows(logs_dir / "benchmark-rollup.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["Config"] == "legacy-agg-run"
    assert row["Total GPU Count"] == "4"


def test_compute_gpu_counts_handles_none_values():
    """_compute_gpu_counts should never raise on null-valued keys."""
    rollup = _load_rollup_module()

    assert rollup._compute_gpu_counts(
        {
            "gpus_per_node": 4,
            "prefill_nodes": None,
            "decode_nodes": None,
            "agg_nodes": None,
            "prefill_workers": None,
            "decode_workers": None,
            "agg_workers": None,
            "gpus_per_agg": None,
        }
    ) == (4, None)

    assert rollup._compute_gpu_counts({}) == (None, None)
    assert rollup._compute_gpu_counts({"gpus_per_node": None}) == (None, None)
