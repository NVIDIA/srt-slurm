# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AIPerf rollup generation (schema/unit parity with sa-bench)."""

import csv
import importlib.util
import json
from pathlib import Path


def _load_rollup_module():
    rollup_path = Path(__file__).parent.parent / "src/srtctl/benchmarks/scripts/aiperf/rollup.py"
    spec = importlib.util.spec_from_file_location("aiperf_rollup", rollup_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_export(artifacts_dir: Path, conc: int, data: dict) -> None:
    # aiperf writes profile_export_aiperf.json under <artifact-dir>/<run>/ ; the rollup globs
    # artifacts/conc_*/**/profile_export_aiperf.json, so nest it one level.
    run_dir = artifacts_dir / f"conc_{conc}" / "profile_export"
    run_dir.mkdir(parents=True)
    (run_dir / "profile_export_aiperf.json").write_text(json.dumps(data))


def _export(*, ttft_avg, ttft_p50, ttft_p99, itl_avg, itl_p50, itl_p99, out_tps, tot_tps, good, total_isl, total_osl):
    """Build a minimal profile_export_aiperf.json-shaped dict (metrics are flat stat objects)."""
    return {
        "output_token_throughput": {"unit": "tokens/sec", "avg": out_tps},
        "total_token_throughput": {"unit": "tokens/sec", "avg": tot_tps},
        "request_throughput": {"unit": "requests/sec", "avg": 1.25},
        "time_to_first_token": {"unit": "ms", "avg": ttft_avg, "p50": ttft_p50, "p99": ttft_p99},
        "inter_token_latency": {"unit": "ms", "avg": itl_avg, "p50": itl_p50, "p99": itl_p99},
        "request_latency": {"unit": "ms", "avg": 200.0},
        "good_request_count": {"unit": "requests", "avg": good},
        "total_isl": {"unit": "tokens", "avg": total_isl},
        "total_output_tokens": {"unit": "tokens", "avg": total_osl},
    }


def test_aiperf_rollup_generates_json_and_csv_without_metadata(tmp_path):
    """Rollup emits sa-bench-shaped JSON + CSV even when job metadata is absent."""
    rollup = _load_rollup_module()

    logs_dir = tmp_path / "logs"
    artifacts = logs_dir / "artifacts"
    artifacts.mkdir(parents=True)

    _write_export(
        artifacts, 16,
        _export(ttft_avg=100.0, ttft_p50=95.0, ttft_p99=150.0, itl_avg=20.0, itl_p50=12.5, itl_p99=30.0,
                out_tps=1234.5, tot_tps=9876.0, good=160, total_isl=131072, total_osl=16384),
    )
    _write_export(
        artifacts, 32,
        _export(ttft_avg=110.0, ttft_p50=96.0, ttft_p99=151.0, itl_avg=21.0, itl_p50=13.0, itl_p99=31.0,
                out_tps=2222.25, tot_tps=11111.0, good=320, total_isl=262144, total_osl=32768),
    )

    rollup.main(logs_dir)

    json_rollup = logs_dir / "benchmark-rollup.json"
    csv_rollup = logs_dir / "benchmark-rollup.csv"
    assert json_rollup.exists()
    assert csv_rollup.exists()

    json_data = json.loads(json_rollup.read_text())
    assert json_data["benchmark_type"] == "aiperf"
    assert [run["concurrency"] for run in json_data["runs"]] == [16, 32]

    first = json_data["runs"][0]
    assert first["ttft_mean_ms"] == 100.0
    assert first["ttft_p99_ms"] == 150.0
    # completed_requests comes from good_request_count (was hardcoded null before the fix).
    assert first["completed_requests"] == 160
    # total_input_tokens comes from total_isl (always present, tokenizer-derived).
    assert first["total_input_tokens"] == 131072
    assert first["total_output_tokens"] == 16384
    assert json_data["runs"][1]["ttft_p99_ms"] == 151.0

    rows = _read_csv_rows(csv_rollup)
    assert [row["Concurrency"] for row in rows] == ["16", "32"]

    row = rows[0]
    assert row["Total GPU Count"] == ""  # no metadata
    assert row["Output Token Throughput"] == "1234.5"
    assert row["Median TTFT"] == "95"
    # aiperf has no distinct TPOT metric: TPOT is defined as ITL, so the columns are equal.
    assert row["Median TPOT"] == "12.5"
    assert row["Median ITL"] == "12.5"
    assert row["P90 Decode Running Requests"] == ""  # no sglang metadata
    # Output Token Throughput per User matches sa-bench's estimator: 1000 / median-TPOT-ms.
    assert row["Output Token Throughput per User"] == "80"


def test_aiperf_rollup_uses_metadata_for_gpu_counts_and_p90(tmp_path):
    """Metadata-driven runs use resource GPU counts and the decode-log P90 (ported from sa-bench)."""
    rollup = _load_rollup_module()

    logs_dir = tmp_path / "logs"
    artifacts = logs_dir / "artifacts"
    artifacts.mkdir(parents=True)

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

    _write_export(
        artifacts, 512,
        _export(ttft_avg=200.0, ttft_p50=200.0, ttft_p99=250.0, itl_avg=25.0, itl_p50=25.0, itl_p99=40.0,
                out_tps=100.0, tot_tps=800.0, good=1024, total_isl=4096, total_osl=1024),
    )

    rollup.main(logs_dir)

    rows = _read_csv_rows(logs_dir / "benchmark-rollup.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["Config"] == "job-metadata-name"
    assert row["Total GPU Count"] == "24"
    assert row["Decode GPU Count"] == "16"
    assert row["P90 Decode Running Requests"] == "32"
    assert row["Output Token Throughput per User"] == "40"  # 1000 / 25
    assert row["Total Token Throughput per GPU"] == "33.333"  # 800 / 24


def test_aiperf_rollup_warns_on_osl_mismatch(tmp_path, capsys):
    """A nonzero osl_mismatch_count is surfaced loudly (fixed-OSL forcing did not hold)."""
    rollup = _load_rollup_module()

    logs_dir = tmp_path / "logs"
    artifacts = logs_dir / "artifacts"
    artifacts.mkdir(parents=True)

    export = _export(ttft_avg=100.0, ttft_p50=95.0, ttft_p99=150.0, itl_avg=20.0, itl_p50=12.5, itl_p99=30.0,
                     out_tps=1234.5, tot_tps=9876.0, good=160, total_isl=131072, total_osl=16384)
    export["osl_mismatch_count"] = {"unit": "requests", "avg": 5}
    export["osl_mismatch_diff_pct"] = {"unit": "%", "avg": -12.0}
    _write_export(artifacts, 16, export)

    rollup.main(logs_dir)

    err = capsys.readouterr().err
    assert "did not honor the requested OSL" in err
