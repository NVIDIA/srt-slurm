# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Log source → shared metric schema, exact-scope references and source evidence."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from test_dsight import write_run

from srtctl.dsight.importer import Importer
from srtctl.dsight.log_metrics import LogMetricDefinition, LogMetricEvent
from srtctl.dsight.log_metrics.reader import read_log_metrics
from srtctl.dsight.log_metrics.tokenspeed import (
    ACTIVE_DECODE,
    ACTIVE_PAGES,
    DECODE_LIMIT,
    POOL_PAGES,
    DynamoTokenSpeedLogMetrics,
)
from srtctl.dsight.query import TraceDataset
from srtctl.dsight.sources import source_identity


def config(second=30, maximum="8", rank=0):
    return (
        f"\x1b[38;20m[2026-09-17 10:58:{second:02d},000  ATTN TP RANK {rank}] - INFO - Scheduler config: "
        f"num_device_pages=129 max_batch_size={maximum} (global max_num_seqs=64, dp_size=8) (event_loop.py:329)\x1b[0m"
    )


def batch(second=33, active=4, pages=80, pool=128, rank=0, kind="Decode"):
    return (
        f"[2026-09-17 10:58:{second:02d},230 ATTN TP RANK {rank}] - INFO - {kind} batch. "
        f"#running-req: {active}, #pages(active/cached/total): {pages}/16/{pool}, #queue-req: 12 (batch_log.py:151)"
    )


def log_run(root: Path, lines: list[str], **options):
    logs, _ = write_run(root)
    (logs / "decode-host_decode_w0.out").write_text("\n".join(lines) + "\n")
    run = Importer(logs, iteration_timezone="UTC", **options)
    return run, run.run()


def metrics(data):
    return {s["name"]: s for s in data["metrics"] if s.get("generator") == "dynamo-tokenspeed"}


def test_tokenspeed_uses_local_batch_limit_and_usable_pool_with_line_evidence(tmp_path):
    run, data = log_run(tmp_path, [config(), batch(), batch(36, active=6, pages=96)])
    series = metrics(data)
    assert set(series) == {ACTIVE_DECODE, DECODE_LIMIT, ACTIVE_PAGES, POOL_PAGES}
    active, limit, pages, pool = (series[name] for name in (ACTIVE_DECODE, DECODE_LIMIT, ACTIVE_PAGES, POOL_PAGES))
    assert [p[:2] for p in active["points"]] == [[2.23, 4], [5.23, 6]]
    assert [p[:2] for p in limit["points"]] == [[-1, 8]]  # Configuration evidence before client start.
    assert [p[1] for p in pages["points"]] == [80, 96]
    assert [p[1] for p in pool["points"]] == [128, 128]  # Not configured 129 or token capacity.
    assert active["reference"]["series_id"] == limit["id"]
    assert pages["reference"]["series_id"] == pool["id"]
    assert active["rank_kind"] == "attention_tp" and active["rank"] == 0
    assert active["worker"] == "decode-0" and active["host"] == "decode-host"
    assert active["worker_process"] is None  # Never borrow an OTel process identity.
    source, line = active["points"][0][2:]
    assert line == 2 and data["sources"][source]["kind"] == "worker_log"
    assert Path(data["sources"][source]["path"]).read_text().splitlines()[line - 1] == batch()
    assert any(s["endpoint"] != "worker_log" for s in data["metrics"])  # Native metrics coexist.
    assert len({s["id"] for s in data["metrics"]}) == len(data["metrics"])
    assert data["audit"]["metric_points"] == sum(len(s["points"]) for s in data["metrics"])
    for series in data["metrics"]:
        if series["worker"] in run.workers:
            assert series["id"] in run.workers[series["worker"]]["metrics"]
    carried = TraceDataset(data).query("metrics", name=DECODE_LIMIT, points=True)["items"][0]
    assert carried["points"] == [] and carried["samples"] == 0
    assert carried["carried_setting"] == limit["points"]


def test_config_changes_and_invalid_restart_do_not_reuse_stale_limit(tmp_path):
    _, data = log_run(
        tmp_path,
        [
            config(20, "64"),
            config(30, "8"),
            batch(),
            config(34, "16"),
            batch(35, active=12),
            config(36, "unknown"),
            batch(37),
            config(38, "32"),
            batch(39),
        ],
    )
    limit = metrics(data)[DECODE_LIMIT]
    assert [p[:2] for p in limit["points"]] == [[-1, 8], [3, 16], [5, None], [7, 32]]
    result = TraceDataset(data).query("metrics", name=DECODE_LIMIT, start=6, end=9, points=True)["items"][0]
    assert result["carried_setting"] == [limit["points"][2]]
    assert result["min"] == result["max"] == 32
    assert result["points"] == [limit["points"][3]]


def test_unknown_limit_is_optional_and_never_zero_or_another_rank(tmp_path):
    _, data = log_run(tmp_path, [config(rank=1), batch(rank=0)])
    active = metrics(data)[ACTIVE_DECODE]
    assert active["reference"]["series_id"] is None
    assert active["points"][0][1] == 4
    # No guessed ratio or utilization series, and source samples remain usable.
    assert not any("utilization" in s["name"] for s in metrics(data).values())


def test_limits_do_not_join_across_workers_or_files(tmp_path):
    logs, _ = write_run(tmp_path)
    (logs / "decode-host_decode_w0.out").write_text(config() + "\n")
    (logs / "decode-host_decode_w1.out").write_text(batch() + "\n")
    (logs / "decode-host_decode_w0_e1.out").write_text(batch(34) + "\n")
    data = Importer(logs, iteration_timezone="UTC").run()
    active = [s for s in data["metrics"] if s["name"] == ACTIVE_DECODE]
    assert len(active) == 2
    assert all(s["reference"]["series_id"] is None for s in active)


def test_config_after_first_sample_is_not_moved_backwards(tmp_path):
    _, data = log_run(tmp_path, [batch(), config(35), batch(36)])
    limit = metrics(data)[DECODE_LIMIT]
    assert limit["points"][0][0] == 4
    assert TraceDataset(data).query("metrics", name=DECODE_LIMIT, end=3)["items"][0]["carried_setting"] == []


def test_duplicate_and_conflicting_limits_keep_shared_quality_semantics(tmp_path):
    _, data = log_run(tmp_path, [config(), config(), config(maximum="16"), batch()])
    limit = metrics(data)[DECODE_LIMIT]
    assert len(limit["points"]) == 2 and limit["conflict_timestamps"] == [-1]
    assert limit["conflicting_samples"] == 2
    catalog = next(f for f in data["metric_catalog"] if f["name"] == DECODE_LIMIT)
    assert "Conflicting raw values" in catalog["quality"]
    assert len(TraceDataset(data).query("metrics", name=DECODE_LIMIT)["items"][0]["carried_setting"]) == 2


def test_missing_timezone_omits_log_metrics_with_explicit_warning(tmp_path):
    logs, _ = write_run(tmp_path)
    (logs / "decode-host_decode_w0.out").write_text(config() + "\n" + batch() + "\n")
    data = Importer(logs).run()
    assert not metrics(data)
    assert data["audit"]["unaligned_log_metric_records"] == 2
    assert any("Log metrics with local timestamps omitted" in w for w in data["meta"]["warnings"])


def test_logs_alone_enable_metrics_without_client_otel_or_tachometer(tmp_path):
    (tmp_path / "worker_decode_w0.out").write_text(config() + "\n" + batch() + "\n" + batch(35) + "\n")
    data = Importer(tmp_path, iteration_timezone="UTC").run()
    assert data["capabilities"]["metrics"] and data["capabilities"]["iterations"]
    assert not data["capabilities"]["requests"] and not data["capabilities"]["request_breakdown"]
    assert data["requests"] == data["server_spans"] == data["profiles"] == []
    assert data["meta"]["duration"] == pytest.approx(2.001)
    assert len(metrics(data)) == 4
    assert metrics(data)[DECODE_LIMIT]["points"][0][0] < 0


def test_prefill_snapshots_supply_pages_without_inventing_decode_activity():
    generator = DynamoTokenSpeedLogMetrics()
    source = source_identity(Path("node_prefill_w0.out"))
    assert source
    event = generator.parse_line(batch(kind="Prefill"), source)
    assert event and dict(event.values) == {ACTIVE_PAGES: 80, POOL_PAGES: 128}
    assert generator.parse_line(config(), source) is None
    assert generator.parse_line("unrelated log line", source) is None


def test_missing_page_fields_are_not_zero(tmp_path):
    line = "[2026-09-17 10:58:33,230 ATTN TP RANK 0] - INFO - Decode batch. #running-req: 3"
    _, data = log_run(tmp_path, [line])
    assert set(metrics(data)) == {ACTIVE_DECODE}
    assert metrics(data)[ACTIVE_DECODE]["reference"]["series_id"] is None


class ExampleGenerator:
    """Different syntax and rank namespace exercise the shared Protocol boundary."""

    name = "example"
    definitions = (
        LogMetricDefinition(
            "log_example_active", "Active", "items", "Recorded active work", reference="log_example_limit"
        ),
        LogMetricDefinition("log_example_limit", "Limit", "items", "Recorded config", temporal="setting"),
    )

    def parse_line(self, line, source):
        if not line.startswith("example "):
            return None
        _, process, value = line.split()
        return LogMetricEvent(
            "2026-09-17T10:58:33.5+00:00",
            (("log_example_active", float(value)), ("log_example_limit", 10)),
            rank=2,
            rank_kind="dp",
            process=process,
            labels=(("partition", "a"),),
        )


def test_new_generator_uses_same_normalizer_without_engine_switches(tmp_path):
    logs, _ = write_run(tmp_path)
    (logs / "decode-host_decode_w0.out").write_text("example p1 3\nexample p2 7\n")
    run = Importer(logs)  # UTC offset in the adapter's events needs no timezone option.
    run.run()
    series = read_log_metrics(run, (ExampleGenerator(),))
    assert len(series) == 4
    by_id = {s["id"]: s for s in series}
    for active in (s for s in series if s["name"] == "log_example_active"):
        reference = by_id[active["reference"]["series_id"]]
        assert reference["worker_process"] == active["worker_process"]
        assert active["rank_kind"] == "dp" and active["labels"]["partition"] == "a"
        assert active["points"][0][0] == 2.5
        assert reference["points"][0][2:] == active["points"][0][2:]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ExampleGenerator.definitions[0].unit = "changed"


def test_reference_units_are_validated_at_interface_boundary(tmp_path):
    run, _ = log_run(tmp_path, [])
    generator = ExampleGenerator()
    generator.definitions = (generator.definitions[0], dataclasses.replace(generator.definitions[1], unit="bytes"))
    with pytest.raises(ValueError, match="Invalid log metric reference"):
        read_log_metrics(run, (generator,))


def test_carriage_return_progress_keeps_physical_log_line_evidence(tmp_path):
    _, data = log_run(tmp_path, ["warming\rprogress\rready", config(), batch()])
    active = metrics(data)[ACTIVE_DECODE]
    snapshot = next(row for row in data["iterations"] if row.get("backend") == "tokenspeed")
    assert active["points"][0][2:] == snapshot["evidence"]
    assert active["points"][0][3] == 3
    assert metrics(data)[DECODE_LIMIT]["points"][0][3] == 2
