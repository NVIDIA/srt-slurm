# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the scrape-timings.jsonl HTML visualizer (the Python side: load, payload, page shell)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from srtctl.analysis.scrape_timings_viz import (
    COST_SEGMENTS,
    build_payload,
    load_timings,
    main,
    render_html,
)
from tests.test_power_collector import _body, _endpoints, _session, exporters  # noqa: F401

T0 = 1_700_000_000.0


def _scrape(
    host: str,
    seq: int,
    *,
    start: float = 0.0,
    duration: float = 0.02,
    lag: float | None = 0.001,
    status: int | None = 200,
    error: str | None = None,
    rows: int = 4,
    parse: float | None = 0.001,
) -> dict:
    started = T0 + start
    return {
        "event": "scrape",
        "job_id": "12345",
        "run_name": "recipe_12345",
        "hostname": host,
        "scrape_seq": seq,
        "request_started_at_unix": started,
        "request_finished_at_unix": started + duration,
        "request_duration_seconds": duration,
        "parse_seconds": parse,
        "schedule_lag_seconds": lag,
        "sample_timestamp_unix": started + duration / 2 if rows else None,
        "http_status": status,
        "error_type": error,
        "row_count": rows,
        "reason_codes": [],
    }


def _write(
    seq: int,
    *,
    scheduled: float | None = None,
    completed: bool = True,
    error: str | None = None,
    lock: float = 0.0001,
    write: float = 0.0005,
) -> dict:
    return {
        "event": "cycle_write",
        "job_id": "12345",
        "run_name": "recipe_12345",
        "scrape_seq": seq,
        "scheduled_at_unix": scheduled,
        "row_count": 8,
        "writer_lock_wait_seconds": lock,
        "sample_write_seconds": write,
        "sample_write_completed": completed,
        "sample_write_error": error,
    }


def _jsonl(tmp_path: Path, records, extra_lines=()) -> Path:
    path = tmp_path / "scrape-timings.jsonl"
    lines = [json.dumps(r) for r in records] + list(extra_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _two_cycles():
    return [
        _scrape("node-b", 0, start=0.001),
        _scrape("node-a", 0, start=0.002),
        _write(0, scheduled=T0),
        _scrape("node-a", 1, start=0.101),
        _write(1, scheduled=T0 + 0.1),
        {"event": "diagnostic_summary", "dropped_records": 0},
    ]


class TestLoadTimings:
    def test_reads_all_three_event_kinds(self, tmp_path):
        t = load_timings(_jsonl(tmp_path, _two_cycles()))
        assert (t.job_id, t.run_name) == ("12345", "recipe_12345")
        assert len(t.scrapes) == 3
        assert sorted(t.writes) == [0, 1]
        assert t.writes[1].scheduled_at == T0 + 0.1
        assert t.dropped_records == 0
        assert (t.bad_lines, t.unknown_events) == (0, 0)

    def test_hosts_and_seqs_are_sorted_and_unioned_with_cycle_writes(self, tmp_path):
        # seq 2 has a cycle_write but no scrape record: it must still be a column.
        t = load_timings(_jsonl(tmp_path, [*_two_cycles(), _write(2, scheduled=T0 + 0.2)]))
        assert t.hosts == ["node-a", "node-b"]
        assert t.seqs == [0, 1, 2]

    def test_missing_host_seq_pair_is_absent_from_by_slot(self, tmp_path):
        t = load_timings(_jsonl(tmp_path, _two_cycles()))
        slots = t.by_slot()
        assert ("node-a", 1) in slots
        assert ("node-b", 1) not in slots  # abandoned at the cycle deadline

    def test_missing_summary_is_none_not_zero(self, tmp_path):
        t = load_timings(_jsonl(tmp_path, _two_cycles()[:-1]))
        assert t.dropped_records is None

    def test_bad_and_unknown_lines_are_counted_not_fatal(self, tmp_path):
        path = _jsonl(tmp_path, [*_two_cycles(), {"event": "future_kind"}], extra_lines=["{not json", ""])
        t = load_timings(path)
        assert (t.bad_lines, t.unknown_events) == (1, 1)
        assert len(t.scrapes) == 3
        page = render_html(t, path)
        assert "unparseable lines" in page and "unknown event kinds" in page

    @pytest.mark.parametrize(
        ("status", "error", "outcome"),
        [
            (200, None, "ok"),
            (503, "HTTPError", "http_error"),
            (None, "ReadTimeout", "timeout"),
            (None, "ConnectTimeout", "timeout"),
            (None, "ConnectionError", "other_error"),
        ],
    )
    def test_outcome_follows_the_collector_exception_classes(self, tmp_path, status, error, outcome):
        t = load_timings(_jsonl(tmp_path, [_scrape("node-a", 0, status=status, error=error, rows=0, parse=None)]))
        assert t.scrapes[0].outcome == outcome

    def test_null_lag_marks_a_bracket_scrape(self, tmp_path):
        t = load_timings(_jsonl(tmp_path, [_scrape("node-a", 0, lag=None), _scrape("node-a", 1)]))
        assert [s.is_bracket for s in t.scrapes] == [True, False]


class TestInterval:
    def test_median_gap_between_scheduled_slots(self, tmp_path):
        records = [_scrape("node-a", q, start=0.1 * q) for q in range(4)]
        records += [_write(q, scheduled=T0 + 0.1 * q) for q in range(4)]
        interval = load_timings(_jsonl(tmp_path, records)).inferred_interval()
        assert interval is not None
        assert interval[0] == pytest.approx(0.1)
        assert interval[1] == "from scheduled slots"

    def test_falls_back_to_request_starts_without_scheduled_slots(self, tmp_path):
        records = [_scrape("node-a", q, start=0.25 * q) for q in range(4)]
        interval = load_timings(_jsonl(tmp_path, records)).inferred_interval()
        assert interval is not None
        assert interval[0] == pytest.approx(0.25)
        assert interval[1] == "inferred from request starts"

    def test_bracket_cycles_do_not_feed_the_fallback(self, tmp_path):
        records = [
            _scrape("node-a", 0, start=0.0),
            _scrape("node-a", 1, start=0.25),
            _scrape("node-a", 2, start=9.0, lag=None),
        ]
        interval = load_timings(_jsonl(tmp_path, records)).inferred_interval()
        assert interval is not None
        assert interval[0] == pytest.approx(0.25)

    def test_single_cycle_has_no_interval(self, tmp_path):
        assert load_timings(_jsonl(tmp_path, [_scrape("node-a", 0)])).inferred_interval() is None


class TestPayload:
    def test_times_are_relative_to_the_earliest_instant(self, tmp_path):
        # The scheduled slot precedes every request start, so it is t0.
        payload = build_payload(load_timings(_jsonl(tmp_path, _two_cycles())))
        assert payload["t0"] == T0
        assert payload["writes"]["0"]["sa"] == 0.0
        assert payload["writes"]["1"]["sa"] == pytest.approx(0.1)
        first = next(s for s in payload["scrapes"] if s["h"] == payload["hosts"].index("node-b"))
        assert first["s"] == pytest.approx(0.001)
        assert first["f"] == pytest.approx(0.021)

    def test_scrape_fields_and_host_indices(self, tmp_path):
        records = [_scrape("node-a", 0, status=None, error="ReadTimeout", rows=0, parse=None), _write(0, scheduled=T0)]
        payload = build_payload(load_timings(_jsonl(tmp_path, records)))
        (s,) = payload["scrapes"]
        assert payload["hosts"][s["h"]] == "node-a"
        assert (s["o"], s["st"], s["err"], s["n"], s["p"], s["ts"]) == ("timeout", None, "ReadTimeout", 0, None, None)

    def test_null_scheduled_slot_stays_null(self, tmp_path):
        payload = build_payload(load_timings(_jsonl(tmp_path, [_scrape("node-a", 0, lag=None), _write(0)])))
        assert payload["writes"]["0"]["sa"] is None

    def test_failed_write_is_carried_with_its_error(self, tmp_path):
        records = [_scrape("node-a", 0), _write(0, scheduled=T0, completed=False, error="OSError")]
        w = build_payload(load_timings(_jsonl(tmp_path, records)))["writes"]["0"]
        assert (w["ok"], w["err"]) == (False, "OSError")

    def test_cost_segments_come_from_the_single_table(self, tmp_path):
        payload = build_payload(load_timings(_jsonl(tmp_path, _two_cycles())))
        assert [seg["key"] for seg in payload["cost_segments"]] == [k for k, _, _ in COST_SEGMENTS]


class TestRenderHtml:
    def test_page_is_self_contained_and_embeds_the_payload(self, tmp_path):
        path = _jsonl(tmp_path, _two_cycles())
        page = render_html(load_timings(path), path)
        assert page.startswith("<!DOCTYPE html>")
        assert (
            "<script src=" not in page
            and "<link " not in page
            and "http://" not in page.replace("http://www.w3.org/2000/svg", "")
        )
        for chart in ("timeline", "lag", "cost", "write", "coverage"):
            assert f'data-chart="{chart}"' in page
        start = page.index('<script id="scrape-data" type="application/json">') + len(
            '<script id="scrape-data" type="application/json">'
        )
        embedded = json.loads(page[start : page.index("</script>", start)].replace("<\\/", "</"))
        assert embedded["hosts"] == ["node-a", "node-b"]

    def test_payload_cannot_close_its_script_tag(self, tmp_path):
        path = _jsonl(tmp_path, [_scrape("node-</script><b>x", 0)])
        page = render_html(load_timings(path), path)
        assert "node-</script>" not in page

    def test_stat_cards_flag_abandoned_slots_and_missing_summary(self, tmp_path):
        path = _jsonl(tmp_path, _two_cycles()[:-1])
        page = render_html(load_timings(path), path)
        assert "abandoned slots" in page
        assert "diagnostic_summary — file may be cut off" in page

    def test_no_scrape_records_is_an_error(self, tmp_path):
        path = _jsonl(tmp_path, [_write(0, scheduled=T0)])
        with pytest.raises(SystemExit, match="no scrape records"):
            render_html(load_timings(path), path)

    def test_main_writes_the_output_file(self, tmp_path):
        out = tmp_path / "out.html"
        assert main([str(_jsonl(tmp_path, _two_cycles())), "-o", str(out)]) == 0
        assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_reads_what_the_power_collector_writes(tmp_path, exporters):  # noqa: F811
    """Contract: a real collector run produces a file the visualizer consumes cleanly."""
    a, b = exporters(_body("a")), exporters(_body("b"))
    session = _session(tmp_path, _endpoints(("node-a", a.url), ("node-b", b.url)), windows=[])
    session.initialize()
    assert session.start_and_wait_for_readiness()
    time.sleep(0.2)
    session.stop_and_finalize()

    t = load_timings(session.power_dir / "scrape-timings.jsonl")
    assert (t.bad_lines, t.unknown_events, t.dropped_records) == (0, 0, 0)
    assert t.hosts == ["node-a", "node-b"]
    assert (t.job_id, t.run_name) == ("12345", "recipe_12345")
    assert any(not s.is_bracket for s in t.scrapes)
    assert all(s.outcome == "ok" and s.row_count == 4 for s in t.scrapes)
    assert {s.seq for s in t.scrapes} <= set(t.writes)
    assert any(w.scheduled_at is not None for w in t.writes.values())
    interval = t.inferred_interval()
    assert interval is not None
    assert interval[1] == "from scheduled slots"
    assert interval[0] == pytest.approx(0.05, rel=1e-6)
    payload = build_payload(t)
    assert all(s["s"] >= 0 for s in payload["scrapes"])
