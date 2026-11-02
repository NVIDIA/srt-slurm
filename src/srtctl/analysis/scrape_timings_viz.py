# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone HTML visualizer for the power collector's ``scrape-timings.jsonl`` sidecar.

Reads the diagnostic file written next to ``power/samples.csv`` (see
``docs/power-telemetry.md`` § "Diagnosing slow scrapes") and emits one
self-contained HTML page. Python parses the records into a JSON payload; the
inline script draws five linked SVG charts from it:

1. per-host request timeline (wall clock) with each cycle's scheduled slot,
2. schedule lag per host against ``scrape_seq``,
3. per-cycle cost decomposition (slowest request, slowest parse, writer lock
   wait, sample write) against the sample interval,
4. ``cycle_write`` health strip,
5. host × seq coverage heatmap of ``row_count``.

Dragging on any chart zooms every chart to that ``scrape_seq`` range (the
timeline maps the range to its wall-clock span); while zoomed, a scrollbar under
each chart (or shift+wheel / a sideways trackpad swipe) pans the window, and
double-click resets. Every mark
has a hover tooltip. A ``(hostname, scrape_seq)`` pair with no ``scrape`` record
is an endpoint abandoned at the cycle deadline and is drawn hatched, not dropped.
Stdlib only; dark theme by default with a persisted light/dark toggle.

Usage::

    python3 src/srtctl/analysis/scrape_timings_viz.py power/scrape-timings.jsonl -o out.html
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Cost-stack segments: (payload key, legend label, colour). One table drives the
# Python legend and the JS drawing so they cannot drift.
COST_SEGMENTS = (
    ("request", "slowest request", "hsl(212 62% 55%)"),
    ("parse", "slowest parse", "hsl(158 50% 48%)"),
    ("lock", "writer lock wait", "hsl(45 80% 55%)"),
    ("write", "sample write", "hsl(18 70% 55%)"),
)


@dataclass
class Scrape:
    hostname: str
    seq: int
    started: float
    finished: float
    duration: float
    parse: float | None
    lag: float | None
    sample_ts: float | None
    http_status: int | None
    error_type: str | None
    row_count: int
    reason_codes: list[str]

    @property
    def is_bracket(self) -> bool:
        return self.lag is None

    @property
    def outcome(self) -> str:
        if self.error_type and "Timeout" in self.error_type:
            return "timeout"
        if self.error_type:
            return "http_error" if self.http_status is not None else "other_error"
        if self.http_status is not None and self.http_status != 200:
            return "http_error"
        return "ok"


@dataclass
class CycleWrite:
    seq: int
    scheduled_at: float | None  # unix; null for bracket/manual cycles
    row_count: int
    lock_wait: float
    write_seconds: float
    completed: bool
    error: str | None


@dataclass
class Timings:
    job_id: str | None = None
    run_name: str | None = None
    scrapes: list[Scrape] = field(default_factory=list)
    writes: dict[int, CycleWrite] = field(default_factory=dict)
    dropped_records: int | None = None
    unknown_events: int = 0
    bad_lines: int = 0

    @property
    def hosts(self) -> list[str]:
        return sorted({s.hostname for s in self.scrapes})

    @property
    def seqs(self) -> list[int]:
        return sorted({s.seq for s in self.scrapes} | set(self.writes))

    def by_slot(self) -> dict[tuple[str, int], Scrape]:
        return {(s.hostname, s.seq): s for s in self.scrapes}

    def by_seq(self) -> dict[int, list[Scrape]]:
        out: dict[int, list[Scrape]] = defaultdict(list)
        for s in self.scrapes:
            out[s.seq].append(s)
        return out

    def inferred_interval(self) -> tuple[float, str] | None:
        """Sample interval and how it was obtained.

        Prefer the median gap between consecutive ``cycle_write.scheduled_at_unix``
        slots (exact: the collector's own cadence). Fall back to the median gap
        between cycles' earliest request start for files written before that
        field existed.
        """
        scheduled = sorted(w.scheduled_at for w in self.writes.values() if w.scheduled_at is not None)
        if len(scheduled) >= 2:
            gaps = [scheduled[i + 1] - scheduled[i] for i in range(len(scheduled) - 1)]
            return statistics.median(gaps), "from scheduled slots"
        starts = sorted(
            min(s.started for s in group) for group in self.by_seq().values() if any(not s.is_bracket for s in group)
        )
        gaps = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
        return (statistics.median(gaps), "inferred from request starts") if gaps else None


def load_timings(path: Path) -> Timings:
    t = Timings()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                t.bad_lines += 1
                continue
            event = rec.get("event")
            if event == "scrape":
                t.job_id = t.job_id or rec.get("job_id")
                t.run_name = t.run_name or rec.get("run_name")
                t.scrapes.append(
                    Scrape(
                        hostname=str(rec["hostname"]),
                        seq=int(rec["scrape_seq"]),
                        started=float(rec["request_started_at_unix"]),
                        finished=float(rec["request_finished_at_unix"]),
                        duration=float(rec["request_duration_seconds"]),
                        parse=rec.get("parse_seconds"),
                        lag=rec.get("schedule_lag_seconds"),
                        sample_ts=rec.get("sample_timestamp_unix"),
                        http_status=rec.get("http_status"),
                        error_type=rec.get("error_type"),
                        row_count=int(rec.get("row_count", 0)),
                        reason_codes=list(rec.get("reason_codes") or []),
                    )
                )
            elif event == "cycle_write":
                t.job_id = t.job_id or rec.get("job_id")
                t.run_name = t.run_name or rec.get("run_name")
                seq = int(rec["scrape_seq"])
                t.writes[seq] = CycleWrite(
                    seq=seq,
                    scheduled_at=rec.get("scheduled_at_unix"),
                    row_count=int(rec.get("row_count", 0)),
                    lock_wait=float(rec.get("writer_lock_wait_seconds") or 0.0),
                    write_seconds=float(rec.get("sample_write_seconds") or 0.0),
                    completed=bool(rec.get("sample_write_completed")),
                    error=rec.get("sample_write_error"),
                )
            elif event == "diagnostic_summary":
                t.dropped_records = int(rec.get("dropped_records", 0))
            else:
                t.unknown_events += 1
    return t


# --------------------------------------------------------------------------- payload


def build_payload(t: Timings) -> dict[str, Any]:
    """Everything the page script needs, with times relative to ``t0`` (seconds)."""
    hosts = t.hosts
    host_index = {h: i for i, h in enumerate(hosts)}
    t0 = min([s.started for s in t.scrapes] + [w.scheduled_at for w in t.writes.values() if w.scheduled_at is not None])
    interval = t.inferred_interval()
    return {
        "job_id": t.job_id,
        "run_name": t.run_name,
        "t0": t0,
        "hosts": hosts,
        "seqs": t.seqs,
        "interval": {"seconds": interval[0], "method": interval[1]} if interval else None,
        "cost_segments": [{"key": k, "label": label, "color": c} for k, label, c in COST_SEGMENTS],
        "scrapes": [
            {
                "h": host_index[s.hostname],
                "q": s.seq,
                "s": round(s.started - t0, 6),
                "f": round(s.finished - t0, 6),
                "d": s.duration,
                "p": s.parse,
                "lag": s.lag,
                "ts": None if s.sample_ts is None else round(s.sample_ts - t0, 6),
                "st": s.http_status,
                "err": s.error_type,
                "n": s.row_count,
                "rc": s.reason_codes,
                "o": s.outcome,
            }
            for s in t.scrapes
        ],
        "writes": {
            str(w.seq): {
                "sa": None if w.scheduled_at is None else round(w.scheduled_at - t0, 6),
                "n": w.row_count,
                "lw": w.lock_wait,
                "ws": w.write_seconds,
                "ok": w.completed,
                "err": w.error,
            }
            for w in t.writes.values()
        },
    }


# --------------------------------------------------------------------------- page

# Colour tokens for both themes. The page defaults to dark and exposes an
# explicit toggle stored in localStorage; the OS preference is not consulted.
_CSS = """
:root[data-theme=dark] { color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink-primary: #ffffff; --ink-secondary: #c3c2b7;
  --ink-muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --slot-0: #3987e5; --slot-1: #d95926; --slot-2: #199e70;
  --ok: #199e70; --err: #e5484d; --timeout: #d95926; --other: #b07cd8; --warn: #f0a04b;
  --lag-lo: #9fd8ff; --lag-mid: #ffe14d; --lag-hi: #ff4fd8; }
:root[data-theme=light] { color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink-primary: #0b0b0b; --ink-secondary: #52514e;
  --ink-muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --slot-0: #2a78d6; --slot-1: #eb6834; --slot-2: #1baf7a;
  --ok: #1baf7a; --err: #d33b3b; --timeout: #eb6834; --other: #8e4fc2; --warn: #b8741a;
  --lag-lo: #3b6c99; --lag-mid: #9a7c00; --lag-hi: #c2189b; }
body { margin: 0; padding: 24px; background: var(--page); color: var(--ink-primary);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
.page-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.page-head .controls { display: flex; gap: 8px; flex: none; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0; }
.subtitle { color: var(--ink-secondary); margin: 0 0 20px; }
.btn { appearance: none; font: inherit; font-size: 12px; font-weight: 600; color: var(--ink-secondary);
  background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; cursor: pointer;
  white-space: nowrap; }
.btn:hover { color: var(--ink-primary); border-color: var(--ink-muted); }
.btn[hidden] { display: none; }
.stat-cards { display: flex; gap: 12px; margin: 4px 0 24px; flex-wrap: wrap; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; min-width: 96px; }
.stat-card-num { font-size: 22px; font-weight: 700; margin: 0; font-variant-numeric: tabular-nums; }
.stat-card-label { color: var(--ink-secondary); font-size: 12px; margin: 2px 0 0; }
.stat-card.warn .stat-card-num { color: var(--warn); }
.chart-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px 8px; margin-bottom: 16px; }
.chart-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin: 0 0 4px; }
.zoom-hint { color: var(--ink-muted); font-size: 11px; white-space: nowrap; flex: none; }
.zoom-hint.zoomed { color: var(--slot-0); font-weight: 600; }
.chart-panel p.blurb { margin: 0 0 8px; color: var(--ink-secondary); font-size: 13px; }
svg.chart { display: block; width: 100%; user-select: none; touch-action: none; }
svg.chart.dragging { cursor: col-resize; }
svg text { fill: var(--ink-primary); font-family: inherit; }
svg text.tick { font-size: 11px; fill: var(--ink-muted); }
svg text.axis { font-size: 11px; font-weight: 600; letter-spacing: .02em; fill: var(--ink-secondary); }
svg text.lane { font-size: 12px; fill: var(--ink-secondary); }
svg text.warn { fill: var(--err); font-weight: 700; }
svg text.ref-label { fill: var(--ink-secondary); paint-order: stroke; stroke: var(--surface); stroke-width: 4px;
  stroke-linejoin: round; }
svg .ax { stroke: var(--axis); }
svg .grid { stroke: var(--grid); }
svg .band { fill: var(--ink-primary); fill-opacity: .035; }
svg .ref { stroke: var(--ink-secondary); }
svg .sched { stroke: var(--ink-primary); stroke-opacity: .55; }
svg .hatch-line { stroke: var(--ink-muted); }
svg .missing { stroke: var(--ink-muted); }
svg .o-ok { fill: var(--ok); } svg .o-http { fill: var(--err); } svg .o-timeout { fill: var(--timeout); } svg .o-other { fill: var(--other); }
svg .bracket { stroke: var(--slot-0); stroke-width: 2; stroke-dasharray: 3 2; }
svg .lag-lo { stroke: var(--lag-lo); }
svg .lag-mid { stroke: var(--lag-mid); }
svg .lag-hi { stroke: var(--lag-hi); }
svg .lag-casing { stroke: var(--page); stroke-opacity: .9; }
svg .head { stroke: var(--page); stroke-width: 1.5px; paint-order: stroke; }
svg .head.lag-lo { fill: var(--lag-lo); } svg .head.lag-mid { fill: var(--lag-mid); } svg .head.lag-hi { fill: var(--lag-hi); }
svg .cell { stroke: var(--grid); }
svg .hit { fill: transparent; }
svg .hit:hover { fill: var(--ink-primary); fill-opacity: .06; }
svg .tipped { cursor: help; }
svg .tipped:hover { filter: brightness(1.25); }
svg .zoom-band { fill: var(--slot-0); fill-opacity: .18; stroke: var(--slot-0); stroke-width: 1; pointer-events: none; }
/* Pan scrollbar: aligned to the plot area (PAD.l = 130, PAD.r = 24 in the script). Styled
   explicitly so macOS overlay scrollbars stay visible instead of auto-hiding. */
.pan { margin: 2px 24px 6px 130px; overflow-x: scroll; overflow-y: hidden; height: 12px; }
.pan[hidden] { display: none; }
.pan-inner { height: 1px; }
.pan::-webkit-scrollbar { height: 10px; }
.pan::-webkit-scrollbar-track { background: var(--grid); border-radius: 5px; }
.pan::-webkit-scrollbar-thumb { background: var(--slot-0); border-radius: 5px; border: 2px solid var(--grid); }
.pan::-webkit-scrollbar-thumb:hover { background: var(--ink-secondary); }
@supports not selector(::-webkit-scrollbar) { .pan { scrollbar-color: var(--slot-0) var(--grid); } }
.legend { display: flex; gap: 6px 14px; flex-wrap: wrap; margin: 4px 0 8px; font-size: 12px; color: var(--ink-secondary); }
.legend .key { display: inline-flex; align-items: center; gap: 6px; padding: 3px 8px; border: 1px solid var(--border); border-radius: 4px; }
.legend .key i { display: inline-block; width: 12px; height: 12px; border-radius: 2px; border: 1px solid var(--border); }
.legend button.key { font: inherit; color: inherit; background: none; cursor: pointer; }
.legend button.key:hover { border-color: var(--ink-muted); color: var(--ink-primary); }
.legend button.key.off { opacity: .4; text-decoration: line-through; }
.legend button.key.off i { background: transparent !important; }
.legend .filter-hint { align-self: center; color: var(--ink-muted); font-size: 11px; }
.legend .filter-hint button { font: inherit; color: var(--slot-0); background: none; border: 0; padding: 0; cursor: pointer; }
.legend .filter-label { align-self: center; color: var(--ink-muted); font-size: 11px; font-weight: 600; min-width: 64px; }
.legend-stack { margin: 4px 0 8px; } .legend-stack .legend { margin: 0 0 4px; }
.legend .key i.ok { background: var(--ok); } .legend .key i.err { background: var(--err); }
.legend .key i.timeout { background: var(--timeout); } .legend .key i.other { background: var(--other); }
.legend .key i.hatch { background: repeating-linear-gradient(45deg, var(--ink-muted) 0 1.5px, transparent 1.5px 6px); }
.legend .key i.bracket { background: var(--ok); outline: 2px dashed var(--slot-0); outline-offset: -2px; }
.legend .key i.sched { width: 0; border: 0; border-left: 1.5px dotted var(--ink-primary); border-radius: 0; height: 14px; }
.legend .key i.lag-lo, .legend .key i.lag-mid, .legend .key i.lag-hi { width: 18px; height: 2px; border: 0; border-radius: 1px;
  box-shadow: 0 0 0 1.5px var(--page); }
.legend .key i.lag-lo { background: var(--lag-lo); }
.legend .key i.lag-mid { background: var(--lag-mid); }
.legend .key i.lag-hi { background: var(--lag-hi); }
.note, .empty { margin: 6px 0 0; color: var(--ink-muted); font-size: 12px; }
.note.warn { color: var(--warn); }
footer { color: var(--ink-muted); font-size: 12px; margin-top: 24px; }
.tooltip { position: fixed; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; padding: 7px 10px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.25); opacity: 0;
  z-index: 10; max-width: 440px; transition: opacity .08s; }
.tooltip.on { opacity: 1; }
.tooltip .t-title { font-weight: 700; margin-bottom: 4px; color: var(--ink-primary); }
.tooltip .t-row { display: flex; justify-content: space-between; gap: 14px; line-height: 1.45; }
.tooltip .t-key { color: var(--ink-muted); white-space: nowrap; }
.tooltip .t-val { font-weight: 600; font-variant-numeric: tabular-nums; text-align: right; }
.tooltip .t-note { margin-top: 5px; padding-top: 5px; border-top: 1px solid var(--grid); color: var(--ink-secondary);
  font-size: 11.5px; white-space: normal; }
"""

# Applied before first paint so a stored light preference never flashes dark.
_THEME_JS = """
(function () {
  var KEY = 'scrape-timings-theme';
  var root = document.documentElement;
  try { var saved = localStorage.getItem(KEY); if (saved === 'light' || saved === 'dark') root.dataset.theme = saved; } catch (e) {}
  function label(btn) { btn.textContent = root.dataset.theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'; }
  document.addEventListener('DOMContentLoaded', function () {
    var btn = document.querySelector('.theme-toggle');
    if (!btn) return;
    label(btn);
    btn.addEventListener('click', function () {
      root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem(KEY, root.dataset.theme); } catch (e) {}
      label(btn);
    });
  });
})();
"""

# Chart drawing. All five charts share one zoom state: a contiguous index range
# into DATA.seqs. Dragging on any chart maps the pixel span back to seq indices
# (the timeline goes via wall-clock time) and redraws everything.
_CHARTS_JS = """
(function () {
  var DATA = JSON.parse(document.getElementById('scrape-data').textContent);
  var NS = 'http://www.w3.org/2000/svg';
  var PAD = { l: 130, r: 24, t: 12, b: 34 };
  var HOST_HUES = [212, 18, 158, 280, 45, 340, 95, 190];
  var OUTCOME_CLASS = { ok: 'o-ok', http_error: 'o-http', timeout: 'o-timeout', other_error: 'o-other' };
  var OUTCOME_LABEL = { ok: 'OK - HTTP 200, body parsed', http_error: 'HTTP error', timeout: 'Request timeout', other_error: 'Request exception' };

  function hostColor(i) { return 'hsl(' + HOST_HUES[i % HOST_HUES.length] + ' 62% 52%)'; }
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(parent, x, y, str, cls, anchor, extra) {
    var t = el('text', { x: x.toFixed(1), y: y.toFixed(1), 'class': cls, 'text-anchor': anchor || 'start' }, parent);
    if (extra) for (var k in extra) t.setAttribute(k, extra[k]);
    t.textContent = str;
    return t;
  }
  function fmtS(s) { return s >= 1 ? s.toFixed(2) + ' s' : (s * 1000).toFixed(1) + ' ms'; }
  function fmtMs(s) { return s === null || s === undefined ? '-' : fmtS(s); }
  function fmtClock(rel) { return new Date((DATA.t0 + rel) * 1000).toISOString().slice(11, 23) + 'Z'; }
  function fmtTick(v) { if (v === 0) return '0'; if (Math.abs(v) >= 100) return v.toFixed(0); return Number(v.toPrecision(3)).toString(); }
  function niceTicks(lo, hi, n) {
    if (hi <= lo) return [lo];
    var raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var step = [1, 2, 2.5, 5, 10].map(function (m) { return m * mag; }).filter(function (s) { return s >= raw; })[0];
    var out = [], v = Math.ceil(lo / step) * step;
    while (v <= hi + 1e-12) { out.push(Number(v.toFixed(10))); v += step; }
    return out;
  }

  // ---- index the payload ---------------------------------------------------
  var seqs = DATA.seqs, hosts = DATA.hosts;
  var bySeq = new Map(), slot = new Map();
  DATA.scrapes.forEach(function (s) {
    if (!bySeq.has(s.q)) bySeq.set(s.q, []);
    bySeq.get(s.q).push(s);
    slot.set(s.h + '|' + s.q, s);
  });
  var writes = DATA.writes;
  function writeOf(q) { return writes[String(q)] || null; }
  var interval = DATA.interval ? DATA.interval.seconds : null;

  // Cycle span per seq (relative seconds): settled scrapes; else the scheduled
  // slot to the next scheduled slot; else interpolated between neighbours.
  var span = new Map();
  seqs.forEach(function (q) {
    var g = bySeq.get(q);
    if (g && g.length) span.set(q, [Math.min.apply(null, g.map(function (s) { return s.s; })), Math.max.apply(null, g.map(function (s) { return s.f; }))]);
  });
  var tEnd = Math.max.apply(null, DATA.scrapes.map(function (s) { return s.f; }));
  seqs.forEach(function (q, i) {
    if (span.has(q)) return;
    var w = writeOf(q);
    if (w && w.sa !== null) {
      var nxt = null;
      for (var j = i + 1; j < seqs.length && nxt === null; j++) { var wn = writeOf(seqs[j]); if (wn && wn.sa !== null) nxt = wn.sa; }
      var b = nxt !== null ? nxt : Math.min(tEnd, w.sa + 0.02);
      span.set(q, [w.sa, Math.max(b, w.sa + 0.005)]);
      return;
    }
    var prev = null, next = null;
    for (var a = i - 1; a >= 0 && !prev; a--) prev = span.get(seqs[a]) || null;
    for (var c = i + 1; c < seqs.length && !next; c++) next = span.get(seqs[c]) || null;
    if (prev && next) span.set(q, [prev[1], next[0]]);
    else if (prev) span.set(q, [prev[1], Math.min(tEnd, prev[1] + (prev[1] - prev[0]) + 0.01)]);
    else if (next) span.set(q, [Math.max(0, next[0] - (next[1] - next[0]) - 0.01), next[0]]);
  });

  // ---- tooltip -------------------------------------------------------------
  var tip = document.createElement('div'); tip.className = 'tooltip'; document.body.appendChild(tip);
  function node(tag, cls, str, parent) { var e = document.createElement(tag); e.className = cls; e.textContent = str; parent.appendChild(e); return e; }
  function setTip(target, title, rows, note) { target.__tip = { title: title, rows: rows, note: note || null }; target.classList.add('tipped'); }
  function showTip(d, ev) {
    tip.textContent = '';
    node('div', 't-title', d.title, tip);
    d.rows.forEach(function (r) { var row = node('div', 't-row', '', tip); node('span', 't-key', r[0], row); node('span', 't-val', String(r[1]), row); });
    if (d.note) node('div', 't-note', d.note, tip);
    tip.classList.add('on'); moveTip(ev);
  }
  function moveTip(ev) {
    var pad = 14, w = tip.offsetWidth, h = tip.offsetHeight, x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = ev.clientY - h - pad;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function hideTip() { tip.classList.remove('on'); }
  function tipTarget(n) { while (n && n.nodeType === 1 && !n.__tip && n.tagName !== 'svg') n = n.parentNode; return n && n.__tip ? n : null; }
  var dragging = false;
  document.addEventListener('pointerover', function (ev) { if (dragging) return; var t = tipTarget(ev.target); if (t) showTip(t.__tip, ev); });
  document.addEventListener('pointermove', function (ev) { if (tip.classList.contains('on')) moveTip(ev); });
  document.addEventListener('pointerout', function (ev) { var t = tipTarget(ev.target); if (t && !(ev.relatedTarget && t.contains(ev.relatedTarget))) hideTip(); });

  function scrapeTip(s) {
    var w = writeOf(s.q);
    var rows = [['host', hosts[s.h]], ['scrape_seq', s.q], ['started', fmtClock(s.s) + '  (+' + s.s.toFixed(3) + ' s)'], ['finished', fmtClock(s.f)],
      ['request duration', fmtS(s.d)], ['parse', fmtMs(s.p)], ['HTTP status', s.st === null ? '-' : s.st]];
    if (s.err) rows.push(['exception', s.err]);
    if (s.lag === null) rows.push(['schedule lag', 'n/a - bracket/manual scrape']);
    else { rows.push(['schedule lag', fmtS(s.lag)]); if (w && w.sa !== null) rows.push(['scheduled slot', fmtClock(w.sa)]); }
    rows.push(['GPU rows', s.n]);
    if (s.ts !== null) rows.push(['sample timestamp', fmtClock(s.ts)]);
    if (s.rc.length) rows.push(['reason codes', s.rc.join(', ')]);
    var note = null;
    if (s.o === 'timeout') note = 'requests raised a Timeout; the cycle deadline had not yet expired so a record exists.';
    else if (s.o === 'http_error') note = 'Non-2xx response; no rows were parsed and no power sample was invented.';
    else if (s.n === 0) note = 'Settled with 0 rows - body parsed but yielded no GPU readings.';
    return { title: OUTCOME_LABEL[s.o], rows: rows, note: note };
  }
  function abandonedTip(h, q, extraRows, note) {
    var rows = [['host', hosts[h]], ['scrape_seq', q]].concat(extraRows || []);
    return { title: 'Abandoned - no scrape record', rows: rows, note: note || 'The request had not settled when the cycle deadline expired; the collector counted an endpoint_timeout and wrote no timing record for this (host, seq).' };
  }

  // ---- shared zoom state ---------------------------------------------------
  var view = { lo: 0, hi: seqs.length - 1 };
  var charts = [];
  function visible() { return seqs.slice(view.lo, view.hi + 1); }
  function setView(lo, hi) {
    lo = Math.max(0, Math.min(lo, seqs.length - 1)); hi = Math.max(lo, Math.min(hi, seqs.length - 1));
    if (lo === view.lo && hi === view.hi) return;
    view.lo = lo; view.hi = hi;
    charts.forEach(function (c) { c.draw(); });
    updateHints();
  }
  function resetView() { setView(0, seqs.length - 1); }
  // ---- horizontal pan when zoomed ------------------------------------------
  // Each panel carries a native scrollbar (div.pan) under its plot area. The
  // scroll content is (all seqs / visible seqs) x the track width, so the thumb
  // is the zoom window; scrolling any of them slides the shared view without
  // changing its width. Shift+wheel or a horizontal trackpad swipe on a chart
  // drives the same scrollbar.
  var panSrc = null;
  function syncPans() {
    var n = view.hi - view.lo + 1, full = n >= seqs.length;
    document.querySelectorAll('.pan').forEach(function (p) {
      p.hidden = full;
      if (full) return;
      var inner = p.firstElementChild, track = p.clientWidth;
      inner.style.width = (track * seqs.length / n) + 'px';
      if (p !== panSrc) p.scrollLeft = view.lo / seqs.length * inner.offsetWidth;
    });
  }
  function onPanScroll(ev) {
    var p = ev.currentTarget, n = view.hi - view.lo + 1;
    if (p.hidden || n >= seqs.length) return;
    var maxLo = seqs.length - n, w = p.firstElementChild.offsetWidth;
    var lo = Math.max(0, Math.min(maxLo, Math.round(p.scrollLeft / w * seqs.length)));
    if (lo === view.lo) return;
    panSrc = p; setView(lo, lo + n - 1); panSrc = null;
  }
  function wirePans() {
    document.querySelectorAll('.pan').forEach(function (p) { p.addEventListener('scroll', onPanScroll, { passive: true }); });
    document.querySelectorAll('section.chart-panel').forEach(function (sec) {
      var svg = sec.querySelector('svg.chart'), p = sec.querySelector('.pan');
      if (!svg || !p) return;
      svg.addEventListener('wheel', function (ev) {
        if (p.hidden) return;
        var dx = ev.shiftKey && !ev.deltaX ? ev.deltaY : ev.deltaX;
        if (!dx || Math.abs(dx) < Math.abs(ev.shiftKey ? 0 : ev.deltaY)) return;
        ev.preventDefault();
        p.scrollLeft += dx;
        onPanScroll({ currentTarget: p });
      }, { passive: false });
    });
    window.addEventListener('resize', syncPans);
  }
  function updateHints() {
    syncPans();
    var full = view.lo === 0 && view.hi === seqs.length - 1;
    document.querySelectorAll('.zoom-hint').forEach(function (h) {
      h.textContent = full ? 'drag to zoom - linked across charts' : 'seq ' + seqs[view.lo] + '-' + seqs[view.hi] + ' of ' + seqs[0] + '-' + seqs[seqs.length - 1] + ' - scroll sideways to pan, double-click to reset';
      h.classList.toggle('zoomed', !full);
    });
    var btn = document.querySelector('.zoom-reset'); if (btn) btn.hidden = full;
  }

  // ---- series filters (lag + cost charts) ----------------------------------
  // One host filter shared by the lag and cost charts, one delay-type filter for
  // the cost stack. Legend chips are buttons: click toggles, shift-click isolates
  // (shift-click the only visible one to show all again).
  var hostOn = hosts.map(function () { return true; });
  var segOn = {}; DATA.cost_segments.forEach(function (seg) { segOn[seg.key] = true; });
  var filterViews = [];
  function hostVisible(h) { return hostOn[h]; }
  function toggleIn(state, keys, key, solo) {
    if (solo) {
      var onlyThis = keys.every(function (k) { return k === key ? state[k] : !state[k]; });
      keys.forEach(function (k) { state[k] = onlyThis || k === key; });
    } else state[key] = !state[key];
  }
  function refilter() {
    filterViews.forEach(function (f) { f(); });
    charts.forEach(function (c) { if (c.id === 'lag' || c.id === 'cost') c.draw(); });
  }
  function buildFilterLegend(container, label, items, state, keyOf) {
    var keys = items.map(keyOf);
    if (label) { var lab = document.createElement('span'); lab.className = 'filter-label'; lab.textContent = label; container.appendChild(lab); }
    var btns = items.map(function (it, idx) {
      var b = document.createElement('button'); b.type = 'button'; b.className = 'key';
      var sw = document.createElement('i'); sw.style.background = it.color; b.appendChild(sw);
      b.appendChild(document.createTextNode(it.label));
      b.title = 'click to show/hide ' + it.label + ' · shift-click to show only ' + it.label;
      b.addEventListener('click', function (ev) { toggleIn(state, keys, keys[idx], ev.shiftKey); refilter(); });
      container.appendChild(b);
      return b;
    });
    var hint = document.createElement('span'); hint.className = 'filter-hint'; container.appendChild(hint);
    function sync() {
      var hidden = 0;
      btns.forEach(function (b, idx) { var on = state[keys[idx]]; b.classList.toggle('off', !on); b.setAttribute('aria-pressed', on ? 'true' : 'false'); if (!on) hidden++; });
      hint.textContent = '';
      if (!hidden) { hint.textContent = 'click to toggle · shift-click to isolate'; return; }
      hint.appendChild(document.createTextNode(hidden + ' of ' + keys.length + ' hidden · '));
      var all = document.createElement('button'); all.type = 'button'; all.textContent = 'show all';
      all.addEventListener('click', function () { keys.forEach(function (k) { state[k] = true; }); refilter(); });
      hint.appendChild(all);
    }
    filterViews.push(sync); sync();
  }
  function buildFilters() {
    var hostItems = hosts.map(function (host, h) { return { label: host, color: hostColor(h), h: h }; });
    document.querySelectorAll('.legend[data-legend="hosts"]').forEach(function (lg) {
      buildFilterLegend(lg, lg.dataset.label || null, hostItems, hostOn, function (it) { return it.h; });
    });
    document.querySelectorAll('.legend[data-legend="segments"]').forEach(function (lg) {
      buildFilterLegend(lg, lg.dataset.label || null, DATA.cost_segments, segOn, function (seg) { return seg.key; });
    });
  }
  function filterNote(c) {
    var nh = hostOn.filter(Boolean).length, ns = DATA.cost_segments.filter(function (s) { return segOn[s.key]; }).length;
    var msg = null;
    if (!nh) msg = 'all hosts hidden - pick one in the legend';
    else if (c.id === 'cost' && !ns) msg = 'all delay types hidden - pick one in the legend';
    if (msg) text(c.gTop, PAD.l + c.plotW / 2, PAD.t + 40, msg, 'tick', 'middle');
  }

  // ---- chart scaffolding ---------------------------------------------------
  var clipSeq = 0;
  function makeChart(id, height, drawFn, seqRangeAt) {
    var svg = document.querySelector('svg.chart[data-chart="' + id + '"]');
    if (!svg) return null;
    svg.style.height = height + 'px';
    var defs = el('defs', {}, svg);
    var pat = el('pattern', { id: 'hatch-' + id, width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
    el('line', { x1: 0, y1: 0, x2: 0, y2: 6, 'class': 'hatch-line', 'stroke-width': 1.5 }, pat);
    var clipId = 'clip-' + id + '-' + (clipSeq++);
    var clip = el('clipPath', { id: clipId }, defs);
    var clipRect = el('rect', { x: PAD.l, y: 0, width: 10, height: height }, clip);
    var gAxes = el('g', { 'class': 'axes' }, svg);
    var gPlot = el('g', { 'class': 'plot', 'clip-path': 'url(#' + clipId + ')' }, svg);
    var gTop = el('g', { 'class': 'annot', 'pointer-events': 'none' }, svg);
    var band = el('rect', { 'class': 'zoom-band', x: 0, y: PAD.t, width: 0, height: height - PAD.t - PAD.b, visibility: 'hidden' }, svg);
    var chart = { id: id, svg: svg, H: height, W: 0, plotW: 0, gAxes: gAxes, gPlot: gPlot, hatch: 'url(#hatch-' + id + ')', gTop: gTop, seqRangeAt: null };
    chart.layout = function () {
      var w = svg.getBoundingClientRect().width || 800;
      chart.W = w; chart.plotW = w - PAD.l - PAD.r;
      svg.setAttribute('viewBox', '0 0 ' + w + ' ' + height);
      clipRect.setAttribute('width', chart.plotW);
    };
    chart.draw = function () {
      if (!chart.W) chart.layout();
      gAxes.textContent = ''; gPlot.textContent = ''; gTop.textContent = '';
      drawFn(chart);
    };
    chart.seqRangeAt = function (xa, xb) { return seqRangeAt(chart, xa, xb); };
    // Band scale over the visible seqs: every seq chart shares it so zoom bands line up.
    chart.n = function () { return view.hi - view.lo + 1; };
    chart.bandW = function () { return chart.plotW / chart.n(); };
    chart.xOfIdx = function (i) { return PAD.l + (i - view.lo + 0.5) / chart.n() * chart.plotW; };
    chart.idxAt = function (x) { return Math.max(view.lo, Math.min(view.hi, view.lo + Math.floor((x - PAD.l) / chart.plotW * chart.n()))); };

    // Drag-to-zoom on the svg itself (marks keep their own pointer events for tooltips).
    var start = null;
    svg.addEventListener('pointerdown', function (ev) {
      if (ev.button !== 0) return;
      start = ev.clientX - svg.getBoundingClientRect().left;
      svg.setPointerCapture(ev.pointerId);
    });
    svg.addEventListener('pointermove', function (ev) {
      if (start === null) return;
      var x = ev.clientX - svg.getBoundingClientRect().left;
      if (!dragging && Math.abs(x - start) < 3) return;
      dragging = true; hideTip(); svg.classList.add('dragging');
      var a = Math.max(PAD.l, Math.min(start, x)), b = Math.min(PAD.l + chart.plotW, Math.max(start, x));
      band.setAttribute('x', a); band.setAttribute('width', Math.max(0, b - a)); band.setAttribute('visibility', 'visible');
    });
    function endDrag(ev) {
      if (start === null) return;
      var wasDragging = dragging;
      var a = band.x.baseVal.value, b = a + band.width.baseVal.value;
      start = null; dragging = false; svg.classList.remove('dragging'); band.setAttribute('visibility', 'hidden');
      if (!wasDragging) return;
      var r = chart.seqRangeAt(a, b);
      if (r) setView(r[0], r[1]);
    }
    svg.addEventListener('pointerup', endDrag);
    svg.addEventListener('pointercancel', endDrag);
    svg.addEventListener('dblclick', resetView);

    var pending = false;
    new ResizeObserver(function () { if (pending) return; pending = true; requestAnimationFrame(function () { pending = false; chart.layout(); chart.draw(); }); }).observe(svg);
    charts.push(chart);
    return chart;
  }

  function seqRangeBand(chart, xa, xb) { return [chart.idxAt(xa), chart.idxAt(xb - 0.001)]; }

  function xAxisSeq(chart, y) {
    var g = chart.gAxes;
    el('line', { x1: PAD.l, y1: y, x2: PAD.l + chart.plotW, y2: y, 'class': 'ax' }, g);
    var vis = visible(), stride = Math.ceil(vis.length / 30);
    vis.forEach(function (q, k) {
      if (k % stride) return;
      var x = chart.xOfIdx(view.lo + k);
      el('line', { x1: x, y1: y, x2: x, y2: y + 4, 'class': 'ax' }, g);
      text(g, x, y + 15, String(q), 'tick', 'middle');
    });
    text(g, PAD.l + chart.plotW / 2, y + 29, 'scrape_seq', 'axis', 'middle');
  }
  function yAxis(chart, yOf, lo, hi, top, bottom, label) {
    var g = chart.gAxes;
    el('line', { x1: PAD.l, y1: top, x2: PAD.l, y2: bottom, 'class': 'ax' }, g);
    niceTicks(lo, hi, 5).forEach(function (v) {
      var y = yOf(v);
      el('line', { x1: PAD.l, y1: y, x2: PAD.l + chart.plotW, y2: y, 'class': 'grid' }, g);
      text(g, PAD.l - 6, y + 4, fmtTick(v), 'tick', 'end');
    });
    var mid = (top + bottom) / 2;
    text(g, 14, mid, label, 'axis', 'middle', { transform: 'rotate(-90 14 ' + mid.toFixed(1) + ')' });
  }

  // ---- 1. timeline ---------------------------------------------------------
  // Lane height adapts to the host count: few hosts get tall lanes so the
  // timeline fills ~220 px of plot; many hosts fall back to compact 26 px lanes.
  var LANE = Math.max(26, Math.min(72, Math.round(220 / Math.max(hosts.length, 1))));
  function timelineWindow() {
    var a = Infinity, b = -Infinity;
    visible().forEach(function (q) {
      var sp = span.get(q); if (sp) { a = Math.min(a, sp[0]); b = Math.max(b, sp[1]); }
      var w = writeOf(q); if (w && w.sa !== null) a = Math.min(a, w.sa);
    });
    if (!isFinite(a)) { a = 0; b = tEnd; }
    var pad = Math.max((b - a) * 0.02, 1e-3);
    return [a - pad, b + pad];
  }
  function drawTimeline(c) {
    var win = timelineWindow(), tA = win[0], tB = win[1];
    var xOfT = function (t) { return PAD.l + (t - tA) / (tB - tA) * c.plotW; };
    var lanesBottom = PAD.t + LANE * hosts.length, arrows = [];
    c.timeAt = function (x) { return tA + (x - PAD.l) / c.plotW * (tB - tA); };
    hosts.forEach(function (host, hi) {
      var y = PAD.t + hi * LANE;
      if (hi % 2) el('rect', { x: PAD.l, y: y, width: c.plotW, height: LANE, 'class': 'band' }, c.gAxes);
      text(c.gAxes, PAD.l - 8, y + LANE / 2 + 4, host, 'lane', 'end');
      visible().forEach(function (q) {
        var s = slot.get(hi + '|' + q);
        if (!s) {
          var sp = span.get(q); if (!sp) return;
          var x0 = xOfT(sp[0]), w0 = Math.max(xOfT(sp[1]) - x0, 4);
          var r0 = el('rect', { x: x0, y: y + 4, width: w0, height: LANE - 8, fill: c.hatch, 'class': 'missing', 'stroke-dasharray': '2 2' }, c.gPlot);
          var wr = writeOf(q), extra = [];
          if (wr && wr.sa !== null) extra.push(['scheduled at', fmtClock(wr.sa)]);
          extra.push(['slot span', '+' + sp[0].toFixed(3) + ' s to +' + sp[1].toFixed(3) + ' s']);
          var at = abandonedTip(hi, q, extra); setTip(r0, at.title, at.rows, at.note);
          return;
        }
        var x = xOfT(s.s), w = Math.max(xOfT(s.f) - x, 2.5);
        // Lag connector: scheduled slot -> request start. Length is the drift off the
        // schedule; colour grades it against the sample interval (or 1 s if unknown).
        // Collected here and drawn after every bar so arrows are never hidden under one.
        var wr2 = writeOf(q);
        if (s.lag !== null && wr2 && wr2.sa !== null && s.lag > 0) {
          var ratio = s.lag / (interval || 1);
          arrows.push({ xs: xOfT(wr2.sa), xe: x, ym: y + LANE / 2, cls: ratio > 1 ? 'lag-hi' : ratio > 0.25 ? 'lag-mid' : 'lag-lo' });
        }
        var r = el('rect', { x: x, y: y + 5, width: w, height: LANE - 10, rx: 1.5, 'class': OUTCOME_CLASS[s.o] + (s.lag === null ? ' bracket' : '') }, c.gPlot);
        var d = scrapeTip(s); setTip(r, d.title, d.rows, d.note);
      });
    });
    // Each arrow is a surface-coloured casing under a coloured core, so it stays
    // legible where it crosses a bar of any outcome colour. During an overrun a
    // cycle's lag spans later slots, so arrows in a lane are packed into tracks
    // (greedy interval colouring): an arrow that starts before the previous one
    // ends drops to the next free track instead of drawing over it. Tracks are
    // centred on the lane and squeezed to fit its height.
    var gArrows = el('g', { 'pointer-events': 'none' }, c.gPlot);
    var byLane = {};
    arrows.forEach(function (a) { (byLane[a.ym] = byLane[a.ym] || []).push(a); });
    Object.keys(byLane).forEach(function (k) {
      var ls = byLane[k].sort(function (a, b) { return a.xs - b.xs; }), ends = [];
      ls.forEach(function (a) {
        var t = 0; while (t < ends.length && ends[t] > a.xs - 2) t++;
        ends[t] = a.xe; a.track = t;
      });
      var n = ends.length, gap = Math.min(6, (LANE - 14) / Math.max(n, 1));
      ls.forEach(function (a) { a.ym = Number(k) + (a.track - (n - 1) / 2) * gap; });
    });
    arrows.forEach(function (a) {
      var len = a.xe - a.xs, head = Math.min(7, Math.max(len * 0.6, 0)), hh = 4;
      var shaftEnd = len > 6 ? a.xe - head + 1 : a.xe;
      el('line', { x1: a.xs, y1: a.ym, x2: shaftEnd, y2: a.ym, 'class': 'lag-casing', 'stroke-width': 5, 'stroke-linecap': 'round' }, gArrows);
      el('line', { x1: a.xs, y1: a.ym, x2: shaftEnd, y2: a.ym, 'class': a.cls, 'stroke-width': 2, 'stroke-linecap': 'round' }, gArrows);
      if (len > 6) {
        var pts = (a.xe - head) + ',' + (a.ym - hh) + ' ' + a.xe + ',' + a.ym + ' ' + (a.xe - head) + ',' + (a.ym + hh);
        el('polygon', { points: pts, 'class': a.cls + ' head', 'stroke-linejoin': 'round' }, gArrows);
      }
    });
    visible().forEach(function (q) {
      var w = writeOf(q); if (!w || w.sa === null) return;
      var x = xOfT(w.sa);
      var ln = el('line', { x1: x, y1: PAD.t, x2: x, y2: lanesBottom, 'class': 'sched', 'stroke-dasharray': '1 3', 'stroke-width': 3, 'stroke-opacity': 0 }, c.gPlot);
      el('line', { x1: x, y1: PAD.t, x2: x, y2: lanesBottom, 'class': 'sched', 'stroke-dasharray': '1 3', 'pointer-events': 'none' }, c.gPlot);
      var g = bySeq.get(q) || [], first = g.length ? Math.min.apply(null, g.map(function (s) { return s.s; })) : null;
      var rows = [['scrape_seq', q], ['scheduled at', fmtClock(w.sa)], ['offset', '+' + w.sa.toFixed(3) + ' s']];
      if (first !== null) rows.push(['first request started', '+' + fmtS(Math.max(0, first - w.sa)) + ' after slot']);
      setTip(ln, 'Scheduled slot', rows);
    });
    // x axis in seconds since t0
    var g = c.gAxes;
    el('line', { x1: PAD.l, y1: lanesBottom, x2: PAD.l + c.plotW, y2: lanesBottom, 'class': 'ax' }, g);
    niceTicks(tA, tB, 8).forEach(function (v) {
      var x = xOfT(v); if (x < PAD.l - 0.5 || x > PAD.l + c.plotW + 0.5) return;
      el('line', { x1: x, y1: lanesBottom, x2: x, y2: lanesBottom + 4, 'class': 'ax' }, g);
      text(g, x, lanesBottom + 15, fmtTick(v), 'tick', 'middle');
    });
    text(g, PAD.l + c.plotW / 2, lanesBottom + 29, 'wall-clock seconds since the first scheduled slot / request (' + fmtClock(0) + ')', 'axis', 'middle');
  }
  function seqRangeTimeline(c, xa, xb) {
    var ta = c.timeAt(xa), tb = c.timeAt(xb), lo = null, hi = null;
    for (var i = view.lo; i <= view.hi; i++) {
      var sp = span.get(seqs[i]); if (!sp) continue;
      if (sp[1] >= ta && sp[0] <= tb) { if (lo === null) lo = i; hi = i; }
    }
    if (lo === null) {
      var best = view.lo, bd = Infinity;
      for (var j = view.lo; j <= view.hi; j++) { var s2 = span.get(seqs[j]); if (!s2) continue; var d = Math.abs((s2[0] + s2[1]) / 2 - (ta + tb) / 2); if (d < bd) { bd = d; best = j; } }
      lo = hi = best;
    }
    return [lo, hi];
  }

  // ---- 2. lag --------------------------------------------------------------
  var LAG_H = 200;
  function drawLag(c) {
    var vis = visible(), lags = [];
    vis.forEach(function (q) { (bySeq.get(q) || []).forEach(function (s) { if (s.lag !== null && hostVisible(s.h)) lags.push(s.lag); }); });
    var hi = lags.length ? Math.max.apply(null, lags) * 1.08 : 0.01; if (hi <= 0) hi = 0.01;
    var yOf = function (v) { return PAD.t + LAG_H - v / hi * LAG_H; };
    yAxis(c, yOf, 0, hi, PAD.t, PAD.t + LAG_H, 'schedule lag (s)');
    hosts.forEach(function (host, h) {
      if (!hostVisible(h)) return;
      var runs = [[]], color = hostColor(h);
      vis.forEach(function (q, k) {
        var s = slot.get(h + '|' + q);
        if (s && s.lag !== null) runs[runs.length - 1].push([view.lo + k, s]);
        else if (runs[runs.length - 1].length) runs.push([]);
      });
      runs.forEach(function (run) {
        if (run.length < 2) return;
        el('polyline', { points: run.map(function (p) { return c.xOfIdx(p[0]).toFixed(1) + ',' + yOf(p[1].lag).toFixed(1); }).join(' '), fill: 'none', stroke: color, 'stroke-width': 1.8, 'pointer-events': 'none' }, c.gPlot);
      });
      runs.forEach(function (run) { run.forEach(function (p) {
        var s = p[1], w = writeOf(s.q);
        var dot = el('circle', { cx: c.xOfIdx(p[0]), cy: yOf(s.lag), r: 3, fill: color }, c.gPlot);
        var rows = [['host', host], ['scrape_seq', s.q], ['schedule lag', fmtS(s.lag)]];
        if (w && w.sa !== null) rows.push(['scheduled slot', fmtClock(w.sa)]);
        rows.push(['request started', fmtClock(s.s)], ['request duration', fmtS(s.d)]);
        var others = (bySeq.get(s.q) || []).filter(function (o) { return o.lag !== null && o.h !== h; }).map(function (o) { return o.lag; });
        if (others.length) rows.push(["other hosts' lag (this seq)", fmtS(Math.min.apply(null, others)) + ' - ' + fmtS(Math.max.apply(null, others))]);
        setTip(dot, 'Schedule lag', rows);
      }); });
      vis.forEach(function (q, k) {
        if (slot.has(h + '|' + q)) return;
        var x = c.xOfIdx(view.lo + k);
        var ln = el('line', { x1: x, y1: PAD.t, x2: x, y2: PAD.t + LAG_H, stroke: color, 'stroke-dasharray': '2 3', opacity: 0.6, 'stroke-width': 3 }, c.gPlot);
        var at = abandonedTip(h, q, [], 'No lag can be measured: the request never settled before the cycle deadline.');
        setTip(ln, at.title, at.rows, at.note);
      });
    });
    xAxisSeq(c, PAD.t + LAG_H);
    filterNote(c);
  }

  // ---- 3. cycle cost -------------------------------------------------------
  // Request and parse terms are per-host maxima, so they follow the host filter;
  // writer lock wait and sample write are per-cycle (one cycle_write per seq) and
  // do not depend on which hosts are shown.
  var COST_H = 220;
  function costStack(q) {
    var g = (bySeq.get(q) || []).filter(function (s) { return hostVisible(s.h); }), w = writeOf(q);
    var slowest = g.reduce(function (m, s) { return !m || s.d > m.d ? s : m; }, null);
    var parsed = g.filter(function (s) { return s.p !== null; });
    return {
      request: g.length ? Math.max.apply(null, g.map(function (s) { return s.d; })) : 0,
      parse: parsed.length ? Math.max.apply(null, parsed.map(function (s) { return s.p; })) : 0,
      lock: w ? w.lw : 0, write: w ? w.ws : 0, slowest: slowest, settled: g.length, write_rec: w
    };
  }
  function drawCost(c) {
    var vis = visible(), stacks = vis.map(costStack);
    var segs = DATA.cost_segments.filter(function (seg) { return segOn[seg.key]; });
    var nHostsOn = hostOn.filter(Boolean).length, filtered = segs.length < DATA.cost_segments.length || nHostsOn < hosts.length;
    var totals = stacks.map(function (st) { return segs.reduce(function (a, seg) { return a + st[seg.key]; }, 0); });
    var hi = Math.max.apply(null, totals.concat([interval || 0])) * 1.1; if (!(hi > 0)) hi = 0.01;
    var yOf = function (v) { return PAD.t + COST_H - v / hi * COST_H; };
    yAxis(c, yOf, 0, hi, PAD.t, PAD.t + COST_H, filtered ? 'seconds in the cycle (filtered)' : 'seconds in the cycle');
    var barW = Math.max(2, c.bandW() * 0.7);
    vis.forEach(function (q, k) {
      var st = stacks[k], total = totals[k], x = c.xOfIdx(view.lo + k) - barW / 2;
      var rows = [['scrape_seq', q], [filtered ? 'total (shown terms)' : 'total', fmtS(total)]];
      DATA.cost_segments.forEach(function (seg) { rows.push([seg.label + (segOn[seg.key] ? '' : ' (hidden)'), fmtS(st[seg.key])]); });
      if (st.slowest) rows.push(['slowest host' + (nHostsOn < hosts.length ? ' (shown)' : ''), hosts[st.slowest.h] + ' (' + fmtS(st.slowest.d) + ', ' + st.slowest.o + ')']);
      rows.push(['hosts settled' + (nHostsOn < hosts.length ? ' (shown)' : ''), st.settled + ' of ' + nHostsOn]);
      if (interval !== null) rows.push(['vs interval', (100 * total / interval).toFixed(0) + '% of ' + fmtS(interval)]);
      var note = null;
      if (!st.write_rec) note = 'No cycle_write record for this seq: lock-wait and write terms are unknown (shown as 0).';
      else if (interval !== null && total > interval) note = 'Cycle cost exceeded the sample interval - the next slot starts late and lag accumulates.';
      if (filtered) note = (note ? note + ' ' : '') + 'Filtered view: the bar stacks only the shown delay types, and request/parse are maxima over the shown hosts only; lock wait and sample write are per-cycle and ignore the host filter.';
      var hit = el('rect', { x: x, y: PAD.t, width: barW, height: COST_H, 'class': 'hit' }, c.gPlot);
      setTip(hit, 'Cycle cost', rows, note);
      var base = 0;
      segs.forEach(function (seg) {
        var v = st[seg.key]; if (v <= 0) return;
        var yt = yOf(base + v), yb = yOf(base);
        el('rect', { x: x, y: yt, width: barW, height: Math.max(yb - yt, 0.5), fill: seg.color, 'pointer-events': 'none' }, c.gPlot);
        base += v;
      });
      if (!st.write_rec) text(c.gPlot, x + barW / 2, PAD.t + COST_H - 3, '?', 'tick warn', 'middle', { 'pointer-events': 'none' });
    });
    if (interval !== null) {
      var y = yOf(interval);
      el('line', { x1: PAD.l, y1: y, x2: PAD.l + c.plotW, y2: y, 'class': 'ref', 'stroke-dasharray': '6 3', 'pointer-events': 'none' }, c.gPlot);
      var label = 'sample interval ≈ ' + fmtS(interval) + ' (' + DATA.interval.method + ')';
      var covered = Math.min(vis.length, Math.max(1, Math.ceil(label.length * 7 / c.bandW())));
      var leftMax = Math.max.apply(null, totals.slice(0, covered)), rightMax = Math.max.apply(null, totals.slice(-covered));
      var onLeft = leftMax <= rightMax;
      text(c.gTop, onLeft ? PAD.l + 6 : PAD.l + c.plotW - 4, y - 4, label, 'tick ref-label', onLeft ? 'start' : 'end');
    }
    xAxisSeq(c, PAD.t + COST_H);
    filterNote(c);
  }

  // ---- 4. write health -----------------------------------------------------
  var STRIP_H = 28;
  function drawWrite(c) {
    var vis = visible(), cellW = Math.max(2, c.bandW() * 0.85);
    text(c.gAxes, PAD.l - 8, PAD.t + STRIP_H / 2 + 4, 'cycle_write', 'lane', 'end');
    vis.forEach(function (q, k) {
      var w = writeOf(q), x = c.xOfIdx(view.lo + k) - cellW / 2, r;
      if (!w) {
        r = el('rect', { x: x, y: PAD.t + 4, width: cellW, height: STRIP_H - 8, fill: c.hatch, 'class': 'missing' }, c.gPlot);
        setTip(r, 'No cycle_write record', [['scrape_seq', q]], 'Either the diagnostics queue dropped it (see dropped_records) or the file was cut off.');
        return;
      }
      var rows = [['scrape_seq', q], ['scheduled slot', w.sa === null ? 'n/a - bracket/manual' : fmtClock(w.sa)], ['rows attempted', w.n],
        ['writer lock wait', fmtS(w.lw)], ['append + flush', fmtS(w.ws)], ['completed', w.ok ? 'yes' : 'no']];
      r = el('rect', { x: x, y: PAD.t + 4, width: cellW, height: STRIP_H - 8, 'class': (w.ok ? 'o-ok' : 'o-http') + ' cell' }, c.gPlot);
      if (w.ok) setTip(r, 'Batch written', rows);
      else { rows.push(['reason', w.err || 'refused (session finalizing)']); setTip(r, 'Batch NOT written', rows, 'sample_write_error names the exception class; null means the session had already disabled artifact mutation and refused the append.'); }
    });
    xAxisSeq(c, PAD.t + STRIP_H);
  }

  // ---- 5. coverage ---------------------------------------------------------
  var ROW_H = 22;
  var maxRows = Math.max(1, Math.max.apply(null, DATA.scrapes.map(function (s) { return s.n; })));
  function drawCoverage(c) {
    var vis = visible(), cellW = c.bandW();
    hosts.forEach(function (host, h) {
      var y = PAD.t + h * ROW_H;
      text(c.gAxes, PAD.l - 8, y + ROW_H / 2 + 4, host, 'lane', 'end');
      vis.forEach(function (q, k) {
        var s = slot.get(h + '|' + q), x = c.xOfIdx(view.lo + k) - cellW / 2, r;
        var attrs = { x: x + 0.5, y: y + 1, width: Math.max(cellW - 1, 1), height: ROW_H - 2 };
        if (!s) {
          attrs.fill = c.hatch; attrs['class'] = 'missing'; r = el('rect', attrs, c.gPlot);
          var at = abandonedTip(h, q, [], 'Unsettled at the cycle deadline; no rows and no timing record.'); setTip(r, at.title, at.rows, at.note);
          return;
        }
        if (s.n === 0) attrs['class'] = 'o-http cell';
        else { attrs['class'] = 'o-ok cell'; attrs['fill-opacity'] = (0.3 + 0.7 * s.n / maxRows).toFixed(2); }
        r = el('rect', attrs, c.gPlot);
        var d = scrapeTip(s); d.rows.splice(2, 0, ['rows vs max seen', s.n + ' / ' + maxRows]); setTip(r, d.title, d.rows, d.note);
      });
    });
    xAxisSeq(c, PAD.t + ROW_H * hosts.length);
  }

  // ---- boot ----------------------------------------------------------------
  buildFilters();
  makeChart('timeline', PAD.t + LANE * hosts.length + PAD.b, drawTimeline, seqRangeTimeline);
  makeChart('lag', PAD.t + LAG_H + PAD.b, drawLag, seqRangeBand);
  makeChart('cost', PAD.t + COST_H + PAD.b, drawCost, seqRangeBand);
  makeChart('write', PAD.t + STRIP_H + PAD.b, drawWrite, seqRangeBand);
  makeChart('coverage', PAD.t + ROW_H * hosts.length + PAD.b, drawCoverage, seqRangeBand);
  charts.forEach(function (c) { c.layout(); c.draw(); });
  updateHints();
  wirePans();
  var resetBtn = document.querySelector('.zoom-reset'); if (resetBtn) resetBtn.addEventListener('click', resetView);
  window.__scrapeViz = { setView: setView, resetView: resetView, view: view, seqs: seqs };
})();
"""


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def _stat_card(num: object, label: str, *, warn: bool = False) -> str:
    cls = "stat-card warn" if warn else "stat-card"
    return (
        f'<div class="{cls}"><p class="stat-card-num">{_esc(num)}</p><p class="stat-card-label">{_esc(label)}</p></div>'
    )


def _timeline_legend() -> str:
    keys = [
        ("ok", "HTTP 200, parsed"),
        ("err", "HTTP error (non-200 / HTTPError)"),
        ("timeout", "request timeout"),
        ("other", "other request exception"),
        ("hatch", "no scrape record — abandoned at the cycle deadline"),
        ("bracket", "bracket / manual scrape (no scheduled slot, no lag)"),
        ("sched", "scheduled slot (cycle_write.scheduled_at_unix)"),
        ("lag-lo", "schedule lag ≤ 25 % of interval"),
        ("lag-mid", "lag 25–100 % of interval"),
        ("lag-hi", "lag > interval — slot overrun"),
    ]
    out = "".join(f'<span class="key"><i class="{cls}"></i>{label}</span>' for cls, label in keys)
    return f'<div class="legend">{out}</div>'


def _cost_legend() -> str:
    # Both rows are filled by the inline script as toggle buttons: delay types
    # (COST_SEGMENTS, via the payload) and hosts (shared with the lag chart).
    return (
        '<div class="legend-stack">'
        '<div class="legend" data-legend="segments" data-label="delay type"></div>'
        '<div class="legend" data-legend="hosts" data-label="hosts"></div></div>'
    )


def _panel(chart_id: str, title: str, blurb: str, before: str = "", after: str = "") -> str:
    return (
        f'<section class="chart-panel"><div class="chart-head"><h2>{_esc(title)}</h2>'
        '<span class="zoom-hint"></span></div>'
        f'<p class="blurb">{_esc(blurb)}</p>{before}'
        f'<svg class="chart" data-chart="{chart_id}"></svg>'
        '<div class="pan" hidden><div class="pan-inner"></div></div>'
        f"{after}</section>"
    )


def render_html(t: Timings, source: Path) -> str:
    hosts, seqs = t.hosts, t.seqs
    if not t.scrapes:
        raise SystemExit(f"{source}: no scrape records found")
    scheduled = [s for s in t.scrapes if not s.is_bracket]
    brackets = [s for s in t.scrapes if s.is_bracket]
    missing = sum(1 for h in hosts for q in seqs if (h, q) not in t.by_slot())
    failed = sum(1 for s in t.scrapes if s.outcome != "ok")
    write_failures = [w for w in t.writes.values() if not w.completed]
    cards = [
        _stat_card(f"{seqs[0]} – {seqs[-1]}", f"scrape_seq range ({len(seqs)} cycles)"),
        _stat_card(len(hosts), "hosts seen"),
        _stat_card(f"{len(scheduled)} + {len(brackets)}", "scrape records: scheduled + bracket"),
        _stat_card(failed, "failed requests", warn=failed > 0),
        _stat_card(missing, "abandoned slots (host × seq, no record)", warn=missing > 0),
        _stat_card(
            f"{len(t.writes)} / {len(write_failures)}", "cycle_write records / failed writes", warn=bool(write_failures)
        ),
    ]
    if t.dropped_records is None:
        cards.append(_stat_card("missing", "diagnostic_summary — file may be cut off", warn=True))
    else:
        cards.append(_stat_card(t.dropped_records, "dropped diagnostic records", warn=t.dropped_records > 0))
    if t.bad_lines:
        cards.append(_stat_card(t.bad_lines, "unparseable lines", warn=True))
    if t.unknown_events:
        cards.append(_stat_card(t.unknown_events, "unknown event kinds", warn=True))

    write_note = (
        '<p class="note warn">'
        + _esc(
            f"{len(write_failures)} failed batch write(s): "
            + "; ".join(f"seq {w.seq}: {w.error or 'refused (session finalizing)'}" for w in write_failures)
        )
        + "</p>"
        if write_failures
        else '<p class="note">Every cycle_write record reports sample_write_completed = true.</p>'
    )

    body = "".join(
        [
            _panel(
                "timeline",
                "Scrape request timeline — one lane per host, one bar per request (request_started_at_unix → request_finished_at_unix)",
                "Bar colour is the request outcome; hatched slots are (host, scrape_seq) pairs with no scrape record, i.e. the "
                "endpoint was still unsettled at the cycle deadline. Dashed blue outline = bracket/manual scrape. Dotted vertical "
                "ticks are each cycle's scheduled slot; the arrow from a tick to its bar is that request's schedule lag "
                "(schedule_lag_seconds) — lengthening arrows mean requests are drifting off the schedule, coloured yellow past "
                "25 % of the sample interval and magenta once they overrun it (hues chosen to stay clear of the outcome colours). "
                "When a cycle's lag runs past later slots, the overlapping arrows are stacked on separate lines within the lane. "
                "Hover for details; drag to zoom.",
                before=_timeline_legend(),
            ),
            _panel(
                "lag",
                "Schedule lag per host by scrape_seq — request start minus the cycle's scheduled slot (schedule_lag_seconds)",
                "Steady growth means cycle overrun is accumulating; a spike on every host at one seq means one slow endpoint stalled "
                "that batch. Vertical dashed ticks mark seqs where that host has no record. The y-axis follows the visible range. "
                "Click a host in the legend to hide or show it (shift-click to isolate); the host filter is shared with the "
                "cost chart below.",
                before='<div class="legend" data-legend="hosts" data-label="hosts"></div>',
            ),
            _panel(
                "cost",
                "Per-cycle cost decomposition by scrape_seq — slowest request, slowest parse, writer lock wait, sample write",
                "The cycle waits for its slowest endpoint, so the max request_duration_seconds across hosts is the dominant term. "
                "Dashed line = sample interval, taken as the median gap between cycle_write.scheduled_at_unix slots (the setting "
                "itself is not in the file). Click a delay type to drop it from the stack, or a host to drop it from the "
                "request/parse maxima (shift-click isolates); lock wait and sample write are per-cycle and ignore the host filter.",
                before=_cost_legend(),
            ),
            _panel(
                "write",
                "Sample batch write health per scrape_seq — cycle_write.sample_write_completed",
                "Green = batch appended and flushed; red = not written (sample_write_error names the exception, or the session was "
                "already finalizing and refused the batch); hatched = no cycle_write record for that seq.",
                after=write_note,
            ),
            _panel(
                "coverage",
                "GPU rows returned per host per scrape_seq — scrape.row_count",
                "Stronger green = more rows; red = a settled request that produced 0 rows (failure); hatched = abandoned (no record).",
            ),
        ]
    )
    payload = json.dumps(build_payload(t), separators=(",", ":")).replace("</", "<\\/")
    title = f"{_esc(t.run_name or '—')}" + (f" · job {_esc(t.job_id)}" if t.job_id else "")
    return (
        '<!DOCTYPE html><html lang="en" data-theme="dark"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>scrape-timings — {_esc(t.run_name or source.name)}</title><style>{_CSS}</style>"
        f"<script>{_THEME_JS}</script></head><body>"
        '<div class="page-head"><div>'
        f"<h1>Power collector scrape timings — {title}</h1>"
        f'<p class="subtitle">Source: {_esc(source)} · diagnostic sidecar written by the power collector; '
        "not publication-validation evidence.</p></div>"
        '<div class="controls"><button type="button" class="btn zoom-reset" hidden>Reset zoom</button>'
        '<button type="button" class="btn theme-toggle">Switch to light mode</button></div></div>'
        f'<div class="stat-cards">{"".join(cards)}</div>{body}'
        "<footer>Generated by srtctl.analysis.scrape_timings_viz · every value is read from the JSONL; "
        "the sample interval is the only derived quantity. Drag on any chart to zoom all of them to a scrape_seq range; "
        "while zoomed, the scrollbar under each chart (or shift+wheel) pans; double-click to reset.</footer>"
        f'<script id="scrape-data" type="application/json">{payload}</script>'
        f"<script>{_CHARTS_JS}</script></body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render scrape-timings.jsonl as a standalone HTML page.")
    parser.add_argument("timings", type=Path, help="path to power/scrape-timings.jsonl")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output .html path")
    args = parser.parse_args(argv)
    timings = load_timings(args.timings)
    args.output.write_text(render_html(timings, args.timings), encoding="utf-8")
    print(f"wrote {args.output} ({len(timings.scrapes)} scrape records, {len(timings.writes)} cycles)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
