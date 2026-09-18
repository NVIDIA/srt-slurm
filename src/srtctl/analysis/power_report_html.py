# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained HTML power report: a concurrency-level stats table plus
GPU/CPU power-over-time charts, built from the same artifacts as
``power_energy_report.py``.

Reuses that module's CSV-loading (``load_gpu_samples``, ``load_cpu_samples``,
``build_reports``) rather than re-parsing the power CSVs, so the two reports never
disagree about what a device's series looks like. CSV *discovery* is its own local
helper here (see ``_discover_power_csvs``): unlike the JSON report, chart-only mode
must work even without a recognized benchmark.out, e.g. mid-run or serve-only jobs.

The front page is a Pareto view: one point per (run, concurrency), with
selectable axes, a per-run frontier guide line, hover stats and a click-to-inspect
panel. Selecting a point swaps in that point's GPU and CPU power-over-time
charts, sliced to the point's *measured window* (the benchmark's own
``start_unix``..``end_unix`` for that concurrency) rather than the whole run.
Each chart combines every device across every node onto one axis (labelled
``host/gpuN`` / ``host/socketN``): the hue encodes the host and the shade the
device index within that host, every line also carries a direct end-of-line
label, and legend entries toggle their line so identity never depends on hue
alone. The "Data table" tab keeps the per-concurrency stats table and each run's
whole-run power charts.

No external JS/CSS: everything (styles, downsampled series data, the crosshair
tooltip) is inlined so the file opens standalone from a browser or an artifact
store with no network access.
"""

from __future__ import annotations

import html
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from srtctl.analysis.power_energy_report import (
    CPU_SAMPLES_DIRNAMES,
    PowerReportError,
    build_reports,
    load_cpu_samples,
    load_gpu_roles,
    load_gpu_samples,
    report_to_dict,
)

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext

logger = logging.getLogger(__name__)

HTML_FILENAME = "power_report.html"

# Only the first three slots of the project's default categorical palette clear
# the all-pairs CVD/contrast gates in both light and dark mode (see palette.md);
# they drive the UI accents (active tab, selected point).
_SLOT_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a")
_SLOT_DARK = ("#3987e5", "#d95926", "#199e70")

# Time-series lines: one hue per host (cycled), lightness spread across that
# host's devices so ``host/gpu0`` .. ``host/gpu3`` read as shades of one colour.
_HOST_HUES = (212, 18, 158, 280, 45, 340, 95, 190)
_DEVICE_LIGHTNESS_RANGE = (36.0, 70.0)

# CSS stroke-dasharray per pattern index: solid, dashed, dotted. Hosts with more
# than 4 devices additionally cycle the stroke pattern so shades alone don't have
# to carry identity.
_STROKE_PATTERNS = ("", "5 4", "1.5 3")
_MAX_SOLID_DEVICES_PER_HOST = 4

# Cap plotted points per series: two per bucket (min + max), so short spikes in an
# otherwise-smooth power trace survive downsampling instead of being averaged away.
_MAX_BUCKETS_PER_SERIES = 1200
# Per-point (measured-window) charts are embedded once per concurrency, so they
# get a tighter cap to keep the standalone file's size in check.
_MAX_BUCKETS_PER_WINDOW_SERIES = 600


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


def _downsample_minmax(
    times: np.ndarray, watts: np.ndarray, max_buckets: int = _MAX_BUCKETS_PER_SERIES
) -> tuple[np.ndarray, np.ndarray]:
    """Bucket into ``max_buckets`` chunks, keeping each bucket's min and max sample.

    Preserves spikes that a plain stride/average downsample would smooth away,
    while keeping the embedded payload bounded regardless of the run's duration
    or sample rate.
    """
    n = len(times)
    if n <= max_buckets * 2:
        return times, watts
    edges = np.linspace(0, n, max_buckets + 1).astype(int)
    out_t: list[float] = []
    out_w: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        if hi <= lo:
            continue
        seg_t, seg_w = times[lo:hi], watts[lo:hi]
        i_min, i_max = int(np.argmin(seg_w)), int(np.argmax(seg_w))
        for i in sorted({i_min, i_max}):
            out_t.append(float(seg_t[i]))
            out_w.append(float(seg_w[i]))
    return np.array(out_t), np.array(out_w)


# ---------------------------------------------------------------------------
# Per-facet series preparation
# ---------------------------------------------------------------------------


def _series_stats(watts: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(watts.mean()),
        "min": float(watts.min()),
        "p50": float(np.percentile(watts, 50)),
        "p95": float(np.percentile(watts, 95)),
        "max": float(watts.max()),
    }


def _series_color(host_position: int, device_position: int, devices_on_host: int) -> str:
    """CSS colour for one device line: hue by host, lightness by device within host."""
    hue = _HOST_HUES[host_position % len(_HOST_HUES)]
    lo, hi = _DEVICE_LIGHTNESS_RANGE
    steps = min(devices_on_host, _MAX_SOLID_DEVICES_PER_HOST)
    step = device_position % steps
    lightness = lo if steps <= 1 else lo + (hi - lo) * step / (steps - 1)
    return f"hsl({hue} 70% {lightness:.0f}%)"


def _build_run_series(
    per_device: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    *,
    label_fmt: str,
    window: tuple[float, float] | None = None,
    origin: float | None = None,
    max_buckets: int = _MAX_BUCKETS_PER_SERIES,
    roles: dict[tuple[str, int], set[str]] | None = None,
    host_roles: dict[str, set[str]] | None = None,
) -> list[dict]:
    """One series per device across the whole run -- every node combined onto a
    single chart -- sorted by ``(hostname, index)``, downsampled and time-shifted
    to seconds since the earliest sample across every device (a shared x-axis).

    ``label_fmt`` receives ``host`` and ``index`` so devices from different nodes
    stay distinguishable once they share one chart (e.g. ``"{host}/gpu{index}"``).

    ``window`` (``start_unix, end_unix``) restricts each device to the samples
    inside that range -- the per-concurrency measured window -- and time-shifts
    to the window start instead of the first sample. Devices with no samples in
    the window are dropped.

    ``origin`` overrides the zero of the relative time axis; pass the same value
    for the GPU and CPU legs of one run so their charts line up sample-for-sample.

    Colour: one hue per host, spread in lightness across that host's devices, so a
    chart of ``N`` nodes x ``M`` GPUs reads as ``N`` colour families.

    ``roles`` (per ``(host, index)``, from the GPU manifest) or ``host_roles`` (per
    host -- used for CPU sockets, which inherit the roles of the GPUs on their node)
    tag each series with its worker roles so the legend can toggle by role.
    """
    populated: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, (times, watts) in per_device.items():
        if window is not None:
            mask = (times >= window[0]) & (times <= window[1])
            times, watts = times[mask], watts[mask]
        if len(times):
            populated[key] = (times, watts)
    if not populated:
        return []
    if origin is not None:
        global_start = origin
    elif window is not None:
        global_start = window[0]
    else:
        global_start = min(times[0] for times, _ in populated.values())

    hosts = sorted({host for host, _ in populated})
    devices_on_host = {host: sum(1 for h, _ in populated if h == host) for host in hosts}

    series = []
    device_position = 0
    previous_host: str | None = None
    for host, index in sorted(populated):
        if host != previous_host:
            device_position, previous_host = 0, host
        times, watts = populated[(host, index)]
        ds_times, ds_watts = _downsample_minmax(times, watts, max_buckets)
        rel_times = (ds_times - global_start).round(2)
        series.append(
            {
                "label": label_fmt.format(host=host, index=index),
                "host": host,
                "roles": sorted((roles or {}).get((host, index)) or (host_roles or {}).get(host) or ()),
                "color": _series_color(hosts.index(host), device_position, devices_on_host[host]),
                "pattern": (device_position // _MAX_SOLID_DEVICES_PER_HOST) % len(_STROKE_PATTERNS),
                "t": rel_times.tolist(),
                "w": [round(v, 2) for v in ds_watts.tolist()],
                "stats": _series_stats(watts),
            }
        )
        device_position += 1
    return series


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_LIGHT_SLOT_VARS = f"--slot-0: {_SLOT_LIGHT[0]}; --slot-1: {_SLOT_LIGHT[1]}; --slot-2: {_SLOT_LIGHT[2]};"
_DARK_SLOT_VARS = f"--slot-0: {_SLOT_DARK[0]}; --slot-1: {_SLOT_DARK[1]}; --slot-2: {_SLOT_DARK[2]};"

_CSS = """
:root, .light { color-scheme: light; }
body {
  margin: 0; padding: 24px; background: var(--page); color: var(--ink-primary);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  --page: #f9f9f7; --surface: #fcfcfb; --ink-primary: #0b0b0b; --ink-secondary: #52514e;
  --ink-muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  __LIGHT_SLOT_VARS__
}
@media (prefers-color-scheme: dark) {
  body {
    --page: #0d0d0d; --surface: #1a1a19; --ink-primary: #ffffff; --ink-secondary: #c3c2b7;
    --ink-muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    __DARK_SLOT_VARS__
  }
}
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 32px 0 12px; }
.subtitle { color: var(--ink-secondary); margin: 0 0 24px; }
table { border-collapse: collapse; width: 100%; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
th, td { text-align: right; padding: 6px 10px; font-variant-numeric: tabular-nums; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
th { color: var(--ink-secondary); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .02em; background: var(--page); }
tr:last-child td { border-bottom: none; }
.legend { display: flex; gap: 8px 12px; flex-wrap: wrap; margin: 4px 0 10px; font-size: 12px; color: var(--ink-secondary); }
.legend-key { display: inline-flex; align-items: center; gap: 6px; padding: 3px 8px; border: 1px solid var(--border); border-radius: 4px; cursor: pointer; user-select: none; }
.legend-key.off { opacity: .4; text-decoration: line-through; }
.legend-key.partial { opacity: .7; border-style: dashed; }
.host-legend { margin: 0 0 12px; padding-bottom: 10px; border-bottom: 1px solid var(--grid); }
.legend-label { color: var(--ink-muted); font-size: 11px; text-transform: uppercase; letter-spacing: .02em; align-self: center; }
.host-key { font-weight: 600; }
.role-legend { margin: 0 0 8px; }
.chart-notices { color: var(--slot-1); font-size: 12px; margin: 0 0 10px; padding: 8px 10px;
  border: 1px solid color-mix(in srgb, var(--slot-1) 45%, transparent); border-radius: 6px;
  background: color-mix(in srgb, var(--slot-1) 8%, transparent); }
.chart-notices div + div { margin-top: 3px; }
.row-warn { color: var(--slot-1); cursor: help; }
.role-key { font-weight: 600; text-transform: capitalize; }
.role-count { color: var(--ink-muted); font-weight: 400; }
.legend-swatch { width: 14px; height: 3px; border-radius: 1px; }
.stat-cards { display: flex; gap: 12px; margin: 4px 0 24px; flex-wrap: wrap; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 18px; min-width: 110px; }
.stat-card-num { font-size: 22px; font-weight: 700; margin: 0; }
.stat-card-label { color: var(--ink-secondary); font-size: 12px; margin: 2px 0 0; }
.chart-group { margin-top: 16px; }
.chart-group-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin: 0 0 8px; }
.zoom-hint { color: var(--ink-muted); font-size: 11px; }
.chart-sub + .chart-sub { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--grid); }
.chart-sub-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
.chart-sub-title { font-weight: 600; font-size: 13px; margin: 0 0 4px; }
.yscale-toggle { display: inline-flex; border: 1px solid var(--line); border-radius: 5px; overflow: hidden; font-size: 11px; }
.yscale-btn { background: transparent; color: var(--ink-muted); border: 0; padding: 1px 8px; cursor: pointer; font: inherit; font-size: 11px; }
.yscale-btn + .yscale-btn { border-left: 1px solid var(--line); }
.yscale-btn.on { background: var(--surface-2, rgba(127,127,127,0.18)); color: var(--ink); font-weight: 600; }
.zoom-band { fill: var(--slot-0); opacity: 0; pointer-events: none; }
.power-chart-run[hidden], .point-charts[hidden] { display: none; }
.chart-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px 4px; margin-bottom: 16px; }
.chart-panel-title { font-weight: 600; font-size: 13px; margin: 0 0 4px; }
.chart-wrap { position: relative; }
svg.chart { width: 100%; height: 220px; display: block; overflow: visible; }  /* viewBox set from pixel width in JS */
.gridline { stroke: var(--grid); stroke-width: 1; }
.axis-text { fill: var(--ink-muted); font-size: 10px; }
.axis-title { fill: var(--ink-secondary); font-size: 11px; font-weight: 600; }
.end-label { font-size: 10px; fill: var(--ink-secondary); }
.crosshair { stroke: var(--axis); stroke-width: 1; pointer-events: none; opacity: 0; }
.tooltip {
  position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; padding: 6px 8px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.15);
  opacity: 0; white-space: nowrap; z-index: 10;
}
.tooltip .t-time { color: var(--ink-muted); margin-bottom: 4px; }
.tooltip .t-row { display: flex; gap: 8px; align-items: center; }
.tooltip .t-key { width: 12px; height: 2px; flex: none; }
.tooltip .t-val { font-weight: 600; font-variant-numeric: tabular-nums; }
.stats-toggle { color: var(--ink-muted); font-size: 11px; cursor: pointer; user-select: none; }
.stats-table { display: none; margin: 8px 0 12px; font-size: 12px; }
.stats-table.open { display: table; }
footer { color: var(--ink-muted); font-size: 12px; margin-top: 32px; }
code { background: var(--page); padding: 1px 4px; border-radius: 3px; }
.tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }
.tab-btn {
  appearance: none; background: none; border: none; color: var(--ink-secondary); cursor: pointer;
  font: inherit; font-size: 13px; font-weight: 600; padding: 8px 4px 10px; margin-bottom: -1px;
  border-bottom: 2px solid transparent;
}
.tab-btn.active { color: var(--ink-primary); border-bottom-color: var(--slot-0); }
.pareto-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px 16px; margin-bottom: 16px; }
.pareto-card h3 { font-size: 14px; margin: 0 0 4px; }
.pareto-subtitle { color: var(--ink-secondary); font-size: 12px; margin: 0 0 12px; }
.pareto-controls { display: flex; gap: 16px; flex-wrap: wrap; align-items: center; font-size: 13px; margin: 0 0 8px; }
.pareto-controls label { display: inline-flex; gap: 6px; align-items: center; }
.pareto-controls select { font: inherit; font-size: 13px; padding: 3px 6px; background: var(--surface); color: var(--ink-primary); border: 1px solid var(--border); border-radius: 4px; }
.run-legend { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 12px; font-size: 12px; }
.run-key { display: inline-flex; align-items: center; gap: 6px; padding: 3px 8px; border: 1px solid var(--border); border-radius: 4px; cursor: pointer; user-select: none; color: var(--ink-secondary); }
.run-key.off { opacity: .4; text-decoration: line-through; }
.run-key[hidden] { display: none; }
.run-swatch { width: 14px; height: 3px; border-radius: 1px; }
.pareto-root { position: relative; }
.pareto-layout { display: grid; grid-template-columns: minmax(0, 3fr) minmax(320px, 2fr); gap: 20px; align-items: start; }
@media (max-width: 1100px) { .pareto-layout { grid-template-columns: 1fr; } }
svg.pareto-svg { width: 100%; height: auto; aspect-ratio: 900 / 520; display: block; overflow: visible; }
.pareto-frontier { fill: none; stroke-width: 1.5; opacity: .7; }
.baseline-ref { stroke: var(--ink-muted); stroke-width: 1; stroke-dasharray: 4 4; }
.scope-toggle { display: inline-flex; gap: 6px; align-items: center; font-size: 12px; color: var(--ink-secondary); cursor: pointer; user-select: none; }
.scope-window[hidden], .scope-run[hidden], .point-charts-table[hidden], .scope-pick[hidden] { display: none; }
.chart-filter { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; margin: 0 0 16px; }
.filter-row { display: flex; gap: 8px 12px; flex-wrap: wrap; align-items: center; font-size: 12px; margin: 4px 0; }
.filter-key { display: inline-flex; gap: 5px; align-items: center; padding: 2px 8px; border: 1px solid var(--border); border-radius: 4px; cursor: pointer; user-select: none; color: var(--ink-secondary); }
.filter-key:has(input:checked) { color: var(--ink-primary); }
.filter-all, .filter-none { appearance: none; background: none; border: none; color: var(--ink-muted); font: inherit; font-size: 11px; cursor: pointer; padding: 0 4px; text-decoration: underline; }
.filter-count { color: var(--ink-muted); font-size: 11px; margin: 6px 0 0; }
.conc-card[hidden], .view-run[hidden], .view-window[hidden] { display: none; }
.scope-global { margin: 0 0 12px; }
.phase-band { stroke: none; }
.phase-idle { fill: var(--ink-muted); fill-opacity: .10; }
.phase-warmup { fill: var(--slot-1); fill-opacity: .16; }
.phase-profile { fill: var(--slot-2); fill-opacity: .14; }
.phase-band.phase-other { fill-opacity: .05; }
.phase-band.phase-focus.phase-warmup { fill-opacity: .28; }
.phase-band.phase-focus.phase-profile { fill-opacity: .24; }
.phase-legend { margin: 0 0 8px; }
.phase-key { cursor: default; }
.phase-swatch { width: 14px; height: 10px; border-radius: 2px; }
.phase-key.phase-idle .phase-swatch { background: var(--ink-muted); opacity: .45; }
.phase-key.phase-warmup .phase-swatch { background: var(--slot-1); opacity: .6; }
.phase-key.phase-profile .phase-swatch { background: var(--slot-2); opacity: .6; }
.scope-controls { display: flex; gap: 20px; align-items: center; flex-wrap: wrap; margin: 0 0 4px; }
.scope-controls select { font: inherit; font-size: 12px; padding: 2px 6px; background: var(--surface); color: var(--ink-primary); border: 1px solid var(--border); border-radius: 4px; }
.baseline-note { color: var(--ink-muted); font-size: 11px; margin: 6px 0 0; min-height: 1em; }
.pareto-point { stroke: var(--surface); stroke-width: 1.5; cursor: pointer; }
.pareto-point.selected { stroke: var(--ink-primary); stroke-width: 2.5; }
.pareto-point.dim { opacity: .18; }
.pareto-tooltip { transform: translate(12px, 12px); }
.chart-wrap .tooltip { max-height: 200px; overflow: hidden; }
.pareto-tooltip .t-row { justify-content: space-between; gap: 16px; }
.pareto-tooltip .t-title { font-weight: 600; margin-bottom: 4px; }
.pareto-inspect h3 { margin-top: 0; }
.pareto-panel-title { font-weight: 600; font-size: 13px; margin: 0 0 8px; color: var(--ink-secondary); }
.pareto-panel { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; font-size: 13px; }
.pareto-panel > div { padding: 7px 10px; border-right: 1px solid var(--grid); border-bottom: 1px solid var(--grid); min-width: 0; overflow-wrap: anywhere; }
.pareto-panel .stat-label { color: var(--ink-muted); font-size: 11px; margin: 0 0 2px; }
.pareto-panel .stat-value { font-weight: 600; font-variant-numeric: tabular-nums; margin: 0; }
.node-power-root { position: relative; }
svg.node-power-svg { width: 100%; display: block; overflow: visible; }
.node-power-tooltip .t-row { justify-content: space-between; gap: 16px; }
.node-power-tooltip .t-title { font-weight: 600; margin-bottom: 4px; }
.pareto-warnings { color: var(--slot-1); font-size: 12px; margin-top: 8px; }
.pareto-panel .pareto-empty { color: var(--ink-muted); grid-column: 1 / -1; }
"""
_CSS = _CSS.replace("__LIGHT_SLOT_VARS__", _LIGHT_SLOT_VARS).replace("__DARK_SLOT_VARS__", _DARK_SLOT_VARS)

_JS = """
function fmtT(sec) {
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60), s = sec % 60;
  return m + ":" + String(s).padStart(2, "0");
}

function fmtNum(v, decimals) {
  if (v === null || v === undefined || !isFinite(v)) return "n/a";
  if (decimals === undefined) decimals = Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : Math.abs(v) >= 1 ? 2 : 4;
  return v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

// Axis labels for big magnitudes: 950, 1.2k, 48k, 1.5M. Tooltips keep full values.
function fmtAxis(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toLocaleString(undefined, { maximumFractionDigits: a >= 1e7 ? 0 : 1 }) + "M";
  if (a >= 1e3) return (v / 1e3).toLocaleString(undefined, { maximumFractionDigits: a >= 1e4 ? 0 : 1 }) + "k";
  return v.toLocaleString(undefined, { maximumFractionDigits: a >= 100 ? 0 : a >= 10 ? 1 : 2 });
}

// Round tick positions for a linear axis: step is 1/2/2.5/5 x 10^n.
function linearTicks(max, n) {
  const rough = max / n, mag = Math.pow(10, Math.floor(Math.log10(rough || 1)));
  const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => s >= rough) || mag;
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(v);
  return out;
}

function nearestIndex(arr, target) {
  let lo = 0, hi = arr.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < target) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && Math.abs(arr[lo - 1] - target) <= Math.abs(arr[lo] - target)) return lo - 1;
  return lo;
}

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs || {}).forEach(([k, v]) => el.setAttribute(k, v));
  return el;
}

const STROKE_PATTERNS = ["", "5 4", "1.5 3"];
const MAX_END_LABELS = 4;
let clipCounter = 0;

function lowerBound(arr, target) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < target) lo = mid + 1; else hi = mid; }
  return lo;
}

// A chart group is one or more stacked time-series charts (GPU over CPU) that share
// an x domain: hovering any one drives the crosshair + tooltip on all of them, and a
// drag on any one zooms all of them. Double-click resets the zoom.
function initChartGroup(group) {
  // Shared series: copy from the source group (same run) instead of re-parsing a
  // second embedded copy. Sources are initialised first (document order).
  let sourceSubs = null;
  if (group.dataset.source) {
    const src = document.querySelector('.chart-group[data-source-id="' + group.dataset.source + '"]');
    if (src && src.__subsData) sourceSubs = src.__subsData;
  }
  const charts = [...group.querySelectorAll(".chart-sub")].map((sub, i) => ({
    sub,
    unit: sub.dataset.unit === undefined ? "W" : sub.dataset.unit,
    data: sub.dataset.series ? JSON.parse(sub.dataset.series) : (sourceSubs ? sourceSubs[i] : []),
    svg: sub.querySelector("svg.chart"),
    overlay: sub.querySelector(".overlay"),
    crosshair: sub.querySelector(".crosshair"),
    band: sub.querySelector(".zoom-band"),
    tooltip: sub.querySelector(".tooltip"),
    hidden: new Set(),
    lineEls: [],
    scale: "linear",
  })).filter(c => c.data.length);
  if (!charts.length) return;
  if (group.dataset.sourceId) group.__subsData = charts.map(c => c.data);
  const hint = group.querySelector(".zoom-hint");
  const phases = group.dataset.phases ? JSON.parse(group.dataset.phases) : [];
  let focus = null;  // {bench, conc} of the selected Pareto point, if any

  // The viewBox tracks the rendered pixel width (1 unit = 1 px) so text and
  // strokes never stretch; only the plot area widens with the window.
  const H = 220, padL = 46, padR = 12, padT = 10, padB = 22;
  const plotH = H - padT - padB;
  let W = 900, plotW = W - padL - padR;
  function layout() {
    W = Math.max(400, Math.round(charts[0].svg.getBoundingClientRect().width || 900));
    plotW = W - padL - padR;
    charts.forEach(c => {
      c.svg.setAttribute("viewBox", "0 0 " + W + " " + H);
      c.overlay.setAttribute("width", plotW);
      c.clipRect.setAttribute("width", plotW);
    });
  }

  let fullMin = Infinity, fullMax = -Infinity;
  charts.forEach(c => c.data.forEach(s => s.t.forEach(v => { if (v < fullMin) fullMin = v; if (v > fullMax) fullMax = v; })));
  if (!isFinite(fullMin)) return;
  let tMin = fullMin, tMax = fullMax;
  const x = t => padL + (tMax > tMin ? (t - tMin) / (tMax - tMin) : 0) * plotW;

  charts.forEach(c => {
    c.overlay.setAttribute("x", padL); c.overlay.setAttribute("y", padT);
    c.overlay.setAttribute("width", plotW); c.overlay.setAttribute("height", plotH);
    c.crosshair.setAttribute("y1", padT); c.crosshair.setAttribute("y2", padT + plotH);
    c.band.setAttribute("y", padT); c.band.setAttribute("height", plotH);
    const clipId = "clip-" + (clipCounter++);
    const clip = svgEl("clipPath", { id: clipId });
    c.clipRect = svgEl("rect", { x: padL, y: 0, width: plotW, height: H });
    clip.appendChild(c.clipRect);
    c.svg.insertBefore(clip, c.overlay);
    c.gGrid = svgEl("g"); c.gLines = svgEl("g", { "clip-path": "url(#" + clipId + ")" }); c.gLabels = svgEl("g");
    c.svg.insertBefore(c.gGrid, c.overlay); c.svg.insertBefore(c.gLines, c.overlay); c.svg.appendChild(c.gLabels);
  });

  charts.forEach(c => {
    c.sub.querySelectorAll(".yscale-btn").forEach(btn => btn.addEventListener("click", () => {
      c.scale = btn.dataset.scale;
      c.sub.querySelectorAll(".yscale-btn").forEach(b => b.classList.toggle("on", b === btn));
      draw();
    }));
  });

  function draw() {
    const zoomed = tMin > fullMin || tMax < fullMax;
    if (hint) hint.textContent = zoomed ? "zoomed \u00b7 double-click for the whole run" : "drag to zoom";
    charts.forEach(c => {
      c.gGrid.innerHTML = ""; c.gLines.innerHTML = ""; c.gLabels.innerHTML = "";
      phases.forEach(b => {
        const a = Math.max(b.t0, tMin), z = Math.min(b.t1, tMax);
        if (z <= a) return;
        let cls = "phase-band phase-" + b.kind;
        if (focus && b.kind !== "idle") cls += (b.bench === focus.bench && b.conc === focus.conc) ? " phase-focus" : " phase-other";
        const rect = svgEl("rect", { class: cls, x: x(a), y: padT, width: x(z) - x(a), height: plotH });
        const title = svgEl("title"); title.textContent = b.label + " (" + fmtT(b.t0) + " \u2013 " + fmtT(b.t1) + ")";
        rect.appendChild(title);
        c.gGrid.appendChild(rect);
      });
      // y scale: max over the visible time range of the *visible* series, so hiding a
      // role or host (or zooming) rescales to what's left. Falls back to every
      // series when all are hidden so the axis doesn't collapse.
      let wMax = 0, wMinPos = Infinity;
      const anyVisible = c.data.some((s, i) => !c.hidden.has(i));
      c.data.forEach((s, i) => {
        if (anyVisible && c.hidden.has(i)) return;
        const lo = lowerBound(s.t, tMin), hi = lowerBound(s.t, tMax + 1e-9);
        for (let j = lo; j < hi; j++) { const v = s.w[j]; if (v > wMax) wMax = v; if (v > 0 && v < wMinPos) wMinPos = v; }
      });
      wMax = wMax <= 0 ? 1 : wMax * 1.08;
      const isLog = c.scale === "log" && isFinite(wMinPos);
      // Log floor: one decade below the smallest positive value, clamped so zeros and
      // gaps still land on the axis instead of at -infinity.
      const logLo = isLog ? Math.pow(10, Math.floor(Math.log10(wMinPos))) : 0;
      const lgLo = isLog ? Math.log10(logLo) : 0, lgSpan = isLog ? Math.log10(wMax) - lgLo || 1 : 1;
      const y = isLog
        ? w => padT + plotH - (w <= logLo ? 0 : (Math.log10(w) - lgLo) / lgSpan) * plotH
        : w => padT + plotH - (w / wMax) * plotH;

      let ticks;
      if (isLog) {
        ticks = [];
        for (let e = Math.ceil(lgLo); Math.pow(10, e) <= wMax; e++) ticks.push(Math.pow(10, e));
        if (!ticks.length || ticks[0] > logLo) ticks.unshift(logLo);
        if (ticks.length <= 2) [2, 5].forEach(m => { const v = m * logLo; if (v < wMax) ticks.push(v); const v2 = m * logLo * 10; if (v2 < wMax) ticks.push(v2); });
        ticks.sort((a, b) => a - b);
      } else ticks = linearTicks(wMax, 4);
      ticks.forEach(v => {
        const gy = y(v);
        c.gGrid.appendChild(svgEl("line", { class: "gridline", x1: padL, x2: W - padR, y1: gy, y2: gy }));
        const label = svgEl("text", { class: "axis-text", x: padL - 6, y: gy + 3, "text-anchor": "end" });
        label.textContent = fmtAxis(v);
        c.gLabels.appendChild(label);
      });
      [0, 0.25, 0.5, 0.75, 1].forEach(f => {
        const gx = padL + f * plotW;
        if (f > 0 && f < 1) c.gGrid.appendChild(svgEl("line", { class: "gridline", x1: gx, x2: gx, y1: padT, y2: padT + plotH }));
        const label = svgEl("text", { class: "axis-text", x: gx, y: H - 4,
          "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle" });
        label.textContent = fmtT(tMin + f * (tMax - tMin));
        c.gLabels.appendChild(label);
      });

      c.lineEls = c.data.map((s, i) => {
        // One point either side of the range keeps lines continuous at the edges; the clip hides the overshoot.
        let lo = Math.max(0, lowerBound(s.t, tMin) - 1), hi = Math.min(s.t.length, lowerBound(s.t, tMax + 1e-9) + 1);
        if (hi <= lo) return [];
        const pts = [];
        for (let j = lo; j < hi; j++) pts.push(x(s.t[j]) + "," + y(s.w[j]));
        const poly = svgEl("polyline", { points: pts.join(" "), fill: "none", stroke: s.color, "stroke-width": "1.5",
          "stroke-linejoin": "round", "stroke-linecap": "round" });
        if (STROKE_PATTERNS[s.pattern]) poly.setAttribute("stroke-dasharray", STROKE_PATTERNS[s.pattern]);
        c.gLines.appendChild(poly);
        const els = [poly];
        if (!zoomed) {
          const lastT = s.t[s.t.length - 1], lastW = s.w[s.w.length - 1];
          const dot = svgEl("circle", { cx: x(lastT), cy: y(lastW), r: 3, fill: s.color, stroke: "var(--surface)", "stroke-width": "1.5" });
          c.gLines.appendChild(dot); els.push(dot);
          if (c.data.length > 1 && c.data.length <= MAX_END_LABELS) {  // a lone series is already named by its title
            const label = svgEl("text", { class: "end-label", x: Math.min(x(lastT) + 6, W - padR - 2), y: y(lastW) + 3 });
            label.textContent = s.label;
            c.gLabels.appendChild(label); els.push(label);
          }
        }
        if (c.hidden.has(i)) els.forEach(el => { el.style.display = "none"; });
        return els;
      });
    });
  }

  function toT(clientX, c) {
    const rect = c.svg.getBoundingClientRect();
    const px = ((clientX - rect.left) / rect.width) * W;
    return Math.min(tMax, Math.max(tMin, tMin + ((px - padL) / plotW) * (tMax - tMin)));
  }

  function showAt(targetT) {
    charts.forEach(c => {
      const cx = x(targetT);
      c.crosshair.style.opacity = 1;
      c.crosshair.setAttribute("x1", cx); c.crosshair.setAttribute("x2", cx);
      let snappedT = targetT;
      const rows = [];
      c.data.forEach((s, i) => {
        if (c.hidden.has(i)) return;
        const idx = nearestIndex(s.t, targetT);
        snappedT = s.t[idx];
        rows.push({ label: s.label, color: s.color, value: s.w[idx] });
      });
      const tip = c.tooltip;
      tip.innerHTML = "";
      const timeEl = document.createElement("div");
      timeEl.className = "t-time";
      timeEl.textContent = fmtT(snappedT);
      tip.appendChild(timeEl);
      rows.forEach(r => {
        const row = document.createElement("div"); row.className = "t-row";
        const key = document.createElement("span"); key.className = "t-key"; key.style.background = r.color;
        const label = document.createElement("span"); label.textContent = r.label;
        const val = document.createElement("span"); val.className = "t-val"; val.style.marginLeft = "auto";
        val.textContent = fmtNum(r.value, 1) + (c.unit ? " " + c.unit : "");
        row.appendChild(key); row.appendChild(label); row.appendChild(val);
        tip.appendChild(row);
      });
      tip.style.opacity = 1;
      // Keep the tooltip inside the chart: flip sides at the midpoint.
      const rect = c.svg.getBoundingClientRect();
      const localX = (cx / W) * rect.width;
      tip.style.top = "4px";
      tip.style.left = (localX < rect.width / 2 ? localX + 14 : localX - tip.offsetWidth - 14) + "px";
    });
  }
  function hideAll() {
    charts.forEach(c => { c.crosshair.style.opacity = 0; c.tooltip.style.opacity = 0; });
  }

  let drag = null;
  charts.forEach(c => {
    c.overlay.addEventListener("pointerdown", ev => {
      drag = { t0: toT(ev.clientX, c) };
      c.overlay.setPointerCapture(ev.pointerId);
    });
    c.overlay.addEventListener("pointermove", ev => {
      const t = toT(ev.clientX, c);
      if (drag) {
        const a = Math.min(drag.t0, t), b = Math.max(drag.t0, t);
        charts.forEach(cc => {
          cc.band.style.opacity = 1;
          cc.band.setAttribute("x", x(a)); cc.band.setAttribute("width", Math.max(0, x(b) - x(a)));
        });
      }
      showAt(t);
    });
    c.overlay.addEventListener("pointerup", ev => {
      if (!drag) return;
      const t1 = toT(ev.clientX, c);
      const a = Math.min(drag.t0, t1), b = Math.max(drag.t0, t1);
      drag = null;
      charts.forEach(cc => { cc.band.style.opacity = 0; });
      if ((b - a) / (tMax - tMin) > 0.01) { tMin = a; tMax = b; draw(); showAt(t1); }
    });
    c.overlay.addEventListener("pointerleave", () => { if (!drag) hideAll(); });
    c.overlay.addEventListener("dblclick", () => { tMin = fullMin; tMax = fullMax; draw(); });

    c.keys = [...c.sub.querySelectorAll(".legend-key")];
    c.keys.forEach((key, i) => {
      key.addEventListener("click", () => { setHidden(c, i, !c.hidden.has(i)); syncHostKeys(); syncRoleKeys(); rescale(); });
    });
  });

  function setHidden(c, i, off) {
    if (off) c.hidden.add(i); else c.hidden.delete(i);
    c.keys[i].classList.toggle("off", off);
    (c.lineEls[i] || []).forEach(el => { el.style.display = off ? "none" : ""; });
  }
  // Visibility changes rescale the y-axis; callers batch their setHidden calls and
  // redraw once.
  function rescale() { draw(); }

  // Host chips: one click hides/shows every device on that host in every chart of
  // the group. A chip reads "partial" when only some of its devices are hidden.
  const hostKeys = [...group.querySelectorAll(".host-key")];
  function hostDevices(host) {
    const out = [];
    charts.forEach(c => c.data.forEach((s, i) => { if (s.host === host) out.push([c, i]); }));
    return out;
  }
  function syncHostKeys() {
    hostKeys.forEach(key => {
      const devs = hostDevices(key.dataset.host);
      const hiddenN = devs.filter(([c, i]) => c.hidden.has(i)).length;
      key.classList.toggle("off", devs.length > 0 && hiddenN === devs.length);
      key.classList.toggle("partial", hiddenN > 0 && hiddenN < devs.length);
    });
  }
  hostKeys.forEach(key => {
    key.addEventListener("click", () => {
      const devs = hostDevices(key.dataset.host);
      const allOff = devs.every(([c, i]) => c.hidden.has(i));
      devs.forEach(([c, i]) => setHidden(c, i, !allOff));
      syncHostKeys(); syncRoleKeys(); rescale();
    });
  });

  // Role chips (prefill / decode ...): toggle every device tagged with that role.
  const roleKeys = [...group.querySelectorAll(".role-key")];
  function roleDevices(role) {
    const out = [];
    charts.forEach(c => c.data.forEach((s, i) => { if ((s.roles || []).includes(role)) out.push([c, i]); }));
    return out;
  }
  function syncRoleKeys() {
    roleKeys.forEach(key => {
      const devs = roleDevices(key.dataset.role);
      const hiddenN = devs.filter(([c, i]) => c.hidden.has(i)).length;
      key.classList.toggle("off", devs.length > 0 && hiddenN === devs.length);
      key.classList.toggle("partial", hiddenN > 0 && hiddenN < devs.length);
    });
  }
  roleKeys.forEach(key => {
    key.addEventListener("click", () => {
      const devs = roleDevices(key.dataset.role);
      const allOff = devs.every(([c, i]) => c.hidden.has(i));
      devs.forEach(([c, i]) => setHidden(c, i, !allOff));
      syncHostKeys(); syncRoleKeys(); rescale();
    });
  });

  // Called by the scatter when a point is selected: zoom to that concurrency's
  // warmup + profile span and emphasise its bands. Double-click still resets.
  group.focusPhase = (bench, conc) => {
    focus = { bench, conc };
    const mine = phases.filter(b => b.kind !== "idle" && b.bench === bench && b.conc === conc);
    if (mine.length) {
      const a = Math.min(...mine.map(b => b.t0)), z = Math.max(...mine.map(b => b.t1));
      const pad = (z - a) * 0.04;
      tMin = Math.max(fullMin, a - pad); tMax = Math.min(fullMax, z + pad);
    }
    layout(); draw();
  };

  if (group.dataset.focusBench) {
    group.focusPhase(group.dataset.focusBench, Number(group.dataset.focusConc));
  } else {
    layout();
    draw();
  }
  if (window.ResizeObserver) {
    let pending = null;
    new ResizeObserver(() => {
      if (pending) return;
      pending = requestAnimationFrame(() => { pending = null; layout(); draw(); });
    }).observe(charts[0].svg);
  }
}

document.querySelectorAll(".chart-group").forEach(initChartGroup);
document.querySelectorAll(".stats-toggle").forEach(t => {
  t.addEventListener("click", () => {
    const table = t.nextElementSibling;
    const open = table.classList.toggle("open");
    t.textContent = (open ? "hide" : "show") + " device stats table";
  });
});

document.querySelectorAll(".tabs").forEach(tabs => {
  const container = tabs.parentElement;
  tabs.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      tabs.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b === btn));
      container.querySelectorAll(".tab-panel").forEach(panel => {
        panel.hidden = panel.dataset.tabPanel !== btn.dataset.tab;
      });
    });
  });
});

// Per-node horizontal stacked bars of window-average power for the selected point.
// Segments: GPU, then the CPU envelope -- split into recorded component rails plus
// the unaccounted remainder when rails exist, else one CPU segment.
const NODE_POWER_SEGMENTS = [
  { key: "gpu",      label: "GPU",              color: "hsl(158 60% 40%)" },
  { key: "cpu_rail", label: "CPU rail",         color: "hsl(35 85% 50%)" },
  { key: "soc",      label: "SoC / SysIO",      color: "hsl(45 90% 62%)" },
  { key: "dram",     label: "DRAM",             color: "hsl(20 80% 55%)" },
  { key: "cpu_rest", label: "CPU envelope (other)", color: "hsl(35 40% 72%)" },
  { key: "cpu",      label: "CPU (socket total)", color: "hsl(35 85% 50%)" },
];
function renderNodePower(card, point) {
  const svg = card.querySelector("svg.node-power-svg");
  const legend = card.querySelector(".node-power-legend");
  const sub = card.querySelector(".node-power-sub");
  const tip = card.querySelector(".node-power-tooltip");
  const nodes = point.node_power || [];
  svg.innerHTML = ""; legend.innerHTML = "";
  card.hidden = !nodes.length;
  if (!nodes.length) return;
  sub.textContent = point.label;
  // Coverage warnings (e.g. "CPU power missing ...") so a GPU-only bar isn't read as
  // the node's whole draw.
  const notices = card.querySelector(".node-power-notices");
  const powerWarnings = (point.warnings || []).filter(w => /power (missing|not collected)/i.test(w));
  notices.innerHTML = "";
  powerWarnings.forEach(w => { const d = document.createElement("div"); d.textContent = "\u26a0 " + w; notices.appendChild(d); });
  notices.hidden = !powerWarnings.length;

  const anyRails = nodes.some(n => n.cpu_rails_w && Object.keys(n.cpu_rails_w).length);
  const rows = nodes.map(n => {
    const segs = [];
    if (n.gpu_w != null) segs.push(["gpu", n.gpu_w]);
    if (n.cpu_w != null) {
      if (anyRails && n.cpu_rails_w && Object.keys(n.cpu_rails_w).length) {
        let acc = 0;
        ["cpu_rail", "soc", "dram"].forEach(k => { if (n.cpu_rails_w[k] != null) { segs.push([k, n.cpu_rails_w[k]]); acc += n.cpu_rails_w[k]; } });
        if (n.cpu_w - acc > 0.5) segs.push(["cpu_rest", n.cpu_w - acc]);
      } else segs.push(["cpu", n.cpu_w]);
    }
    return { host: n.hostname, segs, total: segs.reduce((a, [, v]) => a + v, 0) };
  });
  const used = new Set(rows.flatMap(r => r.segs.map(([k]) => k)));
  NODE_POWER_SEGMENTS.filter(sg => used.has(sg.key)).forEach(sg => {
    const key = document.createElement("span"); key.className = "legend-key"; key.style.cursor = "default";
    const sw = document.createElement("span"); sw.className = "legend-swatch"; sw.style.background = sg.color; sw.style.height = "10px";
    key.appendChild(sw); key.appendChild(document.createTextNode(sg.label)); legend.appendChild(key);
  });

  const W = Math.max(500, Math.round(svg.getBoundingClientRect().width || 900));
  const rowH = 22, padL = 110, padR = 60, padT = 8, padB = 28;
  const H = padT + rows.length * rowH + padB;
  svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.style.height = H + "px";
  const maxW = Math.max(...rows.map(r => r.total)) * 1.05 || 1;
  const x = v => padL + (v / maxW) * (W - padL - padR);
  const unit = maxW >= 2000 ? "kW" : "W", div = unit === "kW" ? 1000 : 1;
  const step = [1, 2, 2.5, 5, 10].map(sf => sf * Math.pow(10, Math.floor(Math.log10(maxW / 5)))).find(sf => sf >= maxW / 6) || 1;
  for (let v = 0; v <= maxW; v += step) {
    svg.appendChild(svgEl("line", { class: "gridline", x1: x(v), x2: x(v), y1: padT, y2: H - padB }));
    const t = svgEl("text", { class: "axis-text", x: x(v), y: H - padB + 14, "text-anchor": "middle" });
    t.textContent = fmtNum(v / div, unit === "kW" ? 1 : 0); svg.appendChild(t);
  }
  const xt = svgEl("text", { class: "axis-title", x: padL + (W - padL - padR) / 2, y: H - 4, "text-anchor": "middle" });
  xt.textContent = "Window-average power (" + unit + ")"; svg.appendChild(xt);

  rows.forEach((r, i) => {
    const y = padT + i * rowH + 3, h = rowH - 6;
    const lbl = svgEl("text", { class: "axis-text", x: padL - 8, y: y + h / 2 + 3, "text-anchor": "end" });
    lbl.textContent = r.host; svg.appendChild(lbl);
    let acc = 0;
    r.segs.forEach(([k, v]) => {
      const sg = NODE_POWER_SEGMENTS.find(sg => sg.key === k);
      const rect = svgEl("rect", { x: x(acc), y, width: Math.max(0, x(acc + v) - x(acc)), height: h, fill: sg.color, stroke: "var(--surface)", "stroke-width": 1 });
      rect.addEventListener("pointerenter", ev => {
        tip.innerHTML = "<div class='t-title'>" + r.host + "</div>";
        r.segs.forEach(([kk, vv]) => { const s2 = NODE_POWER_SEGMENTS.find(sg => sg.key === kk);
          tip.innerHTML += "<div class='t-row'><span><span class='t-key' style='display:inline-block;width:10px;height:10px;margin-right:6px;background:" + s2.color + "'></span>" + s2.label + "</span><span class='t-val'>" + fmtNum(vv, 0) + " W</span></div>"; });
        tip.innerHTML += "<div class='t-row'><span>Total</span><span class='t-val'>" + fmtNum(r.total, 0) + " W</span></div>";
        tip.style.opacity = 1;
        const rc = card.querySelector(".node-power-root").getBoundingClientRect();
        tip.style.left = (ev.clientX - rc.left + 12) + "px"; tip.style.top = (ev.clientY - rc.top + 12) + "px";
      });
      rect.addEventListener("pointerleave", () => { tip.style.opacity = 0; });
      svg.appendChild(rect); acc += v;
    });
    const val = svgEl("text", { class: "axis-text", x: x(r.total) + 6, y: y + h / 2 + 3 });
    val.textContent = fmtNum(r.total / div, unit === "kW" ? 2 : 0) + " " + unit; svg.appendChild(val);
  });
}

// Pareto axes: each option reads a metric off a point's `m` dict. `better` says
// which direction the frontier chases so the guide line connects nondominated points.
const PARETO_AXES = {
  output_tps:      { label: "Output tok/s",              key: "output_tps",      better: "max" },
  total_tps:       { label: "Total tok/s (in + out)",    key: "total_tps",       better: "max" },
  tps_per_gpu:     { label: "Output tok/s / GPU",        key: "tps_per_gpu",     better: "max" },
  total_tps_per_gpu: { label: "Total tok/s / GPU",       key: "total_tps_per_gpu", better: "max" },
  inv_tpot_p90:    { label: "1 / P90 TPOT (tok/s/user)", key: "inv_tpot_p90",    better: "max" },
  inv_tpot_p50:    { label: "1 / P50 TPOT (tok/s/user)", key: "inv_tpot_p50",    better: "max" },
  tpot_p90:        { label: "P90 TPOT (ms)",             key: "tpot_p90",        better: "min" },
  tps_per_gpu_w:   { label: "Output tok/s / GPU W",      key: "tps_per_gpu_w",   better: "max" },
  tps_per_total_w: { label: "Output tok/s / (GPU+CPU) W", key: "tps_per_total_w", better: "max" },
  gpu_w:           { label: "Total GPU power (W)",       key: "gpu_w",           better: "min" },
  gpu_w_per_gpu:   { label: "Power per GPU (W)",         key: "gpu_w_per_gpu",   better: "min" },
  cpu_w:           { label: "Total CPU power (W)",       key: "cpu_w",           better: "min" },
  total_w:         { label: "Total GPU+CPU power (W)",   key: "total_w",         better: "min" },
  concurrency:     { label: "Concurrency",               key: "concurrency",     better: "max" },
};

function initPareto(root) {
  const points = JSON.parse(root.dataset.points);
  if (!points.length) return;
  const card = root.closest(".pareto-card");
  const svg = root.querySelector("svg.pareto-svg");
  const tooltip = root.querySelector(".pareto-tooltip");
  const panel = card.querySelector(".pareto-panel");
  const panelTitle = card.querySelector(".pareto-panel-title");
  // Axis selects are optional: a fixed-axis scatter passes data-x / data-y instead.
  const xSel = card.querySelector("select[data-axis=x]") || { value: root.dataset.x };
  const ySel = card.querySelector("select[data-axis=y]") || { value: root.dataset.y };
  const drawFrontier = root.dataset.frontier !== "off";
  const modelSel = card.querySelector("select[data-model]");
  const modelOk = p => !modelSel || !modelSel.value || p.model === modelSel.value;
  // Baseline mode: Y is divided by the baseline run's Y at the same concurrency.
  const baseSel = card.querySelector("select[data-baseline]");
  const normalize = !!baseSel;
  const runLegend = card.querySelector(".run-legend");
  const W = 900, H = 520, padL = 64, padR = 20, padT = 14, padB = 44;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  // Legend, frontier lines and hide/show all work on the point's group (GPU type x model).
  const runs = [...new Set(points.map(p => p.group))];
  const hiddenRuns = new Set();
  let selected = 0;
  let plotted = [];   // indexes into points that have both metrics
  let xAxis, yAxis, xScale, yScale;

  const gGrid = svgEl("g"), gLines = svgEl("g"), gPoints = svgEl("g"), gAxes = svgEl("g");
  svg.appendChild(gGrid); svg.appendChild(gLines); svg.appendChild(gPoints); svg.appendChild(gAxes);

  runs.forEach(run => {
    const key = document.createElement("span");
    key.className = "run-key";
    const sw = document.createElement("span");
    sw.className = "run-swatch";
    sw.style.background = points.find(p => p.group === run).color;
    key.appendChild(sw);
    key.appendChild(document.createTextNode(run));
    key.dataset.group = run;
    key.addEventListener("click", () => {
      if (hiddenRuns.has(run)) hiddenRuns.delete(run); else hiddenRuns.add(run);
      key.classList.toggle("off", hiddenRuns.has(run));
      draw();
    });
    runLegend.appendChild(key);
  });

  function metric(p, axis) { const v = p.m[axis.key]; return v === null || v === undefined ? null : v; }
  function baselineFor(p) {
    if (!normalize) return null;
    return points.find(b => b.run === baseSel.value && b.m.concurrency === p.m.concurrency
      && b.bench === p.bench) || null;
  }
  function yVal(p) {
    const raw = metric(p, yAxis);
    if (!normalize) return raw;
    const b = baselineFor(p);
    if (raw === null || !b) return null;
    const bv = metric(b, yAxis);
    return bv === null || bv === 0 ? null : raw / bv;
  }
  const yLabel = () => normalize ? yAxis.label + " / baseline" : yAxis.label;

  function niceTicks(min, max, n) {
    const span = max - min || 1, rough = span / n, mag = Math.pow(10, Math.floor(Math.log10(rough)));
    const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => s >= rough) || mag;
    const decimals = Math.max(0, -Math.floor(Math.log10(step)) + (step / Math.pow(10, Math.floor(Math.log10(step))) % 1 ? 1 : 0));
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push([v, decimals]);
    return out;
  }

  function frontier(idxs) {
    // Sort by x ascending; walk keeping points not dominated on y (given each axis's direction).
    const xs = idxs.map(i => [i, metric(points[i], xAxis), yVal(points[i])]);
    xs.sort((a, b) => xAxis.better === "max" ? a[1] - b[1] : b[1] - a[1]);
    // Walk from best-x to worst-x: a point is on the frontier if its y beats every better-x point's y.
    const ordered = xs.reverse();
    const out = [];
    let bestY = null;
    ordered.forEach(([i, , y]) => {
      const beats = bestY === null || (yAxis.better === "max" ? y > bestY : y < bestY);
      if (beats) { out.push(i); bestY = y; }
    });
    return out.reverse();
  }

  function draw() {
    xAxis = PARETO_AXES[xSel.value]; yAxis = PARETO_AXES[ySel.value];
    plotted = points.map((p, i) => i).filter(i => modelOk(points[i]) && metric(points[i], xAxis) !== null && yVal(points[i]) !== null);
    // Legend chips for families outside the chosen model drop out with their points.
    runLegend.querySelectorAll(".run-key").forEach(key => {
      key.hidden = !points.some(p => p.group === key.dataset.group && modelOk(p));
    });
    // Keep the selection on a visible point.
    if (plotted.length && !plotted.includes(selected)) { selected = plotted[0]; renderPanel(points[selected]); }
    gGrid.innerHTML = ""; gLines.innerHTML = ""; gPoints.innerHTML = ""; gAxes.innerHTML = "";
    if (!plotted.length) return;

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    plotted.forEach(i => {
      const px = metric(points[i], xAxis), py = yVal(points[i]);
      if (px < xMin) xMin = px; if (px > xMax) xMax = px;
      if (py < yMin) yMin = py; if (py > yMax) yMax = py;
    });
    xMin = Math.min(0, xMin);
    if (normalize) { yMin = Math.min(yMin, 1); yMax = Math.max(yMax, 1); yMin -= (yMax - yMin) * 0.1 || 0.1; }
    else yMin = Math.min(0, yMin);
    xMax += (xMax - xMin) * 0.06 || 1; yMax += (yMax - yMin) * 0.08 || 1;
    xScale = v => padL + ((v - xMin) / (xMax - xMin)) * plotW;
    yScale = v => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    niceTicks(yMin, yMax, 6).forEach(([v, d]) => {
      if (normalize) d = Math.max(d, 2);
      const gy = yScale(v);
      gGrid.appendChild(svgEl("line", { class: "gridline", x1: padL, x2: W - padR, y1: gy, y2: gy }));
      const t = svgEl("text", { class: "axis-text", x: padL - 8, y: gy + 3, "text-anchor": "end" });
      t.textContent = fmtNum(v, d);
      gAxes.appendChild(t);
    });
    niceTicks(xMin, xMax, 6).forEach(([v, d]) => {
      const gx = xScale(v);
      gGrid.appendChild(svgEl("line", { class: "gridline", x1: gx, x2: gx, y1: padT, y2: padT + plotH }));
      const t = svgEl("text", { class: "axis-text", x: gx, y: padT + plotH + 14, "text-anchor": "middle" });
      t.textContent = fmtNum(v, d);
      gAxes.appendChild(t);
    });
    const xt = svgEl("text", { class: "axis-title", x: padL + plotW / 2, y: H - 6, "text-anchor": "middle" });
    xt.textContent = xAxis.label + " \\u2192";
    gAxes.appendChild(xt);
    const yt = svgEl("text", { class: "axis-title", x: 14, y: padT + plotH / 2, "text-anchor": "middle",
      transform: "rotate(-90 14 " + (padT + plotH / 2) + ")" });
    yt.textContent = yLabel() + " \\u2191";
    gAxes.appendChild(yt);
    if (normalize) {
      gGrid.appendChild(svgEl("line", { class: "baseline-ref", x1: padL, x2: W - padR, y1: yScale(1), y2: yScale(1) }));
      const bl = svgEl("text", { class: "axis-text", x: W - padR - 4, y: yScale(1) - 4, "text-anchor": "end" });
      bl.textContent = "baseline = 1.0";
      gAxes.appendChild(bl);
    }
    const note = card.querySelector(".baseline-note");
    if (note) {
      const dropped = points.filter(p => metric(p, xAxis) !== null && metric(p, yAxis) !== null && yVal(p) === null).length;
      note.textContent = dropped ? dropped + " point(s) hidden: no baseline measurement at the same concurrency." : "";
    }

    if (drawFrontier) runs.forEach(run => {
      if (hiddenRuns.has(run)) return;
      const idxs = plotted.filter(i => points[i].group === run);
      const front = frontier(idxs);
      if (front.length >= 2) {
        const d = front.map(i => xScale(metric(points[i], xAxis)) + "," + yScale(yVal(points[i]))).join(" ");
        gLines.appendChild(svgEl("polyline", { class: "pareto-frontier", points: d, stroke: points[idxs[0]].color }));
      }
    });

    plotted.forEach(i => {
      const p = points[i];
      const c = svgEl("circle", { class: "pareto-point", cx: xScale(metric(p, xAxis)), cy: yScale(yVal(p)),
        r: i === selected ? 8 : 6, fill: p.color });
      if (hiddenRuns.has(p.group)) c.classList.add("dim");
      if (i === selected) c.classList.add("selected");
      c.addEventListener("click", () => select(i));
      c.addEventListener("pointerenter", (ev) => showTip(p, ev));
      c.addEventListener("pointermove", (ev) => moveTip(ev));
      c.addEventListener("pointerleave", () => { tooltip.style.opacity = 0; });
      gPoints.appendChild(c);
    });
  }

  function showTip(p, ev) {
    tooltip.innerHTML = "";
    const title = document.createElement("div");
    title.className = "t-title";
    title.textContent = p.label;
    tooltip.appendChild(title);
    const rows = [[xAxis.label, fmtNum(metric(p, xAxis))], [yLabel(), normalize ? fmtNum(yVal(p), 3) : fmtNum(yVal(p))]];
    if (normalize) rows.push([yAxis.label + " (raw)", fmtNum(metric(p, yAxis))]);
    p.hover.forEach(([k, v]) => rows.push([k, v]));
    rows.forEach(([k, v]) => {
      const row = document.createElement("div");
      row.className = "t-row";
      const kEl = document.createElement("span"); kEl.textContent = k; kEl.style.color = "var(--ink-muted)";
      const vEl = document.createElement("span"); vEl.className = "t-val"; vEl.textContent = v;
      row.appendChild(kEl); row.appendChild(vEl);
      tooltip.appendChild(row);
    });
    tooltip.style.opacity = 1;
    moveTip(ev);
  }
  function moveTip(ev) {
    const rect = root.getBoundingClientRect();
    let left = ev.clientX - rect.left, top = ev.clientY - rect.top;
    if (left > rect.width * 0.6) left -= tooltip.offsetWidth + 24;
    if (top > rect.height * 0.6) top -= tooltip.offsetHeight + 24;
    tooltip.style.left = left + "px"; tooltip.style.top = top + "px";
  }

  function renderPanel(p) {
    if (panelTitle) panelTitle.textContent = p.label;
    panel.innerHTML = "";
    const warnBox = card.querySelector(".pareto-warnings");
    if (warnBox) { warnBox.innerHTML = ""; (p.warnings || []).forEach(w => { const d = document.createElement("div"); d.textContent = "\u26a0 " + w; warnBox.appendChild(d); }); }
    p.fields.forEach(([label, value]) => {
      const cell = document.createElement("div");
      const l = document.createElement("p"); l.className = "stat-label"; l.textContent = label;
      const v = document.createElement("p"); v.className = "stat-value"; v.textContent = value;
      cell.appendChild(l); cell.appendChild(v);
      panel.appendChild(cell);
    });
  }

  function select(i) {
    selected = i;
    renderPanel(points[i]);
    if (root.dataset.drivesCharts !== "off") {
      document.querySelectorAll(".node-power-card").forEach(card => renderNodePower(card, points[i]));
      document.querySelectorAll(".power-scope-card").forEach(card => {
        card.dataset.pointLabel = points[i].label;
        updateScopeTitle(card);
      });
      const id = points[i].id;
      document.querySelectorAll(".point-charts").forEach(el => { el.hidden = el.dataset.pointId !== id; });
      const run = points[i].run;
      document.querySelectorAll(".power-chart-run").forEach(el => {
        el.hidden = el.dataset.paretoRun !== run;
        if (!el.hidden) el.querySelectorAll(".chart-group").forEach(g => {
          if (g.focusPhase) g.focusPhase(points[i].bench, points[i].m.concurrency);
        });
      });
    }
    draw();
  }

  if (modelSel) modelSel.addEventListener("change", () => { draw(); if (plotted.length) select(selected); });
  if (xSel.addEventListener) xSel.addEventListener("change", draw);
  if (ySel.addEventListener) ySel.addEventListener("change", draw);
  if (baseSel) baseSel.addEventListener("change", draw);
  select(0);
}

document.querySelectorAll(".pareto-root").forEach(initPareto);
document.querySelectorAll(".node-power-card[data-point]").forEach(card => renderNodePower(card, JSON.parse(card.dataset.point)));

// Data table: whole-run (zoomed) vs profile-window view for every per-concurrency card.
document.querySelectorAll(".view-window-only").forEach(box => {
  const cards = [...box.closest(".tab-panel, body").querySelectorAll(".conc-card")];
  box.addEventListener("change", () => {
    cards.forEach(card => {
      const win = card.querySelector(".view-window"), run = card.querySelector(".view-run");
      const hasWin = win && win.children.length > 0;
      win.hidden = !(box.checked && hasWin);
      run.hidden = box.checked && hasWin;
    });
  });
});

// Data table: run x concurrency checkbox filter over the per-concurrency chart cards.
document.querySelectorAll(".chart-filter").forEach(filter => {
  const cards = [...filter.parentElement.querySelectorAll(".conc-card")];
  const boxes = [...filter.querySelectorAll("input[type=checkbox]")];
  const count = filter.querySelector(".filter-count");
  const apply = () => {
    const on = { run: new Set(), conc: new Set() };
    boxes.forEach(b => { if (b.checked) on[b.dataset.filter].add(b.value); });
    let shown = 0;
    cards.forEach(card => {
      const show = on.run.has(card.dataset.run) && on.conc.has(card.dataset.conc);
      card.hidden = !show;
      if (show) shown++;
    });
    count.textContent = shown + " of " + cards.length + " charts shown";
  };
  boxes.forEach(b => b.addEventListener("change", apply));
  filter.querySelectorAll(".filter-all, .filter-none").forEach(btn => {
    btn.addEventListener("click", () => {
      const val = btn.classList.contains("filter-all");
      boxes.filter(b => b.dataset.filter === btn.dataset.filter).forEach(b => { b.checked = val; });
      apply();
    });
  });
  apply();
});

// Scope card title: "Power over time \\u2014 <run> \\u00b7 aiperf c=N (profile | whole run)".
// The scatter's select() stores the point label on the card; the checkbox reads it.
function updateScopeTitle(card) {
  const title = card.querySelector("h3");
  const box = card.querySelector(".scope-whole-run");
  const whole = !!(box && box.checked);
  const label = card.dataset.pointLabel;
  if (!label) { title.textContent = "Power over time"; return; }
  title.textContent = "Power over time \\u2014 " + label + (whole ? " (whole run)" : " (profile)");
}
document.querySelectorAll(".power-scope-card").forEach(card => {
  const box = card.querySelector(".scope-whole-run");
  if (!box) return;
  box.addEventListener("change", () => {
    card.querySelector(".scope-window").hidden = box.checked;
    card.querySelector(".scope-run").hidden = !box.checked;
    updateScopeTitle(card);
  });
});
"""


def _fmt(value: float | None, decimals: int = 2) -> str:
    return "n/a" if value is None else f"{value:,.{decimals}f}"


_SUMMARY_TABLE_HEADER = (
    "<tr><th>Run</th><th>Output tok/s</th><th>Tok/s/GPU</th><th>TPOT p50 (ms)</th><th>TPOT p90 (ms)</th>"
    "<th>Total GPU W</th><th>W / GPU</th><th>Total CPU W</th><th>GPU-only tok/s/W</th><th>CPU-only tok/s/W</th>"
    "<th>Combined tok/s/W</th></tr>"
)


def _summary_rows_html(
    reports: list[dict], *, run_label: str | None = None, coverage_warnings: list[str] | None = None
) -> str:
    """``<tr>``s for one run's per-concurrency reports.

    ``run_label`` prefixes the Run cell -- set only in the combined multi-directory
    report, where rows from different runs are interleaved in one table and need
    to say which run they came from. ``coverage_warnings`` (missing power legs)
    put a ⚠ marker on the Run cell with the explanation as its tooltip.
    """
    rows = []
    warn_mark = ""
    if coverage_warnings:
        warn_mark = (
            f' <span class="row-warn" title="{html.escape(" ".join(coverage_warnings), quote=True)}">\u26a0</span>'
        )
    for r in reports:
        w = r
        ppw = r["perf_per_watt"]
        run_cell = (
            f"{run_label} · {r['benchmark_type']} c={r['concurrency']}"
            if run_label
            else (f"{r['benchmark_type']} c={r['concurrency']}")
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(run_cell)}{warn_mark}</td>"
            f"<td>{_fmt(ppw['output_tokens_per_second'])}</td>"
            f"<td>{_fmt(ppw['output_tokens_per_second_per_gpu'])} ({ppw['num_gpus']} gpu)</td>"
            f"<td>{_fmt(w['tpot_p50_ms'])}</td>"
            f"<td>{_fmt(w['tpot_p90_ms'])}</td>"
            f"<td>{_fmt(ppw['gpu_avg_power_w'], 0)}</td>"
            f"<td>{_fmt(_per_gpu(ppw['gpu_avg_power_w'], ppw['num_gpus']), 0)}</td>"
            f"<td>{_fmt(ppw['cpu_avg_power_w'])}</td>"
            f"<td>{_fmt(ppw['output_tokens_per_second_per_gpu_watt'], 4)}</td>"
            f"<td>{_fmt(ppw['output_tokens_per_second_per_cpu_watt'], 4)}</td>"
            f"<td>{_fmt(ppw['output_tokens_per_second_per_combined_watt'], 4)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _summary_table_html(reports: list[dict]) -> str:
    return f"<table><thead>{_SUMMARY_TABLE_HEADER}</thead><tbody>{_summary_rows_html(reports)}</tbody></table>"


def _stats_table_html(series: list[dict], unit: str = "W") -> str:
    rows = []
    for s in series:
        st = s["stats"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(s['label'])}</td>"
            f"<td>{_fmt(st['mean'])}</td><td>{_fmt(st['min'])}</td><td>{_fmt(st['p50'])}</td>"
            f"<td>{_fmt(st['p95'])}</td><td>{_fmt(st['max'])}</td>"
            "</tr>"
        )
    u = f" {unit}" if unit else ""
    header = f"<tr><th>Series</th><th>Mean{u}</th><th>Min{u}</th><th>P50{u}</th><th>P95{u}</th><th>Max{u}</th></tr>"
    return f'<table class="stats-table"><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>'


def _legend_html(series: list[dict]) -> str:
    keys = []
    for s in series:
        style = f"background: {s['color']}"
        if s["pattern"]:
            style += (
                f"; background-image: repeating-linear-gradient(90deg, {s['color']} 0 3px, transparent 3px 6px)"
                "; background-color: transparent"
            )
        keys.append(
            f'<span class="legend-key" title="click to hide/show"><span class="legend-swatch" style="{style}"></span>'
            f"{html.escape(s['label'])}</span>"
        )
    return f'<div class="legend">{"".join(keys)}</div>'


# One colour per run on the Pareto scatter (hue cycles through the host palette so
# a run's points and frontier line share a colour family).
def _run_color(run_position: int) -> str:
    return f"hsl({_HOST_HUES[run_position % len(_HOST_HUES)]} 70% 52%)"


def _inverse_ms(ms: float | None) -> float | None:
    """``1 / TPOT`` in tokens/s/user, the per-user speed a latency-vs-throughput Pareto wants."""
    return None if not ms else 1000.0 / ms


def _point_id(run_label: str | None, report: dict) -> str:
    return f"{run_label or ''}::{report['benchmark_type']}::c{report['concurrency']}"


def _family_label(gpu_type: str | None, model: str | None) -> str:
    """Display key for a (GPU type, model) frontier group; ``"unknown"`` parts stay
    visible rather than silently merging unrelated runs."""
    return f"{gpu_type or 'unknown gpu'} · {model or 'unknown model'}"


def _per_gpu(total: float | None, num_gpus: int | None) -> float | None:
    return None if total is None or not num_gpus else total / num_gpus


def _cpu_sensor_summary(report: dict) -> str:
    """Short provenance line for the inspect panel: which ACPI/DCGM channel the CPU
    socket energy was integrated from, e.g. ``Grace Power Socket N (ACPI envelope)``;
    ``n/a`` when the run has no CPU leg."""
    sensors = report.get("cpu_sensors") or []
    if not sensors:
        return "n/a"
    names = sorted({re.sub(r"\s*\d+\s*$", " N", p["sensor"]).strip() for p in sensors if p["sensor"]})
    unused = sorted({re.sub(r"\s*\d+\s*$", " N", o).strip() for p in sensors for o in p.get("other_sensors", ())})
    text = ", ".join(names) if names else "single channel"
    if unused:
        text += f" (ignored: {', '.join(unused)})"
    return text


def _pareto_points(
    reports: list[dict],
    *,
    run_label: str | None = None,
    run_position: int = 0,
    group: str | None = None,
    group_position: int | None = None,
    model: str | None = None,
    coverage_warnings: list[str] | None = None,
) -> list[dict]:
    """Scatter points for the Pareto view, one per summary-table row.

    Each point carries ``m``, a dict of every metric the axis selectors can plot
    (``PARETO_AXES`` in ``_JS`` reads these by key; ``None`` means "can't plot on
    that axis"), ``hover`` rows for the tooltip and ``fields`` for the inspect
    panel. Rows with no resolvable GPU count (``output_tokens_per_second_per_gpu``
    is ``None``) can't be placed on the default axes and are dropped.

    ``run`` and ``id`` are the keys that link a point back to its run's whole-run
    chart group and its own measured-window chart group -- see ``_pareto_tab_html``.

    ``group`` (default: the run) is the frontier/colour family -- normally the run's
    ``(gpu_type, model)`` from ``_family_label`` -- so every run on the same silicon
    serving the same model shares one Pareto line. ``group_position`` picks its hue.
    """
    group = group if group is not None else (run_label or "")
    colour_position = group_position if group_position is not None else run_position
    points = []
    for r in reports:
        ppw = r["perf_per_watt"]
        output_tps, tps_per_gpu = ppw["output_tokens_per_second"], ppw["output_tokens_per_second_per_gpu"]
        if output_tps is None or tps_per_gpu is None:
            continue
        run_cell = f"{r['benchmark_type']} c={r['concurrency']}"
        label = f"{run_label} · {run_cell}" if run_label else run_cell
        num_gpus = ppw["num_gpus"]
        total_tps = ppw["total_tokens_per_second"]
        combined_w = ppw["combined_avg_power_w"]
        duration = r["timing"]["computed"]["duration_seconds"]
        points.append(
            {
                "id": _point_id(run_label, r),
                "label": label,
                "bench": r["benchmark_type"],
                "run": run_label or "",
                "group": group,
                "model": model or "",
                "color": _run_color(colour_position),
                "m": {
                    "output_tps": output_tps,
                    "total_tps": total_tps,
                    "tps_per_gpu": tps_per_gpu,
                    "total_tps_per_gpu": None if total_tps is None or not num_gpus else total_tps / num_gpus,
                    "inv_tpot_p90": _inverse_ms(r["tpot_p90_ms"]),
                    "inv_tpot_p50": _inverse_ms(r["tpot_p50_ms"]),
                    "tpot_p90": r["tpot_p90_ms"],
                    "tps_per_gpu_w": ppw["output_tokens_per_second_per_gpu_watt"],
                    "tps_per_total_w": ppw["output_tokens_per_second_per_combined_watt"],
                    "gpu_w": ppw["gpu_avg_power_w"],
                    "gpu_w_per_gpu": _per_gpu(ppw["gpu_avg_power_w"], num_gpus),
                    "cpu_w": ppw["cpu_avg_power_w"],
                    "total_w": combined_w,
                    "concurrency": r["concurrency"],
                },
                "hover": [
                    ("Concurrency / GPUs", f"{r['concurrency']} / {num_gpus}"),
                    ("P90 TPOT", f"{_fmt(r['tpot_p90_ms'])} ms"),
                    ("Total GPU power", f"{_fmt(ppw['gpu_avg_power_w'], 0)} W"),
                    ("Per GPU", f"{_fmt(_per_gpu(ppw['gpu_avg_power_w'], num_gpus), 0)} W"),
                    ("Output tok/s / GPU W", _fmt(ppw["output_tokens_per_second_per_gpu_watt"], 3)),
                ],
                "fields": [
                    ("Run", run_label or "—"),
                    ("Concurrency / active GPUs", f"{r['concurrency']} / {num_gpus}"),
                    ("Output tok/s", _fmt(output_tps)),
                    ("Output tok/s / active GPU", _fmt(tps_per_gpu)),
                    ("Total tok/s (in + out)", _fmt(total_tps)),
                    ("P50 TPOT", f"{_fmt(r['tpot_p50_ms'])} ms"),
                    ("P90 TPOT", f"{_fmt(r['tpot_p90_ms'])} ms"),
                    ("1 / P90 TPOT", f"{_fmt(_inverse_ms(r['tpot_p90_ms']), 1)} tok/s/user"),
                    ("Total GPU power (all GPUs, avg)", f"{_fmt(ppw['gpu_avg_power_w'], 0)} W"),
                    ("Power per GPU (avg)", f"{_fmt(_per_gpu(ppw['gpu_avg_power_w'], num_gpus), 1)} W"),
                    ("Total CPU power (all sockets, avg)", f"{_fmt(ppw['cpu_avg_power_w'], 0)} W"),
                    ("Total GPU+CPU power (avg)", f"{_fmt(combined_w, 0)} W"),
                    ("Output tok/s / GPU W", _fmt(ppw["output_tokens_per_second_per_gpu_watt"], 4)),
                    ("Output tok/s / (GPU+CPU) W", _fmt(ppw["output_tokens_per_second_per_combined_watt"], 4)),
                    ("Joules / output token", _fmt(r["joules_per_output_token"], 4)),
                    ("Measured window", f"{_fmt(duration, 1)} s"),
                    ("CPU power source", _cpu_sensor_summary(r)),
                ],
                "warnings": [w for w in r.get("warnings", []) if "CPU energy mismatch" in w]
                + list(coverage_warnings or ()),
                "node_power": r.get("node_power", []),
            }
        )
    return points


_PARETO_AXIS_OPTIONS: tuple[tuple[str, str], ...] = (
    ("output_tps", "Output tok/s"),
    ("total_tps", "Total tok/s (in + out)"),
    ("tps_per_gpu", "Output tok/s / GPU"),
    ("total_tps_per_gpu", "Total tok/s / GPU"),
    ("inv_tpot_p90", "1 / P90 TPOT (tok/s/user)"),
    ("inv_tpot_p50", "1 / P50 TPOT (tok/s/user)"),
    ("tpot_p90", "P90 TPOT (ms)"),
    ("tps_per_gpu_w", "Output tok/s / GPU W"),
    ("tps_per_total_w", "Output tok/s / (GPU+CPU) W"),
    ("gpu_w", "Total GPU power (W)"),
    ("gpu_w_per_gpu", "Power per GPU (W)"),
    ("cpu_w", "Total CPU power (W)"),
    ("total_w", "Total GPU+CPU power (W)"),
    ("concurrency", "Concurrency"),
)
_PARETO_DEFAULT_X = "inv_tpot_p90"
_PARETO_DEFAULT_Y = "tps_per_gpu"


def _axis_select_html(axis: str, default: str) -> str:
    options = "".join(
        f'<option value="{key}"{" selected" if key == default else ""}>{html.escape(label)}</option>'
        for key, label in _PARETO_AXIS_OPTIONS
    )
    return f'<select data-axis="{axis}">{options}</select>'


def _model_select_html(points: list[dict]) -> str:
    """Model filter for the Pareto scatter. With one model it's a no-op (omitted);
    with several it defaults to the first so unrelated models aren't drawn on the
    same frontier by default -- "All models" is still there for a deliberate overlay."""
    models = list(dict.fromkeys(p["model"] for p in points if p.get("model")))
    if len(models) < 2:
        return ""
    options = "".join(
        f'<option value="{html.escape(m, quote=True)}"{" selected" if i == 0 else ""}>{html.escape(m)}</option>'
        for i, m in enumerate(models)
    )
    return f'<label>Model <select data-model><option value="">All models</option>{options}</select></label>'


def _node_power_card_html(point: dict | None = None) -> str:
    """ "Average power by node" stacked-bar card. On the Pareto page it follows the
    selected point (the scatter calls ``renderNodePower`` on click). For a page with
    a single concurrency point there is no scatter, so the point is embedded in
    ``data-point`` and the card renders itself at load."""
    point_attr = f' data-point="{html.escape(json.dumps(point), quote=True)}"' if point is not None else ""
    what = "the selected point" if point is None else "this concurrency"
    return f"""
<div class="pareto-card node-power-card"{point_attr}>
  <div class="chart-group-head">
    <h3>Average power by node</h3>
    <span class="node-power-sub pareto-subtitle" style="margin:0"></span>
  </div>
  <p class="pareto-subtitle">Window-average power for {what}, one bar per allocated node. GPU is the
  node's GPUs summed; CPU is the socket envelope (ACPI total / DCGM). When the collector recorded component rails,
  the CPU bar is drawn as its rails (CPU rail, SoC, DRAM) plus the remainder of the envelope they don't account for.</p>
  <div class="chart-notices node-power-notices" hidden></div>
  <div class="node-power-legend legend"></div>
  <div class="node-power-root"><svg class="node-power-svg"></svg><div class="tooltip node-power-tooltip"></div></div>
</div>
"""


def _pareto_view_html(
    points: list[dict],
    *,
    point_charts: dict[str, str],
    power_charts_by_run: dict[str, str] | None = None,
) -> str:
    """The front page: scatter + axis controls, the inspect panel, then the
    selected point's measured-window power charts (``point_charts``, keyed by
    point ``id``). ``power_charts_by_run`` (whole-run charts keyed by run label)
    is an optional fallback section for points whose window had no samples.
    """
    points_json = html.escape(json.dumps(points), quote=True)

    point_groups = "".join(
        f'<div class="point-charts" data-point-id="{html.escape(pid, quote=True)}" hidden>{chart_html}</div>'
        for pid, chart_html in point_charts.items()
    )
    run_groups = "".join(
        f'<div class="power-chart-run" data-pareto-run="{html.escape(run, quote=True)}" hidden>{chart_html}</div>'
        for run, chart_html in (power_charts_by_run or {}).items()
    )
    charts_section = ""
    if point_groups or run_groups:
        # One card, two scopes: the selected point's measured window (default) or the
        # whole run it came from. A checkbox in the header flips between them so the
        # page doesn't spend two full chart stacks of vertical space on a preference.
        toggle = ""
        if point_groups and run_groups:
            toggle = (
                '<label class="scope-toggle"><input type="checkbox" class="scope-whole-run"> '
                "Show whole run instead of the measured window</label>"
            )
        charts_section = f"""
<div class="pareto-card power-scope-card">
  <div class="chart-group-head">
    <h3>Power over time</h3>
    {toggle}
  </div>
  <p class="pareto-subtitle">Per-device power. Hue = host, shade = device index. Click a legend label
  (or a host) to hide/show lines; drag to zoom, double-click to reset.</p>
  <div class="scope-window">{point_groups}</div>
  <div class="scope-run" hidden>{run_groups}</div>
</div>
"""
    run_section = ""
    return f"""
<div class="pareto-card">
  <h3>Pareto view</h3>
  <p class="pareto-subtitle">One point per run &times; concurrency, coloured by GPU type &times; model. Lines connect
  each family's nondominated points for the chosen axes (visual guide, not interpolation). Hover for the headline
  numbers; click to inspect.</p>
  <div class="pareto-layout">
    <div>
      <div class="pareto-controls">
        {_model_select_html(points)}
        <label>X {_axis_select_html("x", _PARETO_DEFAULT_X)}</label>
        <label>Y {_axis_select_html("y", _PARETO_DEFAULT_Y)}</label>
      </div>
      <div class="run-legend"></div>
      <div class="pareto-root" data-points="{points_json}">
        <svg class="pareto-svg" viewBox="0 0 900 520" preserveAspectRatio="xMidYMid meet"></svg>
        <div class="tooltip pareto-tooltip"></div>
      </div>
    </div>
    <div class="pareto-inspect">
      <h3>Inspect a point</h3>
      <p class="pareto-panel-title"></p>
      <div class="pareto-panel"></div>
      <div class="pareto-warnings"></div>
    </div>
  </div>
</div>
{_node_power_card_html()}
{charts_section}
{run_section}
"""


def _power_scatter_html(points: list[dict]) -> str:
    """ "CPU vs GPU" tab: average CPU socket power against average GPU power, one point
    per run x concurrency, same hover/click/inspect behaviour as the Pareto scatter
    but with fixed axes and no frontier line -- neither axis is a "better" direction."""
    plottable = [p for p in points if p["m"]["cpu_w"] is not None and p["m"]["gpu_w"] is not None]
    if not plottable:
        return "<p>No concurrency points have both CPU and GPU power measurements.</p>"
    points_json = html.escape(json.dumps(plottable), quote=True)
    return f"""
<div class="pareto-card">
  <h3>Total CPU vs total GPU power</h3>
  <p class="pareto-subtitle">Run totals, averaged over each point's measured window: all CPU sockets summed against all GPUs summed.
  Divide by the point's socket / GPU count for per-device figures (the inspect panel shows per-GPU). Over each
  point's measured window. Hover for the headline numbers; click to inspect.</p>
  <div class="pareto-layout">
    <div>
      <div class="run-legend"></div>
      <div class="pareto-root" data-points="{points_json}" data-x="cpu_w" data-y="gpu_w" data-frontier="off"
           data-drives-charts="off">
        <svg class="pareto-svg" viewBox="0 0 900 520" preserveAspectRatio="xMidYMid meet"></svg>
        <div class="tooltip pareto-tooltip"></div>
      </div>
    </div>
    <div class="pareto-inspect">
      <h3>Inspect a point</h3>
      <p class="pareto-panel-title"></p>
      <div class="pareto-panel"></div>
    </div>
  </div>
</div>
"""


_BASELINE_DEFAULT_X = "concurrency"
_BASELINE_DEFAULT_Y = "output_tps"


def _baseline_view_html(points: list[dict]) -> str:
    """ "vs baseline" tab: configurable X/Y, with Y divided by the baseline run's Y at
    the same concurrency (and benchmark type). Needs at least two runs to compare."""
    runs = list(dict.fromkeys(p["run"] for p in points))
    if len(runs) < 2:
        return "<p>A baseline comparison needs at least two runs.</p>"
    points_json = html.escape(json.dumps(points), quote=True)
    options = "".join(
        f'<option value="{html.escape(run, quote=True)}"{" selected" if i == 0 else ""}>{html.escape(run)}</option>'
        for i, run in enumerate(runs)
    )
    return f"""
<div class="pareto-card">
  <h3>Relative to a baseline run</h3>
  <p class="pareto-subtitle">Y is each point's value divided by the baseline run's value at the same concurrency
  (1.0 = identical to baseline). Points whose concurrency the baseline never ran are hidden.</p>
  <div class="pareto-layout">
    <div>
      <div class="pareto-controls">
        <label>Baseline <select data-baseline>{options}</select></label>
        <label>X {_axis_select_html("x", _BASELINE_DEFAULT_X)}</label>
        <label>Y {_axis_select_html("y", _BASELINE_DEFAULT_Y)}</label>
      </div>
      <div class="run-legend"></div>
      <div class="pareto-root" data-points="{points_json}" data-frontier="off" data-drives-charts="off">
        <svg class="pareto-svg" viewBox="0 0 900 520" preserveAspectRatio="xMidYMid meet"></svg>
        <div class="tooltip pareto-tooltip"></div>
      </div>
      <p class="baseline-note"></p>
    </div>
    <div class="pareto-inspect">
      <h3>Inspect a point</h3>
      <p class="pareto-panel-title"></p>
      <div class="pareto-panel"></div>
    </div>
  </div>
</div>
"""


def _concurrency_cards_html(
    bundle: dict,
    *,
    run_label: str | None,
    display_label: str,
    run_key: str,
    run_charts_source_id: str,
    point_charts: dict[str, str],
) -> str:
    """Data-table cards for one run: one card per concurrency holding two views of it.
    ``.view-run`` (default) is the whole-run chart zoomed to that concurrency's
    warmup+profile span with its bands emphasised (double-click for the full run);
    ``.view-window`` is the sliced measured-window chart (``point_charts``, the same
    HTML the Pareto page embeds). A global checkbox flips every card between the
    two. Whole-run series data is embedded once per run (``run_charts_source_id``)
    and shared by every card -- see ``_power_charts_html``."""
    bands = _bundle_phase_bands(bundle)
    cards = []
    for i, r in enumerate(sorted(bundle["reports"], key=lambda r: r["concurrency"])):
        bench, conc = r["benchmark_type"], r["concurrency"]
        title = f"{display_label} — {bench} c={conc}"
        run_view = _power_charts_html(
            bundle["gpu_series"],
            bundle["cpu_series"],
            phase_bands=bands,
            title=title,
            notices=bundle.get("coverage_warnings"),
            source_id=run_charts_source_id if i == 0 else None,
            reuse_source=None if i == 0 else run_charts_source_id,
            focus=(bench, conc),
        )
        window_view = point_charts.get(_point_id(run_label, r), "")
        if window_view:
            # Retitle the shared window chart for this card.
            window_view = window_view.replace(
                '<p class="chart-panel-title">Power over time</p>',
                f'<p class="chart-panel-title">{html.escape(title)} (profile window)</p>',
                1,
            )
        if run_view or window_view:
            cards.append(
                f'<div class="conc-card" data-run="{html.escape(run_key, quote=True)}" data-conc="{conc}">'
                f'<div class="view-run">{run_view}</div>'
                f'<div class="view-window" hidden>{window_view}</div>'
                "</div>"
            )
    return "".join(cards)


def _chart_filter_html(runs: list[str], concurrencies: list[int]) -> str:
    """Checkbox filter for the per-concurrency chart cards: a card is shown when both
    its run and its concurrency are ticked. Everything starts ticked."""
    run_boxes = "".join(
        f'<label class="filter-key"><input type="checkbox" data-filter="run" value="{html.escape(run, quote=True)}" checked> '
        f"{html.escape(run)}</label>"
        for run in runs
    )
    conc_boxes = "".join(
        f'<label class="filter-key"><input type="checkbox" data-filter="conc" value="{conc}" checked> c={conc}</label>'
        for conc in concurrencies
    )
    return f"""
<div class="chart-filter">
  <div class="filter-row"><span class="legend-label">Runs</span>{run_boxes}
    <button type="button" class="filter-all" data-filter="run">all</button>
    <button type="button" class="filter-none" data-filter="run">none</button></div>
  <div class="filter-row"><span class="legend-label">Concurrency</span>{conc_boxes}
    <button type="button" class="filter-all" data-filter="conc">all</button>
    <button type="button" class="filter-none" data-filter="conc">none</button></div>
  <p class="filter-count"></p>
</div>
"""


def _tabs_html(pareto_html: str, table_html: str, power_html: str, baseline_html: str) -> str:
    return f"""
<div class="tabs">
  <button class="tab-btn active" type="button" data-tab="pareto">Pareto view</button>
  <button class="tab-btn" type="button" data-tab="table">Data table</button>
  <button class="tab-btn" type="button" data-tab="power">CPU vs GPU</button>
  <button class="tab-btn" type="button" data-tab="baseline">vs baseline</button>
</div>
<div class="tab-panel" data-tab-panel="pareto">{pareto_html}</div>
<div class="tab-panel" data-tab-panel="table" hidden>{table_html}</div>
<div class="tab-panel" data-tab-panel="power" hidden>{power_html}</div>
<div class="tab-panel" data-tab-panel="baseline" hidden>{baseline_html}</div>
"""


def _chart_sub_html(title: str, series: list[dict], *, embed: bool = True, unit: str = "W") -> str:
    """``embed=False`` leaves ``data-series`` empty; the group's ``data-source`` tells
    the JS which sibling group to copy the (identical) series from -- see
    ``_power_charts_html``."""
    series_json = html.escape(json.dumps(series), quote=True) if embed else ""
    return f"""
<div class="chart-sub" data-series="{series_json}" data-unit="{html.escape(unit, quote=True)}">
  <div class="chart-sub-head">
    <p class="chart-sub-title">{html.escape(title)}</p>
    <span class="yscale-toggle" title="y-axis scale">
      <button type="button" class="yscale-btn on" data-scale="linear">linear</button><button type="button" class="yscale-btn" data-scale="log">log</button>
    </span>
  </div>
  <div class="chart-root chart-wrap">
    <svg class="chart" viewBox="0 0 900 220" preserveAspectRatio="xMinYMin meet">
      <rect class="zoom-band"></rect>
      <rect class="overlay" fill="transparent"></rect>
      <line class="crosshair"></line>
    </svg>
    <div class="tooltip"></div>
  </div>
  {_legend_html(series)}
  <span class="stats-toggle">show stats table</span>
  {_stats_table_html(series, unit)}
</div>
"""


def _notices_html(notices: list[str] | None) -> str:
    if not notices:
        return ""
    items = "".join(f"<div>\u26a0 {html.escape(n)}</div>" for n in notices)
    return f'<div class="chart-notices">{items}</div>'


def _role_legend_html(gpu_series: list[dict], cpu_series: list[dict]) -> str:
    """One chip per worker role (prefill / decode / aggregated ...); clicking toggles
    every device carrying that role across both charts. Omitted when the run has
    fewer than two roles (nothing to separate)."""
    roles: dict[str, int] = {}
    for series in (*gpu_series, *cpu_series):
        for role in series.get("roles", ()):
            roles[role] = roles.get(role, 0) + 1
    if len(roles) < 2:
        return ""
    keys = "".join(
        f'<span class="legend-key role-key" data-role="{html.escape(role, quote=True)}" '
        f'title="click to hide/show every device with this role">{html.escape(role)} '
        f'<span class="role-count">({count})</span></span>'
        for role, count in sorted(roles.items())
    )
    return f'<div class="legend role-legend"><span class="legend-label">Roles</span>{keys}</div>'


def _host_legend_html(gpu_series: list[dict], cpu_series: list[dict]) -> str:
    """One chip per host across both legs; clicking toggles every device on that host
    in every chart of the group (see ``initChartGroup``). Swatch = the host's hue."""
    hosts: dict[str, str] = {}
    for series in (*gpu_series, *cpu_series):
        hosts.setdefault(series["host"], series["color"])
    if len(hosts) < 2:
        return ""
    keys = "".join(
        f'<span class="legend-key host-key" data-host="{html.escape(host, quote=True)}" title="click to hide/show every device on this host">'
        f'<span class="legend-swatch" style="background: {color}"></span>{html.escape(host)}</span>'
        for host, color in sorted(hosts.items())
    )
    return f'<div class="legend host-legend"><span class="legend-label">Hosts</span>{keys}</div>'


# Phase kinds painted behind the whole-run traces, in legend order.
_PHASE_KINDS: tuple[tuple[str, str], ...] = (
    ("idle", "Idle (pre/post/between)"),
    ("warmup", "Warmup"),
    ("profile", "Profile (measured)"),
)


def _phase_bands(reports: list[dict], *, origin: float, run_end: float) -> list[dict]:
    """Classify the run's timeline into idle / warmup / profile bands, in seconds
    since ``origin`` (the same zero the traces use). Profile = each concurrency's
    measured window; warmup = the benchmark's recorded warmup span when it has one;
    everything else -- before the first benchmark, between concurrencies, after the
    last -- is idle. Bands are non-overlapping and sorted by start.
    """
    if not reports:
        return []
    busy: list[tuple[float, float, str, str]] = []
    for r in sorted(reports, key=lambda r: r["start_unix"]):
        label = f"{r['benchmark_type']} c={r['concurrency']}"
        ws, we = r.get("warmup_start_unix"), r.get("warmup_end_unix")
        point = (r["benchmark_type"], r["concurrency"])
        if ws is not None and we is not None and we > ws:
            busy.append((ws, min(we, r["start_unix"]), "warmup", label, point))
        busy.append((r["start_unix"], r["end_unix"], "profile", label, point))
    busy.sort()

    bands: list[dict] = []
    cursor = origin
    for start, end, kind, label, (bench, conc) in busy:
        start = max(start, cursor)
        if start >= end:
            continue
        if start > cursor:
            bands.append({"kind": "idle", "label": "idle", "t0": cursor, "t1": start})
        bands.append({"kind": kind, "label": f"{label} {kind}", "t0": start, "t1": end, "bench": bench, "conc": conc})
        cursor = end
    if run_end > cursor:
        bands.append({"kind": "idle", "label": "idle", "t0": cursor, "t1": run_end})
    for b in bands:
        b["t0"], b["t1"] = round(b["t0"] - origin, 2), round(b["t1"] - origin, 2)
    return bands


def _phase_legend_html(bands: list[dict]) -> str:
    present = {b["kind"] for b in bands}
    keys = "".join(
        f'<span class="legend-key phase-key phase-{kind}"><span class="legend-swatch phase-swatch"></span>{label}</span>'
        for kind, label in _PHASE_KINDS
        if kind in present
    )
    return f'<div class="legend phase-legend"><span class="legend-label">Phases</span>{keys}</div>' if keys else ""


def _power_charts_html(
    gpu_series: list[dict],
    cpu_series: list[dict],
    *,
    phase_bands: list[dict] | None = None,
    title: str = "Power over time",
    source_id: str | None = None,
    reuse_source: str | None = None,
    focus: tuple[str, int] | None = None,
    extra_subs: list[tuple[str, list[dict]]] | None = None,
    notices: list[str] | None = None,
) -> str:
    """GPU chart stacked over CPU chart for one run in a single panel.

    ``notices`` (e.g. "CPU power missing ...") render as a warning strip at the top
    of the panel so a missing leg is explained rather than silently absent. Both legs share
    one time origin (see ``_build_run_series``'s ``origin``), so the JS can drive one
    crosshair and one zoom range across both -- see ``initChartGroup``.

    ``phase_bands`` (whole-run charts only) paints idle/warmup/profile shading behind
    the traces -- see ``_phase_bands``.

    Series payloads are large, so a page that shows one run's whole-run trace several
    times (one card per concurrency) embeds the data once: the first card sets
    ``source_id`` and later cards pass ``reuse_source`` with that id; the JS copies the
    series from the source group at init. ``focus`` (``benchmark_type, concurrency``)
    makes a group start zoomed to that concurrency's warmup+profile span."""
    subs = []
    if gpu_series:
        subs.append(_chart_sub_html("GPU power (W)", gpu_series, embed=reuse_source is None))
    if cpu_series:
        subs.append(_chart_sub_html("CPU socket power (W)", cpu_series, embed=reuse_source is None))
    for title_extra, series in extra_subs or ():
        if series:
            subs.append(_chart_sub_html(title_extra, series, embed=reuse_source is None, unit=""))
    if not subs:
        return ""
    bands = phase_bands or []
    attrs = ""
    if bands:
        attrs += f' data-phases="{html.escape(json.dumps(bands), quote=True)}"'
    if source_id:
        attrs += f' data-source-id="{html.escape(source_id, quote=True)}"'
    if reuse_source:
        attrs += f' data-source="{html.escape(reuse_source, quote=True)}"'
    if focus:
        attrs += f' data-focus-bench="{html.escape(focus[0], quote=True)}" data-focus-conc="{focus[1]}"'
    return f"""
<div class="chart-panel chart-group"{attrs}>
  <div class="chart-group-head">
    <p class="chart-panel-title">{html.escape(title)}</p>
    <span class="zoom-hint">drag to zoom</span>
  </div>
  {_notices_html(notices)}
  {_phase_legend_html(bands)}
  {_role_legend_html(gpu_series, cpu_series)}
  {_host_legend_html(gpu_series, cpu_series)}
  {"".join(subs)}
</div>
"""


_PHASE_TPS_BIN_S = 1.0
_PREFILL_COLOR = "hsl(28 85% 55%)"
_DECODE_COLOR = "hsl(158 60% 42%)"


def _tps_series(label: str, color: str, t: np.ndarray, v: np.ndarray) -> dict:
    t_arr, v_arr = _downsample_minmax(t, v, _MAX_BUCKETS_PER_WINDOW_SERIES)
    return {
        "label": label,
        "host": "",
        "roles": [],
        "color": color,
        "pattern": 0,
        "t": [round(x, 2) for x in t_arr.tolist()],
        "w": [round(x, 1) for x in v_arr.tolist()],
        "stats": _series_stats(v_arr),
    }


def _derive_phase_tps(
    report: dict, *, origin: float, bin_s: float = _PHASE_TPS_BIN_S
) -> tuple[list[dict], list[dict]] | None:
    """Prefill and decode token rates over the measured window, rebuilt from aiperf's
    per-request records (``profile_export.jsonl``).

    Each request's input tokens are spread uniformly over ``request_start -> first
    token`` and its output tokens over ``first token -> last token`` (``ttft +
    decode_duration``); every 1 s bin gets the token-rate x overlap of each request,
    so the series integrate back to the exact token totals aiperf reports. This is
    wall-clock throughput, unlike aiperf's own per-slice fields (which credit a whole
    request to the second it completed, or measure rate-while-active).

    Note "first token" is the client-observed TTFT, so prefill here means *input tokens
    admitted per second* -- it includes frontend queueing and KV transfer, not just
    prefill compute. Returns ``None`` when the run has no per-request export.
    """
    source = report.get("source")
    if not source or not str(source).endswith(".jsonl") or not Path(source).is_file():
        return None
    start, end = report["start_unix"], report["end_unix"]
    n = int(np.ceil((end - start) / bin_s))
    if n <= 0:
        return None
    prefill = np.zeros(n)
    decode = np.zeros(n)
    edges = start + np.arange(n + 1) * bin_s

    def spread(arr: np.ndarray, a: float, b: float, tokens: float) -> None:
        if b <= a or tokens <= 0:
            return
        rate = tokens / (b - a)
        i0 = max(0, int((a - start) // bin_s))
        i1 = min(n - 1, int((b - start) // bin_s))
        for i in range(i0, i1 + 1):
            overlap = min(edges[i + 1], b) - max(edges[i], a)
            if overlap > 0:
                arr[i] += rate * overlap

    found = False
    with Path(source).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            md, m = rec.get("metadata", {}), rec.get("metrics", {})
            if md.get("benchmark_phase", "profiling") != "profiling" or md.get("was_cancelled"):
                continue
            ttft = _metric_value(m.get("time_to_first_token"))
            dec_dur = _metric_value(m.get("decode_duration"))
            isl = _metric_value(m.get("input_sequence_length"))
            osl = _metric_value(m.get("output_sequence_length"))
            req_start_ns = md.get("request_start_ns")
            if req_start_ns is None or ttft is None or isl is None:
                continue
            found = True
            t_start = req_start_ns / 1e9
            t_first = t_start + ttft / 1e3
            spread(prefill, t_start, t_first, isl)
            if dec_dur is not None and osl is not None:
                spread(decode, t_first, t_first + dec_dur / 1e3, osl)
    if not found:
        return None
    mids = (edges[:-1] + edges[1:]) / 2 - origin
    return (
        [_tps_series("prefill (input tok/s)", _PREFILL_COLOR, mids, prefill)],
        [_tps_series("decode (output tok/s)", _DECODE_COLOR, mids, decode)],
    )


def _metric_value(metric: object) -> float | None:
    """aiperf per-request metrics are ``{"value": x, "unit": "ms"}``; tolerate bare numbers."""
    if isinstance(metric, dict):
        metric = metric.get("value")
    return float(metric) if isinstance(metric, (int, float)) else None


# aiperf timeslice metrics plotted under the power charts, in display order.
# (key in the timeslices JSON, series label). Output only by default: aiperf's
# input_token_throughput counts whole prompts accepted per second (ISL x req/s),
# which at agentic ISLs is ~100x output and would flatten it on a shared axis.
_TPS_METRICS: tuple[tuple[str, str], ...] = (("output_token_throughput", "output tok/s"),)
_TPS_COLORS = ("hsl(158 60% 42%)", "hsl(212 70% 55%)")


def _load_tps_series(report: dict, *, origin: float) -> list[dict]:
    """Per-second throughput from aiperf's ``profile_export_aiperf_timeslices.json``
    (sibling of the concurrency's result artifact), as chart series time-shifted to
    ``origin`` so they line up with the point's power charts. Fallback for runs without
    ``profile_export.jsonl`` (see ``_derive_phase_tps``); aiperf credits a request's
    tokens to the second it completed, so this is spikier than the derived series.
    Empty when the run did not export timeslices (older aiperf, or sa-bench)."""
    source = report.get("source")
    if not source:
        return []
    path = Path(source).with_name("profile_export_aiperf_timeslices.json")
    if not path.is_file():
        return []
    try:
        slices = json.loads(path.read_text()).get("timeslices", [])
    except (OSError, ValueError):
        return []
    series: list[dict] = []
    for position, (key, label) in enumerate(_TPS_METRICS):
        times: list[float] = []
        values: list[float] = []
        for sl in slices:
            metric = sl.get(key)
            if not isinstance(metric, dict) or metric.get("avg") is None:
                continue
            times.append((sl["start_ns"] + sl["end_ns"]) / 2e9 - origin)
            values.append(float(metric["avg"]))
        if not values:
            continue
        t_arr, v_arr = _downsample_minmax(np.array(times), np.array(values), _MAX_BUCKETS_PER_WINDOW_SERIES)
        series.append(
            {
                "label": label,
                "host": "",
                "roles": [],
                "color": _TPS_COLORS[position % len(_TPS_COLORS)],
                "pattern": 0,
                "t": [round(v, 2) for v in t_arr.tolist()],
                "w": [round(v, 1) for v in v_arr.tolist()],
                "stats": _series_stats(v_arr),
            }
        )
    return series


def _point_charts_html(bundle: dict, *, run_label: str | None) -> dict[str, str]:
    """Measured-window GPU+CPU chart pair per concurrency report in ``bundle``, keyed
    by the same point ``id`` ``_pareto_points`` emits. Reports whose window holds no
    samples on either leg are omitted (the Pareto view then shows nothing for them)."""
    out: dict[str, str] = {}
    for r in bundle["reports"]:
        window = (r["start_unix"], r["end_unix"])
        gpu = cpu = []
        if bundle["gpu_per_device"] is not None:
            gpu = _build_run_series(
                bundle["gpu_per_device"],
                label_fmt="{host}/gpu{index}",
                roles=bundle.get("gpu_roles"),
                window=window,
                max_buckets=_MAX_BUCKETS_PER_WINDOW_SERIES,
            )
        if bundle["cpu_per_socket"] is not None:
            cpu = _build_run_series(
                bundle["cpu_per_socket"],
                label_fmt="{host}/socket{index}",
                host_roles=bundle.get("host_roles"),
                window=window,
                max_buckets=_MAX_BUCKETS_PER_WINDOW_SERIES,
            )
        phase_tps = _derive_phase_tps(r, origin=window[0])
        if phase_tps is not None:
            extra = [
                ("Prefill throughput (input tok/s)", phase_tps[0]),
                ("Decode throughput (output tok/s)", phase_tps[1]),
            ]
        else:  # no per-request export: fall back to aiperf's completion-attributed slices
            tps = _load_tps_series(r, origin=window[0])
            extra = [("Output throughput (tok/s, aiperf timeslices)", tps)] if tps else []
        notices = list(bundle.get("coverage_warnings") or ())
        if phase_tps is None and r.get("benchmark_type") == "aiperf":
            notices.append(
                "Prefill/decode throughput split unavailable: this run has no per-request export "
                "(profile_export.jsonl). "
                + (
                    "Showing aiperf's aggregate output throughput instead (tokens credited to the second each request completed)."
                    if tps
                    else "No throughput-over-time data for this point."
                )
            )
        charts = _power_charts_html(gpu, cpu, extra_subs=extra or None, notices=notices or None)
        if charts:
            out[_point_id(run_label, r)] = charts
    return out


def _stat_cards_html(cards: list[tuple[str, str]]) -> str:
    if not cards:
        return ""
    tiles = "".join(
        f'<div class="stat-card"><p class="stat-card-num">{html.escape(value)}</p>'
        f'<p class="stat-card-label">{html.escape(label)}</p></div>'
        for label, value in cards
    )
    return f'<div class="stat-cards">{tiles}</div>'


def _page_html(*, title: str, body_parts: list[str]) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
{"".join(body_parts)}
<script>{_JS}</script>
</body>
</html>
"""


def _sources_footer_html(sources: list[tuple[str, Path | None]]) -> str:
    parts = [f"{label}: <code>{html.escape(str(path))}</code>" for label, path in sources if path is not None]
    return (
        "<footer>Time series are downsampled (min/max per bucket) for display; "
        f"full-resolution samples: {' &middot; '.join(parts) if parts else 'n/a'}.</footer>"
    )


def _dedupe_labels(labels: list[str]) -> list[str]:
    """Disambiguate repeated run labels (e.g. two dirs sharing a parent folder name)."""
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return out


def _render_page(bundles: list[dict], *, title: str, subtitle: str, single_run: bool) -> str:
    """Shared page assembly for one run or several.

    Front page: the Pareto view (when 2+ points can be plotted). "Data table":
    the per-concurrency stats table plus every run's whole-run power charts.
    With fewer than two plottable points, the table view is the whole page.
    """
    labels = _dedupe_labels([b["label"] for b in bundles])
    run_labels: list[str | None] = [None] * len(bundles) if single_run else labels

    header_parts = ["<h1>Power &amp; performance report</h1>", f'<p class="subtitle">{html.escape(subtitle)}</p>']
    concurrency_points = sum(len(b["reports"]) for b in bundles)
    gpu_devices = sum(len(b["gpu_series"]) for b in bundles)
    cpu_sockets = sum(len(b["cpu_series"]) for b in bundles)
    cards = [] if single_run else [("Runs", str(len(bundles)))]
    cards.append(("Concurrency points", str(concurrency_points)))
    if gpu_devices:
        cards.append(("GPUs" if single_run else "GPU devices tracked", str(gpu_devices)))
    if cpu_sockets:
        cards.append(("CPU sockets" if single_run else "CPU sockets tracked", str(cpu_sockets)))
    header_parts.append(_stat_cards_html(cards))

    table_parts = []
    if any(b["reports"] for b in bundles):
        rows = "".join(
            _summary_rows_html(b["reports"], run_label=rl, coverage_warnings=b.get("coverage_warnings"))
            for b, rl in zip(bundles, run_labels, strict=True)
        )
        table_parts.append("<h2>Throughput &amp; power by concurrency</h2>")
        table_parts.append(f"<table><thead>{_SUMMARY_TABLE_HEADER}</thead><tbody>{rows}</tbody></table>")
    else:
        which = "this run" if single_run else "any of these runs"
        table_parts.append(f"<p>No concurrency-level benchmark windows were found for {which}.</p>")

    points: list[dict] = []
    point_charts: dict[str, str] = {}
    point_charts_per_run: list[dict[str, str]] = []
    families: dict[str, int] = {}
    for position, (bundle, rl) in enumerate(zip(bundles, run_labels, strict=True)):
        family = _family_label(bundle["gpu_type"], bundle["model"])
        group_position = families.setdefault(family, len(families))
        points.extend(
            _pareto_points(
                bundle["reports"],
                run_label=rl,
                run_position=position,
                group=family,
                group_position=group_position,
                model=bundle["model"] or "unknown model",
                coverage_warnings=bundle.get("coverage_warnings"),
            )
        )
        run_point_charts = _point_charts_html(bundle, run_label=rl)
        point_charts_per_run.append(run_point_charts)
        point_charts.update(run_point_charts)

    sources: list[tuple[str, Path | None]] = []
    power_charts_by_run: dict[str, str] = {}
    conc_cards: list[str] = []
    all_concurrencies: set[int] = set()
    for position, (bundle, label, rl) in enumerate(zip(bundles, labels, run_labels, strict=True)):
        charts_html = _power_charts_html(
            bundle["gpu_series"],
            bundle["cpu_series"],
            phase_bands=_bundle_phase_bands(bundle),
            notices=bundle.get("coverage_warnings"),
        )
        if charts_html:
            power_charts_by_run[label if not single_run else ""] = charts_html
            conc_cards.append(
                _concurrency_cards_html(
                    bundle,
                    run_label=rl,
                    display_label=label,
                    run_key=label,
                    run_charts_source_id=f"run-src-{position}",
                    point_charts=point_charts_per_run[position],
                )
            )
            all_concurrencies.update(r["concurrency"] for r in bundle["reports"])
        prefix = "" if single_run else f"{label} "
        sources.append((f"{prefix}GPU", bundle["gpu_source"]))
        sources.append((f"{prefix}CPU", bundle["cpu_source"]))
    if any(conc_cards):
        table_parts.append("<h2>Power over time — one chart per run &times; concurrency</h2>")
        table_parts.append(
            '<p class="pareto-subtitle">Each chart is the whole run zoomed to that concurrency\'s warmup + profile '
            "span (its phase bands highlighted). Double-click a chart for the full run; drag to zoom.</p>"
        )
        table_parts.append(
            '<div class="scope-global"><label class="scope-toggle">'
            '<input type="checkbox" class="view-window-only"> '
            "Show only the measured profile window (hide warmup / idle context)</label></div>"
        )
        table_parts.append(_chart_filter_html(labels, sorted(all_concurrencies)))
        table_parts.append(f'<div class="conc-cards">{"".join(conc_cards)}</div>')
    table_parts.append(_sources_footer_html(sources))

    if len(points) >= 2:
        pareto_html = _pareto_view_html(points, point_charts=point_charts, power_charts_by_run=power_charts_by_run)
        body_parts = [
            *header_parts,
            _tabs_html(pareto_html, "".join(table_parts), _power_scatter_html(points), _baseline_view_html(points)),
        ]
    else:
        # Single (or no) concurrency point: no scatter to drive the per-node bars, so
        # embed the one point's figures and let the card render itself.
        node_card = _node_power_card_html(points[0]) if points and points[0].get("node_power") else ""
        # Directly under the summary table (h2 + table), before the power-over-time charts.
        table_end = next((i + 1 for i, part in enumerate(table_parts) if part.startswith("<table>")), 0)
        body_parts = [*header_parts, *table_parts[:table_end], node_card, *table_parts[table_end:]]
    return _page_html(title=title, body_parts=body_parts)


def render_html(bundle: dict) -> str:
    """Report page for a single run bundle (see ``_build_run_bundle``)."""
    return _render_page([bundle], title=f"{bundle['label']} — power report", subtitle=bundle["label"], single_run=True)


def render_combined_html(bundles: list[dict]) -> str:
    """Combined comparison report for two or more run bundles (see ``_build_run_bundle``).

    Points from every run share the Pareto scatter (one colour per run, with a
    per-run frontier line); the data table gets a row per run x concurrency. Raw
    power traces from different runs are never overlaid on one axis: each run
    covers a different wall-clock window.
    """
    labels = _dedupe_labels([b["label"] for b in bundles])
    return _render_page(
        bundles,
        title=f"Power report — {len(bundles)} runs",
        subtitle=f"{len(bundles)} runs: {', '.join(labels)}",
        single_run=False,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _discover_power_csvs(
    log_dir: Path, *, cpu_samples_csv: Path | None = None
) -> tuple[Path | None, Path | None, Path | None]:
    """Locate the CPU/GPU ``samples.csv`` legs and the GPU manifest, independent of
    ``power_energy_report.discover_run`` -- that helper also requires a benchmark.out
    with a recognized engine and a matched concurrency window, which a serve-only or
    still-running job may not have. The charts here only need the CSVs themselves.

    Returns ``(cpu_samples_csv, gpu_samples_csv, gpu_manifest)``.
    """
    all_matches = sorted(log_dir.rglob("samples.csv"))
    cpu_matches = [p for p in all_matches if p.parent.name in CPU_SAMPLES_DIRNAMES]
    gpu_matches = [p for p in all_matches if p.parent.name not in CPU_SAMPLES_DIRNAMES]

    if cpu_samples_csv is not None:
        if not cpu_samples_csv.is_file():
            raise PowerReportError(f"{cpu_samples_csv}: not found")
    elif len(cpu_matches) > 1:
        listing = ", ".join(str(p) for p in cpu_matches)
        raise PowerReportError(
            f"multiple CPU power samples.csv found below {log_dir}: {listing}; pick one with --cpu-samples"
        )
    else:
        cpu_samples_csv = cpu_matches[0] if cpu_matches else None

    if len(gpu_matches) > 1:
        raise PowerReportError(f"multiple GPU power samples.csv found below {log_dir}: {gpu_matches}")
    gpu_samples_csv = gpu_matches[0] if gpu_matches else None

    gpu_manifest = None
    if gpu_samples_csv is not None:
        candidate = gpu_samples_csv.with_name("manifest.json")
        gpu_manifest = candidate if candidate.is_file() else None

    return cpu_samples_csv, gpu_samples_csv, gpu_manifest


def _power_coverage_warnings(
    log_dir: Path,
    *,
    cpu_csv: Path | None,
    gpu_csv: Path | None,
    gpu_per_device: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] | None,
    cpu_per_socket: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] | None,
) -> list[str]:
    """Explain missing power legs instead of silently drawing fewer charts.

    A leg counts as *collected* only if its samples.csv has rows. When an exporter
    log (``telemetry_cpu_power_exporter.<host>.out`` / ``telemetry_dcgm_exporter.out``)
    exists but no samples were written, collection was attempted and failed -- the
    common case is one node's exporter dying and Slurm cancelling the whole step.
    Hosts that have GPU samples but no CPU samples are called out individually.
    """
    warnings: list[str] = []
    cpu_logs = sorted(log_dir.glob("telemetry_cpu_power_exporter.*.out"))
    gpu_logs = sorted(log_dir.glob("telemetry_dcgm_exporter*.out"))

    def failed(log_paths: list[Path]) -> str:
        bad = [p for p in log_paths if "TASK FAILURE" in p.read_text(errors="replace")]
        if bad:
            hosts = ", ".join(p.name.split(".")[1] if p.name.count(".") >= 2 else p.name for p in bad)
            return f" (exporter step cancelled: TASK FAILURE on {hosts})"
        return ""

    if not cpu_per_socket:
        if cpu_csv is not None and cpu_logs:
            warnings.append(
                f"CPU power missing: exporter started on {len(cpu_logs)} host(s) but wrote no samples"
                + failed(cpu_logs)
                + ". Totals and tok/s-per-W below exclude CPU."
            )
        elif cpu_csv is not None:
            warnings.append("CPU power missing: power/cpu/samples.csv is empty. Totals below exclude CPU.")
        elif cpu_logs:
            warnings.append(
                f"CPU power missing: {len(cpu_logs)} exporter log(s) but no power/cpu/samples.csv"
                + failed(cpu_logs)
                + "."
            )
        else:
            warnings.append("CPU power not collected for this run (no CPU power exporter).")
    if not gpu_per_device:
        if gpu_csv is not None or gpu_logs:
            warnings.append(
                "GPU power missing: DCGM exporter ran but no GPU samples were written" + failed(gpu_logs) + "."
            )
        else:
            warnings.append("GPU power not collected for this run (no DCGM exporter).")

    if gpu_per_device and cpu_per_socket:
        gpu_hosts = {host for host, _ in gpu_per_device}
        cpu_hosts = {host for host, _ in cpu_per_socket}
        missing = sorted(gpu_hosts - cpu_hosts)
        if missing:
            warnings.append(
                f"CPU power missing on {len(missing)} of {len(gpu_hosts)} GPU host(s): {', '.join(missing)}. "
                "CPU totals cover only the hosts that reported."
            )
    return warnings


def _load_run_config(log_dir: Path) -> dict | None:
    """The run's ``config.yaml`` as a dict, or None. It lives next to ``logs/`` (the job
    dir) and is also copied inside ``logs/`` itself; either location is accepted."""
    for candidate in (log_dir / "config.yaml", log_dir.parent / "config.yaml"):
        if candidate.is_file():
            try:
                config = yaml.safe_load(candidate.read_text())
            except yaml.YAMLError:
                return None
            return config if isinstance(config, dict) else None
    return None


def _load_run_family(config: dict | None) -> tuple[str | None, str | None]:
    """``(gpu_type, model)`` used to group Pareto points into frontiers: runs on the
    same GPU type serving the same model are comparable; a frontier across different
    silicon or different models is not. GPU type comes from ``resources.gpu_type``.
    Model prefers the ``identity.model.repo`` block (the canonical HF name) and falls
    back to the basename of ``model.path`` (stripping any ``hf:`` scheme)."""
    if not config:
        return None, None
    resources = config.get("resources")
    gpu_type = resources.get("gpu_type") if isinstance(resources, dict) else None
    gpu_type = str(gpu_type).lower() if gpu_type else None

    model: str | None = None
    identity = config.get("identity")
    if isinstance(identity, dict) and isinstance(identity.get("model"), dict):
        repo = identity["model"].get("repo")
        model = str(repo) if repo else None
    if model is None:
        model_block = config.get("model")
        if isinstance(model_block, dict) and model_block.get("path"):
            path = str(model_block["path"])
            path = path.split(":", 1)[1] if path.startswith("hf:") else path
            model = path.rstrip("/").rsplit("/", 1)[-1] or None
    return gpu_type, model


def _load_run_topology(log_dir: Path) -> tuple[str, int] | None:
    """Best-effort ``(topology label, total GPU count)`` from the run's ``config.yaml``
    ``resources:`` block, used to classify/sort runs in a combined report. A missing or
    unparseable config is not an error -- callers fall back to the bare directory-name
    label.
    """
    config = _load_run_config(log_dir)
    resources = config.get("resources") if config else None
    if not isinstance(resources, dict):
        return None
    try:
        prefill_workers = int(resources["prefill_workers"])
        gpus_per_prefill = int(resources["gpus_per_prefill"])
        decode_workers = int(resources["decode_workers"])
        gpus_per_decode = int(resources["gpus_per_decode"])
    except (KeyError, TypeError, ValueError):
        return None
    total_gpus = prefill_workers * gpus_per_prefill + decode_workers * gpus_per_decode
    topology = f"P{prefill_workers}x{gpus_per_prefill}+D{decode_workers}x{gpus_per_decode}"
    return topology, total_gpus


def _bundle_phase_bands(bundle: dict) -> list[dict]:
    legs = (bundle["gpu_per_device"], bundle["cpu_per_socket"])
    origin = _earliest_sample(*legs)
    if origin is None:
        return []
    ends = [float(times[-1]) for leg in legs if leg for times, _ in leg.values() if len(times)]
    return _phase_bands(bundle["reports"], origin=origin, run_end=max(ends))


def _host_roles(roles: dict[tuple[str, int], set[str]] | None) -> dict[str, set[str]]:
    """Union of worker roles per host, so CPU sockets can be toggled by the role of the
    GPUs they serve (a host running only decode GPUs is a decode host)."""
    out: dict[str, set[str]] = {}
    for (host, _index), device_roles in (roles or {}).items():
        out.setdefault(host, set()).update(device_roles)
    return out


def _earliest_sample(*legs: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] | None) -> float | None:
    starts = [float(times[0]) for leg in legs if leg for times, _ in leg.values() if len(times)]
    return min(starts) if starts else None


def _build_run_bundle(log_dir: Path, *, cpu_samples_csv: Path | None = None) -> dict:
    """Everything one directory contributes to a report: its stats, its facets, its
    label. Raises ``PowerReportError`` if the directory has nothing to report.

    Shared by :func:`build_report` (one directory -> one page) and
    :func:`build_combined_report` (several directories -> one comparison page), so
    a single directory's report and its row in a combined report are always built
    from identical data.
    """
    cpu_csv, gpu_csv, gpu_manifest = _discover_power_csvs(log_dir, cpu_samples_csv=cpu_samples_csv)

    reports: list[dict] = []
    try:
        reports = [report_to_dict(r) for r in build_reports(log_dir, cpu_samples_csv=cpu_samples_csv)]
    except PowerReportError as e:
        logger.info("power report: concurrency stats unavailable for %s (%s); charts only", log_dir, e)

    gpu_per_device: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] | None = None
    roles: dict[tuple[str, int], set[str]] | None = None
    if gpu_csv is not None:
        roles = load_gpu_roles(gpu_manifest) if gpu_manifest else None
        gpu_per_device = load_gpu_samples(gpu_csv, roles).per_device
    host_roles = _host_roles(roles)

    cpu_per_socket: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] | None = None
    if cpu_csv is not None:
        cpu_per_socket = load_cpu_samples(cpu_csv).per_socket

    # One time origin for both legs so the stacked GPU/CPU charts align.
    origin = _earliest_sample(gpu_per_device, cpu_per_socket)
    gpu_series: list[dict] = []
    if gpu_per_device is not None:
        gpu_series = _build_run_series(gpu_per_device, label_fmt="{host}/gpu{index}", origin=origin, roles=roles)
    cpu_series: list[dict] = []
    if cpu_per_socket is not None:
        cpu_series = _build_run_series(
            cpu_per_socket, label_fmt="{host}/socket{index}", origin=origin, host_roles=host_roles
        )

    if not reports and not gpu_series and not cpu_series:
        raise PowerReportError(f"{log_dir}: no benchmark stats or power samples to report")

    coverage_warnings = _power_coverage_warnings(
        log_dir, cpu_csv=cpu_csv, gpu_csv=gpu_csv, gpu_per_device=gpu_per_device, cpu_per_socket=cpu_per_socket
    )
    for w in coverage_warnings:
        logger.warning("power report: %s: %s", log_dir, w)

    job_label = log_dir.parent.name or str(log_dir)
    gpu_type, model = _load_run_family(_load_run_config(log_dir))
    topology = _load_run_topology(log_dir)
    if topology is not None:
        topology_label, total_gpus = topology
        label = f"{job_label} · {topology_label} · {total_gpus} GPU"
    else:
        label, total_gpus = job_label, None

    return {
        "label": label,
        "total_gpus": total_gpus,
        "gpu_type": gpu_type,
        "model": model,
        "reports": reports,
        "gpu_series": gpu_series,
        "cpu_series": cpu_series,
        # Raw per-device arrays, kept so the Pareto view can slice each concurrency's
        # measured window out of the run without re-reading the CSVs.
        "gpu_per_device": gpu_per_device,
        "cpu_per_socket": cpu_per_socket,
        "gpu_roles": roles,
        "host_roles": host_roles,
        "gpu_source": gpu_csv,
        "cpu_source": cpu_csv,
        "coverage_warnings": coverage_warnings,
    }


def build_report(log_dir: Path, *, cpu_samples_csv: Path | None = None) -> str:
    """Build the report HTML for ``log_dir``. Raises ``PowerReportError`` if nothing applies."""
    return render_html(_build_run_bundle(log_dir, cpu_samples_csv=cpu_samples_csv))


def build_combined_report(log_dirs: list[Path], *, cpu_samples_csv: Path | None = None) -> str:
    """Build one comparison report across several run directories.

    A directory with nothing to report is skipped (logged as a warning) rather
    than failing the whole rollup; ``PowerReportError`` is only raised if *none*
    of the directories had anything.
    """
    bundles: list[dict] = []
    skipped: list[str] = []
    for log_dir in log_dirs:
        try:
            bundles.append(_build_run_bundle(log_dir, cpu_samples_csv=cpu_samples_csv))
        except PowerReportError as e:
            logger.warning("power report: skipping %s: %s", log_dir, e)
            skipped.append(str(e))

    if not bundles:
        raise PowerReportError(f"none of the {len(log_dirs)} directories had a report to build: {'; '.join(skipped)}")
    bundles.sort(key=_bundle_sort_key)
    return render_combined_html(bundles)


def _bundle_sort_key(bundle: dict) -> tuple[float, float, str]:
    """Classify runs by GPU scale, then concurrency, so the table/Pareto rows in a
    combined report read as a topology sweep instead of arrival/job-id order.

    Runs with no resolvable GPU count (``_load_run_topology`` failed) or no
    concurrency-level reports sort last within their tier rather than raising.
    """
    gpus = bundle["total_gpus"] if bundle["total_gpus"] is not None else float("inf")
    concurrencies = [r["concurrency"] for r in bundle["reports"]]
    min_concurrency = min(concurrencies) if concurrencies else float("inf")
    return (gpus, min_concurrency, bundle["label"])


def build(log_dir: Path, *, cpu_samples_csv: Path | None = None, output_path: Path | None = None) -> Path | None:
    """Write the report next to the run's other power artifacts. Returns the path, or None on failure."""
    try:
        content = build_report(log_dir, cpu_samples_csv=cpu_samples_csv)
    except PowerReportError as e:
        logger.debug("power report: skipped: %s", e)
        return None

    out = output_path or (log_dir / HTML_FILENAME)
    out.write_text(content)
    logger.info("power report: %s", out)
    return out


def build_combined(log_dirs: list[Path], *, output_path: Path, cpu_samples_csv: Path | None = None) -> Path | None:
    """Write a combined comparison report for several run directories. Returns the path, or None on failure."""
    try:
        content = build_combined_report(log_dirs, cpu_samples_csv=cpu_samples_csv)
    except PowerReportError as e:
        logger.debug("power report: combined build skipped: %s", e)
        return None

    output_path.write_text(content)
    logger.info("power report (combined, %d run(s)): %s", len(log_dirs), output_path)
    return output_path


def try_build(runtime: RuntimeContext) -> Path | None:
    """Best-effort entry point for :class:`PostProcessStageMixin`.

    Mirrors :func:`srtctl.analysis.perf_dashboard.try_build`'s contract: never
    raises, so a rendering bug cannot affect the outcome of a benchmark that has
    already produced its results.
    """
    try:
        return build(Path(runtime.log_dir))
    except Exception as e:  # noqa: BLE001 - visualisation is never fatal
        logger.warning("power report: skipped after error: %s", e)
        return None


DEFAULT_COMBINED_FILENAME = "power_report_combined.html"


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dirs",
        type=Path,
        nargs="+",
        metavar="LOG_DIR",
        help=(
            "One or more run logs/ directories. A single directory renders its own "
            "power_report.html; more than one rolls them up into a single "
            "combined comparison report instead."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output HTML path. Default: <log_dir>/power_report.html for a single "
            f"directory, ./{DEFAULT_COMBINED_FILENAME} for several."
        ),
    )
    parser.add_argument("--cpu-samples", type=Path, help="Explicit CPU samples.csv, see power_energy_report --help")
    args = parser.parse_args(argv)

    try:
        if len(args.log_dirs) == 1:
            path = build(args.log_dirs[0], cpu_samples_csv=args.cpu_samples, output_path=args.output)
        else:
            output_path = args.output or (Path.cwd() / DEFAULT_COMBINED_FILENAME)
            path = build_combined(args.log_dirs, output_path=output_path, cpu_samples_csv=args.cpu_samples)
    except Exception as e:  # noqa: BLE001 - CLI surface, report and exit non-zero
        print(f"error: {e}", file=sys.stderr)
        return 1
    if path is None:
        print("error: no report could be built (see logs)", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
