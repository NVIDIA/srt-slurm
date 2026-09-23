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
import itertools
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

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
    for lo, hi in itertools.pairwise(edges):
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
.chart-extras { display: flex; flex-direction: column; gap: 4px; margin: 4px 0 6px; }
.chart-fold > summary { display: flex; align-items: center; gap: 8px; cursor: pointer; user-select: none; list-style: none;
  font-size: 12px; font-weight: 600; color: var(--ink-secondary); padding: 5px 10px; border: 1px solid var(--border);
  border-radius: 6px; background: color-mix(in srgb, var(--ink) 3%, transparent); }
.chart-fold > summary:hover { border-color: var(--ink-muted); background: color-mix(in srgb, var(--ink) 7%, transparent); }
.chart-fold > summary::-webkit-details-marker { display: none; }
.chart-fold[open] > summary { border-bottom-left-radius: 0; border-bottom-right-radius: 0; }
.fold-caret { width: 0; height: 0; border-style: solid; border-width: 5px 0 5px 7px; border-color: transparent transparent transparent currentColor;
  flex: none; transition: transform .12s ease; }
.chart-fold[open] .fold-caret { transform: rotate(90deg); }
.fold-hint { font-weight: 400; color: var(--ink-muted); }
.chart-fold > .legend, .chart-fold > .stats-table { margin: 0; padding: 8px 10px; border: 1px solid var(--border); border-top: 0;
  border-radius: 0 0 6px 6px; }
.legend-key.partial { opacity: .7; border-style: dashed; }
/* Summary-table column groups: perf stats (neutral), GPU (green, matches the GPU bar
   segment), CPU (orange, matches the CPU segments), efficiency (blue). A slightly
   stronger wash on the header and a hairline on each group's first column. */
td.cg-perf, th.cg-perf { background: hsl(0 0% 50% / .07); }
td.cg-gpu,  th.cg-gpu  { background: hsl(158 60% 40% / .10); }
td.cg-cpu,  th.cg-cpu  { background: hsl(35 85% 50% / .10); }
td.cg-eff,  th.cg-eff  { background: hsl(210 70% 55% / .10); }
th.cg-perf { background: hsl(0 0% 50% / .14); }
th.cg-gpu  { background: hsl(158 60% 40% / .20); }
th.cg-cpu  { background: hsl(35 85% 50% / .20); }
th.cg-eff  { background: hsl(210 70% 55% / .20); }
td.cg-start, th.cg-start { border-left: 1px solid var(--border); }
.host-legend { margin: 0 0 12px; padding-bottom: 10px; border-bottom: 1px solid var(--grid); }
.legend-label { color: var(--ink-muted); font-size: 11px; text-transform: uppercase; letter-spacing: .02em; align-self: center; }
.host-key { font-weight: 600; }
.chart-notices { color: var(--slot-1); font-size: 12px; margin: 0 0 10px; padding: 8px 10px;
  border: 1px solid color-mix(in srgb, var(--slot-1) 45%, transparent); border-radius: 6px;
  background: color-mix(in srgb, var(--slot-1) 8%, transparent); }
.chart-notices div + div { margin-top: 3px; }
.row-warn { color: var(--slot-1); cursor: help; }
.legend-swatch { width: 14px; height: 3px; border-radius: 1px; }
.stat-cards { display: flex; gap: 12px; margin: 4px 0 24px; flex-wrap: wrap; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 18px; min-width: 110px; }
.stat-card-num { font-size: 22px; font-weight: 700; margin: 0; }
.stat-card-label { color: var(--ink-secondary); font-size: 12px; margin: 2px 0 0; }
.chart-group { margin-top: 16px; }
.chart-group-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin: 0 0 8px; }
.chart-group-tools { display: inline-flex; align-items: center; gap: 12px; }
.granularity-toggle { display: inline-flex; border: 1px solid var(--line); border-radius: 5px; overflow: hidden; font-size: 11px; }
.gran-btn { background: transparent; color: var(--ink-muted); border: 0; padding: 2px 9px; cursor: pointer; font: inherit; font-size: 11px; }
.gran-btn + .gran-btn { border-left: 1px solid var(--line); }
.gran-btn.on { background: var(--surface-2, rgba(127,127,127,0.18)); color: var(--ink); font-weight: 600; }
.zoom-hint { color: var(--ink-muted); font-size: 11px; }
.chart-sub + .chart-sub { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--grid); }
/* Role sections: one tinted box per node type. --role-accent drives the heading
   colour, the left rule and the faint background wash. */
.chart-section { --role-accent: var(--ink-muted); position: relative; margin: 10px 0 0; padding: 10px 14px 6px 16px;
  border: 1px solid color-mix(in srgb, var(--role-accent) 35%, var(--border)); border-left: 4px solid var(--role-accent);
  border-radius: 8px; background: color-mix(in srgb, var(--role-accent) 5%, transparent); }
.chart-section.role-prefill { --role-accent: hsl(28 85% 55%); }
.chart-section.role-decode { --role-accent: hsl(158 60% 42%); }
.chart-section.role-throughput { --role-accent: hsl(212 70% 55%); }
.chart-section.role-all { --role-accent: hsl(212 70% 55%); }
.chart-role-heading { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 12px; letter-spacing: .05em;
  text-transform: uppercase; color: var(--role-accent); margin: 0; cursor: pointer; user-select: none; list-style: none;
  padding: 2px 0; border-radius: 4px; }
.chart-role-heading::-webkit-details-marker { display: none; }
.chart-role-heading:hover { background: color-mix(in srgb, var(--role-accent) 10%, transparent); }
.section-caret { width: 0; height: 0; border-style: solid; border-width: 5px 0 5px 7px;
  border-color: transparent transparent transparent currentColor; flex: none; transition: transform .12s ease; }
.chart-section[open] .section-caret { transform: rotate(90deg); }
.section-hint { font-weight: 500; letter-spacing: 0; text-transform: none; color: var(--ink-muted); margin-left: 2px; }
.chart-section:not([open]) { padding-bottom: 10px; }
.chart-section-body { margin-top: 8px; }
.chart-section .chart-sub + .chart-sub { margin-top: 6px; }
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
.axis-caption { font-size: 11px; fill: var(--ink-secondary); font-weight: 600; letter-spacing: .02em; }
.axis-title { fill: var(--ink-secondary); font-size: 11px; font-weight: 600; }
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
.stats-table { font-size: 12px; width: 100%; }
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
/* Data-table cards: one collapsible box per run x concurrency, tinted per run. */
.conc-card { --card-accent: hsl(215 10% 58%); margin: 0 0 14px; border: 1px solid color-mix(in srgb, var(--card-accent) 40%, var(--border));
  border-left: 5px solid var(--card-accent); border-radius: 8px; background: color-mix(in srgb, var(--card-accent) 4%, var(--surface)); }
.conc-card-head { display: flex; align-items: center; gap: 10px; padding: 10px 14px; cursor: pointer; user-select: none; list-style: none;
  font-size: 13px; color: var(--ink); border-radius: 8px; }
.conc-card-head::-webkit-details-marker { display: none; }
.conc-card-head:hover { background: color-mix(in srgb, var(--card-accent) 10%, transparent); }
.conc-card-head .section-caret { color: var(--card-accent); }
.conc-card[open] > .conc-card-head { border-bottom: 1px solid color-mix(in srgb, var(--card-accent) 25%, var(--border)); border-radius: 8px 8px 0 0; }
.conc-card-run { font-weight: 700; color: var(--ink); }
.conc-card-conc { font-weight: 600; padding: 1px 8px; border-radius: 999px; font-size: 12px;
  background: color-mix(in srgb, var(--card-accent) 16%, transparent); color: var(--ink); }
.conc-card-body { padding: 6px 12px 10px; }
.conc-card-body .chart-panel { border: 0; background: transparent; padding: 6px 0 0; margin: 0; }
.scope-global { margin: 0 0 12px; }
.phase-band { stroke: none; }
.phase-idle { fill: var(--ink-muted); fill-opacity: .10; }
.phase-warmup { fill: var(--slot-1); fill-opacity: .16; }
.phase-profile { fill: var(--slot-2); fill-opacity: .14; }
.phase-drain { fill: var(--slot-2); fill-opacity: .06; stroke: var(--slot-2); stroke-opacity: .5; stroke-dasharray: 3 3; }
.phase-band.phase-focus.phase-drain { fill-opacity: .12; }
.phase-band.phase-other { fill-opacity: .05; }
.phase-band.phase-focus.phase-warmup { fill-opacity: .28; }
.phase-band.phase-focus.phase-profile { fill-opacity: .24; }
.phase-legend { margin: 0 0 8px; }
.phase-key { cursor: default; }
.phase-swatch { width: 14px; height: 10px; border-radius: 2px; }
.phase-key.phase-idle .phase-swatch { background: var(--ink-muted); opacity: .45; }
.phase-key.phase-warmup .phase-swatch { background: var(--slot-1); opacity: .6; }
.phase-key.phase-profile .phase-swatch { background: var(--slot-2); opacity: .6; }
.phase-key.phase-drain .phase-swatch { background: transparent; border: 1px dashed var(--slot-2); opacity: .8; }
.scope-controls { display: flex; gap: 20px; align-items: center; flex-wrap: wrap; margin: 0 0 4px; }
.scope-controls select { font: inherit; font-size: 12px; padding: 2px 6px; background: var(--surface); color: var(--ink-primary); border: 1px solid var(--border); border-radius: 4px; }
.baseline-note, .pareto-note { color: var(--ink-muted); font-size: 11px; margin: 6px 0 0; min-height: 1em; }
.pareto-point { stroke: var(--surface); stroke-width: 1.5; cursor: pointer; }
.pareto-point.selected { stroke: var(--ink-primary); stroke-width: 2.5; }
.pareto-point.dim { opacity: .18; }
.pareto-point.grey { stroke: hsl(0 0% 30%); stroke-width: 1.5; }
.pareto-point.grey.selected { stroke: var(--ink-primary); stroke-width: 2.5; }
/* Budget editor: nested inside the inspect column, so it stacks its inputs. */
.budget-card { margin: 16px 0 0; padding: 0 0 12px; border-top: 1px solid var(--grid); }
.budget-head { display: flex; align-items: center; gap: 10px; cursor: pointer; user-select: none; list-style: none; padding: 12px 0 6px; }
.budget-head::-webkit-details-marker { display: none; }
.budget-head h3 { margin: 0; font-size: 14px; }
.budget-card[open] .section-caret { transform: rotate(90deg); }
.budget-card > .pareto-subtitle { font-size: 11.5px; }
.budget-row { margin: 8px 0 10px; padding: 8px 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); }
.budget-row-head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }
.budget-inputs { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px 12px; }
.budget-inputs label { display: flex; flex-direction: column; gap: 3px; font-size: 11px; color: var(--ink-muted); }
.budget-inputs input { width: 100%; box-sizing: border-box; font: inherit; font-size: 13px; padding: 3px 6px; background: var(--page);
  color: var(--ink-primary); border: 1px solid var(--border); border-radius: 4px; font-variant-numeric: tabular-nums; }
.budget-inputs input.changed { border-color: var(--slot-1); }
.budget-meta, .budget-derived { color: var(--ink-muted); font-size: 11px; }
.budget-derived { margin-top: 6px; }
.budget-missing { color: var(--slot-1); font-size: 11px; margin-left: 6px; }
.budget-reset { font: inherit; font-size: 12px; padding: 3px 10px; background: var(--surface); color: var(--ink-primary);
  border: 1px solid var(--border); border-radius: 4px; cursor: pointer; }
.budget-reset:hover { border-color: var(--ink-muted); }
.est-warn { color: var(--slot-1); cursor: help; margin-right: 4px; font-weight: 600; }
.basis-legend { display: flex; gap: 8px; flex-wrap: wrap; margin: -6px 0 12px; font-size: 12px; }
.basis-legend[hidden] { display: none; }
.basis-swatch { width: 34px; height: 12px; flex: none; overflow: visible; }
.basis-key { display: inline-flex; align-items: center; gap: 8px; }
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
.type-power-root { position: relative; }
.type-power-svg, .node-power-svg { display: block; width: 100%; }
/* The two by-type summaries share a row; each card keeps its own width so the
   bar charts size to their column. Stacks below ~1100px. */
.type-cards-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
.type-cards-row > .pareto-card { margin-bottom: 16px; min-width: 0; }
@media (max-width: 1100px) { .type-cards-row { grid-template-columns: 1fr; } }
.type-power-group { font-size: 11px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; fill: var(--ink-secondary); }
.type-power-tooltip .t-row { justify-content: space-between; gap: 16px; }
.type-power-tooltip .t-title { font-weight: 600; }
.type-power-tooltip .t-sub { color: var(--ink-muted); font-size: 11px; margin-bottom: 4px; }
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

// Stroke for the single "node average" line: a neutral tone that reads on both
// themes and is distinct from the host hues. (SVG stroke attributes can't take a
// CSS variable reference via setAttribute in every renderer, so it is literal.)
const TYPE_LINE_COLOR = "hsl(210 15% 62%)";
const TOTAL_LINE_COLOR = "hsl(38 80% 58%)";
const DEVAVG_LINE_COLOR = "hsl(262 55% 62%)";
const PERGPU_LINE_COLOR = "hsl(190 60% 55%)";

// Round tick positions for a linear axis over [min, max]: step is 1/2/2.5/5 x 10^n.
function linearTicks(min, max, n) {
  const rough = (max - min) / n || 1, mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => s >= rough) || mag;
  const out = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + 1e-9; v += step) out.push(Math.abs(v) < step * 1e-6 ? 0 : v);
  return out;
}

// Linear y-range for visible data in [lo, hi]. Anchors at 0 when the data comes
// within 40% of the range of it (so idle-to-load traces keep their baseline);
// otherwise starts just under the minimum so a band of lines sitting at
// 300-600 W isn't drawn in the top half of an axis that begins at 0.
function linearRange(lo, hi) {
  if (!(hi > 0)) return [0, 1];
  if (!(lo < hi)) lo = hi * 0.9;
  const span = hi - lo;
  const start = lo <= span * 0.4 ? 0 : lo - span * 0.08;
  return [start, hi + span * 0.08];
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
let clipCounter = 0;

function lowerBound(arr, target) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < target) lo = mid + 1; else hi = mid; }
  return lo;
}

// A chart group is one or more stacked time-series charts (GPU over CPU) that share
// an x domain: hovering any one drives the crosshair + tooltip on all of them, and a
// drag on any one zooms all of them. Double-click resets the zoom.
// Page-wide view preferences for every power-over-time panel, so switching
// Pareto points (each point has its own pre-rendered chart group) or moving
// between tabs keeps the chosen granularity and y-scales. Changing a toggle in
// one panel broadcasts to all.
const CHART_PREFS = { granularity: "dev", scales: {} };   // scales: chart title -> "linear" | "log"; default view = device average
const CHART_GROUPS = [];
function broadcastChartPrefs(origin) {
  CHART_GROUPS.forEach(g => { if (g !== origin && g.applyPrefs) g.applyPrefs(); });
}

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
    ylabel: sub.dataset.ylabel || "",
    data: sub.dataset.series ? JSON.parse(sub.dataset.series) : (sourceSubs ? sourceSubs[i] : []),
    deviceData: null,   // per-device series, kept while another view is shown
    nodeData: null,     // per-node sums, built on first use
    typeData: null,     // mean across the chart's nodes (one line), built on first use
    totalData: null,    // sum of every device in the chart (one line), built on first use
    devAvgData: null,   // mean across every device in the chart (one line), built on first use
    perGpuData: null,   // sum of every device in the chart / GPU count of the section (one line), built on first use
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
  const H = 220, padL = 62, padR = 12, padT = 10, padB = 22;   // padL leaves room for the rotated y-axis caption
  const plotH = H - padT - padB;
  let W = 900, plotW = W - padL - padR;
  function layout() {
    // Width from any chart that is currently laid out (a collapsed section's charts measure 0).
    const ref = charts.find(c => c.svg.getBoundingClientRect().width > 0) || charts[0];
    W = Math.max(400, Math.round(ref.svg.getBoundingClientRect().width || 900));
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
    // Clip to the plot area vertically too: with a non-zero y origin, a line can dip below the axis.
    c.clipRect = svgEl("rect", { x: padL, y: padT, width: plotW, height: plotH });
    clip.appendChild(c.clipRect);
    c.svg.insertBefore(clip, c.overlay);
    c.gGrid = svgEl("g"); c.gLines = svgEl("g", { "clip-path": "url(#" + clipId + ")" }); c.gLabels = svgEl("g");
    c.svg.insertBefore(c.gGrid, c.overlay); c.svg.insertBefore(c.gLines, c.overlay); c.svg.appendChild(c.gLabels);
  });

  charts.forEach(c => {
    c.title = (c.sub.querySelector(".chart-sub-title") || {}).textContent || "";
    c.sub.querySelectorAll(".yscale-btn").forEach(btn => btn.addEventListener("click", () => {
      CHART_PREFS.scales[c.title] = btn.dataset.scale;
      setScale(c, btn.dataset.scale);
      draw();
      broadcastChartPrefs(group);
    }));
  });
  function setScale(c, scale) {
    c.scale = scale;
    c.sub.querySelectorAll(".yscale-btn").forEach(b => b.classList.toggle("on", b.dataset.scale === scale));
  }

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
      let wMax = 0, wMin = Infinity, wMinPos = Infinity;
      const anyVisible = c.data.some((s, i) => !c.hidden.has(i));
      c.data.forEach((s, i) => {
        if (anyVisible && c.hidden.has(i)) return;
        const lo = lowerBound(s.t, tMin), hi = lowerBound(s.t, tMax + 1e-9);
        for (let j = lo; j < hi; j++) {
          const v = s.w[j];
          if (v > wMax) wMax = v; if (v < wMin) wMin = v; if (v > 0 && v < wMinPos) wMinPos = v;
        }
      });
      const [yLo, yHi] = linearRange(isFinite(wMin) ? wMin : 0, wMax);
      wMax = isFinite(wMax) && wMax > 0 ? wMax : 1;
      const isLog = c.scale === "log" && isFinite(wMinPos);
      // Log floor: one decade below the smallest positive value, clamped so zeros and
      // gaps still land on the axis instead of at -infinity.
      const logLo = isLog ? Math.pow(10, Math.floor(Math.log10(wMinPos))) : 0;
      const logHi = wMax * 1.08;
      const lgLo = isLog ? Math.log10(logLo) : 0, lgSpan = isLog ? Math.log10(logHi) - lgLo || 1 : 1;
      const y = isLog
        ? w => padT + plotH - (w <= logLo ? 0 : (Math.log10(w) - lgLo) / lgSpan) * plotH
        : w => padT + plotH - ((w - yLo) / (yHi - yLo)) * plotH;

      let ticks;
      if (isLog) {
        ticks = [];
        for (let e = Math.ceil(lgLo); Math.pow(10, e) <= logHi; e++) ticks.push(Math.pow(10, e));
        if (!ticks.length || ticks[0] > logLo) ticks.unshift(logLo);
        if (ticks.length <= 2) [2, 5].forEach(m => { const v = m * logLo; if (v < wMax) ticks.push(v); const v2 = m * logLo * 10; if (v2 < wMax) ticks.push(v2); });
        ticks.sort((a, b) => a - b);
      } else ticks = linearTicks(yLo, yHi, 4);
      ticks.forEach(v => {
        const gy = y(v);
        c.gGrid.appendChild(svgEl("line", { class: "gridline", x1: padL, x2: W - padR, y1: gy, y2: gy }));
        const label = svgEl("text", { class: "axis-text", x: padL - 6, y: gy + 3, "text-anchor": "end" });
        label.textContent = fmtAxis(v);
        c.gLabels.appendChild(label);
      });
      if (c.ylabel) {
        const cy = padT + plotH / 2;
        const cap = svgEl("text", { class: "axis-caption", x: 11, y: cy, "text-anchor": "middle", transform: "rotate(-90 11 " + cy + ")" });
        cap.textContent = c.ylabel + (isLog ? " \u2014 log" : "");
        c.gLabels.appendChild(cap);
      }
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
        // No end-of-line labels or markers: the legend chips and the hover tooltip
        // identify every line, and labels collided on charts with many series.
        const els = [poly];
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
      key.addEventListener("click", () => { setHidden(c, i, !c.hidden.has(i)); syncHostKeys(); rescale(); });
    });
  });

  function setHidden(c, i, off) {
    if (off) c.hidden.add(i); else c.hidden.delete(i);
    if (c.keys[i]) c.keys[i].classList.toggle("off", off);
    (c.lineEls[i] || []).forEach(el => { el.style.display = off ? "none" : ""; });
  }

  // Per-node view: sum each host's device lines into one. Device series are
  // min/max-downsampled independently, so their timestamps don't line up; each
  // is sampled onto a shared grid (union of all timestamps, thinned to <= 2400
  // points) by carrying the last value forward. Only power charts (unit W) with
  // more than one device per host are affected.
  function sharedGrid(series) {
    const allT = []; series.forEach(s => s.t.forEach(v => allT.push(v)));
    allT.sort((a, b) => a - b);
    const grid = []; const stride = Math.max(1, Math.floor(allT.length / 2400));
    for (let i = 0; i < allT.length; i += stride) if (!grid.length || allT[i] > grid[grid.length - 1]) grid.push(allT[i]);
    return grid;
  }
  // Sum of the given series sampled onto grid (last value carried forward), scaled by k.
  function sumOnGrid(series, grid, k) {
    const w = new Float64Array(grid.length);
    series.forEach(s => {
      let j = 0;
      for (let g = 0; g < grid.length; g++) {
        while (j + 1 < s.t.length && s.t[j + 1] <= grid[g]) j++;
        if (s.t[j] <= grid[g]) w[g] += s.w[j];
      }
    });
    return Array.from(w, v => Math.round(v * k * 10) / 10);
  }
  function statsOf(arr) {
    const sorted = [...arr].sort((a, b) => a - b), q = f => sorted[Math.min(sorted.length - 1, Math.floor(f * (sorted.length - 1)))];
    return { mean: arr.reduce((a, b) => a + b, 0) / arr.length, min: sorted[0], p50: q(0.5), p95: q(0.95), max: sorted[sorted.length - 1] };
  }
  function deviceNoun(series) { return series[0].label.includes("socket") ? "sockets" : "GPUs"; }
  // Always applicable: a host with one device just yields that device's line under
  // the node's name, so every chart in the panel shows the same granularity.
  function buildNodeData(series) {
    const byHost = new Map();
    series.forEach(s => { if (s.host) (byHost.get(s.host) || byHost.set(s.host, []).get(s.host)).push(s); });
    if (!byHost.size) return null;
    const grid = sharedGrid(series);
    const out = [];
    byHost.forEach((devs, host) => {
      const arr = sumOnGrid(devs, grid, 1);
      out.push({ label: host + " (" + devs.length + " " + deviceNoun(devs) + ")", host, roles: [...new Set(devs.flatMap(s => s.roles || []))],
        color: devs[0].color, pattern: 0, t: grid, w: arr, stats: statsOf(arr) });
    });
    return out;
  }
  // Node-type average: this chart already holds one role's devices (or every
  // device when the run has a single role), so its hosts *are* the node type.
  // One line = sum over all devices / number of hosts, i.e. the mean node draw.
  // A single-node type is the mean of one: the same line as its node sum, labelled
  // as such, so the view stays consistent across sections.
  function buildTypeData(series, heading) {
    const hosts = new Set(series.map(s => s.host).filter(Boolean));
    if (!hosts.size) return null;
    const grid = sharedGrid(series);
    const arr = sumOnGrid(series, grid, 1 / hosts.size);
    const roles = [...new Set(series.flatMap(s => s.roles || []))];
    const what = (heading || (roles.length ? roles.join("+") : "all")) + (hosts.size === 1 ? " node" : " nodes");
    const count = hosts.size === 1 ? [...hosts][0] : hosts.size + " nodes";
    return [{ label: (hosts.size === 1 ? "node total \u2014 " : "mean per node \u2014 ") + what + " (" + count + ", " + series.length + " " + deviceNoun(series) + ")",
      host: "", roles, color: TYPE_LINE_COLOR, pattern: 0, t: grid, w: arr, stats: statsOf(arr) }];
  }
  function rebuildLegend(c) {
    const legend = c.sub.querySelector(".legend-details .legend");
    const hint = c.sub.querySelector(".legend-details .fold-hint");
    if (!legend) return;
    legend.innerHTML = "";
    c.data.forEach(s => {
      const key = document.createElement("span"); key.className = "legend-key"; key.title = "click to hide/show";
      const sw = document.createElement("span"); sw.className = "legend-swatch"; sw.style.background = s.color;
      key.appendChild(sw); key.appendChild(document.createTextNode(s.label)); legend.appendChild(key);
    });
    if (hint) hint.textContent = "\u2014 " + c.data.length + (c.data.length === 1 ? " line" : " lines") + (c.data.length > 1 ? "; click to show or hide individual " + (c.granularity === "node" ? "nodes" : "GPUs / sockets") : "");
    const table = c.sub.querySelector(".stats-table tbody");
    if (table) {
      table.innerHTML = "";
      c.data.forEach(s => {
        const tr = document.createElement("tr");
        [s.label, s.stats.mean, s.stats.min, s.stats.p50, s.stats.p95, s.stats.max].forEach((v, k) => {
          const td = document.createElement("td"); td.textContent = k === 0 ? v : fmtNum(v, 2); tr.appendChild(td);
        });
        table.appendChild(tr);
      });
    }
    c.keys = [...legend.querySelectorAll(".legend-key")];
    c.keys.forEach((key, i) => key.addEventListener("click", () => { setHidden(c, i, !c.hidden.has(i)); syncHostKeys(); rescale(); }));
  }
  // Total: every device in the chart summed -- e.g. all decode GPUs, or all
  // prefill sockets -- so the section's aggregate draw can be read directly.
  function buildTotalData(series, heading) {
    if (!series.length) return null;
    const hosts = new Set(series.map(s => s.host).filter(Boolean));
    const grid = sharedGrid(series);
    const arr = sumOnGrid(series, grid, 1);
    const roles = [...new Set(series.flatMap(s => s.roles || []))];
    const what = (heading || (roles.length ? roles.join("+") : "all")) + (hosts.size === 1 ? " node" : " nodes");
    return [{ label: "total \u2014 " + what + " (" + hosts.size + (hosts.size === 1 ? " node, " : " nodes, ") + series.length + " " + deviceNoun(series) + ")",
      host: "", roles, color: TOTAL_LINE_COLOR, pattern: 0, t: grid, w: arr, stats: statsOf(arr) }];
  }
  // Device average: sum over every device in the chart / device count -- the mean
  // draw of one GPU (or one socket) on this node type, comparable across topologies.
  function buildDevAvgData(series, heading) {
    if (!series.length) return null;
    const hosts = new Set(series.map(s => s.host).filter(Boolean));
    const grid = sharedGrid(series);
    const arr = sumOnGrid(series, grid, 1 / series.length);
    const roles = [...new Set(series.flatMap(s => s.roles || []))];
    const what = (heading || (roles.length ? roles.join("+") : "all")) + (hosts.size === 1 ? " node" : " nodes");
    const noun = deviceNoun(series) === "sockets" ? "socket" : "GPU";
    return [{ label: "mean per " + noun + " \u2014 " + what + " (" + series.length + " " + deviceNoun(series) + " on " + hosts.size + (hosts.size === 1 ? " node)" : " nodes)"),
      host: "", roles, color: DEVAVG_LINE_COLOR, pattern: 0, t: grid, w: arr, stats: statsOf(arr) }];
  }
  // Strictly per GPU: every W chart normalised by the *GPU* count of its section,
  // so the CPU line reads "CPU watts per GPU" (total CPU / GPUs) rather than per
  // socket. For the GPU chart this equals the device average. GPU count comes from
  // the section's GPU chart; a CPU-only section has none and keeps the device average.
  function sectionGpuCount(c) {
    // The GPU chart living in the same role section (<details.chart-section>) as
    // chart c; the panel holds several sections, each with its own GPU count.
    const section = c.sub.closest(".chart-section");
    const gpuChart = charts.find(o => o.unit === "W" && (!section || o.sub.closest(".chart-section") === section)
      && (o.deviceData || o.data).length && deviceNoun(o.deviceData || o.data) === "GPUs");
    return gpuChart ? (gpuChart.deviceData || gpuChart.data).length : 0;
  }
  function buildPerGpuData(c, series, heading) {
    if (!series.length) return null;
    const gpus = sectionGpuCount(c);
    if (!gpus) return null;
    const hosts = new Set(series.map(s => s.host).filter(Boolean));
    const grid = sharedGrid(series);
    const arr = sumOnGrid(series, grid, 1 / gpus);
    const roles = [...new Set(series.flatMap(s => s.roles || []))];
    const what = (heading || (roles.length ? roles.join("+") : "all")) + (hosts.size === 1 ? " node" : " nodes");
    const noun = deviceNoun(series) === "sockets" ? "CPU watts" : "GPU watts";
    return [{ label: noun + " per GPU \u2014 " + what + " (" + series.length + " " + deviceNoun(series) + " \u00f7 " + gpus + " GPUs)",
      host: "", roles, color: PERGPU_LINE_COLOR, pattern: 0, t: grid, w: arr, stats: statsOf(arr) }];
  }
  function chartHeading(c) {
    // "Prefill nodes — GPU power (W)" -> "prefill"; plain "GPU power (W)" -> null
    const title = c.sub.querySelector(".chart-sub-title");
    const m = title && title.textContent.match(/^(.*?) nodes? \u2014/);
    return m ? m[1].toLowerCase() : null;
  }
  let granularity = "device";
  function setGranularity(gran) {
    granularity = gran;
    charts.forEach(c => {
      if (c.unit !== "W") return;
      const base = c.deviceData || c.data;
      let next = null;
      if (gran === "node") {
        if (c.nodeData === null) c.nodeData = buildNodeData(base) || false;
        next = c.nodeData || null;
      } else if (gran === "type") {
        if (c.typeData === null) c.typeData = buildTypeData(base, chartHeading(c)) || false;
        next = c.typeData || null;
      } else if (gran === "total") {
        if (c.totalData === null) c.totalData = buildTotalData(base, chartHeading(c)) || false;
        next = c.totalData || null;
      } else if (gran === "dev") {
        if (c.devAvgData === null) c.devAvgData = buildDevAvgData(base, chartHeading(c)) || false;
        next = c.devAvgData || null;
      } else if (gran === "gpu") {
        if (c.perGpuData === null) c.perGpuData = buildPerGpuData(c, base, chartHeading(c)) || false;
        if (!c.perGpuData) { if (c.devAvgData === null) c.devAvgData = buildDevAvgData(base, chartHeading(c)) || false; }
        next = c.perGpuData || c.devAvgData || null;
      } else next = c.deviceData;
      if (!next) { if (gran === "device") return; next = base; }   // only when series carry no host at all
      if (!c.deviceData) c.deviceData = c.data;
      c.data = next;
      c.granularity = gran;
      c.hidden.clear();
      rebuildLegend(c);
    });
    group.querySelectorAll(".gran-btn").forEach(b => b.classList.toggle("on", b.dataset.gran === gran));
    syncHostKeys();
    draw();
  }
  group.querySelectorAll(".gran-btn").forEach(btn => btn.addEventListener("click", () => {
    CHART_PREFS.granularity = btn.dataset.gran;
    setGranularity(btn.dataset.gran);
    broadcastChartPrefs(group);
  }));
  // Bring this panel in line with the page-wide preferences (called at init and
  // whenever another panel changes a toggle). Cheap when nothing changed.
  group.applyPrefs = () => {
    let changed = false;
    charts.forEach(c => {
      const want = CHART_PREFS.scales[c.title];
      if (want && want !== c.scale) { setScale(c, want); changed = true; }
    });
    if (CHART_PREFS.granularity !== granularity) { granularity = CHART_PREFS.granularity; setGranularity(granularity); changed = false; }
    else if (changed) draw();
  };
  CHART_GROUPS.push(group);
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
      syncHostKeys(); rescale();
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

  // Adopt page-wide prefs first (setGranularity draws), then the initial layout/draw.
  charts.forEach(c => { const want = CHART_PREFS.scales[c.title]; if (want) setScale(c, want); });
  if (CHART_PREFS.granularity !== "device") setGranularity(CHART_PREFS.granularity);
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
  // Charts inside a section that was collapsed at init (or while the panel was
  // hidden) were never measured; lay out again when a section opens.
  const relayoutOnOpen = el => el.addEventListener("toggle", () => { if (el.open) { layout(); draw(); } });
  group.querySelectorAll("details.chart-section").forEach(relayoutOnOpen);
  const card = group.closest("details.conc-card");
  if (card) relayoutOnOpen(card);
}

document.querySelectorAll(".chart-group").forEach(initChartGroup);

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
  { key: "cpu_est",  label: "CPU (estimated, not measured)", color: "hsl(35 30% 60%)" },
  { key: "overhead", label: "Rack overhead (projected, assumed)", color: "hsl(262 45% 60%)" },
];
// Assumed additions from the point's power budget (see GPU_POWER_BUDGETS in Python):
// per-GPU rack overhead, and a per-socket CPU stand-in when the run has no CPU leg.
function overheadPerGpu(point) { return point && point.budget ? point.budget.overhead_w_per_gpu || 0 : 0; }
function cpuEstimatePerGpu(point) {
  return point && point.budget && point.cpu_estimated ? point.budget.cpu_estimate_w_per_gpu || 0 : 0;
}
// Category visibility for the by-type chart, shared across points (a legend
// click persists like the granularity toggle).
const TYPE_POWER_HIDDEN = new Set();
const TYPE_ROLE_ORDER = ["prefill", "decode"];
function roleKeyOf(n) {
  const r = (n.roles || []).filter(Boolean).sort();
  return r.length ? r.join("+") : "";
}
function roleHeading(key) {
  if (key === "prefill") return "Prefill nodes";
  if (key === "decode") return "Decode nodes";
  if (!key) return "Nodes without a worker role";
  return key.charAt(0).toUpperCase() + key.slice(1) + " nodes";
}

function rerenderBreakdownCards() {
  document.querySelectorAll(".type-power-card, .node-power-card").forEach(c => {
    if (c.__point) (c.classList.contains("type-power-card") ? renderTypePower : renderNodePower)(c, c.__point);
  });
}

function renderTypePower(card, point) {
  card.__point = point;
  const svg = card.querySelector("svg.type-power-svg");
  const legend = card.querySelector(".type-power-legend");
  const sub = card.querySelector(".type-power-sub");
  const tip = card.querySelector(".type-power-tooltip");
  const notices = card.querySelector(".type-power-notices");
  const nodes = point.node_power || [];
  svg.innerHTML = ""; legend.innerHTML = "";
  card.hidden = !nodes.length;
  if (!nodes.length) return;
  sub.textContent = point.label;
  const powerWarnings = (point.warnings || []).filter(w => /power (missing|not collected)/i.test(w));
  notices.innerHTML = ""; powerWarnings.forEach(w => { const d = document.createElement("div"); d.textContent = "\u26a0 " + w; notices.appendChild(d); });
  notices.hidden = !powerWarnings.length;

  // Group nodes by role; per group compute mean W per GPU and per socket (with rails).
  const groups = new Map();
  nodes.forEach(n => { const k = roleKeyOf(n); (groups.get(k) || groups.set(k, []).get(k)).push(n); });
  const keys = [...groups.keys()].sort((a, b) => {
    const ia = TYPE_ROLE_ORDER.indexOf(a), ib = TYPE_ROLE_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || (a === "" ? 1 : b === "" ? -1 : a.localeCompare(b));
  });
  const anyRails = nodes.some(n => n.cpu_rails_w && Object.keys(n.cpu_rails_w).length);
  const rows = [];   // { group, label, segs:[[key, w]], total, meta }
  keys.forEach(k => {
    const ns = groups.get(k);
    const gpuN = ns.reduce((a, n) => a + (n.gpu_count || 0), 0);
    const gpuW = ns.reduce((a, n) => a + (n.gpu_w || 0), 0);
    const sockN = ns.reduce((a, n) => a + (n.socket_count || 0), 0);
    const cpuW = ns.reduce((a, n) => a + (n.cpu_w || 0), 0);
    const heading = roleHeading(k);
    if (gpuN > 0) {
      rows.push({ group: heading, label: "per GPU", segs: [["gpu", gpuW / gpuN]], meta: gpuN + " GPUs on " + ns.length + (ns.length === 1 ? " node" : " nodes") });
    }
    if (sockN === 0 && cpuEstimatePerGpu(point) > 0 && gpuN > 0) {
      const b = point.budget, perSocket = cpuEstimatePerGpu(point) * b.gpus_per_node / b.sockets_per_node;
      rows.push({ group: heading, label: "per CPU socket", segs: [["cpu_est", perSocket]],
        meta: "assumed: " + fmtNum(cpuEstimatePerGpu(point), 0) + " W/GPU \u00d7 " + b.gpus_per_node + " GPUs / " + b.sockets_per_node + " sockets; CPU not measured" });
    }
    if (sockN > 0) {
      const segs = [];
      if (anyRails) {
        let acc = 0;
        ["cpu_rail", "soc", "dram"].forEach(rk => {
          const tot = ns.reduce((a, n) => a + ((n.cpu_rails_w || {})[rk] || 0), 0);
          if (tot > 0) { segs.push([rk, tot / sockN]); acc += tot / sockN; }
        });
        if (cpuW / sockN - acc > 0.5) segs.push(["cpu_rest", cpuW / sockN - acc]);
        if (!segs.length) segs.push(["cpu", cpuW / sockN]);
      } else segs.push(["cpu", cpuW / sockN]);
      rows.push({ group: heading, label: "per CPU socket", segs, meta: sockN + " sockets on " + ns.length + (ns.length === 1 ? " node" : " nodes") });
    }
    // Assumed rack overhead gets its own row so the measured per-GPU bar stays a measurement.
    if (gpuN > 0 && overheadPerGpu(point) > 0)
      rows.push({ group: heading, label: "overhead per GPU", segs: [["overhead", overheadPerGpu(point)]], meta: "assumed rack overhead (projected basis); not measured" });
  });

  const used = new Set(rows.flatMap(r => r.segs.map(([k]) => k)));
  NODE_POWER_SEGMENTS.filter(sg => used.has(sg.key)).forEach(sg => {
    const key = document.createElement("span"); key.className = "legend-key" + (TYPE_POWER_HIDDEN.has(sg.key) ? " off" : "");
    key.title = "click to hide/show this category";
    const sw = document.createElement("span"); sw.className = "legend-swatch"; sw.style.background = sg.color; sw.style.height = "10px";
    key.appendChild(sw); key.appendChild(document.createTextNode(sg.label)); legend.appendChild(key);
    key.addEventListener("click", () => {
      if (TYPE_POWER_HIDDEN.has(sg.key)) TYPE_POWER_HIDDEN.delete(sg.key); else TYPE_POWER_HIDDEN.add(sg.key);
      rerenderBreakdownCards();
    });
  });
  card.__point = point;

  const visRows = rows.map(r => ({ ...r, segs: r.segs.filter(([k]) => !TYPE_POWER_HIDDEN.has(k)) }))
    .map(r => ({ ...r, total: r.segs.reduce((a, [, v]) => a + v, 0) }));
  const maxW = Math.max(...visRows.map(r => r.total), 0) * 1.08 || 1;
  const W = Math.max(400, Math.round(svg.getBoundingClientRect().width || 900));
  const rowH = 22, groupGap = 14, padL = 120, padR = 64, padT = 6, padB = 28;
  const groupsInOrder = [...new Set(visRows.map(r => r.group))];
  const H = padT + visRows.length * rowH + (groupsInOrder.length) * groupGap + padB;
  svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.style.height = H + "px";
  const x = v => padL + (v / maxW) * (W - padL - padR);
  linearTicks(0, maxW, 5).forEach(v => {
    const gx = x(v);
    svg.appendChild(svgEl("line", { class: "gridline", x1: gx, x2: gx, y1: padT, y2: H - padB }));
    const t = svgEl("text", { class: "axis-text", x: gx, y: H - padB + 12, "text-anchor": "middle" }); t.textContent = fmtAxis(v); svg.appendChild(t);
  });
  const xt = svgEl("text", { class: "axis-title", x: padL + (W - padL - padR) / 2, y: H - 4, "text-anchor": "middle" });
  xt.textContent = "average watts per device (measured window)"; svg.appendChild(xt);

  let yCursor = padT, lastGroup = null;
  visRows.forEach(r => {
    if (r.group !== lastGroup) {
      yCursor += groupGap;
      const gl = svgEl("text", { class: "type-power-group", x: 4, y: yCursor - 3 }); gl.textContent = r.group; svg.appendChild(gl);
      lastGroup = r.group;
    }
    const y = yCursor + 3, h = rowH - 6;
    const lbl = svgEl("text", { class: "axis-text", x: padL - 8, y: y + h / 2 + 3, "text-anchor": "end" }); lbl.textContent = r.label; svg.appendChild(lbl);
    let acc = 0;
    r.segs.forEach(([k, v]) => {
      const sg = NODE_POWER_SEGMENTS.find(s2 => s2.key === k);
      const rect = svgEl("rect", { x: x(acc), y, width: Math.max(0, x(acc + v) - x(acc)), height: h, fill: sg.color, stroke: "var(--surface)", "stroke-width": 1 });
      rect.addEventListener("pointerenter", ev => {
        tip.innerHTML = "<div class='t-title'>" + r.group + " \u2014 " + r.label + "</div><div class='t-sub'>" + r.meta + "</div>";
        r.segs.forEach(([k2, vv]) => { const s2 = NODE_POWER_SEGMENTS.find(z => z.key === k2);
          tip.innerHTML += "<div class='t-row'><span><span class='t-key' style='display:inline-block;width:10px;height:10px;margin-right:6px;background:" + s2.color + "'></span>" + s2.label + "</span><span class='t-val'>" + fmtNum(vv, 1) + " W</span></div>"; });
        tip.innerHTML += "<div class='t-row'><span>total shown</span><span class='t-val'>" + fmtNum(r.total, 1) + " W</span></div>";
        tip.style.opacity = 1;
      });
      rect.addEventListener("pointermove", ev => { const rc = card.querySelector(".type-power-root").getBoundingClientRect(); tip.style.left = (ev.clientX - rc.left + 12) + "px"; tip.style.top = (ev.clientY - rc.top + 12) + "px"; });
      rect.addEventListener("pointerleave", () => { tip.style.opacity = 0; });
      svg.appendChild(rect); acc += v;
    });
    const val = svgEl("text", { class: "axis-text", x: x(acc) + 6, y: y + h / 2 + 3 }); val.textContent = fmtNum(r.total, 0) + " W"; svg.appendChild(val);
    yCursor += rowH;
  });
}

// Mean node per worker role: average each component across the role's nodes.
function meanNodesByType(nodes) {
  const groups = new Map();
  nodes.forEach(n => { const k = roleKeyOf(n); (groups.get(k) || groups.set(k, []).get(k)).push(n); });
  const keys = [...groups.keys()].sort((a, b) => {
    const ia = TYPE_ROLE_ORDER.indexOf(a), ib = TYPE_ROLE_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || (a === "" ? 1 : b === "" ? -1 : a.localeCompare(b));
  });
  return keys.map(k => {
    const ns = groups.get(k);
    const mean = pick => { const vals = ns.map(pick).filter(v => v != null); return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null; };
    const railKeys = [...new Set(ns.flatMap(n => Object.keys(n.cpu_rails_w || {})))];
    const rails = {}; railKeys.forEach(rk => { const m = mean(n => (n.cpu_rails_w || {})[rk]); if (m != null) rails[rk] = m; });
    return { hostname: roleHeading(k) + " (mean of " + ns.length + (ns.length === 1 ? " node)" : " nodes)"),
      gpu_w: mean(n => n.gpu_w), cpu_w: mean(n => n.cpu_w), cpu_rails_w: rails, roles: k ? k.split("+") : [],
      gpu_count: mean(n => n.gpu_count), socket_count: mean(n => n.socket_count) };
  });
}

function renderNodePower(card, point) {
  card.__point = point;
  if (card.classList.contains("type-node-card")) point = { ...point, node_power: meanNodesByType(point.node_power || []) };
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
  // Group by node type: prefill, decode, other roles, then role-less hosts; the
  // by-type card's rows are already one per type so grouping is a no-op there.
  const byType = !card.classList.contains("type-node-card") && nodes.some(n => (n.roles || []).length);
  const roleRank = k => { const i = TYPE_ROLE_ORDER.indexOf(k); return i < 0 ? (k === "" ? 999 : 99) : i; };
  const ordered = byType
    ? [...nodes].sort((a, b) => roleRank(roleKeyOf(a)) - roleRank(roleKeyOf(b)) || roleKeyOf(a).localeCompare(roleKeyOf(b)) || a.hostname.localeCompare(b.hostname))
    : nodes;
  const rows = ordered.map(n => {
    const segs = [];
    if (n.gpu_w != null) segs.push(["gpu", n.gpu_w]);
    if (n.cpu_w != null) {
      if (anyRails && n.cpu_rails_w && Object.keys(n.cpu_rails_w).length) {
        let acc = 0;
        ["cpu_rail", "soc", "dram"].forEach(k => { if (n.cpu_rails_w[k] != null) { segs.push([k, n.cpu_rails_w[k]]); acc += n.cpu_rails_w[k]; } });
        if (n.cpu_w - acc > 0.5) segs.push(["cpu_rest", n.cpu_w - acc]);
      } else segs.push(["cpu", n.cpu_w]);
    } else if (cpuEstimatePerGpu(point) > 0 && (n.socket_count == null || n.socket_count === 0) && n.gpu_count)
      segs.push(["cpu_est", cpuEstimatePerGpu(point) * n.gpu_count]);
    if (overheadPerGpu(point) > 0 && n.gpu_count) segs.push(["overhead", overheadPerGpu(point) * n.gpu_count]);
    return { host: n.hostname, group: byType ? roleHeading(roleKeyOf(n)) : null, segs, total: segs.reduce((a, [, v]) => a + v, 0) };
  });
  const groupCount = byType ? new Set(rows.map(r => r.group)).size : 0;
  const used = new Set(rows.flatMap(r => r.segs.map(([k]) => k)));
  NODE_POWER_SEGMENTS.filter(sg => used.has(sg.key)).forEach(sg => {
    const key = document.createElement("span"); key.className = "legend-key" + (TYPE_POWER_HIDDEN.has(sg.key) ? " off" : "");
    key.title = "Click to hide/show this category on all power-breakdown charts";
    const sw = document.createElement("span"); sw.className = "legend-swatch"; sw.style.background = sg.color; sw.style.height = "10px";
    key.appendChild(sw); key.appendChild(document.createTextNode(sg.label)); legend.appendChild(key);
    key.addEventListener("click", () => {
      if (TYPE_POWER_HIDDEN.has(sg.key)) TYPE_POWER_HIDDEN.delete(sg.key); else TYPE_POWER_HIDDEN.add(sg.key);
      rerenderBreakdownCards();
    });
  });
  // Hidden categories drop out of the bars and the totals; the axis rescales to what's left.
  rows.forEach(r => { r.segs = r.segs.filter(([k]) => !TYPE_POWER_HIDDEN.has(k)); r.total = r.segs.reduce((a, [, v]) => a + v, 0); });

  const W = Math.max(400, Math.round(svg.getBoundingClientRect().width || 900));
  const rowH = 22, groupGap = 16, padL = card.classList.contains("type-node-card") ? 180 : 110, padR = 60, padT = 8, padB = 28;
  const H = padT + rows.length * rowH + groupCount * groupGap + padB;
  svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.style.height = H + "px";
  const maxW = Math.max(0, ...rows.map(r => r.total)) * 1.05 || 1;
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

  let yCursor = padT, lastGroup = null;
  rows.forEach(r => {
    if (byType && r.group !== lastGroup) {
      yCursor += groupGap;
      const gl = svgEl("text", { class: "type-power-group", x: 4, y: yCursor - 4 }); gl.textContent = r.group; svg.appendChild(gl);
      lastGroup = r.group;
    }
    const y = yCursor + 3, h = rowH - 6;
    yCursor += rowH;
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
  tps_per_gpu_w:   { label: "Output tok/s per GPU watt", key: "tps_per_gpu_w",   better: "max" },
  tps_per_total_w: { label: "Output tok/s per watt (GPU+CPU)", key: "tps_per_total_w", better: "max" },
  gpu_w:           { label: "Total GPU watts",           key: "gpu_w",           better: "min" },
  gpu_w_per_gpu:   { label: "Watts per GPU",             key: "gpu_w_per_gpu",   better: "min" },
  cpu_w:           { label: "Total CPU watts",           key: "cpu_w",           better: "min" },
  total_w:         { label: "Total watts (GPU+CPU)",     key: "total_w",         better: "min" },
  concurrency:     { label: "Concurrency",               key: "concurrency",     better: "max" },
  // Basis-split axes: one series (own frontier, own line style) per power basis.
  // The metric key is suffixed with the variant key ("total_tps_per_mw__static").
  total_tps_per_mw:  { label: "Total TPS / MW",  key: "total_tps_per_mw",  better: "max", split: true },
  input_tps_per_mw:  { label: "Input TPS / MW",  key: "input_tps_per_mw",  better: "max", split: true },
  output_tps_per_mw: { label: "Output TPS / MW", key: "output_tps_per_mw", better: "max", split: true },
  node_w_per_gpu:    { label: "Watts per GPU",   key: "node_w_per_gpu",    better: "min", split: true },
};

// Power bases for the split axes (mirrors _POWER_VARIANTS in Python). Each has its
// own frontier line style and point rendering so the series read apart even when
// they share a family colour. `legend(p)` builds the chip text from the point's
// budget figures so the assumed constants are always visible next to the data.
const POWER_VARIANTS = [
  { key: "measured", label: "Measured CPU+GPU", dash: "", hollow: false, tone: "base",
    legend: () => "Measured CPU+GPU" },
  { key: "projected", label: "Projected avg-rack", dash: "6 4", hollow: false, tone: "light",
    legend: p => "Projected avg-rack" + (p ? " \u00b7 +" + fmtNum(p.budget.overhead_w_per_gpu, 0) + " W/GPU" : "") },
  { key: "static", label: "Static budget", dash: "2 4", hollow: false, tone: "grey",
    legend: p => "Static budget" + (p ? " \u00b7 " + fmtNum(p.budget.static_w_per_gpu, 0) + " W/GPU" : "") },
];
const NO_VARIANT = { key: "", dash: "", hollow: false, tone: "base" };
// Legend/marker fill for a basis. (SVG fill attributes can't take a CSS variable
// via setAttribute in every renderer -- an earlier hollow "var(--surface)" fill rendered black.)
function variantFill(base, v) { return variantColor(base, v); }
// Marker: circle for fully measured points, diamond when the point's CPU power is an
// estimate from the budget (no CPU leg collected) -- the estimate flows into every
// basis of that point, so all of its markers take the diamond.
function paretoMarker(p, cx, cy, r, fill) {
  if (!p.cpu_estimated) return svgEl("circle", { class: "pareto-point", cx, cy, r, fill });
  const d = r * 1.25;
  return svgEl("polygon", { class: "pareto-point estimated", fill,
    points: cx + "," + (cy - d) + " " + (cx + d) + "," + cy + " " + cx + "," + (cy + d) + " " + (cx - d) + "," + cy });
}
function variantStroke(base, v) { return v.tone === "grey" ? "hsl(0 0% 30%)" : "var(--surface)"; }
// Family colours are "hsl(H 70% 52%)"; derive the projected (lighter) and static
// (desaturated grey) tones from the hue so the series stay tied to their family.
function variantColor(base, variant) {
  // No regex here: this JS lives in a non-raw Python string and backslash escapes get mangled.
  if (!base || base.indexOf("hsl(") !== 0) return base;
  const hue = base.slice(4).split(" ")[0];
  if (variant.tone === "light") return "hsl(" + hue + " 60% 70%)";
  if (variant.tone === "grey") return "hsl(" + hue + " 10% 62%)";
  return base;
}

// Every scatter registers here so the budget editor can redraw them all.
const PARETO_VIEWS = [];
const BUDGET_FIELD_LABELS = ["Measured GPU + estimated CPU (avg)", "Projected avg-rack power (approx.)", "Static power budget"];
const BUDGET_WARNING_RE = /^(No power budget for GPU type|CPU power not measured for this run)/;
const POWER_BASES = ["measured", "projected", "static"];

// Mirror of _power_variant_watts / _power_variant_metrics / the inspect-field and
// warning text in Python: rewrite one point in place for a (possibly null) budget so
// every consumer (scatter, legend chips, inspect panel, bar cards) sees the new
// assumption without a rebuild.
function applyBudget(p, b) {
  p.budget = b;
  const gpuW = p.m.gpu_w, cpuW = p.m.cpu_w, n = p.num_gpus;
  const w = { measured: null, projected: null, static: null };
  let est = false;
  if (gpuW != null && n) {
    if (cpuW != null) w.measured = gpuW + cpuW;
    else if (b && b.cpu_estimate_w_per_gpu != null) { w.measured = gpuW + b.cpu_estimate_w_per_gpu * n; est = true; }
    if (b) {
      if (w.measured != null) w.projected = w.measured + b.overhead_w_per_gpu * n;
      w.static = b.static_w_per_gpu * n;
    }
  }
  p.power_basis_w = w; p.cpu_estimated = est;
  const out = p.m.output_tps, tot = p.m.total_tps, inp = (out == null || tot == null) ? null : tot - out;
  const perMw = (r, ww) => (r == null || !ww) ? null : r / (ww / 1e6);
  POWER_BASES.forEach(k => {
    p.m["total_tps_per_mw__" + k] = perMw(tot, w[k]); p.m["input_tps_per_mw__" + k] = perMw(inp, w[k]);
    p.m["output_tps_per_mw__" + k] = perMw(out, w[k]); p.m["node_w_per_gpu__" + k] = (w[k] == null || !n) ? null : w[k] / n;
  });
  // Inspect-panel rows, spliced in after the measured total.
  const f0 = fmtNum;
  const fields = p.fields.filter(([k]) => !BUDGET_FIELD_LABELS.includes(k));
  const add = [];
  if (est && b) add.push(["Measured GPU + estimated CPU (avg)", f0(w.measured, 0) + " W (" + f0(b.cpu_estimate_w_per_gpu, 0) + " W per GPU assumed)"]);
  if (b) {
    add.push(["Projected avg-rack power (approx.)", f0(w.projected, 0) + " W (+" + f0(b.overhead_w_per_gpu, 0) + " W/GPU overhead)"]);
    add.push(["Static power budget", f0(w.static, 0) + " W (" + f0(b.static_node_w, 0) + " W/node \u00f7 " + b.gpus_per_node + " GPUs)"]);
  }
  let at = fields.findIndex(([k]) => k === "Total watts, GPU+CPU (avg)");
  at = at < 0 ? fields.length : at + 1;
  fields.splice(at, 0, ...add);
  p.fields = fields;
  const warnings = (p.warnings || []).filter(x => !BUDGET_WARNING_RE.test(x));
  if (!b) warnings.push("No power budget for GPU type '" + (p.gpu_type_key || "unknown") + "': projected and static series unavailable (add it to GPU_POWER_BUDGETS).");
  else if (est) warnings.push("CPU power not measured for this run: the CPU+GPU, projected and static-vs-measured comparisons use an assumed " + f0(b.cpu_estimate_w_per_gpu, 0) + " W per GPU.");
  p.warnings = warnings;
}

function initBudgetCard(card) {
  if (!card) return;
  const rows = [...card.querySelectorAll(".budget-row[data-gpu-type]")];
  const num = el => { const v = parseFloat(el.value); return el.value.trim() === "" || !isFinite(v) ? null : v; };
  function budgetFromRow(row) {
    const d = JSON.parse(row.dataset.budgetDefaults);
    const get = f => num(row.querySelector('[data-budget-field="' + f + '"]'));
    const staticPerGpu = get("static_w_per_gpu"), overhead = get("overhead_w_per_gpu"), cpuEst = get("cpu_estimate_w_per_gpu");
    if (staticPerGpu == null && overhead == null && cpuEst == null) return null;   // no budget at all
    return { static_w_per_gpu: staticPerGpu || 0, static_node_w: (staticPerGpu || 0) * d.gpus_per_node, overhead_w_per_gpu: overhead || 0,
      cpu_estimate_w_per_gpu: cpuEst, gpus_per_node: d.gpus_per_node, sockets_per_node: d.sockets_per_node };
  }
  function apply() {
    rows.forEach(row => {
      const gt = row.dataset.gpuType, b = budgetFromRow(row), d = JSON.parse(row.dataset.budgetDefaults);
      row.querySelectorAll("input").forEach(el => {
        const dv = d[el.dataset.budgetField]; el.classList.toggle("changed", (dv == null ? "" : String(dv)) !== (num(el) == null ? "" : String(num(el))));
      });
      const derived = row.querySelector(".budget-derived");
      derived.textContent = b ? fmtNum(b.static_node_w, 0) + " W static per node \u00b7 " + fmtNum(b.overhead_w_per_gpu * b.gpus_per_node, 0) + " W overhead per node"
        + (b.cpu_estimate_w_per_gpu != null ? " \u00b7 " + fmtNum(b.cpu_estimate_w_per_gpu * b.gpus_per_node / b.sockets_per_node, 0) + " W CPU est. per socket" : "") : "no budget: measured series only";
      PARETO_VIEWS.forEach(v => v.points.forEach(p => { if ((p.gpu_type_key || "") === gt) applyBudget(p, b); }));
    });
    PARETO_VIEWS.forEach(v => v.refresh());
    rerenderBreakdownCards();
  }
  card.querySelectorAll("input").forEach(el => { el.addEventListener("change", apply); el.addEventListener("input", apply); });
  function resetToDefaults() {
    rows.forEach(row => { const d = JSON.parse(row.dataset.budgetDefaults);
      row.querySelectorAll("input").forEach(el => { const v = d[el.dataset.budgetField]; el.value = v == null ? "" : String(v); }); });
  }
  card.querySelector(".budget-reset").addEventListener("click", () => { resetToDefaults(); apply(); });
  // Every page load starts from the defaults: browsers restore form values across a
  // reload / back-forward, which would silently carry an edited assumption into a
  // fresh view of the page while the embedded metrics still hold the defaults.
  resetToDefaults();
  window.addEventListener("pageshow", ev => { if (ev.persisted) { resetToDefaults(); apply(); } });
  // Initial pass only fills the "Derived" column (points already carry the Python-computed values).
  rows.forEach(row => { const b = budgetFromRow(row); row.querySelector(".budget-derived").textContent = b
    ? fmtNum(b.static_node_w, 0) + " W static per node \u00b7 " + fmtNum(b.overhead_w_per_gpu * b.gpus_per_node, 0) + " W overhead per node"
      + (b.cpu_estimate_w_per_gpu != null ? " \u00b7 " + fmtNum(b.cpu_estimate_w_per_gpu * b.gpus_per_node / b.sockets_per_node, 0) + " W CPU est. per socket" : "")
    : "no budget: measured series only"; });
}

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
  const frontierOnlyBox = card.querySelector("input[data-frontier-only]");
  const frontierOnly = () => !!(frontierOnlyBox && frontierOnlyBox.checked);
  const modelSel = card.querySelector("select[data-model]");
  const modelOk = p => !modelSel || !modelSel.value || p.model === modelSel.value;
  // Baseline mode: Y is divided by the baseline run's Y at the same concurrency.
  const baseSel = card.querySelector("select[data-baseline]");
  const normalize = !!baseSel;
  const runLegend = card.querySelector(".run-legend");
  const W = 900, H = 520, padR = 20, padT = 14, padB = 44;
  // Left padding is set per redraw from the widest y tick label so the rotated
  // axis title never sits under the numbers (e.g. 9-digit TPS/MW ticks).
  let padL = 64, plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // Legend, frontier lines and hide/show all work on the point's group (GPU type x model).
  const runs = [...new Set(points.map(p => p.group))];
  const hiddenRuns = new Set();
  const hiddenVariants = new Set();
  const basisLegend = card.querySelector(".basis-legend");
  let selected = 0;
  let plotted = [];   // indexes into points that have both metrics
  let items = [];     // [{ i, v }] -- one drawn marker per (point, power basis); v = NO_VARIANT on plain axes
  let xAxis, yAxis, xScale, yScale;
  const variantsFor = axis => axis.split ? POWER_VARIANTS : [NO_VARIANT];
  const variantKey = (axis, v) => axis.split ? axis.key + "__" + v.key : axis.key;

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

  function metric(p, axis, v) { const val = p.m[variantKey(axis, v || NO_VARIANT)]; return val === null || val === undefined ? null : val; }
  function baselineFor(p) {
    if (!normalize) return null;
    return points.find(b => b.run === baseSel.value && b.m.concurrency === p.m.concurrency
      && b.bench === p.bench) || null;
  }
  function yVal(p, v) {
    const raw = metric(p, yAxis, v);
    if (!normalize) return raw;
    const b = baselineFor(p);
    if (raw === null || !b) return null;
    const bv = metric(b, yAxis, v);
    return bv === null || bv === 0 ? null : raw / bv;
  }
  // Basis chips: shown only on a split axis; the label carries the budget figure of
  // the first point that has one so the assumption is visible on the chart.
  function renderBasisLegend() {
    if (!basisLegend) return;
    basisLegend.innerHTML = "";
    basisLegend.hidden = !yAxis.split;
    if (!yAxis.split) return;
    POWER_VARIANTS.forEach(v => {
      const has = points.filter(p => modelOk(p) && metric(p, yAxis, v) !== null);
      if (!has.length) return;
      const withBudget = has.find(p => p.budget) || null;
      const key = document.createElement("span");
      key.className = "run-key basis-key" + (hiddenVariants.has(v.key) ? " off" : "");
      const sw = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      sw.setAttribute("viewBox", "0 0 34 12"); sw.setAttribute("class", "basis-swatch");
      const line = svgEl("line", { x1: 1, x2: 33, y1: 6, y2: 6, stroke: variantColor(has[0].color, v), "stroke-width": 2 });
      if (v.dash) line.setAttribute("stroke-dasharray", v.dash);
      sw.appendChild(line);
      sw.appendChild(svgEl("circle", { cx: 17, cy: 6, r: 4, fill: variantFill(has[0].color, v),
        stroke: variantStroke(has[0].color, v), "stroke-width": 1.5 }));
      key.appendChild(sw);
      key.appendChild(document.createTextNode(v.legend(withBudget)));
      key.title = "Click to hide or show this power basis";
      key.addEventListener("click", () => {
        if (hiddenVariants.has(v.key)) hiddenVariants.delete(v.key); else hiddenVariants.add(v.key);
        draw();
      });
      basisLegend.appendChild(key);
    });
    const est = points.filter(p => modelOk(p) && p.cpu_estimated);
    if (est.length) {
      const key = document.createElement("span");
      key.className = "run-key basis-key basis-shape-key";
      const sw = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      sw.setAttribute("viewBox", "0 0 34 12"); sw.setAttribute("class", "basis-swatch");
      sw.appendChild(svgEl("polygon", { points: "17,1 22,6 17,11 12,6", fill: est[0].color, stroke: "var(--surface)", "stroke-width": 1 }));
      key.appendChild(sw);
      const b = est[0].budget;
      key.appendChild(document.createTextNode("\u25c6 = CPU not measured; " + fmtNum(b.cpu_estimate_w_per_gpu, 0) + " W/GPU assumed (" + est.length + " point" + (est.length === 1 ? "" : "s") + ")"));
      key.title = "CPU power collection failed on these runs; the CPU leg is the budget's per-socket estimate";
      key.style.cursor = "default";
      basisLegend.appendChild(key);
    }
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

  function frontier(its) {
    // Sort by x ascending; walk keeping items not dominated on y (given each axis's direction).
    const xs = its.map(it => [it, metric(points[it.i], xAxis), yVal(points[it.i], it.v)]);
    xs.sort((a, b) => xAxis.better === "max" ? a[1] - b[1] : b[1] - a[1]);
    // Walk from best-x to worst-x: an item is on the frontier if its y beats every better-x item's y.
    const ordered = xs.reverse();
    const out = [];
    let bestY = null;
    ordered.forEach(([it, , y]) => {
      const beats = bestY === null || (yAxis.better === "max" ? y > bestY : y < bestY);
      if (beats) { out.push(it); bestY = y; }
    });
    return out.reverse();
  }

  function draw() {
    xAxis = PARETO_AXES[xSel.value]; yAxis = PARETO_AXES[ySel.value];
    renderBasisLegend();
    items = [];
    points.forEach((p, i) => {
      if (!modelOk(p) || metric(p, xAxis) === null) return;
      variantsFor(yAxis).forEach(v => { if (yVal(p, v) !== null) items.push({ i, v }); });
    });
    plotted = [...new Set(items.map(it => it.i))];
    // Legend chips for families outside the chosen model drop out with their points.
    runLegend.querySelectorAll(".run-key").forEach(key => {
      key.hidden = !points.some(p => p.group === key.dataset.group && modelOk(p));
    });
    // Keep the selection on a visible point.
    if (plotted.length && !plotted.includes(selected)) { selected = plotted[0]; renderPanel(points[selected]); }
    gGrid.innerHTML = ""; gLines.innerHTML = ""; gPoints.innerHTML = ""; gAxes.innerHTML = "";
    if (!plotted.length) return;

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    items.forEach(it => {
      const px = metric(points[it.i], xAxis), py = yVal(points[it.i], it.v);
      if (px < xMin) xMin = px; if (px > xMax) xMax = px;
      if (py < yMin) yMin = py; if (py > yMax) yMax = py;
    });
    xMin = Math.min(0, xMin);
    if (normalize) { yMin = Math.min(yMin, 1); yMax = Math.max(yMax, 1); yMin -= (yMax - yMin) * 0.1 || 0.1; }
    else yMin = Math.min(0, yMin);
    xMax += (xMax - xMin) * 0.06 || 1; yMax += (yMax - yMin) * 0.08 || 1;
    // Tick labels: SI suffix once values reach the thousands (unless normalized, where
    // the ratios stay small), matching the time-series axes.
    const yTicks = niceTicks(yMin, yMax, 6).map(([v, d]) => [v, normalize ? fmtNum(v, Math.max(d, 2)) : (Math.abs(yMax) >= 1e4 ? fmtAxis(v) : fmtNum(v, d))]);
    const tickW = Math.max(0, ...yTicks.map(([, s]) => s.length)) * 6.6;   // ~11px axis-text; measured without a layout pass
    padL = Math.round(Math.max(64, 30 + tickW + 8));   // 30 = rotated title + gap
    plotW = W - padL - padR;
    xScale = v => padL + ((v - xMin) / (xMax - xMin)) * plotW;
    yScale = v => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    yTicks.forEach(([v, s]) => {
      const gy = yScale(v);
      gGrid.appendChild(svgEl("line", { class: "gridline", x1: padL, x2: W - padR, y1: gy, y2: gy }));
      const t = svgEl("text", { class: "axis-text", x: padL - 8, y: gy + 3, "text-anchor": "end" });
      t.textContent = s;
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
    const note = card.querySelector(".baseline-note, .pareto-note");
    if (note) {
      const dropped = normalize ? points.filter(p => metric(p, xAxis) !== null && metric(p, yAxis) !== null && yVal(p) === null).length : 0;
      note.textContent = dropped ? dropped + " point(s) hidden: no baseline measurement at the same concurrency." : "";
    }

    // One frontier per (family, power basis): on a split axis the measured,
    // projected and static series of one family each get their own line style.
    const frontierSet = new Set();
    const itemHidden = it => hiddenRuns.has(points[it.i].group) || hiddenVariants.has(it.v.key);
    if (drawFrontier) runs.forEach(run => {
      if (hiddenRuns.has(run)) return;
      variantsFor(yAxis).forEach(v => {
        if (hiddenVariants.has(v.key)) return;
        const its = items.filter(it => points[it.i].group === run && it.v === v);
        const front = frontier(its);
        front.forEach(it => frontierSet.add(it));
        if (front.length >= 2) {
          const d = front.map(it => xScale(metric(points[it.i], xAxis)) + "," + yScale(yVal(points[it.i], it.v))).join(" ");
          const line = svgEl("polyline", { class: "pareto-frontier", points: d, stroke: variantColor(points[its[0].i].color, v) });
          if (v.dash) line.setAttribute("stroke-dasharray", v.dash);
          gLines.appendChild(line);
        }
      });
    });
    // "Frontier points only": dominated items are left out entirely (the axis
    // range and the frontier are computed from the full set, so nothing shifts).
    const shown = frontierOnly() && drawFrontier ? items.filter(it => frontierSet.has(it) || itemHidden(it)) : items;
    if (frontierOnly() && drawFrontier && note) {
      const n = items.length - shown.length;
      if (n) note.textContent = (note.textContent ? note.textContent + " " : "") + n + " dominated point(s) hidden (frontier only).";
    }

    shown.forEach(it => {
      const p = points[it.i], v = it.v, col = variantColor(p.color, v);
      const c = paretoMarker(p, xScale(metric(p, xAxis)), yScale(yVal(p, v)), it.i === selected ? 8 : 6, variantFill(p.color, v));
      if (v.tone === "grey") c.classList.add("grey");
      if (itemHidden(it)) c.classList.add("dim");
      if (it.i === selected) c.classList.add("selected");
      c.addEventListener("click", () => select(it.i));
      c.addEventListener("pointerenter", (ev) => showTip(p, ev, v));
      c.addEventListener("pointermove", (ev) => moveTip(ev));
      c.addEventListener("pointerleave", () => { tooltip.style.opacity = 0; });
      gPoints.appendChild(c);
    });
  }

  // Values that rest on the CPU estimate get a hoverable warning glyph so a reader
  // scanning numbers can't miss that part of the watts is assumed, not measured.
  const EST_WARN_TEXT = p => "CPU power was not measured for this run; " + fmtNum(p.budget.cpu_estimate_w_per_gpu, 0) + " W per GPU (from the power budget) is assumed in this value.";
  const EST_FIELD_PREFIXES = ["Measured GPU + estimated CPU", "Projected avg-rack power", "Output tok/s / (GPU+CPU) W", "Total watts, GPU+CPU"];
  function estWarn(p) {
    const w = document.createElement("span"); w.className = "est-warn"; w.textContent = "\\u26a0"; w.title = EST_WARN_TEXT(p);
    return w;
  }
  function showTip(p, ev, v) {
    tooltip.innerHTML = "";
    const title = document.createElement("div");
    title.className = "t-title";
    title.textContent = p.label;
    tooltip.appendChild(title);
    // A basis whose watts include the CPU estimate: measured and projected (static is budget-only).
    const estBasis = p.cpu_estimated && v && v.key && v.key !== "static";
    const rows = [[xAxis.label, fmtNum(metric(p, xAxis))], [yLabel(), normalize ? fmtNum(yVal(p, v), 3) : fmtNum(yVal(p, v)), estBasis]];
    if (yAxis.split && v && v.key) rows.push(["Power basis", v.legend(p.budget ? p : null), estBasis],
      ["Total watts on this basis", fmtNum(p.power_basis_w[v.key], 0) + " W", estBasis]);
    if (p.cpu_estimated) rows.push(["CPU power", "not measured \\u2014 " + fmtNum(p.budget.cpu_estimate_w_per_gpu, 0) + " W/GPU assumed", true]);
    if (normalize) rows.push([yAxis.label + " (raw)", fmtNum(metric(p, yAxis, v)), estBasis]);
    p.hover.forEach(([k, v]) => rows.push([k, v]));
    rows.forEach(([k, v, est]) => {
      const row = document.createElement("div");
      row.className = "t-row";
      const kEl = document.createElement("span"); kEl.textContent = k; kEl.style.color = "var(--ink-muted)";
      const vEl = document.createElement("span"); vEl.className = "t-val"; vEl.textContent = v;
      if (est) vEl.insertBefore(estWarn(p), vEl.firstChild);
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
      if (p.cpu_estimated && value !== "n/a" && !String(value).startsWith("n/a ") && EST_FIELD_PREFIXES.some(pre => label.startsWith(pre))) v.insertBefore(estWarn(p), v.firstChild);
      cell.appendChild(l); cell.appendChild(v);
      panel.appendChild(cell);
    });
  }

  function select(i) {
    selected = i;
    renderPanel(points[i]);
    if (root.dataset.drivesCharts !== "off") {
      document.querySelectorAll(".node-power-card").forEach(card => renderNodePower(card, points[i]));
      document.querySelectorAll(".type-power-card").forEach(card => renderTypePower(card, points[i]));
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

  // Budget editor hook: recompute is done on the shared point objects; this just redraws.
  PARETO_VIEWS.push({ points, refresh: () => { draw(); if (plotted.length) select(selected); } });
  if (modelSel) modelSel.addEventListener("change", () => { draw(); if (plotted.length) select(selected); });
  if (xSel.addEventListener) xSel.addEventListener("change", draw);
  if (frontierOnlyBox) frontierOnlyBox.addEventListener("change", draw);
  if (ySel.addEventListener) ySel.addEventListener("change", draw);
  if (baseSel) baseSel.addEventListener("change", draw);
  select(0);
}

document.querySelectorAll(".pareto-root").forEach(initPareto);
initBudgetCard(document.querySelector(".budget-card"));
document.querySelectorAll(".node-power-card[data-point]").forEach(card => renderNodePower(card, JSON.parse(card.dataset.point)));
document.querySelectorAll(".type-power-card[data-point]").forEach(card => renderTypePower(card, JSON.parse(card.dataset.point)));
// Bar cards derive their pixel width at render time; redraw when their column resizes.
if (window.ResizeObserver) {
  let barPending = null;
  const barObserver = new ResizeObserver(() => {
    if (barPending) return;
    barPending = requestAnimationFrame(() => {
      barPending = null;
      document.querySelectorAll(".node-power-card, .type-power-card").forEach(card => {
        if (card.hidden || !card.__point) return;
        if (card.classList.contains("type-power-card")) renderTypePower(card, card.__point); else renderNodePower(card, card.__point);
      });
    });
  });
  document.querySelectorAll(".node-power-card, .type-power-card").forEach(card => barObserver.observe(card));
}

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
    '<tr><th>Run</th><th class="cg-perf cg-start">Output tok/s</th><th class="cg-perf">Tok/s/GPU</th>'
    '<th class="cg-perf">TPOT p50 (ms)</th><th class="cg-perf">TPOT p90 (ms)</th>'
    '<th class="cg-gpu cg-start">Total GPU watts</th><th class="cg-gpu">Watts per GPU</th>'
    '<th class="cg-cpu cg-start">Total CPU watts</th><th class="cg-cpu">Watts per CPU socket</th><th class="cg-cpu">CPU watts per GPU</th>'
    '<th class="cg-eff cg-start">Tok/s per GPU watt</th><th class="cg-eff">Tok/s per CPU watt</th>'
    '<th class="cg-eff">Tok/s per watt (GPU+CPU)</th></tr>'
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
            f"<td class='cg-perf cg-start'>{_fmt(ppw['output_tokens_per_second'])}</td>"
            f"<td class='cg-perf'>{_fmt(ppw['output_tokens_per_second_per_gpu'])} ({ppw['num_gpus']} gpu)</td>"
            f"<td class='cg-perf'>{_fmt(w['tpot_p50_ms'])}</td>"
            f"<td class='cg-perf'>{_fmt(w['tpot_p90_ms'])}</td>"
            f"<td class='cg-gpu cg-start'>{_fmt(ppw['gpu_avg_power_w'], 0)}</td>"
            f"<td class='cg-gpu'>{_fmt(_per_gpu(ppw['gpu_avg_power_w'], ppw['num_gpus']), 0)}</td>"
            f"<td class='cg-cpu cg-start'>{_fmt(ppw['cpu_avg_power_w'], 0)}</td>"
            f"<td class='cg-cpu'>{_fmt(_per_gpu(ppw['cpu_avg_power_w'], _socket_count(r)), 0)}</td>"
            f"<td class='cg-cpu'>{_fmt(_per_gpu(ppw['cpu_avg_power_w'], ppw['num_gpus']), 1)}</td>"
            f"<td class='cg-eff cg-start'>{_fmt(ppw['output_tokens_per_second_per_gpu_watt'], 4)}</td>"
            f"<td class='cg-eff'>{_fmt(ppw['output_tokens_per_second_per_cpu_watt'], 4)}</td>"
            f"<td class='cg-eff'>{_fmt(ppw['output_tokens_per_second_per_combined_watt'], 4)}</td>"
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


def _hosts_summary(hosts: list[str] | None, *, full: bool = False) -> str:
    """Allocated hosts, compact for the hover tooltip and complete for the inspect
    panel. Tooltip form collapses a same-prefix numeric range: ``nvl72d090-T10…T18 (9)``."""
    if not hosts:
        return "—"
    hosts = sorted(set(hosts))
    if full or len(hosts) <= 3:
        return ", ".join(hosts) + (f" ({len(hosts)})" if len(hosts) > 3 else "")
    m_first, m_last = re.match(r"^(.*?)(\d+)$", hosts[0]), re.match(r"^(.*?)(\d+)$", hosts[-1])
    if m_first and m_last and m_first.group(1) == m_last.group(1):
        return f"{hosts[0]}…{m_last.group(2)} ({len(hosts)})"
    return f"{hosts[0]} … {hosts[-1]} ({len(hosts)})"


def _socket_count(report: dict) -> int:
    """Number of CPU sockets that reported power in this window (one per-socket energy row each)."""
    return len(report.get("cpu_per_socket") or ())


def _per_gpu(total: float | None, num_gpus: int | None) -> float | None:
    return None if total is None or not num_gpus else total / num_gpus


# -- Power budgets per GPU type --------------------------------------------------
#
# The Pareto view plots efficiency against three power bases side by side:
#   measured   -- window-average CPU+GPU watts from the collectors (nothing assumed);
#   projected  -- measured plus a fixed per-GPU overhead for everything the collectors
#                 don't see (NVLink switches, fans, PSU loss, ...), an approximation of
#                 the rack's average draw;
#   static     -- a fixed nameplate budget per node, i.e. what a datacenter provisions,
#                 independent of what the run actually drew.
# The numbers are assumptions, not measurements. Edit them here (one record per GPU
# type, keyed by the lower-cased ``resources.gpu_type`` from the run's config.yaml).
# A GPU type with no entry gets the measured series only, plus a warning.


@dataclass(frozen=True)
class GpuPowerBudget:
    """Fixed power assumptions for one GPU type (see the block comment above)."""

    static_node_w: float
    """Provisioned / nameplate power of one node, watts."""
    gpus_per_node: int
    overhead_w_per_gpu: float
    """Unmeasured rack overhead attributed to each active GPU, watts (added to measured CPU+GPU)."""
    sockets_per_node: int = 2
    cpu_estimate_w_per_gpu: float | None = None
    """Amortised CPU watts per active GPU, the stand-in for the CPU leg when the run has
    no CPU measurements at all; ``None`` leaves such points without a measured value."""

    @property
    def static_w_per_gpu(self) -> float:
        return self.static_node_w / self.gpus_per_node

    @property
    def cpu_estimate_w_per_socket(self) -> float | None:
        """The per-GPU estimate expressed per socket (display only)."""
        if self.cpu_estimate_w_per_gpu is None:
            return None
        return self.cpu_estimate_w_per_gpu * self.gpus_per_node / self.sockets_per_node


GPU_POWER_BUDGETS: dict[str, GpuPowerBudget] = {
    # Source for every number: Kyle Liang (SemiAnalysis), Slack, Sep 2026 -- the
    # "avg-per-GPU rack power" (static), "avg everything-else power" (overhead) and the
    # amortised CPU-per-GPU figures used on SA's dashboard.
    #
    # GB300 NVL72 compute tray: 4 GPUs + 2 Grace sockets.
    #   static   2,120 W/GPU  -> 8,480 W per 4-GPU node
    #   overhead   670 W/GPU  ("avg everything-else power")
    #   CPU stand-in when CPU collection failed: 50 W/GPU (amortised CPU power per GPU)
    "gb300": GpuPowerBudget(
        static_node_w=8_480.0,
        gpus_per_node=4,
        overhead_w_per_gpu=670.0,
        sockets_per_node=2,
        cpu_estimate_w_per_gpu=50.0,
    ),
    # VR NVL72 (Vera Rubin) compute tray, assumed 4 GPUs + 2 Vera sockets.
    #   static   3,300 W/GPU  -> 13,200 W per 4-GPU node
    #   overhead   900 W/GPU
    #   CPU stand-in: 100 W/GPU
    # No VR run has been reported yet, so the config.yaml ``gpu_type`` spelling is a
    # guess; ``GPU_TYPE_ALIASES`` maps the likely variants onto this entry.
    "vr200": GpuPowerBudget(
        static_node_w=13_200.0,
        gpus_per_node=4,
        overhead_w_per_gpu=900.0,
        sockets_per_node=2,
        cpu_estimate_w_per_gpu=100.0,
    ),
}

# Alternate ``resources.gpu_type`` spellings -> budget key.
GPU_TYPE_ALIASES: dict[str, str] = {
    "vr": "vr200",
    "vr-nvl72": "vr200",
    "vr_nvl72": "vr200",
    "vera_rubin": "vr200",
    "vera-rubin": "vr200",
    "rubin": "vr200",
}


# Power bases in display order. ``key`` suffixes every per-basis Pareto metric
# (``total_tps_per_mw__projected``); ``POWER_VARIANTS`` in ``_JS`` mirrors this list
# and owns the line styles.
_POWER_VARIANTS: tuple[str, ...] = ("measured", "projected", "static")


def _power_budget_for(gpu_type: str | None) -> GpuPowerBudget | None:
    if not gpu_type:
        return None
    key = gpu_type.lower()
    return GPU_POWER_BUDGETS.get(GPU_TYPE_ALIASES.get(key, key))


def _power_variant_watts(
    *, gpu_w: float | None, cpu_w: float | None, num_gpus: int | None, budget: GpuPowerBudget | None
) -> tuple[dict[str, float | None], bool]:
    """``({basis: total watts}, cpu_estimated)`` for one concurrency point.

    ``measured`` is GPU + CPU; when the CPU leg is missing and the budget carries a
    per-socket estimate, that estimate stands in and ``cpu_estimated`` is True (the
    chart draws the point with a different marker). With neither, ``measured`` is
    None -- never silently GPU-only. ``projected`` adds the per-GPU overhead to
    ``measured``; ``static`` is the budget alone times the active GPU count.
    """
    out: dict[str, float | None] = dict.fromkeys(_POWER_VARIANTS)
    if gpu_w is None or not num_gpus:
        return out, False
    estimated = False
    if cpu_w is not None:
        out["measured"] = gpu_w + cpu_w
    elif budget is not None and budget.cpu_estimate_w_per_gpu is not None:
        out["measured"] = gpu_w + budget.cpu_estimate_w_per_gpu * num_gpus
        estimated = True
    if budget is None:
        return out, estimated
    if out["measured"] is not None:
        out["projected"] = out["measured"] + budget.overhead_w_per_gpu * num_gpus
    out["static"] = budget.static_w_per_gpu * num_gpus
    return out, estimated


def _per_mw(rate: float | None, watts: float | None) -> float | None:
    return None if rate is None or not watts else rate / (watts / 1e6)


def _power_variant_metrics(
    *,
    output_tps: float | None,
    total_tps: float | None,
    num_gpus: int | None,
    watts: dict[str, float | None],
) -> dict[str, float | None]:
    """``m`` entries for the basis-split axes: ``<axis>__<variant>`` for total, input
    and output tok/s per MW plus watts per GPU (node power / active GPUs)."""
    input_tps = None if output_tps is None or total_tps is None else total_tps - output_tps
    m: dict[str, float | None] = {}
    for variant in _POWER_VARIANTS:
        w = watts.get(variant)
        m[f"total_tps_per_mw__{variant}"] = _per_mw(total_tps, w)
        m[f"input_tps_per_mw__{variant}"] = _per_mw(input_tps, w)
        m[f"output_tps_per_mw__{variant}"] = _per_mw(output_tps, w)
        m[f"node_w_per_gpu__{variant}"] = _per_gpu(w, num_gpus)
    return m


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
    gpu_type: str | None = None,
    hosts: list[str] | None = None,
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
        budget = _power_budget_for(gpu_type)
        variant_w, cpu_estimated = _power_variant_watts(
            gpu_w=ppw["gpu_avg_power_w"], cpu_w=ppw["cpu_avg_power_w"], num_gpus=num_gpus, budget=budget
        )
        basis_fields: list[tuple[str, str]] = []
        if cpu_estimated and budget is not None:
            basis_fields.append(
                (
                    "Measured GPU + estimated CPU (avg)",
                    f"{_fmt(variant_w['measured'], 0)} W ({_fmt(budget.cpu_estimate_w_per_gpu, 0)} W per GPU assumed)",
                )
            )
        if budget is not None:
            basis_fields.append(
                (
                    "Projected avg-rack power (approx.)",
                    f"{_fmt(variant_w['projected'], 0)} W (+{_fmt(budget.overhead_w_per_gpu, 0)} W/GPU overhead)",
                )
            )
            basis_fields.append(
                (
                    "Static power budget",
                    f"{_fmt(variant_w['static'], 0)} W ({_fmt(budget.static_node_w, 0)} W/node ÷ {budget.gpus_per_node} GPUs)",
                )
            )
        point_warnings = [w for w in r.get("warnings", []) if "CPU energy mismatch" in w] + list(
            coverage_warnings or ()
        )
        if budget is None:
            point_warnings.append(
                f"No power budget for GPU type {gpu_type or 'unknown'!r}: projected and static series unavailable "
                "(add it to GPU_POWER_BUDGETS)."
            )
        elif cpu_estimated:
            point_warnings.append(
                "CPU power not measured for this run: the CPU+GPU, projected and static-vs-measured comparisons use "
                f"an assumed {_fmt(budget.cpu_estimate_w_per_gpu, 0)} W per GPU."
            )
        points.append(
            {
                "id": _point_id(run_label, r),
                "label": label,
                "bench": r["benchmark_type"],
                "run": run_label or "",
                "group": group,
                "model": model or "",
                "gpu_type_key": (gpu_type or "").lower(),
                "num_gpus": num_gpus,
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
                    **_power_variant_metrics(
                        output_tps=output_tps, total_tps=total_tps, num_gpus=num_gpus, watts=variant_w
                    ),
                },
                "power_basis_w": variant_w,
                "cpu_estimated": cpu_estimated,
                "budget": None
                if budget is None
                else {
                    "static_node_w": budget.static_node_w,
                    "static_w_per_gpu": budget.static_w_per_gpu,
                    "overhead_w_per_gpu": budget.overhead_w_per_gpu,
                    "cpu_estimate_w_per_gpu": budget.cpu_estimate_w_per_gpu,
                    "gpus_per_node": budget.gpus_per_node,
                    "sockets_per_node": budget.sockets_per_node,
                },
                "hover": [
                    ("GPU type / hosts", f"{gpu_type or '—'} · {_hosts_summary(hosts)}"),
                    ("Concurrency / GPUs", f"{r['concurrency']} / {num_gpus}"),
                    ("P90 TPOT", f"{_fmt(r['tpot_p90_ms'])} ms"),
                    ("Total GPU watts", f"{_fmt(ppw['gpu_avg_power_w'], 0)} W"),
                    ("Watts per GPU", f"{_fmt(_per_gpu(ppw['gpu_avg_power_w'], num_gpus), 0)} W"),
                    ("Output tok/s per GPU watt", _fmt(ppw["output_tokens_per_second_per_gpu_watt"], 3)),
                ],
                "fields": [
                    ("Run", run_label or "—"),
                    ("GPU type", gpu_type or "—"),
                    ("Hosts", _hosts_summary(hosts, full=True)),
                    ("Concurrency / active GPUs", f"{r['concurrency']} / {num_gpus}"),
                    ("Output tok/s", _fmt(output_tps)),
                    ("Output tok/s / active GPU", _fmt(tps_per_gpu)),
                    ("Total tok/s (in + out)", _fmt(total_tps)),
                    ("P50 TPOT", f"{_fmt(r['tpot_p50_ms'])} ms"),
                    ("P90 TPOT", f"{_fmt(r['tpot_p90_ms'])} ms"),
                    ("1 / P90 TPOT", f"{_fmt(_inverse_ms(r['tpot_p90_ms']), 1)} tok/s/user"),
                    ("Total GPU watts (all GPUs, avg)", f"{_fmt(ppw['gpu_avg_power_w'], 0)} W"),
                    ("Watts per GPU (avg)", f"{_fmt(_per_gpu(ppw['gpu_avg_power_w'], num_gpus), 1)} W"),
                    ("Total CPU watts (all sockets, avg)", f"{_fmt(ppw['cpu_avg_power_w'], 0)} W"),
                    (
                        "Watts per CPU socket (avg)",
                        f"{_fmt(_per_gpu(ppw['cpu_avg_power_w'], _socket_count(r)), 1)} W"
                        + (f" ({_socket_count(r)} sockets)" if _socket_count(r) else ""),
                    ),
                    ("CPU watts per GPU (avg)", f"{_fmt(_per_gpu(ppw['cpu_avg_power_w'], num_gpus), 1)} W"),
                    ("Total watts, GPU+CPU (avg)", f"{_fmt(combined_w, 0)} W"),
                    *basis_fields,
                    ("Output tok/s per GPU watt", _fmt(ppw["output_tokens_per_second_per_gpu_watt"], 4)),
                    ("Output tok/s / (GPU+CPU) W", _fmt(ppw["output_tokens_per_second_per_combined_watt"], 4)),
                    ("Joules / output token", _fmt(r["joules_per_output_token"], 4)),
                    ("Measured window", f"{_fmt(duration, 1)} s"),
                    ("CPU power source", _cpu_sensor_summary(r)),
                ],
                "warnings": point_warnings,
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
    ("tps_per_gpu_w", "Output tok/s per GPU watt"),
    ("tps_per_total_w", "Output tok/s / (GPU+CPU) W"),
    ("total_tps_per_mw", "Total TPS / MW · measured vs projected vs static"),
    ("input_tps_per_mw", "Input TPS / MW · measured vs projected vs static"),
    ("output_tps_per_mw", "Output TPS / MW · measured vs projected vs static"),
    ("node_w_per_gpu", "Watts per GPU · measured vs projected vs static"),
    ("gpu_w", "Total GPU watts"),
    ("gpu_w_per_gpu", "Watts per GPU (GPU only)"),
    ("cpu_w", "Total CPU watts"),
    ("total_w", "Total watts (GPU+CPU)"),
    ("concurrency", "Concurrency"),
)
_PARETO_DEFAULT_X = "inv_tpot_p90"
_PARETO_DEFAULT_Y = "total_tps_per_mw"


def _axis_select_html(axis: str, default: str) -> str:
    options = "".join(
        f'<option value="{key}"{" selected" if key == default else ""}>{html.escape(label)}</option>'
        for key, label in _PARETO_AXIS_OPTIONS
    )
    return f'<select data-axis="{axis}">{options}</select>'


def _model_select_html(points: list[dict]) -> str:
    """Model dropdown for the Pareto scatter, always rendered so the page states which
    model is plotted. Defaults to the first model so unrelated models aren't drawn on
    the same frontier by default -- "All models" is still there for a deliberate overlay.
    Omitted only when no point carries a model name at all."""
    models = list(dict.fromkeys(p["model"] for p in points if p.get("model")))
    if not models:
        return ""
    options = "".join(
        f'<option value="{html.escape(m, quote=True)}"{" selected" if i == 0 else ""}>{html.escape(m)}</option>'
        for i, m in enumerate(models)
    )
    return f'<label>Model <select data-model><option value="">All models</option>{options}</select></label>'


def _type_power_card_html(point: dict | None = None) -> str:
    """ "Average power per device by node type" card: for each worker role (prefill /
    decode / ...) one bar for the mean GPU draw per GPU and one for the mean CPU draw
    per socket, the latter split into rails when the collector recorded them. Legend
    keys toggle categories. Same data flow as the per-node card (``renderTypePower``)."""
    point_attr = f' data-point="{html.escape(json.dumps(point), quote=True)}"' if point is not None else ""
    what = "the selected point" if point is None else "this concurrency"
    return f"""
<div class="pareto-card type-power-card"{point_attr}>
  <div class="chart-group-head">
    <h3>Average power per device by node type</h3>
    <span class="type-power-sub pareto-subtitle" style="margin:0"></span>
  </div>
  <p class="pareto-subtitle">Window-average watts for {what}, averaged over every device of that kind on nodes of the
  same worker role: one bar per GPU (mean across the role's GPUs) and one per CPU socket (mean across its sockets; drawn
  as rails when recorded). The rack-overhead row is the budget's assumed per-GPU overhead, not a
  measurement. Click a legend key to hide or show that category.</p>
  <div class="chart-notices type-power-notices" hidden></div>
  <div class="type-power-legend legend"></div>
  <div class="type-power-root"><svg class="type-power-svg"></svg><div class="tooltip type-power-tooltip"></div></div>
</div>
"""


def _node_power_card_html(point: dict | None = None, *, by_type: bool = False) -> str:
    """ "Average power by node" stacked-bar card.

    ``by_type=True`` renders the sibling "Average node power by node type" card: one
    bar per worker role whose segments are each component's mean across that role's
    nodes (so the bar total is the mean node draw). Same markup and renderer; the JS
    collapses the nodes per role before drawing (``meanNodesByType``). On the Pareto page it follows the
    selected point (the scatter calls ``renderNodePower`` on click). For a page with
    a single concurrency point there is no scatter, so the point is embedded in
    ``data-point`` and the card renders itself at load."""
    point_attr = f' data-point="{html.escape(json.dumps(point), quote=True)}"' if point is not None else ""
    what = "the selected point" if point is None else "this concurrency"
    if by_type:
        cls = "pareto-card node-power-card type-node-card"
        heading = "Average node power by node type"
        blurb = (
            f"Window-average power for {what}, one bar per node type (worker role). Each segment is that component's "
            "mean across the type's nodes -- GPUs summed per node then averaged, likewise the CPU envelope and its rails -- "
            "so the bar total is the mean draw of one node of that type. The rack-overhead segment is the assumed "
            "per-GPU overhead from the GPU type's power budget (not measured), so the bar total matches the projected "
            "avg-rack basis. Legend keys toggle categories across all breakdown charts."
        )
    else:
        cls = "pareto-card node-power-card"
        heading = "Average power by node"
        blurb = (
            f"Window-average power for {what}, one bar per allocated node. GPU is the node's GPUs summed; CPU is the "
            "socket envelope (ACPI total / DCGM). When the collector recorded component rails, the CPU bar is drawn as "
            "its rails (CPU rail, SoC, DRAM) plus the remainder of the envelope they don't account for. "
            "Rack overhead is the assumed per-GPU overhead from the power budget (not measured); a CPU (estimated) "
            "segment appears only when the run has no CPU measurements. "
            "Click a legend key to hide or show that category on every breakdown chart."
        )
    return f"""
<div class="{cls}"{point_attr}>
  <div class="chart-group-head">
    <h3>{heading}</h3>
    <span class="node-power-sub pareto-subtitle" style="margin:0"></span>
  </div>
  <p class="pareto-subtitle">{blurb}</p>
  <div class="chart-notices node-power-notices" hidden></div>
  <div class="node-power-legend legend"></div>
  <div class="node-power-root"><svg class="node-power-svg"></svg><div class="tooltip node-power-tooltip"></div></div>
</div>
"""


def _power_basis_blurb(points: list[dict]) -> str:
    """Intro sentence for the basis-split Y axes, spelling out the assumed constants
    for every GPU type on the page so the chart never shows a number whose origin
    isn't written next to it."""
    text = (
        "The <b>measured vs projected vs static</b> Y-axis options draw one series per power basis. "
        "<b>Measured</b> uses benchmark-window averages from the assigned GPUs plus every CPU socket on "
        "participating nodes; where a run has no CPU measurements at all, a per-socket estimate stands in "
        "and the point is drawn as a diamond instead of a circle."
    )
    seen: list[tuple[str, GpuPowerBudget]] = []
    for p in points:
        gt = (p.get("gpu_type_key") or "").lower()
        b = _power_budget_for(gt)
        if b is not None and all(k != gt for k, _ in seen):
            seen.append((gt, b))
    for gt, b in seen:
        text += (
            f" <b>{html.escape(gt.upper())}</b>: <b>projected avg-rack</b> adds {b.overhead_w_per_gpu:,.0f} W per active GPU "
            f"to measured power; <b>static budget</b> is {b.static_node_w:,.0f} W per node "
            f"({b.static_w_per_gpu:,.0f} W/GPU); CPU estimate {b.cpu_estimate_w_per_gpu or 0:,.0f} W per GPU."
        )
    text += " Edit the constants in <code>GPU_POWER_BUDGETS</code>."
    return text


def _budget_card_html(points: list[dict]) -> str:
    """ "Power budget assumptions" editor under the Pareto scatter: one row per GPU type
    present on the page with the static W/GPU, overhead W/GPU and CPU-estimate
    W/GPU inputs. Edits are session-only -- the JS recomputes every basis metric,
    marker, frontier, inspect field and bar-card segment in place (``applyBudget``);
    the permanent values live in ``GPU_POWER_BUDGETS``."""
    gpu_types = list(dict.fromkeys(p.get("gpu_type_key") or "" for p in points))
    if not gpu_types:
        return ""
    rows = []
    for gt in gpu_types:
        b = _power_budget_for(gt)
        gpn = b.gpus_per_node if b else 4
        spn = b.sockets_per_node if b else 2
        defaults = {
            "static_w_per_gpu": b.static_w_per_gpu if b else None,
            "overhead_w_per_gpu": b.overhead_w_per_gpu if b else None,
            "cpu_estimate_w_per_gpu": b.cpu_estimate_w_per_gpu if b else None,
            "gpus_per_node": gpn,
            "sockets_per_node": spn,
        }
        n_points = sum(1 for p in points if (p.get("gpu_type_key") or "") == gt)

        def inp(field: str, step: str = "10", d: dict = defaults) -> str:
            v = d[field]
            val = "" if v is None else f"{v:g}"
            return (
                f'<input type="number" min="0" step="{step}" data-budget-field="{field}" value="{val}" '
                f'placeholder="n/a" autocomplete="off">'
            )

        name = html.escape(gt.upper() if gt else "unknown GPU type")
        status = "" if b else ' <span class="budget-missing" title="No entry in GPU_POWER_BUDGETS">no default</span>'
        rows.append(
            f'<div class="budget-row" data-gpu-type="{html.escape(gt, quote=True)}" '
            f'data-budget-defaults="{html.escape(json.dumps(defaults), quote=True)}">'
            f'<div class="budget-row-head"><b>{name}</b>{status}<span class="budget-meta">{n_points} point'
            f"{'s' if n_points != 1 else ''} · {gpn} GPUs / {spn} sockets per node</span></div>"
            f'<div class="budget-inputs">'
            f"<label>Static W per GPU{inp('static_w_per_gpu')}</label>"
            f"<label>Overhead W per GPU{inp('overhead_w_per_gpu')}</label>"
            f"<label>CPU estimate W per GPU{inp('cpu_estimate_w_per_gpu', '5')}</label>"
            f"</div>"
            f'<div class="budget-derived"></div></div>'
        )
    return f"""
<details class="budget-card" open>
  <summary class="budget-head"><span class="section-caret"></span><h3>Power budget assumptions</h3>
    <span class="section-hint">edits apply to every chart</span></summary>
  <p class="pareto-subtitle">Static = provisioned rack power per GPU (the <b>static budget</b> basis, × active GPUs).
  Overhead = unmeasured "everything else" per active GPU, added to measured CPU+GPU for the <b>projected avg-rack</b>
  basis and the violet bar segment. CPU estimate = amortised CPU watts per GPU, used only for runs whose CPU power was
  not collected (◆ points). Live but not saved: make changes permanent in <code>GPU_POWER_BUDGETS</code>.</p>
  {"".join(rows)}
  <div class="budget-actions"><button type="button" class="budget-reset">Reset to defaults</button></div>
</details>
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
  <p class="pareto-subtitle">{_power_basis_blurb(points)}</p>
  <div class="pareto-layout">
    <div>
      <div class="pareto-controls">
        {_model_select_html(points)}
        <label>X {_axis_select_html("x", _PARETO_DEFAULT_X)}</label>
        <label>Y {_axis_select_html("y", _PARETO_DEFAULT_Y)}</label>
        <label class="frontier-only" title="Hide points that another point of the same group beats on both axes"><input type="checkbox" data-frontier-only checked> Frontier points only</label>
      </div>
      <div class="run-legend"></div>
      <div class="basis-legend run-legend" hidden></div>
      <div class="pareto-root" data-points="{points_json}">
        <svg class="pareto-svg" viewBox="0 0 900 520" preserveAspectRatio="xMidYMid meet"></svg>
        <p class="pareto-note"></p>
        <div class="tooltip pareto-tooltip"></div>
      </div>
      <div class="pareto-warnings"></div>
    </div>
    <div class="pareto-inspect">
      <h3>Inspect a point</h3>
      <p class="pareto-panel-title"></p>
      <div class="pareto-panel"></div>
      {_budget_card_html(points)}
    </div>
  </div>
</div>
<div class="type-cards-row">
{_type_power_card_html()}
{_node_power_card_html(by_type=True)}
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
    accent: str | None = None,
) -> str:
    """Data-table cards for one run: one card per concurrency holding two views of it.
    ``accent`` (a CSS colour) tints the card so a run's cards read as a group.
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
            style = f' style="--card-accent: {html.escape(accent, quote=True)}"' if accent else ""
            cards.append(
                f'<details class="conc-card" open data-run="{html.escape(run_key, quote=True)}" data-conc="{conc}"{style}>'
                f'<summary class="conc-card-head"><span class="section-caret"></span>'
                f'<span class="conc-card-run">{html.escape(display_label)}</span>'
                f'<span class="conc-card-conc">{html.escape(bench)} c={conc}</span></summary>'
                f'<div class="conc-card-body">'
                f'<div class="view-run">{run_view}</div>'
                f'<div class="view-window" hidden>{window_view}</div>'
                "</div></details>"
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


def _ylabel_from_title(title: str) -> str:
    """ "Prefill throughput (input tok/s)" -> "Input tok/s"; "Output throughput (tok/s, aiperf timeslices)" -> "Tok/s"."""
    m = re.search(r"\(([^)]*)\)", title)
    if not m:
        return ""
    inner = m.group(1).split(",")[0].strip()
    return inner[:1].upper() + inner[1:] if inner else ""


def _chart_sub_html(
    title: str, series: list[dict], *, embed: bool = True, unit: str = "W", ylabel: str | None = None
) -> str:
    """``embed=False`` leaves ``data-series`` empty; the group's ``data-source`` tells
    the JS which sibling group to copy the (identical) series from -- see
    ``_power_charts_html``. ``ylabel`` is the rotated y-axis caption; defaults to
    "Power (W)" for power charts."""
    series_json = html.escape(json.dumps(series), quote=True) if embed else ""
    if ylabel is None:
        ylabel = "Power (W)" if unit == "W" else ""
    return f"""
<div class="chart-sub" data-series="{series_json}" data-unit="{html.escape(unit, quote=True)}" data-ylabel="{html.escape(ylabel, quote=True)}">
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
  <div class="chart-extras">
    <details class="chart-fold legend-details">
      <summary><span class="fold-caret"></span>Series toggles <span class="fold-hint">— {len(series)} line{"s" if len(series) != 1 else ""}{"; click to show or hide individual GPUs / sockets" if len(series) > 1 else ""}</span></summary>
      {_legend_html(series)}
    </details>
    <details class="chart-fold stats-details">
      <summary><span class="fold-caret"></span>Stats table <span class="fold-hint">— mean / min / p50 / p95 / max per line</span></summary>
      {_stats_table_html(series, unit)}
    </details>
  </div>
</div>
"""


_ROLE_ORDER = ("prefill", "decode")
_ROLE_HEADINGS = {"prefill": "Prefill nodes", "decode": "Decode nodes", "": "Nodes without a worker role"}


def _split_by_role(gpu_series: list[dict], cpu_series: list[dict]) -> list[tuple[str, list[dict], list[dict]]]:
    """Partition series by worker role for one chart section per role.

    Returns ``[]`` when the run has fewer than two roles (a single-role run is
    drawn as one section without headings). A device tagged with several roles
    appears in each; devices with no role fall into a trailing '' section so
    frontend-only or unmanifested hosts are still shown.
    """
    roles: set[str] = set()
    for series in (*gpu_series, *cpu_series):
        roles.update(series.get("roles") or ())
    if len(roles) < 2:
        return []
    ordered = [r for r in _ROLE_ORDER if r in roles] + sorted(roles - set(_ROLE_ORDER))
    out: list[tuple[str, list[dict], list[dict]]] = []
    for role in ordered:
        out.append(
            (
                role,
                [s for s in gpu_series if role in (s.get("roles") or ())],
                [s for s in cpu_series if role in (s.get("roles") or ())],
            )
        )
    untagged_gpu = [s for s in gpu_series if not s.get("roles")]
    untagged_cpu = [s for s in cpu_series if not s.get("roles")]
    if untagged_gpu or untagged_cpu:
        out.append(("", untagged_gpu, untagged_cpu))
    return out


def _notices_html(notices: list[str] | None) -> str:
    if not notices:
        return ""
    items = "".join(f"<div>\u26a0 {html.escape(n)}</div>" for n in notices)
    return f'<div class="chart-notices">{items}</div>'


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
    ("drain", "Drain (in-flight tail, excluded)"),
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
    busy: list[tuple[float, float, str, str, tuple[str, int]]] = []
    for r in sorted(reports, key=lambda r: r["start_unix"]):
        label = f"{r['benchmark_type']} c={r['concurrency']}"
        ws, we = r.get("warmup_start_unix"), r.get("warmup_end_unix")
        point = (r["benchmark_type"], r["concurrency"])
        if ws is not None and we is not None and we > ws:
            busy.append((ws, min(we, r["start_unix"]), "warmup", label, point))
        busy.append((r["start_unix"], r["end_unix"], "profile", label, point))
        de = r.get("drain_end_unix")
        if de is not None and de > r["end_unix"]:
            busy.append((r["end_unix"], de, "drain", label, point))
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
    embed = reuse_source is None
    extras = [(t, ser) for t, ser in (extra_subs or ()) if ser]

    def tps_chart(title_extra: str, series: list[dict]) -> str:
        return _chart_sub_html(title_extra, series, embed=embed, unit="", ylabel=_ylabel_from_title(title_extra))

    def section(role_key: str, heading: str, charts: list[str]) -> str:
        # A collapsible box per node type: the coloured heading is the toggle.
        cls = f"chart-section role-{html.escape(role_key or 'none', quote=True)}"
        n = len(charts)
        hint = f"{n} chart{'s' if n != 1 else ''}"
        return (
            f'<details class="{cls}" open><summary class="chart-role-heading">'
            f'<span class="section-caret"></span>{html.escape(heading)} <span class="section-hint">{hint}</span></summary>'
            f'<div class="chart-section-body">{"".join(charts)}</div></details>'
        )

    role_groups = _split_by_role(gpu_series, cpu_series)
    if role_groups:
        # One box per worker role (prefill, decode, then anything else, then hosts
        # with no role), each holding that role's GPU power, CPU power and -- when
        # the throughput chart's title names the role -- its throughput chart.
        placed: set[str] = set()
        for role, role_gpu, role_cpu in role_groups:
            heading = _ROLE_HEADINGS.get(role, role.capitalize() if role else "No worker role")
            charts: list[str] = []
            if role_gpu:
                charts.append(
                    _chart_sub_html(f"{heading} — GPU power (W)", role_gpu, embed=embed, ylabel="GPU power (W)")
                )
            if role_cpu:
                charts.append(
                    _chart_sub_html(f"{heading} — CPU socket power (W)", role_cpu, embed=embed, ylabel="CPU power (W)")
                )
            for title_extra, series in extras:
                if role and title_extra.lower().startswith(role.lower()):
                    charts.append(tps_chart(title_extra, series))
                    placed.add(title_extra)
            subs.append(section(role, heading, charts))
        leftover = [(t, ser) for t, ser in extras if t not in placed]
        if leftover:
            subs.append(section("throughput", "Throughput", [tps_chart(t, ser) for t, ser in leftover]))
    else:
        charts = []
        if gpu_series:
            charts.append(_chart_sub_html("GPU power (W)", gpu_series, embed=embed, ylabel="GPU power (W)"))
        if cpu_series:
            charts.append(_chart_sub_html("CPU socket power (W)", cpu_series, embed=embed, ylabel="CPU power (W)"))
        if charts:
            subs.append(section("all", "Power", charts))
        if extras:
            subs.append(section("throughput", "Throughput", [tps_chart(t, ser) for t, ser in extras]))
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
    <span class="chart-group-tools">
      <span class="granularity-toggle" title="Draw one line per device; sum each host's devices into one line per node; average the node lines of each chart (one line per node type); average every device in the chart (mean watts per GPU / per socket); divide every chart by the section's GPU count (GPU watts per GPU, CPU watts per GPU); or sum every device into one total line">
        <button type="button" class="gran-btn" data-gran="device">per GPU / socket</button><button type="button" class="gran-btn" data-gran="node">per node</button><button type="button" class="gran-btn" data-gran="type">node average</button><button type="button" class="gran-btn on" data-gran="dev">device average</button><button type="button" class="gran-btn" data-gran="gpu">per GPU</button><button type="button" class="gran-btn" data-gran="total">total</button>
      </span>
      <span class="zoom-hint">drag to zoom</span>
    </span>
  </div>
  {_notices_html(notices)}
  {_phase_legend_html(bands)}
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
        metric = cast(dict[str, object], metric).get("value")
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
    run_labels: list[str | None] = [None] * len(bundles) if single_run else list(labels)

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
                gpu_type=bundle.get("gpu_type"),
                hosts=bundle.get("hosts"),
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
        node_card = (
            '<div class="type-cards-row">'
            + _type_power_card_html(points[0])
            + _node_power_card_html(points[0], by_type=True)
            + "</div>"
            + _node_power_card_html(points[0])
            if points and points[0].get("node_power")
            else ""
        )
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
        "hosts": sorted({s["host"] for s in (*gpu_series, *cpu_series) if s.get("host")}),
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
