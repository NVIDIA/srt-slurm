#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diff two ``perf_dashboard.json`` payloads: what moved between two runs.

The HTML is often unreadable where it lands -- a headless node, a CI log, an S3
prefix -- and comparing two runs by eye across ~80 panels is not a thing anyone
actually does. This turns an A/B into one command over the machine-readable payload
that the page embeds.

Built for the ablation shape: change ONE variable, run both, and ask which signals
responded. That is also the honest test of whether a panel earns its place -- a panel
that never moves between a healthy run and a known-broken one is not diagnostic.

    python3 tools/dashboard_diff.py BASELINE.json IMPROVED.json [--top N] [--min-pct P]

Reports, in order:
  * KPI deltas
  * tab availability changes (a leg present in one run and not the other)
  * per-panel deltas, ranked by relative movement
  * panels that exist in only one payload
  * request/session population changes

Comparison is on the MEDIAN of each series, not the mean: these series are routinely
heavy-tailed (one 10s iteration among thousands of 13ms ones), and a mean would report
the tail moving as though the body had. Where the median is zero on BOTH sides it
cannot discriminate -- sparse counter rates and event counters sit at zero through
their idle windows -- so the comparison falls back to the peak, and every row states
which statistic it used. Without that fallback an 8x concurrency change reported 30
panels as "identical" when only 8 genuinely were.

Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys


def _median(series: dict) -> float | None:
    """Median across every point of every series in a panel."""
    vals = [v for pts in series.values() for _, v in pts if v is not None]
    return statistics.median(vals) if vals else None


def _peak(series: dict) -> float | None:
    """Max across every point of every series in a panel."""
    vals = [v for pts in series.values() for _, v in pts if v is not None]
    return max(vals) if vals else None


def _compare(series_a: dict, series_b: dict):
    """(stat_name, a, b) -- median normally, peak when the median cannot discriminate.

    Sparse and bursty series (a counter rate with many idle windows, an event counter)
    sit at a median of 0 in BOTH runs even when their peaks differ by orders of
    magnitude. Reporting those as "identical" is worse than saying nothing: it implies
    the panel was checked and found unchanged. Fall back to the peak, and label which
    statistic was used so the reader is never guessing.
    """
    ma, mb = _median(series_a), _median(series_b)
    if (ma or 0) == 0 and (mb or 0) == 0:
        return "peak", _peak(series_a), _peak(series_b)
    return "median", ma, mb


def _pct(a: float | None, b: float | None) -> float | None:
    """Relative change b vs a, as a percentage. None when it is not expressible."""
    if a is None or b is None:
        return None
    if a == 0:
        return None if b == 0 else float("inf")
    return (b - a) / abs(a) * 100.0


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    if v == float("inf"):
        return "0 -> nonzero"
    return f"{v:+.1f}%"


def _num(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def diff(base: dict, new: dict, top: int, min_pct: float) -> int:
    out: list[str] = []

    out.append("=" * 78)
    out.append(f"BASELINE  {base.get('meta', {}).get('src', '?')}")
    out.append(f"COMPARED  {new.get('meta', {}).get('src', '?')}")
    out.append("=" * 78)

    # --- KPIs -------------------------------------------------------------
    bk, nk = base.get("kpi", {}) or {}, new.get("kpi", {}) or {}
    keys = sorted(set(bk) | set(nk))
    if keys:
        out.append("\nKPI")
        for k in keys:
            p = _pct(bk.get(k), nk.get(k))
            out.append(f"  {k:16s} {_num(bk.get(k)):>12s} -> {_num(nk.get(k)):>12s}   {_fmt(p)}")

    # --- tabs -------------------------------------------------------------
    bt, nt = base.get("tabs", {}) or {}, new.get("tabs", {}) or {}
    changed = [t for t in sorted(set(bt) | set(nt)) if bool(bt.get(t)) != bool(nt.get(t))]
    if changed:
        out.append("\nTAB AVAILABILITY CHANGED  (a capture leg differed between the runs)")
        for t in changed:
            out.append(f"  {t:14s} {bool(bt.get(t))} -> {bool(nt.get(t))}")

    # --- panels -----------------------------------------------------------
    bp, np_ = base.get("panels", {}) or {}, new.get("panels", {}) or {}
    only_b = sorted(set(bp) - set(np_))
    only_n = sorted(set(np_) - set(bp))
    moved: list[tuple[float, str, float | None, float | None, str]] = []
    flat: list[str] = []
    for pid in sorted(set(bp) & set(np_)):
        stat, a, b = _compare(bp[pid].get("series", {}), np_[pid].get("series", {}))
        p = _pct(a, b)
        title = f"{np_[pid].get('title', pid)}  [{stat}]"
        if p is None:
            if a == b:
                flat.append(pid)
            continue
        if p == float("inf") or abs(p) >= min_pct:
            rank = 1e9 if p == float("inf") else abs(p)
            moved.append((rank, pid, a, b, title))
    moved.sort(reverse=True)

    if moved:
        out.append(f"\nPANELS THAT MOVED  (>= {min_pct:g}%, top {top}; statistic noted per row)")
        for _, pid, a, b, title in moved[:top]:
            out.append(f"  {pid:26s} {_num(a):>12s} -> {_num(b):>12s}   "
                       f"{_fmt(_pct(a, b)):>14s}   {title}")
    if flat:
        # Not noise: a panel identical across a run that changed behaviour is either
        # measuring something the change did not touch, or is not measuring at all.
        out.append(f"\nPANELS IDENTICAL IN BOTH RUNS ({len(flat)})")
        out.append("  (median AND peak both equal -- these genuinely did not respond)")
        out.append("  " + ", ".join(flat))
    if only_b or only_n:
        out.append("\nPANELS PRESENT IN ONLY ONE RUN")
        for pid in only_b:
            out.append(f"  baseline only: {pid}")
        for pid in only_n:
            out.append(f"  compared only: {pid}")

    # --- populations ------------------------------------------------------
    brt, nrt = base.get("rt", {}) or {}, new.get("rt", {}) or {}
    out.append("\nPOPULATION")
    for label, key in (("profiling requests", None), ("request cards", "requests"),
                       ("sessions", "sessions")):
        if key is None:
            a = base.get("meta", {}).get("n")
            b = new.get("meta", {}).get("n")
        else:
            a, b = len(brt.get(key) or []), len(nrt.get(key) or [])
        out.append(f"  {label:20s} {_num(a):>8s} -> {_num(b):>8s}")

    # --- provenance ---------------------------------------------------------
    # An A/B is only an A/B if the two runs differed in the one variable you think
    # they did. Everything above reports what MOVED; this reports what CHANGED, which
    # is the claim the reader is actually being asked to accept. A framework version
    # differing between baseline and arm invalidates the comparison outright, and it
    # is otherwise completely invisible -- both runs simply work.
    bpv, npv = base.get("meta", {}).get("provenance"), new.get("meta", {}).get("provenance")
    out.append("\nPROVENANCE")
    if not bpv or not npv:
        out.append("  unavailable on at least one side (no fingerprint_*.json in the bundle)")
        out.append("  -> the single-variable claim of this comparison is UNVERIFIED")
    else:
        bf, nf = bpv.get("frameworks") or {}, npv.get("frameworks") or {}
        drift = [(k, bf.get(k), nf.get(k)) for k in sorted(set(bf) | set(nf))
                 if bf.get(k) != nf.get(k)]
        if drift:
            out.append("  FRAMEWORK DRIFT between the two runs -- this is not a clean A/B:")
            for k, a, b in drift:
                out.append(f"    {k}: {a} -> {b}")
        else:
            out.append(f"  frameworks identical across both runs: {bf}")
        for tag, pv in (("baseline", bpv), ("compared", npv)):
            if pv.get("framework_disagreement"):
                out.append(f"  {tag}: WORKERS DISAGREE INTERNALLY {pv['framework_disagreement']}")

        # The single-variable check. For an ablation this should list exactly the one
        # setting the experiment moved; anything else in the list is a confound that
        # would otherwise be argued about rather than seen.
        bc, nc = bpv.get("config"), npv.get("config")
        if bc is None or nc is None:
            out.append("  config unavailable on at least one side "
                       "(no config.yaml in the bundle) -- variables changed are UNKNOWN")
        else:
            keys = sorted(set(bc) | set(nc))
            delta = [(k, bc.get(k, "<absent>"), nc.get(k, "<absent>"))
                     for k in keys if bc.get(k, "<absent>") != nc.get(k, "<absent>")]
            if not delta:
                out.append("  config IDENTICAL between the two runs "
                           "-- any movement above is run-to-run variance, not a change")
            else:
                label = ("SINGLE-VARIABLE: this is a clean ablation"
                         if len(delta) == 1 else
                         f"{len(delta)} settings differ -- NOT a single-variable comparison")
                out.append(f"  CONFIG DELTA ({label})")
                for k, a, b in delta[:20]:
                    out.append(f"    {k}: {a!r} -> {b!r}")
                if len(delta) > 20:
                    out.append(f"    ... and {len(delta) - 20} more")

    bb, nb = brt.get("belief"), nrt.get("belief")
    if bb or nb:
        out.append("\nROUTER BELIEF vs ENGINE REALITY")
        for tag, v in (("baseline", bb), ("compared", nb)):
            if v:
                out.append(f"  {tag}: {v['n'] - v['disagree']}/{v['n']} agree, "
                           f"worst error {v['worst'] * 100:.1f}%")

    print("\n".join(out))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline")
    ap.add_argument("compared")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-pct", type=float, default=5.0,
                    help="ignore panels that moved less than this (default 5%%)")
    a = ap.parse_args(argv)
    with open(a.baseline) as f:
        base = json.load(f)
    with open(a.compared) as f:
        new = json.load(f)
    return diff(base, new, a.top, a.min_pct)


if __name__ == "__main__":
    sys.exit(main())
