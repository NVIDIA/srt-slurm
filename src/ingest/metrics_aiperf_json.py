# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2 processor: AIPerf ``server_metrics_export.json`` -> ``server_metrics_export.jsonl``.

AIPerf (>= 0.12) writes an aggregate **and per-second-timeslice** export of every
Prometheus endpoint it scraped. That is the same information srt-slurm's in-job RAW
scraper captures, in a different container: ``raw_prometheus.jsonl`` holds one verbatim
exposition body per ``(sweep, endpoint)``, this holds one pre-binned time series per
``(family, series)``.

Why this exists
---------------
It is the **only** server-metrics source on runs that predate ``observability.enabled``.
Those runs are where the before/after pairs live -- a fix landed against a baseline that
was never instrumented is otherwise unrenderable, and re-running it is not always
possible. This module makes the historical corpus readable without re-running anything.

Input shape (pretty-printed; the ``metrics`` object is the bulk of the file)::

    {"metrics": {"<family>": {
        "type": "counter" | "gauge" | "histogram",
        "series": [{"endpoint_url": ..., "labels": {...}, "stats": {...},
                    "timeslices": [{"start_ns":..., "end_ns":..., ...}, ...]}]}}}

Output is schema 2, byte-compatible with ``metrics_prometheus``::

    {"timestamp_ns": <int>, "metrics": {"<name>": [{"labels": {...}, "value": <num>}]}}

Two transpositions are required, and both are load-bearing.

CUMULATIVE vs INTERVAL
    The renderer's ``counter_rate`` and ``hist_mean`` kinds both take DELTAS of a
    **cumulative** series. AIPerf's timeslices may be either cumulative or per-interval,
    and a single slice cannot tell you which. So this module **detects** it, per series,
    by asking which reading reproduces the run total in ``stats``: do the slices *sum*
    to it (interval), or does the *last* slice equal it (cumulative)? Interval series
    are then accumulated into cumulative ones.

    Guessing instead of detecting fails silently and plausibly: read intervals as
    cumulative and every rate panel becomes a jagged step function around zero; read
    cumulative as intervals and the series climbs like a hockey stick. Both look like
    data.

NAME SUFFIXES
    AIPerf keys a family by its base name; Prometheus exposition -- and therefore every
    panel -- uses ``_total`` / ``_count`` / ``_sum`` / ``_bucket``. The suffixes are
    re-attached here so the two metric sources are interchangeable at the panel layer.

Streaming
---------
The export routinely exceeds 2 GB, so nothing is loaded whole. Families are sliced out
one at a time by indentation (the writer pretty-prints, so a family key is the only
thing at a 4-space indent), and the per-timestamp regroup spills into time-contiguous
shard files rather than a dict or an external ``sort`` -- a 1 h run is ~15 M samples,
and both of those get the process OOM-killed on a shared login node.

Stdlib only (runs under a bare cluster python3).

Usage::

    python3 -m src.ingest.metrics_aiperf_json IN.json OUT.jsonl [--buckets]
        [--only dynamo_frontend_tokenize_seconds --only ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile

logger = logging.getLogger("metrics_aiperf_json")

# A family key is the only thing the writer puts at exactly 4 spaces of indent.
# Matching on that, rather than counting braces, keeps a '{' inside a description
# string from desynchronising the scan.
_FAMILY_KEY = re.compile(r'^    "([^"]+)": \{$')

# Gauge slices carry a summary of the samples seen in that second; prefer the most
# instantaneous reading available, and fall back through progressively lossier ones.
_GAUGE_FIELDS = ("last", "value", "avg", "mean", "max")


def _iter_families(path, section="metrics"):
    """Yield ``(name, obj)`` per family inside ``section``, one at a time.

    Anchoring on the section matters: ``summary`` is also a top-level object, so its
    children sit at the same 4-space indent as a metrics family. Scanning without the
    anchor picks both up and silently merges two different things -- on the reference
    export that is 193 apparent families against 93 real ones.
    """
    anchor = f'  "{section}": {{'
    inside = False
    cur, buf = None, []
    with open(path) as f:
        for line in f:
            if not inside:
                inside = line.startswith(anchor)
                continue
            m = _FAMILY_KEY.match(line)
            if m:
                if cur is not None:
                    yield cur, _load_family(cur, buf)
                cur, buf = m.group(1), [line]
            elif cur is not None:
                if line.startswith("  }"):  # the section object itself closing
                    yield cur, _load_family(cur, buf)
                    return
                buf.append(line)
            elif line.startswith("  }"):
                return
    if cur is not None:
        yield cur, _load_family(cur, buf)


def _load_family(name, buf):
    text = "".join(buf).rstrip().rstrip(",")
    return json.loads("{" + text + "}")[name]


def _is_interval(slices, field, run_total):
    """True when per-slice values are interval deltas rather than a running total.

    Decided by which reading reproduces ``stats``: intervals SUM to the run total,
    a cumulative series ENDS at it. Ties and missing totals fall back to interval,
    which is what every AIPerf version observed so far actually emits.
    """
    if not slices or run_total is None:
        return True
    total = 0.0
    last = 0.0
    for sl in slices:
        v = sl.get(field)
        if isinstance(v, (int, float)):
            total += v
            last = v
    return abs(total - run_total) <= abs(last - run_total)


def _series_labels(series):
    labels = dict(series.get("labels") or {})
    # metrics_prometheus injects worker_id/dynamo_component when the scrape's role is
    # known. Here the role is not knowable from the export alone, and the labels that
    # Dynamo does publish are already on the series, so nothing is invented.
    return labels


def _emit_family(name, fam, write, want_buckets, recon=None):
    """Push every ``(ts_ns, out_name, labels, value)`` of one family through ``write``.

    ``recon`` collects ``(family, field, expected, produced)`` for every series whose
    run total is known, so the cumulative/interval decision is checked against ``stats``
    rather than trusted. A wrong decision is invisible in the output but shows up here
    as an end-value that misses the run total.
    """
    typ = fam.get("type")
    n = 0
    for series in fam.get("series", []):
        labels = _series_labels(series)
        slices = series.get("timeslices") or []
        stats = series.get("stats") or {}
        if not slices:
            continue

        if typ == "histogram":
            fields = [("count", f"{name}_count"), ("sum", f"{name}_sum")]
            modes = {f: _is_interval(slices, f, stats.get(f)) for f, _ in fields}
            run = {f: 0.0 for f, _ in fields}
            bucket_run: dict[str, float] = {}
            for sl in slices:
                ts = _slice_ts(sl)
                for f, out_name in fields:
                    v = sl.get(f)
                    if not isinstance(v, (int, float)):
                        continue
                    if modes[f]:
                        run[f] += v
                        v = run[f]
                    run[f] = v
                    write(ts, out_name, labels, v)
                    n += 1
                if want_buckets:
                    for le, cnt in (sl.get("buckets") or {}).items():
                        if not isinstance(cnt, (int, float)):
                            continue
                        bucket_run[le] = bucket_run.get(le, 0.0) + cnt
                        bl = dict(labels)
                        bl["le"] = le
                        write(ts, f"{name}_bucket", bl, bucket_run[le])
                        n += 1
            if recon is not None:
                for f, _ in fields:
                    if isinstance(stats.get(f), (int, float)):
                        recon.append((name, f, float(stats[f]), run[f]))

        elif typ == "counter":
            interval = _is_interval(slices, "total", stats.get("total"))
            run = 0.0
            # A counter family may already carry the _total suffix in its key; emit the
            # alias only when it would differ, so a panel naming either form resolves.
            out_names = [name] if name.endswith("_total") else [name, f"{name}_total"]
            for sl in slices:
                ts = _slice_ts(sl)
                v = sl.get("total")
                if not isinstance(v, (int, float)):
                    continue
                if interval:
                    run += v
                    v = run
                for out_name in out_names:
                    write(ts, out_name, labels, v)
                    n += 1
            if recon is not None and isinstance(stats.get("total"), (int, float)):
                recon.append((name, "total", float(stats["total"]), run if interval else v))

        else:  # gauge (and anything else: an instantaneous reading is the safe default)
            for sl in slices:
                ts = _slice_ts(sl)
                for f in _GAUGE_FIELDS:
                    v = sl.get(f)
                    if isinstance(v, (int, float)):
                        write(ts, name, labels, v)
                        n += 1
                        break
    return n


def _slice_ts(sl):
    """Slice -> whole-second epoch ns.

    Families do not share a start instant, so each is snapped to the nearest second.
    That puts every family on one grid at the cost of at most half a second of skew,
    which is below the resolution of a 1 Hz panel.
    """
    end = sl.get("end_ns") or sl.get("start_ns") or 0
    return int(round(end / 1e9)) * 1_000_000_000


# Spill shard width. Samples are regrouped per timestamp, which needs them adjacent;
# bucketing by wall-clock minute makes each shard time-contiguous, so the shards can be
# emitted in numeric order and only one minute of samples is ever resident.
_SHARD_NS = 60_000_000_000


def process(raw_path: str, out_path: str, *, buckets: bool = False, only=None) -> int:
    """Convert an AIPerf server-metrics JSON export to schema 2. Returns lines written.

    Neither the input nor the intermediate is held in memory. A 1 h export is ~15 M
    samples, and an external ``sort`` over that gets OOM-killed on a shared login node,
    so the regroup is done by spilling into time-contiguous shards instead: bounded
    memory, no subprocess, no temp-space assumptions beyond the output directory.
    """
    keep = set(only) if only else None
    tmp_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    spill_dir = tempfile.mkdtemp(prefix=".aiperf_sm_", dir=tmp_dir)
    shards: dict[int, object] = {}
    families = samples = 0
    recon: list = []
    try:

        def write(ts, name, labels, value):
            key = ts // _SHARD_NS
            fh = shards.get(key)
            if fh is None:
                fh = shards[key] = open(os.path.join(spill_dir, f"{key}.tsv"), "w")
            fh.write(f"{ts}\t{name}\t{json.dumps(labels, sort_keys=True)}\t{value!r}\n")

        for name, fam in _iter_families(raw_path):
            if keep is not None and name not in keep:
                continue
            families += 1
            samples += _emit_family(name, fam, write, buckets, recon)
        for fh in shards.values():
            fh.close()
        logger.info(
            "read %d famil(ies) -> %d samples across %d shard(s)", families, samples, len(shards)
        )

        # Reconcile against `stats`: the cumulative/interval decision is the one thing
        # here that fails silently, so it is checked rather than trusted.
        bad = [r for r in recon if abs(r[3] - r[2]) > max(1e-6, abs(r[2]) * 1e-6)]
        logger.info(
            "run-total reconciliation: %d/%d series match stats%s",
            len(recon) - len(bad),
            len(recon),
            "" if not bad else f"; {len(bad)} off by <=1 slice, e.g. {bad[:2]}",
        )

        lines = 0
        with open(out_path, "w") as out:
            for key in sorted(shards):
                by_ts: dict[int, dict] = {}
                with open(os.path.join(spill_dir, f"{key}.tsv")) as sp:
                    for line in sp:
                        ts_s, name, labels_s, value_s = line.rstrip("\n").split("\t", 3)
                        merged = by_ts.setdefault(int(ts_s), {})
                        merged.setdefault(name, []).append(
                            {"labels": json.loads(labels_s), "value": float(value_s)}
                        )
                for ts in sorted(by_ts):
                    out.write(json.dumps({"timestamp_ns": ts, "metrics": by_ts[ts]}) + "\n")
                    lines += 1
        return lines
    finally:
        for fh in shards.values():
            if not fh.closed:
                fh.close()
        for fn in os.listdir(spill_dir):
            os.remove(os.path.join(spill_dir, fn))
        os.rmdir(spill_dir)


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("raw_path", help="input AIPerf server_metrics_export.json")
    ap.add_argument("out_path", help="output server_metrics_export.jsonl (schema 2)")
    ap.add_argument(
        "--buckets",
        action="store_true",
        help="also emit <name>_bucket{le=...}; multiplies output size ~10x and no panel "
        "kind reads buckets today",
    )
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        help="restrict to this metric family (repeatable)",
    )
    args = ap.parse_args(argv)
    n = process(args.raw_path, args.out_path, buckets=args.buckets, only=args.only)
    logger.info("wrote %s: %d timestamps", args.out_path, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
