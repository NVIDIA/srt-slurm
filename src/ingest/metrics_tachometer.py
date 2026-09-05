# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L2 processor: Tachometer parquet -> ``server_metrics_export.jsonl`` (schema 2).

Tachometer (``src/tachometer``, the Rust scraper srt-slurm starts in-job) polls every
``/metrics`` endpoint and writes rows into parquet; on shutdown it compacts them into
``final.parquet`` under its storage leaf. This module is the inverse of that capture:
it turns the parquet rows back into the fixed schema-2 stream that L3 reads, so a
tachometer-instrumented run renders with the exact same panels as a
``raw_prometheus.jsonl`` run.

Input discovery (``raw_path`` may be a parquet file, a glob, or a run log dir):

    tachometer/raw/scrape/final.parquet     post-run compaction output (preferred)
    tachometer/raw/scrape/*.parquet         incomplete-N shards (compaction never ran)
    tachometer/final/final.parquet          the direct-host explicit-compact output
    tachometer/local/final.parquet          local copy kept after a successful upload
    tachometer/local/*.parquet              out-N/incomplete-N leftovers (last resort)

Parquet row schema (tachometer-writer ``writer.rs::Row``): ``scraper_endpoint`` Utf8,
``metric_name`` Utf8, ``metric_value`` Float64, ``histogram_bucket_lower`` /
``histogram_bucket_upper`` / ``histogram_sum`` / ``histogram_count`` nullable Float64,
``time_since_start`` Float64, ``timestamp_ns`` Int64 (epoch ns), plus one Utf8 column
per metadata key (union across endpoints, "" when an endpoint lacks the key; on worker
endpoints: hostname/worker_index/worker_process/worker_role, on frontend endpoints:
frontend_index/hostname). Post-compaction files also carry ``metric_name_clean``.

Parquet row -> schema 2 mapping (derived from tachometer-scraper ``parse.rs`` +
``filters.rs``; the ``backend``/``frontend`` filters srt-slurm configures format a row's
name as ``<base>{k="v",...}`` where ``<base>`` is the exposition sample name with any
``_bucket``/``_count``/``_sum`` suffix stripped):

* PLAIN row (all histogram columns null, no ``le`` label): ``name{labels}`` is parsed
  back into (name, labels) and emitted verbatim -- ``metric_value`` is the raw scraped
  sample value, so counters stay cumulative exactly as in ``raw_prometheus.jsonl``.
* HISTOGRAM BUCKET row (``le`` in the inline labels, or ``histogram_bucket_upper``
  non-null): ``parse.rs`` folds one exposition ``<fam>_bucket{le=...}`` sample into one
  row whose ``metric_value`` is the sample's CUMULATIVE bucket count, copied verbatim
  (``samples_to_rows_with_filter`` pushes ``sample.value`` unchanged). Re-emitted as
  ``<fam>_bucket`` with the ``le`` label kept -- cumulative semantics round-trip 1:1
  with what ``metrics_prometheus`` produces. (Unfiltered endpoints keep the ``_bucket``
  suffix in the stored name; it is then not re-appended.)
* ``<fam>_sum`` / ``<fam>_count``: ``parse.rs`` DROPS the exposition ``_sum``/``_count``
  samples and instead attaches their values to every bucket row of the family via the
  ``histogram_sum``/``histogram_count`` columns. Its fix-up misses the ``+Inf`` bucket
  on filtered endpoints (the row's upper is null and its name no longer contains
  "_bucket", so ``fix_histogram_bucket_bounds_and_stats`` never groups it), so the
  reconstruction takes the first non-null column value across the family's bucket rows
  per (second, labels-ex-le) group. ``_count`` prefers the ``+Inf`` bucket's own value
  (by exposition semantics ``le="+Inf"`` == count) and falls back to the column.
* Known capture-side loss (documented, not repaired): on filtered endpoints the writer
  strips ``_bucket``/``_count``/``_sum`` from EVERY name (``parse.rs::strip_suffix`` is
  unconditional), so a plain gauge named ``foo_count`` is stored -- and re-emitted here
  -- as ``foo``, and summary ``_sum``/``_count`` samples collapse onto the quantile
  series name. The suffix cannot be recovered from the parquet.

Label enrichment mirrors :mod:`.metrics_prometheus`: every non-empty per-row metadata
column is folded into the entry's labels via ``setdefault`` (never clobbering a real
scraped label) so per-node/per-GPU series stay distinguishable, and rows whose
``worker_role`` metadata maps prefill -> "prefill" / decode -> "backend" additionally
get ``dynamo_component`` and ``worker_id`` (= the ``hostname`` metadata). Frontend
endpoints carry no ``worker_role``, so they get no invented worker labels.

Timestamps are snapped to whole seconds (``metrics_aiperf_json``'s slice snapping) and
all rows sharing a second merge into one schema-2 line, deduped with the same
idempotent (labels, value) fold, emitted in timestamp order. The regroup spills into
time-contiguous per-minute shards (the compacted parquet is sorted by metric name, not
time, and a whole run does not fit in memory on a shared login node).

Needs ``pyarrow`` (the only non-stdlib dependency in this package; imported lazily so
the registry and the stdlib-only siblings stay importable without it).

Usage:
    python3 -m src.ingest.metrics_tachometer LOG_DIR_OR_PARQUET OUT_PATH
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - only on the bare-script path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingest.metrics_prometheus import _dedup  # noqa: E402  (shared idempotent fold)

logger = logging.getLogger("metrics_tachometer")

# Columns owned by the writer schema (writer.rs) + compaction (metric_name_clean).
# Everything else is a per-endpoint metadata column.
_FIXED_COLUMNS = frozenset({
    "scraper_endpoint",
    "metric_name",
    "metric_name_clean",
    "metric_value",
    "histogram_bucket_lower",
    "histogram_bucket_upper",
    "histogram_sum",
    "histogram_count",
    "time_since_start",
    "timestamp_ns",
})

# worker_role metadata -> dynamo_component label. Mirrors metrics_prometheus's
# _ROLE_COMPONENT; roles outside the map (agg, "") get no injection.
_ROLE_COMPONENT = {"prefill": "prefill", "decode": "backend"}

# Search order under a run log dir. final.parquet is authoritative when present;
# local/ leftovers only matter when compaction never ran.
_SEARCH_ORDER = (
    "tachometer/raw/scrape/final.parquet",
    "tachometer/raw/scrape/*.parquet",
    "tachometer/final/final.parquet",
    "tachometer/local/final.parquet",
    "tachometer/local/*.parquet",
)

# Spill shard width, same rationale as metrics_aiperf_json: per-minute shards keep the
# per-timestamp regroup bounded to one minute of samples.
_SHARD_NS = 60_000_000_000

_INF_LE = ("+Inf", "Inf", "inf")


def find_parquets(path) -> list[str]:
    """Resolve ``path`` (parquet file | glob | run log dir) to an ordered file list."""
    p = os.fspath(path)
    if glob.has_magic(p):
        return sorted(glob.glob(p))
    if os.path.isfile(p):
        return [p]
    if os.path.isdir(p):
        for pat in _SEARCH_ORDER:
            hits = sorted(glob.glob(os.path.join(p, pat)))
            if hits:
                return hits
        return sorted(glob.glob(os.path.join(p, "*.parquet")))
    return []


def _parse_name(name: str) -> tuple[str, dict]:
    """``base{k="v",k2=v2}`` -> (base, labels). Handles both quoting styles the
    scraper's filters emit (backend/frontend quote values, node_exporter does not).
    Same limitation as the writer itself: a comma inside a quoted value mis-splits."""
    brace = name.find("{")
    if brace < 0:
        return name, {}
    base = name[:brace]
    end = name.rfind("}")
    body = name[brace + 1 : end if end > brace else len(name)]
    labels: dict = {}
    for pair in body.split(","):
        k, eq, v = pair.partition("=")
        if not eq:
            continue
        v = v.strip()
        if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
            v = v[1:-1]
        labels[k.strip()] = v
    return base, labels


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _import_parquet():
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415 - lazy: keep the package stdlib-importable
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "metrics_tachometer needs pyarrow to read the tachometer parquet "
            "(pip install pyarrow)"
        ) from e
    return pq


def process(raw_path, out_path: str) -> int:
    """Convert tachometer parquet row(s) to schema 2. Returns lines written.

    ``raw_path`` may be a parquet file, a glob, a run log dir (searched in
    ``_SEARCH_ORDER``), or a list of parquet paths (shards, read in order).
    """
    if isinstance(raw_path, (str, os.PathLike)):
        paths = find_parquets(raw_path)
    else:
        paths = [os.fspath(p) for p in raw_path]
    if not paths:
        raise FileNotFoundError(f"no tachometer parquet found under {raw_path!r}")
    pq = _import_parquet()

    tmp_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    spill_dir = tempfile.mkdtemp(prefix=".tach_sm_", dir=tmp_dir)
    shards: dict[int, object] = {}
    # (snapped_ts, family, labels-ex-le json) -> {"sum", "count", "inf"} for the
    # _sum/_count reconstruction (see the module docstring's mapping).
    hist_stats: dict = {}
    rows_read = samples = 0
    try:

        def write(ts, name, labels, value):
            key = ts // _SHARD_NS
            fh = shards.get(key)
            if fh is None:
                fh = shards[key] = open(os.path.join(spill_dir, f"{key}.tsv"), "w")
            fh.write(f"{ts}\t{name}\t{json.dumps(labels, sort_keys=True)}\t{value!r}\n")

        for path in paths:
            pf = pq.ParquetFile(path)
            col_names = list(pf.schema_arrow.names)
            meta_cols = [c for c in col_names if c not in _FIXED_COLUMNS]
            for batch in pf.iter_batches():
                names = batch.column("metric_name").to_pylist()
                values = batch.column("metric_value").to_pylist()
                uppers = batch.column("histogram_bucket_upper").to_pylist()
                sums = batch.column("histogram_sum").to_pylist()
                counts = batch.column("histogram_count").to_pylist()
                tss = batch.column("timestamp_ns").to_pylist()
                metas = {c: batch.column(c).to_pylist() for c in meta_cols}
                for i in range(batch.num_rows):
                    rows_read += 1
                    value = values[i]
                    if not _finite(value):
                        continue
                    value = float(value)
                    # Snap to the second grid, like metrics_aiperf_json's _slice_ts.
                    ts = int(round(tss[i] / 1e9)) * 1_000_000_000
                    base, labels = _parse_name(names[i])
                    meta = {}
                    for c in meta_cols:
                        v = metas[c][i]
                        if v not in (None, ""):
                            meta[c] = v
                    component = _ROLE_COMPONENT.get(meta.get("worker_role", ""))
                    if component is not None:
                        labels.setdefault("dynamo_component", component)
                        if meta.get("hostname"):
                            labels.setdefault("worker_id", meta["hostname"])
                    for k, v in meta.items():
                        labels.setdefault(k, v)  # never clobber a real scraped label

                    upper = uppers[i]
                    if "le" in labels or upper is not None:
                        # Histogram bucket row: metric_value is the verbatim
                        # cumulative bucket count (see mapping in the docstring).
                        le = labels.get("le") or ("+Inf" if upper is None else format(upper, "g"))
                        out_name = base if base.endswith("_bucket") else f"{base}_bucket"
                        fam = out_name[: -len("_bucket")]
                        blabels = dict(labels)
                        blabels["le"] = le
                        write(ts, out_name, blabels, value)
                        samples += 1
                        group_labels = {k: v for k, v in labels.items() if k != "le"}
                        key = (ts, fam, json.dumps(group_labels, sort_keys=True))
                        st = hist_stats.setdefault(key, {"sum": None, "count": None, "inf": None})
                        if st["sum"] is None and _finite(sums[i]):
                            st["sum"] = float(sums[i])
                        if st["count"] is None and _finite(counts[i]):
                            st["count"] = float(counts[i])
                        if le in _INF_LE:
                            st["inf"] = value
                    else:
                        write(ts, base, labels, value)
                        samples += 1

        # Reconstruct <fam>_sum / <fam>_count once per (second, family, labels) group.
        for (ts, fam, labels_key), st in hist_stats.items():
            labels = json.loads(labels_key)
            count = st["inf"] if st["inf"] is not None else st["count"]
            if count is not None:
                write(ts, f"{fam}_count", labels, count)
                samples += 1
            if st["sum"] is not None:
                write(ts, f"{fam}_sum", labels, st["sum"])
                samples += 1
        for fh in shards.values():
            fh.close()
        logger.info(
            "read %d parquet row(s) from %d file(s) -> %d samples across %d shard(s)",
            rows_read, len(paths), samples, len(shards),
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
                    metrics = by_ts[ts]
                    for name in metrics:
                        metrics[name] = _dedup(metrics[name])
                    out.write(json.dumps({"timestamp_ns": ts, "metrics": metrics}) + "\n")
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
    ap.add_argument("raw_path", help="run log dir, tachometer parquet file, or glob")
    ap.add_argument("out_path", help="output server_metrics_export.jsonl (schema 2)")
    args = ap.parse_args(argv)
    n = process(args.raw_path, args.out_path)
    logger.info("wrote %s: %d timestamps", args.out_path, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
