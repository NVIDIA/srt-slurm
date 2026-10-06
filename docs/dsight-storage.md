# DSight storage and query interface

DSight writes `trace-data.sqlite` for indexed local queries and an HTML catalog
with compressed detail files for static browser delivery. Both contain the same
normalized evidence. The browser never downloads or opens the SQLite database;
it needs only `index.html` and the complete `detail/` directory. Keep
`manifest.json` with a published report to identify its generation and files.

See [DSight](dsight.md) for input discovery, generation commands, source coverage,
and timing definitions. `--single-file` retains the embedded HTML and
`trace-data.json.gz` format for direct `file://` viewing.

## SQLite schema, version 1

`PRAGMA user_version` is **1**. This physical schema version is independent of
the normalized JSON contract, `srtctl-trace/1`. Unsupported versions are rejected;
rebuild from preserved inputs instead of modifying the cache in place.

| Table | Columns | Key and meaning |
| --- | --- | --- |
| `catalog` | `json TEXT NOT NULL`, `generation TEXT NOT NULL` | One row containing normalized metadata and a random generation ID. Requests, lifecycle, sessions, workers, batch observations, unmatched spans, source inventory, warnings and coverage remain in this JSON. Profile events/CPU samples and metric points are empty arrays; profile/sample counts and name/stack dictionaries remain. |
| `events` | `profile INTEGER`, `ordinal INTEGER`, `bucket INTEGER`, `start REAL`, `end REAL`, `name INTEGER`, `tid TEXT`, `source_row INTEGER` | `PRIMARY KEY (profile, ordinal) WITHOUT ROWID`. One imported NVTX interval. `ordinal` is its position in the normalized profile; `name` indexes that profile's name dictionary. `tid` preserves the global thread ID as text; `source_row` retains the original export row ID. |
| `durations` | `profile INTEGER`, `bucket INTEGER`, `maximum REAL` | `PRIMARY KEY (profile, bucket) WITHOUT ROWID`. Maximum observed interval duration in each duration class. Derived from `events`; no extra evidence is inferred. |
| `samples` | `profile INTEGER`, `ordinal INTEGER`, `time REAL`, `payload TEXT` | `PRIMARY KEY (profile, ordinal) WITHOUT ROWID`. CPU sample as JSON `[time, global_tid, stack_index, source_row]`; the stack and symbol dictionaries are in the profile's catalog entry. |
| `points` | `series INTEGER`, `ordinal INTEGER`, `time REAL`, `payload TEXT` | `PRIMARY KEY (series, ordinal) WITHOUT ROWID`. Metric point as JSON `[time, value, source_id, source_row]`. Series identity, labels, units, sample/setting semantics and configuration evidence remain in the catalog. Null values and duplicate-timestamp evidence are preserved. |

Indexes are `event_window(profile, bucket, start)`,
`sample_window(profile, time)` and `point_window(series, time)`. Profile and series
IDs refer to their catalog entries; `ordinal` is a zero-based array position,
not a source row ID. Source IDs refer to the catalog's source inventory. NVTX
source identity comes from the profile's `evidence_source`.

Times remain the normalized floating-point seconds relative to
`catalog.meta.origin_ns`, which is an exact string. The database does not recover
raw nanoseconds discarded by an earlier import or correct clock skew. Negative
setting timestamps, interval boundaries and source precision remain unchanged.

For NVTX, `bucket = ceil(log2(max(end - start, 1e-9)))`. An overlap query uses
each class's recorded maximum to bound both ends of an indexed start-time scan,
then applies `start <= to AND end >= from`. This includes long intervals that
started before the window without making every short interval scan reach back
to the beginning of the capture. The lower bound is rounded outward with
`nextafter` so floating-point subtraction cannot omit a boundary event. Results
sort by `(start, profile, source_row)` before pagination; counts include all
matching imported rows.

The cache opens with SQLite `mode=ro`. Query scratch tables use temporary storage.
Readers check the generation ID before indexed queries and reject a replaced
generation rather than combining old catalog metadata with new rows. Reopen the
dataset after a rebuild. MCP detects replacement from file identity and reloads.
The builder stages the database, detail files, HTML and manifest before replacing
an existing managed report; failed builds retain the preceding generation.

## Local query interface

The existing CLI, Python and MCP interfaces now accept either a report directory
or an explicit `trace-data.sqlite` path. Directory lookup prefers SQLite;
explicit legacy JSON/gzip paths still work. Opening SQLite reads the catalog,
without loading the event, sample or point tables into Python lists.

```bash
uv run srtctl dsight query ./report summary
uv run srtctl dsight query ./report profiles --worker decode-0
uv run srtctl dsight query ./report nsys --profile 3 --from 600 --to 601 --limit 100
uv run srtctl dsight query ./report/trace-data.sqlite metrics \
  --name "<recorded_metric_name>" --from 600 --to 601 --points
```

Choose profile IDs and metric names from the report's catalogs; the values above
are examples. CLI `--from`/`--to` map to Python/MCP `start`/`end`:

```python
from srtctl.dsight.query import TraceDataset

trace = TraceDataset.from_path("./report")
profiles = trace.query("profiles", worker="decode-0")
page = trace.query("nsys", profile=profiles["items"][0]["id"],
                   start=600, end=601, offset=0, limit=100)
```

MCP exposes `query_trace(dataset, kind="summary", **filters)` with the same
parameters and response shapes. The path belongs to the MCP server's filesystem.
It is read-only and does not build reports or start profiling.

| Query kinds | Data access and filters |
| --- | --- |
| `nsys` | Indexed interval overlap; `profile`, `worker`, recorded `rank`, case-insensitive name substring. Returns exact total, bounded rows, provenance and `partial` when any selected profile was truncated at import. |
| `cpu` | Indexed inclusive sample window; `profile`, `worker`, recorded `rank`. Aggregates inclusive symbol hits, counting a symbol once per sample; returns `total_samples` and sorted hotspots. |
| `metrics` | Indexed inclusive point window for selected series; `worker`, recorded `rank`, exact metric `name`. Streams aggregate statistics and optionally returns raw points. |
| `profiles` | Catalog metadata and imported counts; `profile`, `worker`, recorded `rank`. |
| `summary`, `requests`, `request`, `lifecycle`, `iterations`, `server_spans`, `sources` | Existing catalog queries. Request filters include `session`, `agent`, `worker`, `search`, `min_ttft_ms`; `request` and `lifecycle` require `request_id`. These collections have not been moved into relational tables. |

All ranges require `0 <= start < end <= duration` and default to the whole run.
Intervals overlap inclusively; point/sample selection includes both endpoints.
List responses contain `total`, `items`, `offset`, `limit` and `range`. Offset is
nonnegative, limit defaults to 100 and is capped at 1,000; zero returns counts.
For metrics, pagination selects series. `points=True` includes at most 1,000 raw
points per series and reports `points_total`/`points_truncated`. Statistics use
all non-null samples in the selected window. Settings additionally return every
point at the latest timestamp strictly preceding the window in `carried_setting`;
conflicting settings are not resolved by arbitrarily selecting one row.

To inspect the physical schema directly without changing it:

```python
import sqlite3
from pathlib import Path

with sqlite3.connect(Path("report/trace-data.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
    print(db.execute("PRAGMA user_version").fetchone())
    print(db.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('table', 'index')").fetchall())
```

Prefer `TraceDataset.query` for evidence queries; it owns dictionary resolution,
interval indexing, source references, validation and response compatibility.

## Static browser delivery and API

`index.html` embeds UI assets plus compressed catalog metadata, a 512-bin NVTX
density overview per profile, and indexes of exact detail shards. Each shard has
at most 8,192 rows and targets at most 512 KiB of decoded JSON (a single oversized
row is kept intact). Names are the SHA-256 of compressed content. Descriptors
retain compressed/decoded byte sizes, both hashes, row count and
`[min_start, min_end, max_start, max_end]`. Point bounds repeat their timestamps.

The browser fetches selected metric families/windows, CPU samples and NVTX
detail as needed. Whole selected shards can contribute exact NVTX counts without
downloading their rows; boundary/name-filtered shards are decoded to check exact
overlap. A short window renders exact ranges when candidate rows fit the 20,000
row display budget. Broader windows show an explicitly labeled density overview:
overlapping interval counts, not CPU utilization or per-request execution time.
Exact paginated queries remain available regardless of that display budget.

The shared LRU cache allows 32 MiB of decoded JSON source bytes. This is a cache
accounting limit, not a bound on total JavaScript heap, chart arrays or an
explicit full-window metric query. New views cancel stale rendering fetches.
Missing or malformed shards produce errors; secure contexts (HTTPS/localhost)
also verify decoded SHA-256. Static hosts may serve gzip files with or without
`Content-Encoding: gzip`.

Browser API **3.1** makes Nsight and CPU detail methods awaitable for progressive
reports. Use `await` with either delivery format:

```javascript
const x = window.traceExplorer;
const profiles = x.listProfiles();                // synchronous metadata
const page = await x.queryNsys({profile: profiles[0].id, from: 600, to: 601, limit: 100});
await x.inspectNsys({profile: profiles[0].id, from: 600, to: 601});
await x.whenDetailsReady();                        // current rendered Nsight view
await x.whenMetricsReady();                        // current visible metric charts
const samples = await x.queryCpu({from: 600, to: 601});
const metrics = await x.queryMetrics({from: 600, to: 601, points: true});
const exported = await x.exportSelection();
const cache = x.detailDataStatus();
```

Browser queries keep their existing response shapes and selected-profile scope;
they are not a SQL endpoint. `queryNsys` preserves profile source order, while
local multi-profile queries sort by time/profile/source row. Catalog, request,
lifecycle and batch queries remain synchronous. Query data is exact within the
imported evidence; chart density/downsampling never substitutes for query rows.

## Measured HTML loading impact

Measured on 2026-09-29 with unmodified main `e7e8d007` and this change's renderer
`9dd5fa98`, using identical normalized inputs for each pair. The inputs are two
preserved TokenSpeed inference captures; no inference benchmark was rerun.

| Capture | Client requests | Imported NVTX rows | Metric points | Main HTML | Progressive HTML |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before capacity fix | 17 | 66,710 | 8,012,886 | 114.107 MB | 1.805 MB |
| After capacity fix | 406 | 4,162,788 | 7,616,598 | 185.104 MB | 3.988 MB |

MB uses decimal bytes. The HTML reduction is 98.4% and 97.8%, respectively;
exact detail remains in adjacent files and the local SQLite cache.

| Capture | Main ready | Progressive ready | Main with visible metrics | Progressive with visible metrics |
| --- | ---: | ---: | ---: | ---: |
| Before capacity fix | 560 ms | 152 ms | 633 ms | 222 ms |
| After capacity fix | 5,132 ms | 328 ms | 5,198 ms | 388 ms |

These are medians of three cache-disabled HTTP navigations per renderer in
isolated headless Chrome 154.0.8037.57, Linux x86-64, Intel Core Ultra 9 285K
(24 logical CPUs). Reports were served from localhost by Python's
`ThreadingHTTPServer`; filesystem caches were not flushed and the browser process
was reused. Ready measures navigation start to `trace-explorer:ready`, including
HTML transfer, parse/decompression and initial UI rendering. The visible-metrics
measurement additionally awaits `whenMetricsReady()`, current Nsight detail when
available, and two animation frames; it includes browser-driver observation
overhead. It does not load every metric family or every Nsight row.

One additional progressive navigation per capture, throttled to 10 Mbit/s
and 20 ms latency in Chrome, measured **1.63 s / 3.53 s** to ready and
**2.09 s / 3.99 s** with visible metrics. These controlled local measurements
are not authenticated remote-host timings or a guarantee for other captures.
The larger catalog is still about 4 MB, so initial transfer remains material
on a constrained connection.

Both pages also passed a selected-window query check against SQLite: the tested
one-second windows contained 44 and 3,009 ranges. Browser page counts and source
row IDs matched. End-to-end window selection plus exact rendering took 34 ms
and 128 ms on localhost in those single checks (after resetting throttling).

Reproduce the browser measurement with two report directories containing the
same evidence and the respective HTML renderers:

```bash
uv run --with websockets python tests/dsight_bundle_check.py \
  --out /tmp/dsight-browser-check \
  --baseline /path/to/main-report --report /path/to/progressive-report --repeats 3
```

The harness starts an isolated Chrome and local HTTP server, saves navigation
timings, resource sizes and screenshots, then compares exact browser/SQLite
results. It also checks synthetic crossing windows, pagination, cancellation,
cache accounting, exports, CPU samples, carried/conflicting settings and saved
views. `--dataset /path/to/local-report` supplies the query cache when `--report`
is a browser-only publication. Without report arguments it runs the synthetic
regression fixture alone. A Chrome/Chromium executable must be on `PATH`.

## Limits

The importer still materializes normalized inputs during a build. The default
250,000-event limit per Nsight report and its partial-import warnings are
unchanged. Request/lifecycle catalogs and shard indexes still grow with the
capture. SQLite and browser shards duplicate large arrays on disk; this trades
build time and storage for faster reopening and lower initial browser work.
Large full-range queries can still require substantial I/O, aggregation or
memory. Original Nsight exports remain the source for excluded NVTX categories,
CUDA kernel timing and evidence not imported into DSight.
