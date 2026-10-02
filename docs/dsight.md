# DSight: offline inference trace explorer

DSight aligns client requests, Dynamo lifecycle spans, worker metrics, hardware
samples, and existing Nsight exports on one timeline.

See the [data-flow guide](dsight-data-flow.md) for diagrams connecting each source
file to its UI view, request-identity joins, and the limits of each source.
The [storage and query reference](dsight-storage.md) documents the SQLite schema,
indexed query semantics, static detail format and browser API.

## Agent skills

Before using DSight, agents must read this guide and load the applicable skills
below by reading their `SKILL.md` files. This applies when building or querying
reports, analyzing existing results, preparing dashboard views, or changing
DSight code. The skills are maintained alongside DSight; no global installation
is required.

| Skill | When to load it |
| --- | --- |
| [dsight-query](../src/srtctl/dsight/skills/dsight-query/SKILL.md) | Query an existing report, inspect source coverage, compare runs, or gather evidence for a dashboard view. Prefer the normalized SQLite cache through the read-only CLI, Python or MCP interface. |

## Generate on a cluster login node

Run these commands manually on the login node after the run's artifacts have
been preserved. Use a Bash shell with `uv` on `PATH`, Python 3.10 or newer, and
a writable srt-slurm checkout that includes DSight. The run directory must be
readable and the report's parent directory writable from that node.

Replace the quoted placeholders with your paths. Absolute paths work from any
checkout; relative paths resolve from your current working directory.

```bash
cd "<path_to_srt_slurm_checkout>"
uv run --no-dev srtctl dsight build "<path_to_run_directory>" \
  --output "<path_to_report_directory>"

# Skip OTel processing, even when trace files exist:
uv run --no-dev srtctl dsight build "<path_to_run_directory>" \
  --output "<path_to_report_directory>" --no-otel

# Optional existing profiles and a known timezone for iteration/batch logs:
uv run --no-dev srtctl dsight build "<path_to_run_directory>" \
  --output "<path_to_report_directory>" \
  --nsys-sqlite "<path_to_nsys_sqlite_exports>" \
  --iteration-timezone "<iteration_log_timezone>"
```

`uv run --no-dev` prepares the checkout's Python environment without development
dependencies; its first invocation needs access to the required packages or a
populated package cache. Choose the timezone recorded by the iteration logs,
using an IANA timezone name; it is independent of the login node's timezone.
The optional flags can be combined with `--no-otel`.

Generation is CPU-only and requires no Slurm allocation, running deployment,
GPU, container, or browser. It can also run on another machine with access to
the same artifacts. Large captures can require substantial CPU, memory, and
filesystem reads; follow your site's login-node resource limits and use a CPU
job when needed.

Publish the complete report directory on a static HTTP host and open `index.html`.
For local viewing, serve it with Python (no database service is needed):

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory "<path_to_report_directory>"
# Open http://127.0.0.1:8000/index.html
```

The HTML embeds the request catalog and UI assets. Metric samples and Nsight
detail load from adjacent compressed files as needed. This works without internet
access; a modern browser with `DecompressionStream` support is required. Copying
only the HTML omits its detail files. Direct `file://` viewing requires the larger
embedded format: add `--single-file` to the build command.

Generation is **CLI-only**: DSight has no submission,
benchmark, cleanup, or upload hook. It does not enable profiling, change recipes,
launch GPU jobs, or export `.nsys-rep` files. `dashboard` is an alias for `dsight`.

| Output | Purpose |
| --- | --- |
| `index.html` | UI assets, request/lifecycle catalog, source metadata and detail indexes |
| `detail/*.json.gz` | Content-addressed exact metric/NVTX/CPU shards, fetched on demand |
| `trace-data.sqlite` | Indexed normalized evidence for local CLI and MCP queries; not downloaded by the browser |
| `manifest.json` | Schema, counts, warnings, source inventory and output hashes |

`--single-file` instead writes the legacy embedded `index.html` and
`trace-data.json.gz`. Both formats retain the same normalized evidence. Directory
queries prefer SQLite; explicit legacy JSON/gzip paths remain supported.

A rebuild stages the complete output before replacing an earlier DSight
directory. Import/render failures preserve the previous generation. Existing
directories that are not DSight outputs are rejected. Use preserved inputs:
the importer checks registered source files for changes during generation.
The HTML and all detail files form one generation. Publish them together. Detail
URLs contain content hashes; a missing file reports an error rather than an empty
measurement. HTTPS/localhost viewers also verify decoded content hashes. Existing
open pages may need a reload after their directory is rebuilt.

## Inputs

Pass a run directory containing `logs/`, or the log directory itself. Multiple
client exports require `--client "<path_to_client_export>"`; DSight does not silently
mix concurrency sweeps or duplicated exports.

Each source is optional. Views and controls appear only for usable observations:
no OTel means no request breakdown; no matching NVTX/CPU samples means no Nsight
section; no metrics means no metric selector.
Missing CPU samples never become an empty hotspot table. An empty time selection
keeps available controls and reports that no observations overlap the window.

Client-only, metrics-only, Nsight-only, OTel-only and timestamped-batch-only
captures are supported. Without a client export (or with an empty export), the
window is the union of recorded source timestamps. DSight does not synthesize
client requests or TTFT. Unjoined server spans remain available through
`queryServerSpans()` and evidence export, without separate timeline rows.
OTel-only captures retain source coverage and query access; their timeline shows
an empty-state message. A nonempty client export
whose rows all fail the requested phase filter remains an error. Missing explicitly
supplied paths, malformed inputs and captures without a positive recorded time
range remain errors, rather than silently appearing complete.

OTel is imported automatically when available. Use `--no-otel`
to skip reading OTel files entirely. A request without supported, correlated
OTel activity has no lifecycle expansion button or stage rows.
This also applies to untraced requests in a partially traced run.
Client request bars and their measured TTFT remain available, along with any
independent worker logs, metrics, and Nsight exports. Empty request paths and
identity bridges are omitted.

| Input | Discovery / selection | Contribution |
| --- | --- | --- |
| AgentX / AIPerf | `profile_export.jsonl`, `agentic/*/[aiperf_artifacts/]profile_export.jsonl`, `artifacts/*/profile_export.jsonl` | Client timing, tokens and recorded session/agent identities |
| Native AgentPerf | `requests.jsonl`, `agentperf/requests.jsonl`, `agentperf/*/requests.jsonl` | HTTP identity; companion phase-analysis JSONL supplies fully decoded timing where present |
| AgentPerf manifest | `phase_manifest.jsonl` beside the export | Measured request starts in `[settling_end, actual_phase_end)` |
| Frontend logs | `*_frontend_*.out` | Explicit client header → Dynamo UUID bridge from text or original JSON records |
| Dynamo OTel | `otel/traces.jsonl` or `otel/*/traces.jsonl`, OTLP JSON resource/scope spans | Original timestamps, parents, trace/request/process identities and route attributes |
| Worker logs | `*_{prefill,decode,agg}_w*[_e<k>].out` | Engine ID maps when recorded, Dynamo request/process bindings, iterations or periodic batch snapshots; log-derived metrics and scheduler settings |
| Tachometer | `tachometer/local`, or `--metrics <capture-leaf-or-file>` | All captured metric families, with source labels and samples within the client trace interval |
| Nsight SQLite | `--nsys-sqlite <directory-or-file>` | Selected NVTX ranges and available frontend CPU samples, aligned by session UTC anchor |

`--phase profiling` excludes explicit AIPerf warmup rows; `--phase all` includes
them. Missing phase metadata is reported as unclassified. AgentPerf's manifest
selects measured requests; without it, the measurement window is not inferred.
Phase-analysis records join the request log on phase, request ID, user,
conversation and conversation index. Timing and HTTP-identity references remain
separate. Missing analysis records are labeled as liveness-log timing. AgentPerf
streams explicitly marked `has_output=false` are unsuccessful even when HTTP
`success=true`. If a phase-analysis record omits cache usage, the exactly matched
request-log value is retained with its own source reference; an explicit null
remains unknown. AgentPerf sessions group phase/user/conversation; agent nesting
is not inferred. Prompts,
response text and SSE payloads are excluded from the normalized dataset.

Worker IDs can span multiple hosts, as in a two-node SGLang decode worker.
The worker catalog retains all observed hosts; log samples, process bindings,
and configuration references remain scoped to their recorded host and source.

For metrics, `final.parquet` supersedes compacted Parquet shards. An Arrow tail
is also read, with identical samples deduplicated within complete series
identities. The upload mirror is excluded. Absolute `timestamp_ns` is required
for alignment. The catalog includes every captured family, including families
with no samples in the client trace interval. Presentation categories reuse the
Tachometer taxonomy: **Frontend, Router, Workers, GPU, Host**, with subgroups
such as engine scheduling, KV cache, and host memory. Unknown families remain
selectable with their raw names and stored values.

Worker logs can also supply metrics without Tachometer. The shared
`LogMetricGenerator` interface normalizes these into the same series/catalog used
by the metric UI. The Dynamo–TokenSpeed implementation supplies active decode
batch size, configured batch limit, active KV pages and usable page pool size.
The SGLang implementation supplies batch snapshots and per-request cache/timing
events. Default second-resolution logs, optional fractional timestamps and rank
prefixes, and batches with or without counters are supported. Batch and request
events logged at the same timestamp remain separate in raw queries;
the chart displays their median with an event-count label.
See [metric sources, schema and log generators](dsight-log-metrics.md) for exact
names, units, evidence and extension rules. Local log timestamps require
`--iteration-timezone`; missing configuration stays unknown.

Known counters display captured cumulative values. Histogram lines show recorded
bucket observation counts, with bucket bounds retained in their identities;
attached histogram sum/count fields do not manufacture additional series.
No rate or quantile is inferred. Conflicting values at an identical source and
timestamp remain in the query evidence and are marked as chart gaps.

Nsight worker filenames follow
`<host>_<role>_w<index>_profile_rank<rank>.sqlite` for MPI ranks and
`<host>_<role>_w<index>_profile_gpu<devices>.sqlite` for per-process workers
(including TokenSpeed and SGLang captures); the role is `prefill`, `decode`, or `agg` as in the worker logs,
and a failover shadow engine's `_e<k>` suffix is retained as the report's
engine. Frontend names follow `<host>_frontend_<index>.sqlite`. A `_window<n>`
suffix is accepted. Unknown
names remain unmapped. Imported NVTX categories are the frontend
`preprocess.*`/`route.*`/`transport.*` ranges, TRT-LLM executor and scheduling
ranges, SGLang `scheduler.*` stages, and TokenSpeed forward/graph-replay,
input preparation, sampling, cache and commit annotations. OS PID/TID, GPU device
sets and distributed ranks stay separate; a GPU-set filename does not imply rank 0.
The default
NVTX limit is 250,000 events per report; `--max-profile-events` changes it.
Truncation and the imported time range are explicit. Operator ranges and CUDA kernels remain in the source
report; a CUDA table's presence is reported separately from imported data.

## Using the timeline

- Drag in the overview **or Client sessions & agents**, or enter From/To.
  All tracks follow the same time range.
- In **Request & execution path**, click **Fit Session** to zoom out to every
  recorded request in the selected request's session, including its main agent
  and subagents. The range uses the session's full recorded start/end, independent
  of the current time window or search filter, with the same padding as
  **Fit request**. Selection, expansions and pinned metrics are preserved;
  **Previous time range** (↶) returns to the previous zoom. This works without
  OTel. It covers the imported requests, not unrecorded session activity.
- Expand session → agent → request. An expanded agent shows every request
  matching the current time range and search, in chronological order.
  For requests with OTel activity, **Expand
  lifecycle** reveals **Progress milestones**: cumulative rows ending at
  chronological recorded boundaries. The breakdown ends at **Client complete**.
  **Fit TTFT** selects the client TTFT window and expands the lifecycle only when
  available.
- Click a milestone for boundaries and source references. Click **Inspect Nsight**
  on a recorded request-path card to open that worker's report directly, keeping
  the shared time range. Cards without a usable report are disabled and say
  **No Nsight report**. The metric selector at the top right searches all captured
  families by name and title, grouped by component and subgroup. The selected
  metric appears as one chart across its sources, using the same offline uPlot
  library as the Tachometer dashboard. Click a legend entry to hide or show its line. Every
  recorded series remains available in the legend; **Labels** exposes its full
  identity. Hover values include the actual sample timestamp. Chart dragging
  changes the shared time range; saved views preserve line visibility per metric.
- Click **Pin** beside the metric selector to keep that chart in the Metrics
  section while browsing other metrics. Pinned charts stack in the order you pin
  them, followed by the current unpinned metric. Selecting an already pinned
  metric uses its existing panel. Use **Move up** (↑) and **Move down** (↓) on a
  pinned card to reorder it; the controls are disabled at the first and last
  positions. Saved views and selection exports preserve your chosen order.
  **Unpin** on a card or beside the selector
  removes its pin; the selected metric stays visible. Each chart keeps its own
  legend choices, and all charts follow the shared time range.
- **Inspect phase in Nsight** follows the recorded worker. Select a rank or
  compare frontend + request workers. Router DP rank is retained as evidence;
  it is not assumed to map to a global process rank.
- **Request & execution path** contains **Request**, **Nsight**, and **Agent API**
  tabs. Request and Nsight appear only when their data is available; Agent API
  remains available for queries and exports. Saved links targeting a removed or
  unavailable tab open the first available tab.
- **Agent API** exposes recorded batch observations through `queryIterations()`.
  TokenSpeed periodic snapshots retain running/queued requests and cache pages;
  absent iteration counters or device timers remain unknown. Supply the log's
  timezone to align timestamps that have no offset.
- **Copy view link** saves range, request, expansions, pinned metrics and line
  visibility in the URL fragment. **Export selection** saves evidence JSON with
  the same view state, including bounded pages of independent server activity
  and batch observations with total counts for pagination.

Sessions are paginated; details expand on demand. A broad Nsight selection shows
512 precomputed density bins per report, labeled with their time resolution.
These count overlapping intervals and do not represent CPU utilization. When
the candidate shards contain at most 20,000 rows, the timeline loads exact ranges
and available CPU samples. Larger selections retain the overview until zoomed in;
exact paginated queries remain available at any range.

Browser API v3.1 supports `await traceExplorer.queryNsys(...)`,
`await traceExplorer.queryCpu(...)` and `await traceExplorer.inspectNsys(...)`.
Use `await traceExplorer.whenDetailsReady()` after changing a view to wait for its
Nsight tracks, and `whenMetricsReady()` for charts. Legacy embedded reports also
accept these awaited calls. `exportSelection()` awaits exact evidence queries.

Static shards contain at most 8,192 rows or approximately 512 KiB of decoded JSON.
Time bounds include intervals beginning before the viewport, so long crossing
ranges are retained. Metric loads include neighboring samples and all records at
the latest preceding setting timestamp. The shared LRU cache budgets 32 MiB of
decoded JSON bytes; this is a cache budget, not a total JavaScript heap limit.
Pan/zoom cancels stale view fetches. A requested wide exact query can still read
many shards, especially with a name filter; use local SQLite for broad analysis.

SQLite keeps the existing normalized values, source references, relative-second
timestamps, clock notes and partial-import warnings. It indexes NVTX ranges by
profile, duration class and start time, and metric/CPU samples by series/profile
and time. Requests and lifecycle metadata still load as a catalog; ingestion
still normalizes a run in memory. Neither delivery format removes the importer's
explicit event limit or upgrades incomplete source coverage.

## Timing definitions

Cumulative blocks measure **elapsed time between milestones**, not the inclusive
duration of a similarly named span. For example, “First frontend SSE ready” can
cover waiting after decode response-stream creation. It is not automatically
queue time, KV transfer, or first-token compute.

| Measurement | Source | Meaning |
| --- | --- | --- |
| TTFT / output reception | AIPerf or AgentPerf | Client start → first-token boundary → client end |
| Preprocessing | `request.preprocessing` OTel | Frontend preparation/tokenization |
| Worker selection | `kv_router.select_worker` OTel | Phase association explicitly marked as inferred from the next same-parent route |
| Ingress / transport setup | `worker.admission` OTel | Envelope decoding and response transport setup; not engine scheduler admission |
| Backend stream creation | `request.dispatch` OTel | Runtime `segment.generate`; the Python path creates a response stream without proving engine submission or first-token completion |
| Worker operation | `worker.operation.*` OTel | Inclusive parent of dispatch and response pumping, including backend waits |
| Response pump | `response.streaming.<role>` OTel | Begins before awaiting the first item; includes initial wait, generation and publishing |
| Frontend response stream | `response.streaming` OTel | First final SSE event available → completion/drop; concurrent with worker generation |
| Worker binding | Dynamo structured worker logs | Recorded request UUID, host, role and process epoch; distinct from engine-local IDs |
| Engine bridge | `Engine ID map` log | UUID → worker/process-local client ID → disaggregated ID; post-submission observation |
| Iteration context | TRT-LLM log | Shared batches, host-loop time, delayed device time; one-second timestamps |
| Batch snapshot | TokenSpeed log | Periodic scheduler state, millisecond timestamp precision and attention TP rank; not a forward iteration |
| NVTX / CPU | Nsight SQLite | Shared process/rank activity; overlap does not prove request ownership |

Definitions follow the lifecycle instrumentation introduced in
[Dynamo #14101](https://github.com/ai-dynamo/dynamo/pull/14101). Multiple attempts,
repeated milestones, non-monotonic clocks, or milestones outside client TTFT
suppress the derived server partition; client timing and raw spans remain.
Chronological ordering accommodates concurrent prefill/decode setup; it does not
assert causality. Missing first-token timing leaves an unsplit neutral request bar,
with raw server activity still available. Conflicting worker bindings remain in
evidence and are excluded from definite paths. A unique recorded host/role can
link activity to a worker when an explicit process binding is unavailable; this
weaker basis is labeled. Collector directory names are never worker identities.
No cross-host clock correction is invented. Engine queue/compute/KV timing needs
additional recorded per-request evidence.

Iteration counters and previous-device timers can lag the forward pass under
overlap scheduling. Original counters are preserved; no universal shift or
per-request assignment of shared batch time is applied.

## Session zoom example

Build the small client-only capture in [examples/dsight/fit-session](../examples/dsight/fit-session):

```bash
uv run --no-dev srtctl dsight build examples/dsight/fit-session --output /tmp/dsight-fit-session
```

Open `/tmp/dsight-fit-session/index.html`, select `child-turn`, then click
**Fit request** followed by **Fit Session**. The latter includes `parent-turn`
and `sibling-turn`, from 1 to 9 seconds plus padding, while keeping `child-turn`
selected. The unrelated sessions at either end are outside the fitted range.

## Agent, CLI and Python access

```bash
uv run --no-dev srtctl dsight query "<path_to_report_directory>" summary
uv run --no-dev srtctl dsight query "<path_to_report_directory>" requests --from 29 --to 34 --worker decode-0 --limit 20
uv run --no-dev srtctl dsight query "<path_to_report_directory>" lifecycle --request "<client_request_id>"
uv run --no-dev srtctl dsight query "<path_to_report_directory>" nsys --from 32 --to 33 --worker decode-0 --rank 0
uv run --no-dev srtctl dsight query "<path_to_report_directory>" iterations --from 32 --to 33 --worker decode-0 --rank 0
```

Times are seconds relative to the exact string `meta.origin_ns`. List queries
return total, offset, limit, range and items. Limits are at most 1,000; metric
`--points` includes up to 1,000 points per series with a truncation flag. Kinds:
`summary`, `requests`, `request`, `lifecycle`, `metrics`, `profiles`, `nsys`,
`cpu`, `iterations`, `server_spans`, `sources`. Rank filters follow the recorded
`rank_kind`: TRT-LLM global rank (with local rank retained separately), TokenSpeed
attention TP rank, or the Nsight report's recorded distributed rank. Prefer
`--profile` for captures that only identify a GPU set.

Lifecycle queries return `available: false` and empty `stages`, `activities`,
`milestones`, and `rows` when no supported OTel activity is joined. No fallback
breakdown is synthesized from client timing. The browser's `getLifecycle()`
returns the same model; `expandRequest()` keeps these requests unexpanded.

```python
from srtctl.dsight.query import TraceDataset

trace = TraceDataset.from_path("<path_to_report_directory>")
rows = trace.query("requests", start=29, end=34, min_ttft_ms=1000, limit=20)
detail = trace.query("lifecycle", request_id=rows["items"][0]["id"])
```

`srtctl-mcp` exposes the read-only **`query_trace`** tool with the same filters
and pagination. Its dataset path is local to the MCP server. It never builds
a dashboard or starts a job. The browser uses the same normalized lifecycle
model and controls the visible selection:

```javascript
const x = window.traceExplorer;
x.selectRange(29, 34);
x.selectRequest("<client-request-id>", {expand: true});
x.getLifecycle("<client-request-id>");
await x.inspectNsys({worker: "decode-0", rank: 0, from: 32, to: 33});
x.listMetricFamilies(); // Synchronous catalog, including coverage and categories.
x.listMetricSeries(); // Synchronous source identities without point decoding.
x.setState({pinnedMetrics: ["trtllm_num_requests_running"]});
await x.queryMetrics({name: "trtllm_num_requests_running", worker: "decode-0"});
x.queryIterations({worker: "decode-0", rank: 0});
await x.exportSelection();
await x.whenMetricsReady(); // Wait for all visible metric charts after a UI action.
await x.whenDetailsReady(); // Wait for the current Nsight detail rendering.
```

Browser API version 3.1 makes `queryNsys()`, `inspectNsys()` and `queryCpu()`
awaitable for progressive reports, alongside `queryMetrics()` and
`exportSelection()`. Await these methods even when a family or window was
previously viewed. Exports capture the selected view and range before loading
samples, so changing the view during loading does not mix selections. Range,
request, lifecycle, catalog and batch query methods remain synchronous.
`pinnedMetrics` in `getState()` / `setState()` is an ordered array of metric family
names. Restoring it removes duplicates and unknown names; an empty array clears
all pins. State updates that omit it preserve the current pins.

Agent workflow: inspect coverage → find slow requests in a bounded window →
inspect lifecycle/source evidence → compare worker metrics and shared execution
context → save a view for human review. Verify an optimization hypothesis
against a specific source before claiming a cause.

## Engine extension boundary

The [data-flow guide](dsight-data-flow.md#what-the-shared-engine-interface-contributes)
maps each source through the shared readers to its UI view.

The normalized `srtctl-trace/1` contract adds `worker_bindings`, independent
`server_spans`, source capabilities, observation kinds/rank scopes, metric
metadata and Nsight thread identities. Existing request, lifecycle, metric and
profile collections retain their roles. CLI, Python, MCP and the browser consume
the same records; `summary.capabilities` / `traceExplorer.describe().available`
report usable sources. Empty collections are valid. Unknown values stay null.

The implementation separates these responsibilities:

- `sources.py` parses shared filenames and discovers OTel captures;
  `identities.py` decodes common Dynamo identities from text and JSON logs.
  Both are independent of the backend engine.
- `engines.py` contains frozen `EngineDialect` descriptors for TRT-LLM, TokenSpeed
  and SGLang. Each can provide any subset of NVTX names/prefixes, a single-line
  log decoder and exact metric definitions. Log decoders return typed identity, iteration and snapshot observations.
  Decoding has no clocks, joins, filesystem access or UI state.
- `log_metrics/base.py` defines the `LogMetricGenerator` abstract base class and immutable
  metric definitions/events. `log_metrics/tokenspeed.py` implements its
  Dynamo–TokenSpeed dialect, and `log_metrics/sglang.py` implements SGLang batch
  snapshots and completed-request statistics. `log_metrics/reader.py` normalizes all registered
  generators into the shared metric schema and joins limits in their exact scope.
- Source readers and `Importer` own UTC alignment, bounded imports, provenance,
  identity joins and auditing. `window.py` derives a source-only time envelope;
  `capabilities.py` determines which views have usable evidence.
- The viewer selects controls from capabilities and metric metadata. It has no
  engine-name switches and never assumes every engine records TRT-LLM timers.

To add another engine, first preserve representative source fixtures. Add its
vocabulary to an `EngineDialect`, documenting metric units and observation/rank
semantics. Reuse common identity and source readers. Add fixture tests for missing
sources, unknown IDs and timing precision; use the optional-source browser matrix
below. Introduce a new common observation kind only when existing kinds cannot
represent the evidence. Do not infer IDs/ranks/timings to satisfy a shape.
SGLang batch and request logs contribute metric observations, not GPU execution
intervals or a client-request identity join. There is no vLLM-specific vocabulary.

## Development checks

```bash
uv run pytest tests/test_dsight.py tests/test_dsight_agentperf.py tests/test_dsight_engines.py tests/test_dsight_log_metrics.py
uv run ty check src/srtctl/dsight
node --check src/srtctl/dsight/assets/explorer.js
```

The optional browser checks use an already-running isolated Chrome DevTools
port, a generated traced artifact, and no GPU:

```bash
uv run --with websockets python tests/dsight_browser_check.py "<path_to_report_directory>/index.html" \
  --port 9338 --out "<path_to_browser_check_output>" --request "<joined_client_request_id>"
```

Check missing, empty, unjoined, disabled, and mixed OTel inputs with synthetic
source files (uses the same isolated Chrome port):

```bash
uv run --with websockets python tests/dsight_optional_otel_check.py \
  --port 9338 --out "<path_to_browser_check_output>"
```

Check the complete optional-source matrix (TokenSpeed overlap, independently
missing OTel/Nsight/metrics, and each source alone), restored view links including
removed inspector tabs, and available API examples:

```bash
uv run --with websockets python tests/dsight_sources_browser_check.py \
  --port 9338 --out "<fresh_path_to_browser_check_output>"
```

Check session zoom, subagents, filtered views, range history and missing sources:

```bash
uv run --with websockets python tests/dsight_fit_session_check.py \
  --port 9338 --out "<fresh_path_to_browser_check_output>"
```

For very large SQLite sorts, set `SQLITE_TMPDIR` to a writable temporary directory
with sufficient capacity. This affects temporary sorting only; exports are opened
read-only. Plan memory/disk headroom for the normalized dataset and staged output.
