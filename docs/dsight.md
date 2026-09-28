# DSight: offline inference trace explorer

DSight aligns client requests, Dynamo lifecycle spans, worker metrics, hardware
samples, and existing Nsight exports on one timeline.

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

Open `<path_to_report_directory>/index.html` in a browser on your own machine,
after copying or publishing the generated HTML. The HTML embeds its data and
assets; it works offline, including from `file://`, in a modern browser with
`DecompressionStream` support. Generation is **CLI-only**: DSight has no submission,
benchmark, cleanup, or upload hook. It does not enable profiling, change recipes,
launch GPU jobs, or export `.nsys-rep` files. `dashboard` is an alias for `dsight`.

| Output | Purpose |
| --- | --- |
| `index.html` | Self-contained interactive UI |
| `trace-data.json.gz` | Exact normalized dataset embedded in the HTML |
| `manifest.json` | Schema, counts, warnings, source inventory and output hashes |

A rebuild stages the complete output before replacing an earlier DSight
directory. Import/render failures preserve the previous generation. Existing
directories that are not DSight outputs are rejected. Use preserved inputs:
the importer checks registered source files for changes during generation.

## Inputs

Pass a run directory containing `logs/`, or the log directory itself. Multiple
client exports require `--client "<path_to_client_export>"`; DSight does not silently
mix concurrency sweeps or duplicated exports.

Each source is optional. Views and controls appear only for usable observations:
no OTel means no request breakdown; no matching NVTX/CPU samples means no Nsight
section; no metrics means no metric selector; no batch records means no batch tab.
Missing CPU samples never become an empty hotspot table. An empty time selection
keeps available controls and reports that no observations overlap the window.

Client-only, metrics-only, Nsight-only, OTel-only and timestamped-batch-only
captures are supported. Without a client export (or with an empty export), the
window is the union of recorded source timestamps. DSight does not synthesize
client requests or TTFT. Unjoined server spans have a separate section and query,
including in runs that also contain measured clients. A nonempty client export
whose rows all fail the requested phase filter remains an error. Missing explicitly
supplied paths, malformed inputs and captures without a positive recorded time
range remain errors, rather than silently appearing complete.

OTel is imported automatically when available. Use `--no-otel`
to skip reading OTel files entirely. A request without supported, correlated
OTel activity has no lifecycle expansion button, stage rows, or source-measurement
breakdown. This also applies to untraced requests in a partially traced run.
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
| Worker logs | `*_{prefill,decode,agg}_w*[_e<k>].out` | Engine ID maps when recorded, Dynamo request/process bindings, iterations or periodic batch snapshots |
| Tachometer | `tachometer/local`, or `--metrics <capture-leaf-or-file>` | Selected running/waiting/in-flight, KV, GPU and host gauges with recorded labels |
| Nsight SQLite | `--nsys-sqlite <directory-or-file>` | Selected NVTX ranges and available frontend CPU samples, aligned by session UTC anchor |

`--phase profiling` excludes explicit AIPerf warmup rows; `--phase all` includes
them. Missing phase metadata is reported as unclassified. AgentPerf's manifest
selects measured requests; without it, the measurement window is not inferred.
Phase-analysis records join the request log on phase, request ID, user,
conversation and conversation index. Timing and HTTP-identity references remain
separate. Missing analysis records are labeled as liveness-log timing. AgentPerf
sessions group phase/user/conversation; agent nesting is not inferred. Prompts,
response text and SSE payloads are excluded from the normalized dataset.

For metrics, `final.parquet` supersedes compacted Parquet shards. An Arrow tail
is also read, with identical samples deduplicated within complete series
identities. The upload mirror is excluded. Absolute `timestamp_ns` is required
for alignment. Common families are listed in `src/srtctl/dsight/metrics.py`, engine families in
`src/srtctl/dsight/engines.py`;
this context view does not replace the complete Tachometer metric catalog.

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
- Expand session → agent → request. For requests with OTel activity, **Expand
  lifecycle** reveals **Activity spans**, retaining original overlap and nesting.
  **Progress milestones** switches to cumulative rows ending at chronological
  recorded boundaries. **Fit TTFT** selects the client TTFT window and expands the
  lifecycle only when available.
- Click a milestone or raw span for boundaries and source references. Expand
  workers for operation/dispatch/response-pump nesting. Select a metric series
  explicitly when a worker has several rank/label combinations.
- **Inspect phase in Nsight** follows the recorded worker. Select a rank or
  compare frontend + request workers. Router DP rank is retained as evidence;
  it is not assumed to map to a global process rank.
- **Batch context** shows the measurements actually recorded by each engine.
  TokenSpeed periodic snapshots show running/queued requests and cache pages;
  absent iteration counters or device timers are not filled with zeros. Supply
  the log's timezone to align timestamps that have no offset.
- **Copy view link** saves range, request and expansions in the URL fragment.
  **Export selection** saves evidence JSON, including bounded pages of independent
  server activity and batch observations with total counts for pagination.

Sessions are paginated; details expand on demand. Dense Nsight lanes show event
density until zoomed in. Queries retain exact imported intervals.

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
x.inspectNsys({worker: "decode-0", rank: 0, from: 32, to: 33});
x.queryMetrics({worker: "decode-0"});
x.queryIterations({worker: "decode-0", rank: 0});
x.exportSelection();
```

Agent workflow: inspect coverage → find slow requests in a bounded window →
inspect lifecycle/source evidence → compare worker metrics and shared execution
context → save a view for human review. Verify an optimization hypothesis
against a specific source before claiming a cause.

## Engine extension boundary

The normalized `srtctl-trace/1` contract adds `worker_bindings`, independent
`server_spans`, source capabilities, observation kinds/rank scopes, metric
metadata and Nsight thread identities. Existing request, lifecycle, metric and
profile collections retain their roles. CLI, Python, MCP and the browser consume
the same records; `summary.capabilities` / `traceExplorer.describe().available`
report usable sources. Empty collections are valid. Unknown values stay null.

The implementation separates four responsibilities:

- `sources.py` parses shared filenames, Dynamo JSON identity records and OTel
  discovery. This is independent of the backend engine.
- `engines.py` contains frozen `EngineDialect` descriptors for TRT-LLM, TokenSpeed
  and SGLang. Each can provide any subset of NVTX names/prefixes, a single-line
  log decoder and exact metric definitions. Log decoding has no clocks, joins,
  filesystem access or UI state.
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
SGLang's existing NVTX and metrics remain supported; this does not claim a
SGLang batch-log decoder or vLLM-specific vocabulary has been implemented.

## Development checks

```bash
uv run pytest tests/test_dsight.py tests/test_dsight_agentperf.py tests/test_dsight_engines.py
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
missing OTel/Nsight/metrics, and each source alone), restored view links and
available API examples:

```bash
uv run --with websockets python tests/dsight_sources_browser_check.py \
  --port 9338 --out "<fresh_path_to_browser_check_output>"
```

For very large SQLite sorts, set `SQLITE_TMPDIR` to a writable temporary directory
with sufficient capacity. This affects temporary sorting only; exports are opened
read-only. Plan memory/disk headroom for the normalized dataset and staged output.
