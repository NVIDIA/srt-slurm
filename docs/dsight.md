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

# Optional existing profiles and a known timezone for TRT-LLM iteration logs:
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

OTel is optional and is imported automatically when available. Use `--no-otel`
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
| Frontend logs | `*_frontend_*.out` | Explicit client header → Dynamo UUID bridge |
| Dynamo OTel | `otel/*/traces.jsonl`, OTLP JSON resource/scope spans | Original timestamps, parents, trace/request/process identities and route attributes |
| Worker logs | `*_{prefill,decode,aggregated}_w*.out` | Engine ID maps, worker identity and iteration summaries |
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
for alignment. Imported families are listed in `src/srtctl/dsight/metrics.py`;
this context view does not replace the complete Tachometer metric catalog.

Nsight worker filenames follow
`<host>_<role>_w<index>_profile_rank<rank>.sqlite`; frontend names follow
`<host>_frontend_<index>.sqlite`. Unknown names remain unmapped. The default
NVTX limit is 250,000 events per report; `--max-profile-events` changes it.
Truncation is explicit. Operator ranges and CUDA kernels remain in the source
report; a CUDA table's presence is reported separately from imported data.

## Using the timeline

- Drag in the overview **or Client sessions & agents**, or enter From/To.
  All tracks follow the same time range.
- Expand session → agent → request. For requests with OTel activity, **Expand
  lifecycle** reveals cumulative rows: each appends elapsed time ending at its
  named milestone. **Fit TTFT** selects the client TTFT window and expands the
  lifecycle only when available.
- Click a milestone or raw span for boundaries and source references. Expand
  workers for operation/dispatch/response-pump nesting. Select a metric series
  explicitly when a worker has several rank/label combinations.
- **Inspect phase in Nsight** follows the recorded worker. Select a rank or
  compare frontend + request workers. Router DP rank is retained as evidence;
  it is not assumed to map to a global process rank.
- **Iterations** shows shared batch context. Supply the log's timezone to align
  timestamps that have no offset.
- **Copy view link** saves range, request and expansions in the URL fragment.
  **Export selection** saves evidence JSON.

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
| Engine bridge | `Engine ID map` log | UUID → worker/process-local client ID → disaggregated ID; post-submission observation |
| Iteration context | TRT-LLM log | Shared batches, host-loop time, delayed device time; one-second timestamps |
| NVTX / CPU | Nsight SQLite | Shared process/rank activity; overlap does not prove request ownership |

Definitions follow the lifecycle instrumentation introduced in
[Dynamo #14101](https://github.com/ai-dynamo/dynamo/pull/14101). Multiple attempts,
repeated milestones, non-monotonic clocks, or milestones outside client TTFT
suppress the derived server partition; client timing and raw spans remain.
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
`cpu`, `iterations`, `sources`.

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

## Development checks

```bash
uv run pytest tests/test_dsight.py tests/test_dsight_agentperf.py
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
