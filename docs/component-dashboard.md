# Component Performance Dashboard

A single self-contained HTML page with **Overview / Frontend / Router / Engine /
Session / Log analysis** tabs, built offline from the artifacts an srt-slurm job
already captures.

It answers component questions the aggregate benchmark numbers cannot: where TTFT
went (admission queue vs. routing vs. prefill vs. KV transfer), whether KV cache or
in-flight batch was the ceiling, how evenly the router spread load, and whether the
frontend event loop was stalling.

Two pieces, vendored from the `dynamo-benchmark-perf-dashboard` repo:

| Layer | Path | Role |
| ----- | ---- | ---- |
| **L2 ingest** | `src/ingest/` | RAW capture -> five fixed intermediate schemas (a *bundle*) |
| **L3 render** | `src/visualization/` | bundle -> one self-contained `.html` |

Both are stdlib-only and are **not** part of the installed `srtctl` wheel. Run them
from a repo checkout.

---

## Quick start — nothing to turn on

**Every job builds its own dashboard.** There is no knob and nothing to remember:
post-processing writes, into the run's log dir:

```
perf_dashboard.html          self-contained page (D3 inlined, no network needed)
perf_dashboard.json          the page's DATA payload — same object the HTML embeds
perf_dashboard_bundle/       the intermediate schemas, re-renderable in seconds
```

`perf_dashboard.json` exists because the HTML is often unreadable where it lands — a
headless cluster, a CI log, an S3 prefix. It is the machine-readable form of
everything the page shows, so a run can be diffed against another, asserted on in a
test, or simply read without a browser. It is built before the S3 sync, so all three
artifacts ship with the rest of the log dir.

Driven by `srtctl.analysis.perf_dashboard`, which is best-effort: a rendering failure
is logged and never changes the outcome of a benchmark that already produced results.

Nothing else is required. One `srtctl` submission collects the data, post-processes it
and renders the page, in that order, inside the job.

What `observability.enabled` changes is **which tabs the page carries**, not whether
there is a page:

```yaml
observability:
  enabled: true          # adds the server-side capture legs — see the table below
```

Without it the page is built from the client's own metrics export, the per-iteration
worker logs and the frontend log: **Frontend / Router / Engine / Log-analysis**.
Turning it on adds the scraped `/metrics` stream and the `SPAN_CLOSED` traces, and
with them the **Overview** tab. Either way there is a dashboard, because the question
it answers is asked *after* the run, when opting in is no longer possible.

### Comparing two runs

```bash
python3 tools/dashboard_diff.py BASELINE/perf_dashboard.json COMPARED/perf_dashboard.json
```

Reports KPI deltas, tab-availability changes, per-panel movement ranked by magnitude,
and population changes. Built for the ablation shape -- change one variable, run both,
ask which signals responded -- which is also the honest test of whether a panel earns
its place: a panel that does not move between a healthy run and a known-degraded one
is not diagnostic.

It compares medians, falling back to peaks when both medians are zero (sparse counter
rates sit at zero through their idle windows), and labels which statistic each row
used. Without that fallback an 8x concurrency change reported 30 panels as unchanged
when only 8 genuinely were.

It also prints a **PROVENANCE** section, which answers the question the rest of the
report assumes: *were these two runs actually comparable?*

* **Framework drift** — the versions found inside the running workers, from
  `fingerprint_*.json`. If baseline and compared ran different `tensorrt_llm` builds,
  nothing else in the report means what it appears to. This is invisible otherwise:
  both runs simply work.
* **CONFIG DELTA** — every differing key of the resolved `config.yaml`, flattened to
  dotted paths. For an ablation this should name exactly one setting, and the section
  says outright whether the comparison is `SINGLE-VARIABLE` or how many settings
  differ. A confound becomes something you see rather than something you argue about.

Both degrade honestly: when either bundle lacks the provenance files the section says
the changed variables are **UNKNOWN** rather than implying the runs matched.

### Running a job against a modified checkout

If you are testing dashboard changes, the compute node runs whatever `srtctl_root` in
`srtslurm.yaml` points at -- not the tree you edited. Two things bite:

* **`srtctl_root` is the only lever.** Staging a second checkout and `cd`-ing into it
  does not select it. Point `srtctl_root` at the new tree and confirm the job's own
  `uv sync` line reports `srtctl==… (from file:///<your tree>)`.
* **The runtime state is gitignored, so a fresh checkout does not have it.**
  `srtslurm.yaml`, `configs/nats-server`, `configs/etcd`, `configs/etcdctl`,
  `wheelhouse/dynamo` and `bin/uv` all live outside git. Copy or symlink them from a
  working checkout, and do not `rsync --delete` over the result -- it removes exactly
  those files, and the job then fails somewhere unrelated (a model path resolving
  against the wrong root, for instance).

### If the job hit its wall clock

A SLURM `TIMEOUT` kills the job hard, so **post-processing never runs and no dashboard is
written** — even though the capture itself completed. This matters because the runs most
worth looking at are often the ones that ran out of time.

Nothing is lost. Every raw input survives the kill, and the bundle rebuilds from the run
directory in under a minute:

```bash
cd /path/to/srt-slurm
python3 -m src.ingest.ingest --run-dir outputs/<job_id>/logs --out /tmp/<job_id>-bundle
python3 -m src.visualization.build_dynamo_bench_dash \
    /tmp/<job_id>-bundle /tmp/<job_id>.html \
    --frontend-log outputs/<job_id>/logs/<node>_frontend_0.out
```

Verified on job 2753007 (TIMEOUT at 20:12): `raw_prometheus.jsonl`, `host_samples.jsonl`,
`dynamo-request-trace`, all four `fingerprint_*.json` and `resource_snapshot.json` were
intact, and the bundle rebuilt in **53 s** with `aiperf=True traces=True metrics=True
request_trace=True`.

> Rebuild on a compute node, not the login node, for a full-length run. The login node's
> memory limit will OOM-kill a render whose `server_metrics_export.jsonl` is a few hundred
> MB (observed at exit 137 on a 243-request run), and its `/tmp` is a 2 GB tmpfs.

## Running it by hand

Against any past run — including one captured before this existed:

```bash
cd /path/to/srt-slurm

python3 -m src.ingest.ingest \
    --run-dir outputs/<job_id>/logs \
    --out     /tmp/<job_id>-bundle \
    --worker  prefill=dep:4:2 --worker decode=tep:4:4

python3 -m src.visualization.build_dynamo_bench_dash \
    /tmp/<job_id>-bundle  /tmp/<job_id>.html \
    --frontend-log outputs/<job_id>/logs/<node>_frontend_0.out
```

Open the `.html` in a browser. D3 is inlined by default, so the page works with no
network — copy it anywhere.

---

## Capturing the inputs

The dashboard reads whatever the run happens to have. Four of the eight legs below do
not depend on `observability.enabled` at all; that knob is what adds the other four:

```yaml
observability:
  enabled: true
```

It expands (see `ObservabilityConfig` in `src/srtctl/core/schema.py`) into the legs
marked with it below. Without it the Metrics leg falls back to the client's own
export, and the page is built without the traces — so it keeps every time-series tab
and loses Overview.

| Leg | Recipe requirement | Artifact | Bundle output | Feeds |
| --- | ------------------ | -------- | ------------- | ----- |
| **Metrics** | `observability.enabled`, else the client's own export | `<log_dir>/raw_prometheus.jsonl`, else `<log_dir>/agentic/*/…/server_metrics_export.json` | `server_metrics_export.jsonl` | every time-series panel |
| **Request trace** | `observability.enabled` | `<log_dir>/dynamo-request-trace` | `request_trace.jsonl` | per-request card, per-session view, the waterfall's KV-transfer band |
| **Per-iteration** | `print_iter_log: true` in the engine config | `SPAN`-free lines in `<log_dir>/*_w*.out` | `iter_bins.json` | batch composition, host/device step time |
| **Traces** | `observability.enabled` **and** an AIPerf benchmark | `SPAN_CLOSED` lines in `<log_dir>/*.out` | `tempo_traces/<xid>.json` | Overview, routing outcome on the card |
| **Client** | an AIPerf benchmark at export level `records` (default) | `<log_dir>/agentic/*/aiperf_artifacts/` or `artifacts/*/` | `profile_export.jsonl` | Overview, warmup filtering |
| **Engine config** | TRT-LLM backend | `<log_dir>/trtllm_config_*.yaml` | copied verbatim | in-flight-batch ceilings |
| **Frontend log** | *nothing* — always written | `<log_dir>/<node>_frontend_<i>.out` | *(read directly)* | Log analysis |
| **Host telemetry** | `observability.enabled` | `<log_dir>/host_samples.jsonl` | `host_series.json` | host/process CPU, ctx switches, fd headroom |

### Host telemetry — what the metrics stream cannot say

`srtctl.analysis.host_sampler` runs in-process beside the metrics scraper, on the same
opt-in, reading only `/proc`. It exists because every other leg describes what Dynamo
and TRT-LLM *choose to publish*, and three failure modes live below that line:

* **Host CPU saturation / lock convoys.** A frontend pinned at 100% of one core is
  indistinguishable, in every published metric, from an idle one — both report low queue
  depth and low in-flight. The discriminator is host CPU busy % together with
  **involuntary** context switches per thread: a convoy is a thread descheduled against
  its will, which no application-level counter expresses.
* **File-descriptor exhaustion.** An accept loop out of descriptors logs nothing;
  connections are refused before the application sees them. Open fds against
  `RLIMIT_NOFILE` is the only available warning, and `fd_headroom_pct` is reported even
  when comfortable — *"the peak was 2% of the limit"* is the answer that rules the
  hypothesis out.
* **A load generator that has become the bottleneck.** Client 100% / server 0% is the
  whole diagnosis and is invisible from the server side by construction.

The sampler stores **cumulative** jiffie counters; the ingest derives percentages by
differencing, so the capture cadence is never baked into a stored number. Per-process
CPU is percent of **one** core, so >100 means genuinely more than one.

**Scope limit, stated because it bounds every conclusion drawn from it:** the sampler
observes the node driving the sweep — the one hosting the frontend — and the Dynamo /
AIPerf processes on it. It does **not** reach the worker nodes; that would cost an srun
round-trip per sample, which is the expense the scraper's opt-in exists to avoid. Worker-
side host health is not covered.

### Which tabs a run can populate

A tab is rendered only when its input exists. An empty Router tab would read as
"this run had no queueing"; an absent one reads as "this run was not instrumented",
which is the true statement — so the tab is dropped.

| Benchmark type | Overview | Router / Engine / Frontend | Log analysis |
| -------------- | :------: | :------------------------: | :----------: |
| `mooncake-router`, `router`, `trace-replay` (AIPerf) | yes | yes | yes |
| `sa-bench`, `sglang-bench` | **no** | yes | yes |
| any, with `observability.enabled: false` | no | no | yes |

**Overview needs the client and trace legs joined on `x_request_id`.** `sa-bench` runs
vLLM's `benchmark_serving.py`, which writes aggregate `results_concurrency_*.json`
only — there are no per-request records to join, so Overview is dropped. The
Log-analysis tab exists exactly for this case: it reconstructs per-request TTFT and a
shallow stage breakdown from the frontend log alone, and feeds the *same* renderer as
the span-sourced Overview panel.

> **`--export-level summary` silently removes the Overview tab.** AIPerf writes
> `profile_export.jsonl` only at export level `records` or `raw`. `records` is the
> default, so the stock `bench.sh` invocations are fine — but a recipe that passes
> `--export-level summary` keeps `profile_export_aiperf.json` (which the rollup reads)
> and drops the per-request export, and the join key goes with it. AIPerf is also what
> sets the `X-Request-ID` header the Dynamo spans carry, which is what makes the client
> and trace legs joinable in the first place.

---

## `src/ingest/ingest.py` — building the bundle

```bash
python3 -m src.ingest.ingest --run-dir <log_dir> --out <bundle> [flags]
```

Produces:

```
<bundle>/profile_export.jsonl        client axis         (schema 1)
<bundle>/server_metrics_export.jsonl metrics axis        (schema 2)
<bundle>/tempo_traces/<xid>.json     traces axis         (schema 3)
<bundle>/request_trace.jsonl         request-trace axis  (schema 4)
<bundle>/iter_bins.json              per-iteration axis  (schema 5)
<bundle>/trtllm_config_*.yaml        engine ceilings, copied verbatim
<bundle>/profile_export_aiperf.json  client run summary: workload cache ceiling, validity
<bundle>/config.yaml                 the RESOLVED recipe — what was configured
<bundle>/fingerprint_*.json          per-worker ground truth: framework versions, GPU, CUDA
<bundle>/resource_snapshot.json      the allocation the numbers came from
<bundle>/benchmark_status.json       exit code, error lines, AIPerf phase census
<bundle>/log_signals.json            log-only signals: counts, timestamps, samples
<bundle>/run_lifecycle.json          time-to-ready, readiness gap, terminal cause
<bundle>/host_series.json            host + per-process CPU, RSS, ctx switches, fds
<bundle>/dashboard.yaml              generated sidecar (header labels + topology)
```

| Flag | Default | Notes |
| ---- | ------- | ----- |
| `--client` | `aiperf` | `none` to skip. The AIPerf export is already schema 1, so this is a passthrough |
| `--client-input` | `agentic/*/aiperf_artifacts/…` then `artifacts/*/…` | both harness layouts are tried; AgentX nests one level deeper and shards by concurrency |
| `--traces` | `spanlog` | `none` to skip |
| `--span-logs` | `*.out` | srt-slurm's worker/frontend log naming |
| `--metrics` | `prometheus` | parses `raw_prometheus.jsonl` |
| `--request-trace` | `dynamo` | parses `dynamo-request-trace`; the only source of KV-transfer cost and `session_id` |
| `--iter-log` | `trtllm` | parses `print_iter_log` lines from the worker logs; local->UTC offset is derived per run, not hardcoded |
| *(automatic)* | — | `trtllm_config_*.yaml` are copied from the run dir into the bundle, giving the Engine tab its real in-flight-batch ceilings instead of the `--max-batch-*` defaults |
| *(automatic)* | — | `profile_export_aiperf.json` — AIPerf's run summary. Carries `theoretical_prefix_cache_hit` (the ceiling the *workload* offered), `error_summary` / `was_cancelled` / `branch_stats` (run validity), and `effective_concurrency`. Its **absence** is a signal too: AIPerf writes it when a concurrency finishes, so a run killed by its wall clock has none |
| *(automatic)* | — | `config.yaml`, `fingerprint_*.json`, `resource_snapshot.json` — run provenance. Without these, two bundles can be compared for what *moved* but never for what *changed* |
| *(automatic)* | — | `benchmark_status.json` — why an empty dashboard is empty: exit code, error lines, and AIPerf's per-phase completed/cancelled census. A run can exit non-zero with no error-marker line at all, and the census is then the only account of what happened |
| *(automatic)* | — | `log_signals.json` — signals whose only evidence is a log line and which have no metric behind them: worker crash, Dynamo recompile storm, OOM, KV-index desync, KV-transfer timeout, NCCL error, engine stall, etcd disconnect, client cancel. Counts + timestamps + 2 sample lines each, bounded regardless of log size. **Zero counts are kept** — "no crashes" is a statement; absence is not |
| *(automatic)* | — | `run_lifecycle.json` — time-to-ready, the readiness gap (frontend accepting before the router can place), and the run's terminal cause. On run 2752632 that was **650 s of 28 held GPUs serving nothing**, invisible on an x-axis starting at the first request |
| *(automatic)* | — | `host_series.json` — host CPU %, memory, per-process CPU / RSS / **involuntary** context switches / open fds, and fd headroom against `RLIMIT_NOFILE`. Requires the sampler (below); everything else in the bundle describes what Dynamo publishes, not the machine under it |
| `--worker` | — | repeatable `ROLE=PARALLELISM:RANK:COUNT`, e.g. `prefill=dep:4:6` |
| `--jobs` | `4` | parallelism for the `SPAN_CLOSED` pre-grep |

`--worker` fills in `dashboard.yaml`'s topology. It is worth passing: the renderer sums
`rank x worker_count` for the **tok/s/GPU** denominator, and falls back to 1 GPU with a
warning when it cannot work the count out.

> The trace leg pre-greps `SPAN_CLOSED` lines out of the worker logs in parallel before
> parsing, because `DYN_LOG=debug` logs are routinely multi-GB.

## `src/visualization/build_dynamo_bench_dash.py` — rendering

```bash
python3 -m src.visualization.build_dynamo_bench_dash <bundle> <out.html> [flags]
```

| Flag | Default | Notes |
| ---- | ------- | ----- |
| `--frontend-log PATH` | — | adds the Log-analysis tab |
| `--d3 PATH` | vendored `src/visualization/d3.v7.min.js` | inlined for a self-contained page |
| `--d3-cdn` | off | load D3 from the CDN instead (smaller file, needs network to view) |
| `--max-batch-prefill / --max-batch-decode` | `128` / `256` | in-flight-batch ceilings drawn on the Engine tab. **Only used when the bundle has no `trtllm_config_*.yaml`** — ingest copies those in automatically, and the real values win. On AgentX run 2739690 the true decode ceiling is `1`, so the `256` default would misdraw that panel by 256x |
| `--gpus N` | from `dashboard.yaml` | tok/s/GPU denominator |
| `--dump-json PATH` | — | also write the DATA payload as indented JSON (what the automatic build produces as `perf_dashboard.json`) |
| `--include-warmup` | off | by default only the profiling phase is kept |

The bundle argument is optional — with only `--frontend-log`, you get a
Log-analysis-only page for a run that had no observability at all:

```bash
python3 -m src.visualization.build_dynamo_bench_dash /tmp/out.html \
    --frontend-log outputs/<job_id>/logs/<node>_frontend_0.out
```

### Same-run enforcement

Passing a `--frontend-log` from a different run than the bundle **fails the build**,
naming both runs and the `x_request_id` overlap. It is not a warning: the result would
be a page whose header and Overview describe one workload while the Log-analysis tab
describes another, and a build-time warning is invisible to whoever opens the HTML.

Where the artifacts do correspond, that same pivot transfers the bundle's
warmup/phase filter onto the log source.

---

## The three dashboard entities

Everything the page draws is one of exactly three things. Nothing is bespoke to a run.

### 1. Time-series panels — `src/visualization/panels.py`

77 panels declared as data rows, evaluated by one generic function. Adding a signal
means adding a dict; there is no per-panel rendering code, which is what stops a panel
acquiring behaviour another panel lacks. Four kinds cover every catalogued signal:

| kind | arithmetic |
| ---- | ---------- |
| `gauge` | the sample as-is |
| `counter_rate` | successive differences / dt. A backwards step means the process restarted and is **dropped**, not drawn as a negative spike |
| `hist_mean` | `Δ_sum / Δ_count` — the **interval** mean. The cumulative form flattens over a long run and hides late degradation |
| `ratio` | `a / (a+b)` on interval deltas, for genuine partner counters only |

Two more panel kinds are computed rather than scraped (`kind: "derived"`), and are
emitted in the same shape so the same renderer draws them: in-flight imbalance across
ranks/workers, and batch composition.

A panel whose series never moves is **stated in words**, not drawn. A flat line reads
as a broken chart; "constant 0 for the whole run" is the finding.

### 2. Per-request card — `DATA.rt.requests[<x_request_id>]`

Four bands that sum to `total_ms` by construction: admission+routing, prefill compute,
KV transfer, steady decode. Plus what the request *was* (`isl`, `osl`, `cached_tokens`,
`kv_hit_rate`, `turn_index`, `prefix_reuse_ratio`) and where it went
(`routing.worker_id`, `.dp_rank`, `.overlap_blocks`, `.router_ms`, `.admission_ms`).

`routing` is an **outcome, not a rationale**. The router's cost comparison lives on a
free-text selector line carrying no `request_id`, so "which worker, with how much
overlap" is joinable per request while "why it beat the alternatives" is not.

`routing.belief_error` compares the router's `overlap_blocks × block_size` against the
engine's reported `cached_tokens`. They are the same quantity, so a divergence means
the router is routing on a prefix the engine does not hold.

### 3. Per-session view — `DATA.rt.sessions`

One row per session, identical columns for every session and every run: turns,
span/busy/**idle**, per-turn TTFT / KV-hit / prefix-reuse series, and which decode
workers and prefill ranks it touched.

Idle is broken out because a session can spend most of its wall-clock waiting on the
harness rather than the server; a session-level latency figure that absorbs that is
measuring think time.

## Known limitations

- **Interpretive captions on the older panels** were developed on an
  admission/queue-bound reference run. The numbers are data-driven, but the framing
  reads best on queue-bound runs. The declarative panels in `panels.py` do not have
  this problem — their captions are fixed prose assembled from `why` / `source` /
  `caveat` and never interpolate a run's values.
- **Router *rationale* is not recoverable.** The selector line carrying logit, overlap
  credit and prefill load scale has no `request_id`, so it cannot be joined per
  request. Only the outcome is shown.
- **`iter_bins.json` is rank 0 only**, so it cannot show TP/EP-rank divergence; and
  `iter` restarts per engine lifecycle, so it is not a usable x-axis.
- **Engine-side `dp_rank` is a replicated broadcast.** Splitting an engine metric by
  rank renders a flat family of lines that reads as "balanced" on an imbalanced run.
  Per-rank truth is frontend-side only; a test enforces that no `trtllm_*` panel
  splits by `dp_rank`.
- **Some queue gauges read a constant 0** on a run that never queued. They are still
  specified: queue depth is the strongest single TTFT predictor when it is not zero.
- **The KV-routing panels can describe a small slice of the traffic.** Session affinity
  pins a request to the worker already holding its conversation, and a pinned request
  never reaches the KV scorer — on run 2751593 that was 248 of 274 decisions (90.5%).
  The router tab states its own coverage from `ro.coverage`, but only when the
  frontend-log leg is present; without it, no decision count exists and no coverage
  claim is made.
- **`meta.n = 0` does not mean the run served nothing.** It counts *profiling*-phase
  requests, and a run that hits its wall clock during warmup has none — reference run
  2750618 has 118 client records, all `benchmark_phase: warmup`. Scrape-derived KPIs
  (`tok_cache`, `kv_hit_true`) are still valid on such a run; the per-request and
  per-session cards are the parts that need `meta.n > 0`. `meta.phase_filter` states
  which case you are looking at.
- **Worker fingerprints do not cover the frontend.** They are written per worker, so a
  frontend-only setting such as `DYN_TOKENIZER_CACHE` appears in no fingerprint. Its
  record is `meta.provenance.config`, taken from the resolved `config.yaml` — a
  configured value rather than an observation from inside the running process.

## Provenance and re-syncing

Vendored from `dynamo-benchmark-perf-dashboard` at commit
`22f49fea243e43403690b38e70a8d4092dec4cc8`. Only what the component dashboard needs was
copied; the upstream panel dashboard (`dashboard.py`, `template/`), the framework
adapters (`src/common/`, `src/trtllm/`, `src/vllm/`), the capture layer (`src/capture/`
— srt-slurm's own `srtctl.analysis.metrics_scraper` already implements that contract),
and two unreachable processors (`agentperf` client, live `tempo` trace scrape) were
deliberately left behind.

Deltas applied on top of upstream, all of them layout/wiring:

1. `ingest.py` moved under `src/ingest/`, the renderer under `src/visualization/`.
2. Client default glob `artifacts/*/profile_export.jsonl`; span-log default `*.out`.
3. D3 inlined from the vendored sibling by default (`--d3-cdn` opts out).
4. `agentperf` / `tempo` registry entries dropped, with `get_processor` naming the
   valid options when one is requested.

Both directories are excluded from `ruff` and `ty` in `pyproject.toml` so the files stay
byte-comparable with upstream; reformatting them would rewrite ~100 lines of deliberately
dense style and destroy the diff. Comments inside them that cite `dashboard.py`,
`render_fast.sh` or `src/common/*` are upstream provenance — those files are not here,
but the reasoning they record still applies.

Coverage lives in `tests/test_component_dashboard.py`, which pins the *seam*: the
artifact layout srt-slurm writes must be what the vendored defaults look for.
