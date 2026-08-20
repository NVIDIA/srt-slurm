# Component Performance Dashboard

A single self-contained HTML page with **Overview / Router / Engine / Frontend** tabs,
plus an optional **Log analysis** tab, built offline from the artifacts an srt-slurm
job already captures.

It answers component questions the aggregate benchmark numbers cannot: where TTFT
went (admission queue vs. routing vs. prefill vs. KV transfer), whether KV cache or
in-flight batch was the ceiling, how evenly the router spread load, and whether the
frontend event loop was stalling.

Two pieces, vendored from the `dynamo-benchmark-perf-dashboard` repo:

| Layer | Path | Role |
| ----- | ---- | ---- |
| **L2 ingest** | `src/ingest/` | RAW capture -> three fixed intermediate schemas (a *bundle*) |
| **L3 render** | `src/visualization/` | bundle -> one self-contained `.html` |

Both are stdlib-only and are **not** part of the installed `srtctl` wheel. Run them
from a repo checkout.

---

## Quick start

Point the ingest at a job's log directory, then render the bundle:

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

Everything the dashboard reads is produced by **one recipe knob**:

```yaml
observability:
  enabled: true
```

That expands (see `ObservabilityConfig` in `src/srtctl/core/schema.py`) into the three
capture legs below. Without it you still get the Log-analysis tab, and nothing else.

| Leg | Recipe requirement | Artifact | Feeds |
| --- | ------------------ | -------- | ----- |
| **Metrics** | `observability.enabled` | `<log_dir>/raw_prometheus.jsonl` | Router, Engine, Frontend |
| **Traces** | `observability.enabled` **and** an AIPerf-based benchmark | `SPAN_CLOSED` lines in `<log_dir>/*.out` | Overview |
| **Client** | an AIPerf-based benchmark | `<log_dir>/artifacts/<run>/profile_export.jsonl` | Overview |
| **Frontend log** | *nothing* — always written | `<log_dir>/<node>_frontend_<i>.out` | Log analysis |

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

---

## `src/ingest/ingest.py` — building the bundle

```bash
python3 -m src.ingest.ingest --run-dir <log_dir> --out <bundle> [flags]
```

Produces:

```
<bundle>/profile_export.jsonl        client axis   (schema 1)
<bundle>/tempo_traces/<xid>.json     traces axis   (schema 3)
<bundle>/server_metrics_export.jsonl metrics axis  (schema 2)
<bundle>/dashboard.yaml              generated sidecar (header labels + topology)
```

| Flag | Default | Notes |
| ---- | ------- | ----- |
| `--client` | `aiperf` | `none` to skip. The AIPerf export is already schema 1, so this is a passthrough |
| `--client-input` | `artifacts/*/profile_export.jsonl` | reaches into the per-run artifact dir `bench.sh` creates |
| `--traces` | `spanlog` | `none` to skip |
| `--span-logs` | `*.out` | srt-slurm's worker/frontend log naming |
| `--metrics` | `prometheus` | parses `raw_prometheus.jsonl` |
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
| `--max-batch-prefill / --max-batch-decode` | `128` / `256` | in-flight-batch ceilings drawn on the Engine tab |
| `--gpus N` | from `dashboard.yaml` | tok/s/GPU denominator |
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

## Known limitations

- **Engine ceilings are inputs, not measurements.** The renderer looks for
  `trtllm_config_{prefill,decode}.yaml` in the bundle; srt-slurm dumps engine config as
  `<log_dir>/<node>_config.json` instead, so that lookup misses and the
  `--max-batch-*` defaults apply. Pass them explicitly to match your run — the defaults
  (128 / 256) are frequently wrong by orders of magnitude, and the Engine tab's captions
  are drawn against them.
- **Interpretive captions were developed on an admission/queue-bound reference run.**
  The numbers are data-driven, but the framing reads best on queue-bound runs.
- **Nothing is wired into the job path.** These are offline tools run against a
  finished log directory; no stage of a benchmark invokes them.

---

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
