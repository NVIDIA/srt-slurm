# Raw Tachometer dashboard

For TensorRT-LLM runs using either the Dynamo frontend or native `trtllm_serve`, srt-slurm automatically builds
`logs/dashboard.html` and `logs/dashboard.data/` after process cleanup, before artifact upload and legacy analysis.
Measurements come only from the raw capture under `logs/<observability.tachometer.storage_subdir>/local`.
The dashboard uses the existing capture; enabling it does not change engine instrumentation or metric defaults.

`logs/dashboard-status.json` records `building`, `ready`, `disabled`, `missing`, or `failed`, with the source/output
paths, selected and excluded files, source warnings, and any error. A failed rebuild preserves the previous HTML and
records `previous_artifact: true` when it exists. Generation is a bounded,
best-effort subprocess. Missing or unusable raw data is reported explicitly and does not select the legacy dashboard.
Other backends retain the separately marked legacy dashboard path. A ready dashboard means the HTML was rendered; it
does not certify capture completeness. The UI warns when its selected raw files contain no `final.parquet`, and lists
excluded files. The presence of `final.parquet` alone is not a completeness check.

For regular benchmark runs, capture covers the benchmark. A later `RUN_EVAL` phase is outside that capture. Manual,
eval-only, and serve-only runs retain their session capture. The dashboard displays the timestamps actually recorded;
it does not infer phases from benchmark logs.

## Manual build

Build or rebuild the same portable dashboard directly from a Tachometer capture:

```bash
srtctl-dashboard /path/to/run/logs/tachometer/local \
  --out /path/to/dashboard.html --title 'My capture' --resolution 10
```

From an uninstalled checkout, use `PYTHONPATH=src python3 -m srtctl.analysis.tachometer_dashboard` with the same arguments.
The output embeds uPlot, its styles, and compressed per-metric data. Open the HTML directly in a modern browser; a server
or internet connection is not required. The sibling `.data` directory retains the reduced data and source manifest.

## Navigation

The component tabs are Frontend, Router, Workers, GPU, and Host. Collapsible rows group related subsystems. All captured
metric families have panels by default; search and the full metric catalog help navigate them. An optional Featured only
filter provides a smaller view with explicit displayed/total counts. Series retain endpoint,
inline labels, and capture metadata, allowing filters to select hosts, roles, process groups, GPUs, and other dimensions.
Time selection applies across tabs. Panel inspection exposes the metric's definition, source labels, and reduced data.

Frontend dispatch is separate from worker admission. Router values describe its decisions/estimates, even when they are
exported by the frontend process. Host includes process CPU, memory, NUMA, scheduling, network, and collector diagnostics.
Exporter self-metrics describe the exporter, not inference-worker resource use.

## Component rows

| Tab | Related groups |
| --- | --- |
| Frontend | Requests and latency; tokenization; tokenizer cache; dispatch and streaming; detokenization; Tokio runtime |
| Router | Routing decisions; queues and backpressure; worker feedback; KV matching; KV index events |
| Workers | Engine scheduling, iteration, KV cache and transfer, latency, speculation; worker lifecycle, admission, handler pools and transport |
| GPU | Utilization; framebuffer; power, clocks and temperature; NVLink/PCIe; hardware events |
| Host | Process CPU; host scheduling; host/process/NUMA memory; threads; paging; network; collection health |

These are logical component groups. Exporting endpoint, worker role, hostname, GPU and process group remain series
labels rather than extra tabs. The component names do not imply that missing telemetry is available.

Histogram inventory names ending in `_bucket`, `_count` and `_sum` find the same parent-family panel. Historical
attached sums and counts are not displayed as trustworthy measurements. Every captured family has a group and remains
represented by a panel by default and available through search or **Explore metrics**. Collapsing a row preserves
its panels and count. Tab badges match panel totals; filtered views explicitly show displayed/total families.

## Input boundary

Only raw Tachometer `.parquet` and Arrow IPC `.arrow` inputs supply measurements. The reader does not consume processed
JSONL, client results, logs, run configuration, old dashboard bundles, or resource snapshots. Display metadata in the
catalog describes known metrics; it does not add missing topology, capacities, or workload-phase boundaries.

The capture's own timestamps define the time range. Relative-only captures remain relative. Separate captures with
incompatible time origins are rejected instead of aligned using a log or file modification time.

## Interpretation

- Full series identity is retained before arithmetic: metric, endpoint, inline labels, and capture metadata.
- Known counters use elapsed-time deltas before display reduction. Resets and scrape gaps interrupt rate calculations.
- Gauges expose reduced stored values; unknown types are not inferred from monotonicity.
- Histograms use recorded cumulative bucket samples. Percentiles are estimates; overflow beyond the last finite bucket
  cannot establish an exact value. Historical attached histogram sum/count/lower-bound fields are not used.
- Missing observations, measured zero, constant values, resets, and known invalid semantics remain distinct.
- Conflicting samples with the same complete identity and timestamp are flagged. Their stored envelopes remain
  inspectable, but rates and percentiles are disabled; discarded upstream labels cannot be reconstructed.
- The default 10-second display bins preserve stored mean/min/max/last and counter deltas. The HTML is a reduced
  view, not a lossless copy of every raw sample. Adjust `--resolution` when rebuilding.
- Panels initially show at most 12 series ranked by full-capture peak. The displayed count makes truncation explicit;
  label filters and the series chooser can select any captured series.
- The historical Tokio `busy_ratio`, `gpu_mem_total` alias, and cumulative CPU quantiles carry interpretation notes.
  They do not become CPU utilization or GPU capacity panels.
- Worker-admitted requests do not establish engine occupancy. A capture without engine-internal telemetry cannot supply
  engine batch size, KV utilization, or iteration duration.

uPlot 1.6.32 is included under its MIT license in the dashboard assets. The interface follows Grafana-style dashboard
navigation but does not require or run a Grafana service.


## Local Chrome validation

The capture-independent browser regression below validates every family and component count, including empty tabs. It uses the locally installed
`/usr/bin/google-chrome` in headless mode, renders screenshots, opens all metric payloads, and exercises tabs,
labels, histogram aliases, inspector, shared time range, legends and narrow layout with the network disabled.
Install `playwright-core` in a separate tooling directory and point `NODE_PATH` to its `node_modules`:

```bash
NODE_PATH=/path/to/browser-tools/node_modules node tools/tachometer_dashboard_check.cjs \
  /path/to/dashboard.html /path/to/screenshots
```

Python checks: `PYTHONPATH=src pytest tests/test_tachometer_dashboard_*.py`.
