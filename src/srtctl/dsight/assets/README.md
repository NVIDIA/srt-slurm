# DSight browser assets

`metric-charts.js` provides multi-series metric plots with direct legend controls.
The chart library is the same **uPlot 1.6.32** used by the September 11 Tachometer
dashboard, vendored unchanged under the MIT license:

- Source: https://github.com/leeoniya/uPlot/tree/1.6.32
- JavaScript: https://raw.githubusercontent.com/leeoniya/uPlot/1.6.32/dist/uPlot.iife.min.js
- CSS: https://raw.githubusercontent.com/leeoniya/uPlot/1.6.32/dist/uPlot.min.css
- License: `uPlot.LICENSE`

The application loads local files only; no CDN or JavaScript build is required.

## Metric component

```javascript
const chart = DSightMetricCharts.mount(host, {
  series, // normalized DSight series; points: [elapsed_seconds, value, ...evidence]
  references, // optional series matched by each observed series' reference.series_id
  from, to,
  selection: {hidden: []},
  onSelectionChange(selection) { /* adapter persists hidden series IDs */ },
  onRangeChange(from, to) { /* adapter updates the shared time range */ },
});
chart.destroy();
```

Each mount accepts one metric family in a single unit across all supplied sources.
Optional `title`, `height`, and `syncKey` control the caption, plot height, and shared
cursor group. Every series appears in the legend; there is no series cap, filter,
or separate chooser. Clicking a legend entry (or pressing Enter/Space on its button)
updates its line with `uPlot.setSeries` and preserves the plot and keyboard focus.
IDs are compared as strings. Only `selection.hidden` is read and emitted; obsolete
`ids` and `filters` fields are ignored so saved views cannot make sources inaccessible.
The adapter owns persistence and must destroy a mount before replacing its host.
A selection callback may synchronously destroy the component.

Legend captions use worker, host, GPU, and rank identities, adding varying labels
when needed to distinguish sources. Tooltips retain full labels; each row's Labels
disclosure exposes normalized identity plus complete raw metadata and label JSON.

The chart uses `uPlot.join` to align the original timestamps. Explicit nulls stay
null and alignment holes are undefined, as documented by the upstream
[`join` implementation](https://github.com/leeoniya/uPlot/blob/1.6.32/src/utils.js).
It does not resample sample observations. A series with `temporal: "setting"`
has separate display coordinates that hold configuration until its next recorded
change and clip it to the view boundaries; raw points remain unchanged. Hover values show the
nearest recorded sample with its actual timestamp. Sample evidence remains in
the unchanged input objects; this component only reads timestamps and values.
When a series declares `conflict_timestamps`, those timestamps become explicit
plot gaps; conflicting numeric observations remain in the raw query evidence.
For `temporal: "event"`, every source-line observation remains in raw queries.
The chart needs one value per timestamp and uses the median of simultaneous
events only for display; its hover text gives the event count. It does not add
a raw point or alter per-request query statistics.

## Metric catalog and loading

`metric-data.js` indexes series metadata. Progressive reports delegate selected
time windows to `detail-data.js`, which fetches content-addressed gzip shards from
the same static host. Its shared LRU cache budgets 32 MiB of decoded JSON bytes,
and each fetch accepts cancellation. Shards retain exact timestamps, values and
source references, including nulls and conflicting settings across shard edges.
The browser never downloads the SQLite cache used by local CLI/MCP queries.

`--single-file` retains the older inert base64 family payloads, serialized and
coalesced decompression, and a 500,000-point family cache. Legacy inline points
also remain supported. Rendering checks its generation after loading so an
earlier selection cannot replace a newer chart.

The metric picker consumes `metric_catalog`, including the shared presentation
semantics in `srtctl.analysis.metric_catalog` adapted from the Tachometer dashboard
([PR #447](https://github.com/NVIDIA/srt-slurm/pull/447), source commit
`b2509c17c68f4b2326f7656b3e33738770e1d575`). Every family is selectable, including
families without samples in the trace interval. These charts display captured
counter and histogram bucket values without computing rates or percentiles.

Browser API v3.1 exposes synchronous `listMetricFamilies()` and `listMetricSeries()`
metadata, asynchronous `queryMetrics()` and `exportSelection()` evidence, and
`whenMetricsReady()` for chart readiness. Callers must await the asynchronous
methods regardless of whether the family is cached.
Nsight `queryNsys()`, `queryCpu()` and `inspectNsys()` should also be awaited;
`whenDetailsReady()` waits for the active window's exact tracks or density view.
Overview bins are labeled summaries, never substitutes for exact API results.

## Capacity references

An observed series can declare `reference: {name, label, series_id}`. The caller
loads that reference family and supplies the exact paired series via `references`.
The component checks units and renders a dashed limit in the observed series'
color; its legend toggle controls both lines. `temporal: "setting"` references use
the latest preceding setting, including explicit null/conflict invalidations.
Sample references require an exact timestamp match for usage calculations and
are never extended to view boundaries.

Per-source summaries show the range's observed peak, recorded limit or changed
range, highest observed sample/limit ratio, and unavailable limit counts. They do
not infer continuous occupancy or aggregate workers/ranks. All evidence remains
in the input objects. Missing `references` is valid; observed values still render.
