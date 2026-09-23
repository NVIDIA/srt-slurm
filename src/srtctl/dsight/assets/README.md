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
It does not resample, interpolate, or add boundary samples. Hover values show the
nearest recorded sample with its actual timestamp. Sample evidence remains in
the unchanged input objects; this component only reads timestamps and values.
