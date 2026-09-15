# Dashboard assets

`dashboard.html`, `dashboard.css`, and `dashboard.js` are the offline Tachometer
dashboard shell. Python embeds them, the catalog, and per-metric gzip payloads into
one HTML file. No Node build or web server is required to generate or open it.

The vendored plotting library is **uPlot 1.6.32**, MIT licensed:

- Source: https://github.com/leeoniya/uPlot/tree/1.6.32
- JavaScript: https://raw.githubusercontent.com/leeoniya/uPlot/1.6.32/dist/uPlot.iife.min.js
- CSS: https://raw.githubusercontent.com/leeoniya/uPlot/1.6.32/dist/uPlot.min.css
- License: `uPlot.LICENSE`, copied verbatim from that tag.

The browser needs `DecompressionStream` support for gzip, Canvas 2D, and JavaScript.
There are no CDN requests, web fonts, telemetry, or external data fetches.
