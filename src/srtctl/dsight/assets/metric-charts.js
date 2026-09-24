/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */
(() => {
  "use strict";
  const COLORS = ["#2670b5", "#087f8c", "#b77512", "#7154b4", "#bd465b", "#467c38", "#bb5e21", "#5368a4", "#9e5185", "#327f72", "#887222", "#665c8b"];
  const LABEL_ORDER = ["endpoint", "host", "hostname", "worker", "worker_role", "worker_index", "worker_process", "gpu", "rank", "rank_kind"];

  function element(tag, className = "", text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  }

  function numberText(value) {
    if (value == null || !Number.isFinite(value)) return "—";
    const magnitude = Math.abs(value);
    if (magnitude >= 1e9) return (value / 1e9).toFixed(2) + "G";
    if (magnitude >= 1e6) return (value / 1e6).toFixed(2) + "M";
    if (magnitude >= 1e3) return (value / 1e3).toFixed(2) + "k";
    if (magnitude > 0 && magnitude < 0.001) return value.toExponential(2);
    return value.toLocaleString("en-US", { maximumFractionDigits: 4 });
  }

  function labelMap(series) {
    // Retain collisions between recorded labels and normalized identity fields.
    const labels = { ...(series.metadata || {}), ...(series.labels || {}) };
    for (const [key, value] of Object.entries(series.metadata || {})) labels["capture." + key] = value;
    for (const [key, value] of Object.entries(series.labels || {})) labels["label." + key] = value;
    for (const key of ["endpoint", "host", "worker", "gpu", "rank", "rank_kind", "worker_process"]) {
      const value = series[key];
      if (value === undefined || value === null || value === "") continue;
      labels["series." + key] = String(value);
      if (!(key in labels)) labels[key] = String(value);
    }
    return Object.fromEntries(Object.entries(labels).filter(([, value]) => value != null).map(([key, value]) => [key, String(value)]));
  }

  function seriesLabel(series) {
    const all = labelMap(series);
    const keys = Object.keys(all).filter(key => {
      const scope = /^(?:label|capture|series)\.(.+)$/.exec(key);
      return !scope || all[scope[1]] !== all[key];
    });
    keys.sort((a, b) => {
      const ai = LABEL_ORDER.indexOf(a), bi = LABEL_ORDER.indexOf(b);
      return (ai < 0 ? LABEL_ORDER.length : ai) - (bi < 0 ? LABEL_ORDER.length : bi) || a.localeCompare(b);
    });
    return keys.map(key => `${key}=${JSON.stringify(all[key])}`).join(", ") || `series=${series.id}`;
  }

  function setCaptions(series) {
    const groups = new Map();
    for (const item of series) {
      const raw = item.raw, parts = [];
      if (raw.worker) parts.push(raw.worker);
      else if (raw.host) parts.push(raw.host);
      if (raw.gpu != null && raw.gpu !== "") parts.push("GPU " + raw.gpu);
      if (raw.rank != null) parts.push(`${raw.rank_kind || "rank"} ${raw.rank}`);
      item.caption = parts.join(" · ") || raw.endpoint || `Series ${item.id}`;
      if (!groups.has(item.caption)) groups.set(item.caption, []);
      groups.get(item.caption).push(item);
    }
    // Add only the dimensions needed to distinguish otherwise identical captions.
    for (const group of groups.values()) {
      if (group.length < 2) continue;
      const keys = [...new Set(group.flatMap(item => Object.keys(item.labels)))];
      keys.sort((a, b) => {
        const order = ["rank", "dp_rank", "global_rank", "worker_process", "pid", "gpu", "host", "endpoint", "stage", "mode"];
        const ai = order.indexOf(a), bi = order.indexOf(b);
        return (ai < 0 ? order.length : ai) - (bi < 0 ? order.length : bi) || a.localeCompare(b);
      });
      for (const key of keys) {
        if (new Set(group.map(item => item.labels[key])).size < 2) continue;
        for (const item of group) item.caption += ` · ${key}=${item.labels[key] ?? "—"}`;
        if (new Set(group.map(item => item.caption)).size === group.length) break;
      }
      const counts = new Map();
      for (const item of group) counts.set(item.caption, (counts.get(item.caption) || 0) + 1);
      for (const item of group) if (counts.get(item.caption) > 1) item.caption += ` · series ${item.id}`;
    }
  }

  function nearestPoint(points, time, from, to) {
    let low = 0, high = points.length;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (points[middle][0] < time) low = middle + 1;
      else high = middle;
    }
    return [points[low - 1], points[low]].filter(point => point && point[0] >= from && point[0] <= to)
      .sort((a, b) => Math.abs(a[0] - time) - Math.abs(b[0] - time))[0];
  }

  /**
   * Mount one metric family across all supplied sources. Points: [seconds, value, ...evidence].
   * selection: {hidden: string[]}; legacy ids/filters are ignored. Callbacks own persistence.
   */
  function mount(host, options) {
    let destroyed = false, plot = null, resize = null;
    const hidden = new Set((options.selection?.hidden || []).map(String));
    const from = Number(options.from), to = Number(options.to);
    if (!Number.isFinite(from) || !Number.isFinite(to) || to <= from) throw new Error("Metric charts need a finite increasing time range.");
    const series = (options.series || []).map(raw => {
      const conflicts = new Set(raw.conflict_timestamps || []), seen = new Set();
      const points = [];
      for (const point of raw.points || []) {
        if (!Number.isFinite(point[0]) || seen.has(point[0])) continue;
        seen.add(point[0]);
        points.push([point[0], conflicts.has(point[0]) || !Number.isFinite(point[1]) ? null : point[1]]);
      }
      points.sort((a, b) => a[0] - b[0]);
      return {raw, id: String(raw.id), labels: labelMap(raw), fullLabel: seriesLabel(raw), points};
    }).sort((a, b) => a.id.localeCompare(b.id, "en", { numeric: true }));
    setCaptions(series);
    const title = options.title || series[0]?.raw.label || series[0]?.raw.name || "Metric";
    const height = Math.max(150, Number(options.height) || 220);
    const colors = new Map(series.map((item, index) => [item.id, COLORS[index % COLORS.length]]));
    const root = element("section", "ds-metric-chart");
    root.setAttribute("aria-label", title + " metric series");
    root.dataset.drawnIds = JSON.stringify(series.map(item => item.id));
    const tools = element("div", "ds-metric-tools");
    const count = element("span", "ds-metric-count");
    count.setAttribute("aria-live", "polite");
    const chart = element("div", "ds-metric-canvas");
    const legend = element("div", "ds-metric-legend");
    const hasConflicts = series.some(item => item.raw.conflict_timestamps?.length);
    const note = element("p", "ds-metric-note", "Click a legend entry to show or hide its line. Drag across the plot to set the shared time range. Hover values show the nearest recorded sample and its timestamp." + (hasConflicts ? " Conflicting values at the same timestamp are shown as gaps; raw evidence retains every value." : ""));
    tools.append(element("h3", "ds-metric-title", title), count);
    root.append(tools, chart, legend, note);
    host.append(root);

    function updateCount() {
      const visible = series.filter(item => !hidden.has(item.id)).length;
      count.textContent = `${visible} shown / ${series.length} recorded series`;
      chart.setAttribute("role", "img");
      chart.setAttribute("aria-label", `${title}; ${visible} series shown, elapsed ${from} to ${to} seconds. Legend entries control visibility; source labels and sampled values appear below.`);
    }

    const legendRows = series.map((item, index) => {
      const row = element("div", "ds-metric-legend-row");
      row.dataset.seriesId = item.id;
      const toggle = element("button", "ds-metric-legend-toggle");
      toggle.type = "button";
      toggle.title = item.fullLabel;
      toggle.setAttribute("aria-label", `Toggle series ${item.caption}`);
      toggle.setAttribute("aria-pressed", String(!hidden.has(item.id)));
      toggle.addEventListener("click", () => {
        if (hidden.has(item.id)) hidden.delete(item.id); else hidden.add(item.id);
        const show = !hidden.has(item.id);
        toggle.setAttribute("aria-pressed", String(show));
        plot?.setSeries(index + 1, {show});
        updateCount();
        options.onSelectionChange?.({hidden: [...hidden]});
      });
      const swatch = element("span", "ds-metric-swatch"); swatch.style.background = colors.get(item.id);
      toggle.append(swatch, element("span", "ds-metric-caption", item.caption));
      const value = element("span", "ds-metric-value");
      const details = element("details", "ds-metric-label-details");
      details.append(element("summary", "", "Labels"), element("pre", "", JSON.stringify({
        id: item.raw.id, name: item.raw.name, raw_name: item.raw.raw_name,
        endpoint: item.raw.endpoint, host: item.raw.host, worker: item.raw.worker,
        gpu: item.raw.gpu, rank: item.raw.rank, rank_kind: item.raw.rank_kind,
        metadata: item.raw.metadata, labels: item.raw.labels,
        conflict_timestamps: item.raw.conflict_timestamps,
        conflicting_samples: item.raw.conflicting_samples,
      }, null, 2)));
      row.append(toggle, value, details); legend.append(row);
      return {item, value};
    });

    function updateValues(time = null) {
      for (const {item, value} of legendRows) {
        const sample = nearestPoint(item.points, time ?? to, from, to);
        const unit = item.raw.unit ? " " + item.raw.unit : "";
        value.textContent = sample ? `${numberText(sample[1])}${sample[1] === null ? "" : unit} @ ${numberText(sample[0])}s` : "No sample in range";
        const unavailable = item.raw.conflict_timestamps?.includes(sample?.[0]) ? "conflicting recorded values" : "unavailable";
        value.title = sample ? `Recorded sample at ${sample[0]} elapsed seconds: ${sample[1] ?? unavailable}${unit}` : "No recorded sample in the selected range";
      }
    }
    updateCount(); updateValues();
    if (!series.length) chart.append(element("div", "ds-metric-empty", "No recorded series for this metric."));
    else if (typeof window.uPlot !== "function") chart.append(element("div", "ds-metric-empty", "The bundled chart library could not be loaded."));
    else if (!series.some(item => item.points.some(point => point[0] >= from && point[0] <= to && point[1] !== null))) {
      const conflictsInRange = series.some(item => item.raw.conflict_timestamps?.some(time => time >= from && time <= to));
      chart.append(element("div", "ds-metric-empty", conflictsInRange
        ? "No unambiguous metric value in this time range. Conflicting observations remain in the raw evidence."
        : "No metric sample in this time range. Widen the shared time range."));
    } else {
      // uPlot.join retains explicit nulls and uses undefined only for alignment holes.
      // No resampling, sample interpolation, or fabricated boundary samples are used.
      const data = window.uPlot.join(series.map(item => [item.points.map(point => point[0]), item.points.map(point => point[1])]));
      const chartOptions = {
        width: Math.max(180, chart.clientWidth), height, padding: [10, 0, 0, 0],
        scales: {x: {time: false, min: from, max: to}},
        legend: {show: false},
        cursor: {drag: {x: true, y: false, setScale: false}, sync: {key: options.syncKey || "dsight-metrics", scales: ["x", null]}},
        series: [{label: "Elapsed seconds"}, ...series.map(item => ({
          label: item.caption, stroke: colors.get(item.id), width: 1.5, spanGaps: false,
          show: !hidden.has(item.id), points: {show: item.points.filter(point => point[0] >= from && point[0] <= to && point[1] !== null).length === 1, size: 5},
        }))],
        axes: [
          {stroke: "#5f7187", grid: {stroke: "#d8e1ed88"}, ticks: {stroke: "#d8e1ed"}, font: "11px sans-serif", size: 30, values: (_, ticks) => ticks.map(value => numberText(value) + "s")},
          {stroke: "#5f7187", grid: {stroke: "#d8e1ed88"}, ticks: {stroke: "#d8e1ed"}, font: "11px sans-serif", size: 52, values: (_, ticks) => ticks.map(numberText)},
        ],
        hooks: {
          setCursor: [u => updateValues(u.cursor.left >= 0 ? u.posToVal(u.cursor.left, "x") : null)],
          setSelect: [u => {
            if (u.select.width < 4) return;
            const start = Math.max(from, u.posToVal(u.select.left, "x"));
            const end = Math.min(to, u.posToVal(u.select.left + u.select.width, "x"));
            u.setSelect({left: 0, top: 0, width: 0, height: 0}, false);
            if (end > start) options.onRangeChange?.(start, end);
          }],
        },
      };
      plot = new window.uPlot(chartOptions, data, chart);
      plot.setScale("x", {min: from, max: to});
      resize = new ResizeObserver(() => {
        const nextWidth = Math.max(180, chart.clientWidth);
        if (!destroyed && plot && Math.abs(nextWidth - plot.width) > 1) plot.setSize({width: nextWidth, height});
      });
      resize.observe(chart);
    }
    return {
      destroy() {
        destroyed = true;
        resize?.disconnect(); resize = null;
        plot?.destroy(); plot = null;
        root.remove();
      },
    };
  }

  window.DSightMetricCharts = Object.freeze({mount});
})();
