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

  function normalizedSeries(raw) {
    const conflicts = new Set(raw.conflict_timestamps || []), seen = new Set(), points = [];
    for (const point of raw.points || []) {
      if (!Number.isFinite(point[0]) || seen.has(point[0])) continue;
      seen.add(point[0]);
      points.push([point[0], conflicts.has(point[0]) || !Number.isFinite(point[1]) ? null : point[1], ...point.slice(2)]);
    }
    points.sort((a, b) => a[0] - b[0]);
    return {raw, id: String(raw.id), labels: labelMap(raw), fullLabel: seriesLabel(raw), points};
  }

  function referenceAt(item, time) {
    if (!item) return null;
    let low = 0, high = item.points.length;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (item.points[middle][0] <= time) low = middle + 1; else high = middle;
    }
    const point = item.points[low - 1];
    return point && (item.raw.temporal === "setting" || point[0] === time) ? point : null;
  }

  function plotPoints(item, from, to) {
    if (item.raw.temporal !== "setting") return item.points;
    // Only the display projects a recorded setting to the view boundaries.
    // Raw/query points keep the original configuration timestamp and evidence.
    const points = item.points.filter(point => point[0] >= from && point[0] <= to);
    const prior = referenceAt(item, from);
    if (prior && prior[0] < from) points.unshift([from, prior[1]]);
    if (points.length && points[points.length - 1][0] < to) points.push([to, points[points.length - 1][1]]);
    return points;
  }

  function capacityNumber(value) {
    return Number.isFinite(value) ? value.toLocaleString("en-US", {maximumFractionDigits: 4}) : "—";
  }

  function capacitySummary(item, from, to) {
    const samples = item.points.filter(p => p[0] >= from && p[0] <= to && p[1] !== null);
    const label = item.raw.reference.label || "Limit";
    if (!samples.length) return ["No observed value in range", label + " shown where recorded"];
    const peak = samples.reduce((value, p) => Math.max(value, p[1]), -Infinity);
    let usage = null, unknown = 0;
    for (const point of samples) {
      const reference = referenceAt(item.reference, point[0]);
      if (reference?.[1] > 0) usage = Math.max(usage ?? 0, point[1] / reference[1]);
      else unknown++;
    }
    const referencePoints = item.reference ? plotPoints(item.reference, from, to).filter(p => p[0] >= from && p[0] <= to) : [];
    const limits = [...new Set(referencePoints.filter(p => p[1] !== null).map(p => p[1]))].sort((a, b) => a - b);
    const limit = !limits.length ? "unavailable" : limits.length === 1 ? capacityNumber(limits[0])
      : `${capacityNumber(limits[0])}–${capacityNumber(limits[limits.length - 1])} (changed)`;
    return [
      `Peak observed ${capacityNumber(peak)} ${item.raw.unit || ""}`, `${label} ${limit}`,
      ...(usage === null ? [] : [`Highest observed usage ${(100 * usage).toLocaleString("en-US", {maximumFractionDigits: 1})}%`]),
      ...(unknown ? [`Limit unavailable for ${unknown}/${samples.length} samples`] : []),
    ];
  }

  function configurationSeries(item) {
    return (item.raw.configuration || []).flatMap((config, index) => {
      const comparison = config.comparison;
      if (!comparison || comparison.unit !== item.raw.unit || !Number.isFinite(comparison.value) || comparison.value <= 0 || !Number.isFinite(comparison.start)) return [];
      const display = normalizedSeries({id: `config-${item.id}-${index}`, label: config.label,
        unit: comparison.unit, temporal: "setting", points: [[comparison.start, comparison.value]]});
      display.configuration = config;
      return [display];
    });
  }

  function configurationDetails(config, item, from, to, sources) {
    const box = element("details", "ds-metric-configuration");
    const summary = element("summary");
    const value = config.source?.value;
    summary.append(element("span", "", config.label + " "),
      element("strong", "", `${typeof value === "number" ? capacityNumber(value) : value == null ? "unset" : String(value)}${value == null ? "" : " " + config.unit}`));
    const field = element("code", "ds-metric-config-field", config.source?.field || "");
    summary.append(field);
    box.append(summary);
    const facts = element("dl", "ds-metric-config-facts");
    function fact(label, value) {
      if (value == null || value === "") return;
      facts.append(element("dt", "", label), element("dd", "", value));
    }
    const source = sources?.find(source => source.id === config.source?.source_id);
    if (config.comparison) {
      fact("Comparison", config.comparison.basis);
      fact("Valid from", `${numberText(config.comparison.start)} elapsed seconds`);
      const start = Math.max(from, config.comparison.start);
      const observed = item.points.filter(point => point[0] >= start && point[0] <= to && point[1] !== null);
      const peak = observed.length ? observed.reduce((peak, point) => Math.max(peak, point[1]), -Infinity) : null;
      if (peak !== null) fact("Peak / configured limit", `${capacityNumber(peak)} / ${capacityNumber(config.comparison.value)} (${(100 * peak / config.comparison.value).toLocaleString("en-US", {maximumFractionDigits: 1})}%)`);
      const limits = item.reference && start <= to ? plotPoints(item.reference, start, to).filter(point => point[0] >= start && point[0] <= to && point[1] !== null) : [];
      if (limits.length) {
        const match = limits.every(point => point[1] === config.comparison.value);
        const status = element("span", "ds-metric-config-status", match ? "Recorded limits match" : "Recorded limits differ");
        status.dataset.match = String(match);
        summary.append(status);
      }
    }
    fact("Scope", config.scope);
    if (config.note) fact("Note", config.note);
    fact("Source", `${source?.path || config.source?.file || "Source " + config.source?.source_id}${config.source?.line ? ":" + config.source.line : ""}`);
    fact("Field", config.source?.field);
    for (const evidence of config.comparison?.evidence || []) {
      const file = sources?.find(source => source.id === evidence.source_id);
      fact("Scope evidence", `${evidence.field}=${evidence.value}; ${file?.path || "source " + evidence.source_id}${evidence.line ? ":" + evidence.line : ""}`);
    }
    if (source?.sha256) fact("SHA-256", source.sha256);
    box.append(facts);
    return box;
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
    const references = new Map((options.references || []).map(raw => [String(raw.id), normalizedSeries(raw)]));
    const series = (options.series || []).map(normalizedSeries).sort((a, b) => a.id.localeCompare(b.id, "en", {numeric: true}));
    for (const item of series) {
      const reference = references.get(String(item.raw.reference?.series_id));
      if (reference && reference.raw.unit === item.raw.unit) item.reference = reference;
      item.configurations = configurationSeries(item);
    }
    setCaptions(series);
    const title = options.title || series[0]?.raw.label || series[0]?.raw.name || "Metric";
    const height = Math.max(150, Number(options.height) || 220);
    const colors = new Map(series.map((item, index) => [item.id, COLORS[index % COLORS.length]]));
    const root = element("section", "ds-metric-chart");
    root.setAttribute("aria-label", title + " metric series");
    root.dataset.drawnIds = JSON.stringify(series.map(item => item.id));
    root.dataset.referenceIds = JSON.stringify(series.filter(item => item.reference).map(item => item.reference.id));
    const tools = element("div", "ds-metric-tools");
    const count = element("span", "ds-metric-count");
    count.setAttribute("aria-live", "polite");
    const chart = element("div", "ds-metric-canvas");
    const legend = element("div", "ds-metric-legend");
    const hasConflicts = series.some(item => item.raw.conflict_timestamps?.length || item.reference?.raw.conflict_timestamps?.length);
    const hasCapacity = series.some(item => item.raw.reference || item.configurations.length);
    const plotItems = [
      ...series.map(item => ({item, owner: item, reference: false})),
      ...series.filter(item => item.reference).map(item => ({item: item.reference, owner: item, reference: true})),
      ...series.flatMap(owner => owner.configurations.map(item => ({item, owner, configuration: true}))),
    ];
    const note = element("p", "ds-metric-note", "Click a legend entry to show or hide its line. Drag across the plot to set the shared time range. Hover values show the nearest recorded sample and its timestamp." + (hasConflicts ? " Conflicting values at the same timestamp are shown as gaps; raw evidence retains every value." : ""));
    if (hasCapacity) note.textContent += " Solid lines show observations; matching dashed lines show limits. Settings are held until the next recorded configuration; sampled limits are not extended. Peak and usage summarize recorded samples in this range, not continuous occupancy.";
    else if (series.some(item => item.raw.temporal === "setting")) note.textContent += " Dashed steps hold the recorded configuration until it changes; hover shows the original setting timestamp.";
    if (series.some(item => item.configurations.length)) note.textContent += " Dotted lines show comparable limits from the saved recipe. Configuration details identify the exact field, source and comparison scope.";
    root.dataset.configurationComparisons = JSON.stringify(series.flatMap(item => item.configurations.map(config => ({series: item.id, field: config.configuration.source.field, value: config.points[0][1], start: config.points[0][0]}))));
    tools.append(element("h3", "ds-metric-title", title), count);
    root.append(tools, chart, legend, note);
    host.append(root);

    function updateCount() {
      const visible = series.filter(item => !hidden.has(item.id)).length;
      count.textContent = `${visible} shown / ${series.length} recorded series`;
      chart.setAttribute("role", "img");
      chart.setAttribute("aria-label", `${title}; ${visible} series shown, elapsed ${from} to ${to} seconds. Legend entries control visibility; source labels and sampled values appear below.`);
    }

    const legendRows = series.map(item => {
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
        for (let position = 0; position < plotItems.length; position++)
          if (plotItems[position].owner.id === item.id) plot?.setSeries(position + 1, {show});
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
        source_kind: item.raw.source_kind, generator: item.raw.generator,
        temporal: item.raw.temporal, reference: item.raw.reference,
        reference_source_ids: item.reference?.raw.source_ids, configuration: item.raw.configuration,
      }, null, 2)));
      row.append(toggle, value, details);
      if (item.raw.reference) {
        const capacity = element("div", "ds-metric-capacity");
        capacitySummary(item, from, to).forEach((part, index) => {
          if (index) capacity.append(document.createTextNode(" · "));
          capacity.append(element(index < 2 ? "strong" : "span", "", part));
        });
        capacity.style.borderLeftColor = colors.get(item.id);
        row.append(capacity);
      }
      for (const config of item.raw.configuration || []) row.append(configurationDetails(config, item, from, to, options.sources));
      legend.append(row);
      return {item, value};
    });

    function updateValues(time = null) {
      for (const {item, value} of legendRows) {
        const sample = item.raw.temporal === "setting" ? referenceAt(item, time ?? to) : nearestPoint(item.points, time ?? to, from, to);
        const unit = item.raw.unit ? " " + item.raw.unit : "";
        value.textContent = sample ? `${numberText(sample[1])}${sample[1] === null ? "" : unit} @ ${numberText(sample[0])}s` : "No sample in range";
        const unavailable = item.raw.conflict_timestamps?.includes(sample?.[0]) ? "conflicting recorded values" : "unavailable";
        value.title = sample ? `Recorded ${item.raw.temporal === "setting" ? "setting" : "sample"} at ${sample[0]} elapsed seconds: ${sample[1] ?? unavailable}${unit}` : "No recorded sample in the selected range";
        if (sample && item.raw.reference) {
          const reference = referenceAt(item.reference, sample[0]);
          value.textContent += ` · limit ${capacityNumber(reference?.[1])}`;
          if (reference) value.title += `; ${item.raw.reference.label}: ${reference[1] ?? "unavailable"}, recorded at ${reference[0]}s (source ${reference[2]}, line ${reference[3]})`;
        }
      }
    }
    updateCount(); updateValues();
    if (!series.length) chart.append(element("div", "ds-metric-empty", "No recorded series for this metric."));
    else if (typeof window.uPlot !== "function") chart.append(element("div", "ds-metric-empty", "The bundled chart library could not be loaded."));
    else if (!plotItems.some(({item}) => plotPoints(item, from, to).some(point => point[0] >= from && point[0] <= to && point[1] !== null))) {
      const conflictsInRange = series.some(item => item.raw.conflict_timestamps?.some(time => time >= from && time <= to));
      chart.append(element("div", "ds-metric-empty", conflictsInRange
        ? "No unambiguous metric value in this time range. Conflicting observations remain in the raw evidence."
        : "No metric sample in this time range. Widen the shared time range."));
    } else {
      // uPlot.join retains explicit nulls and uses undefined only for alignment holes.
      // Configuration display boundaries are separate from preserved raw points.
      const data = window.uPlot.join(plotItems.map(({item}) => {
        const points = plotPoints(item, from, to);
        return [points.map(point => point[0]), points.map(point => point[1])];
      }));
      const chartOptions = {
        width: Math.max(180, chart.clientWidth), height, padding: [10, 0, 0, 0],
        scales: {x: {time: false, min: from, max: to}, ...(hasCapacity ? {y: {range: (u, min, max) => [Math.min(0, min), Math.max(1, max * 1.08)]}} : {})},
        legend: {show: false},
        cursor: {drag: {x: true, y: false, setScale: false}, sync: {key: options.syncKey || "dsight-metrics", scales: ["x", null]}},
        series: [{label: "Elapsed seconds"}, ...plotItems.map(({item, owner, reference, configuration}) => ({
          label: configuration ? item.configuration.label : reference ? owner.raw.reference.label : owner.caption,
          stroke: colors.get(owner.id), width: reference ? 1 : 1.5, spanGaps: false,
          dash: configuration ? [2, 3] : reference || item.raw.temporal === "setting" ? [6, 4] : [],
          ...(reference || item.raw.temporal === "setting" ? {paths: window.uPlot.paths.stepped({align: 1})} : {}),
          show: !hidden.has(owner.id), points: {show: !configuration && !reference && item.raw.temporal !== "setting" && (item.raw.temporal === "sample" || item.points.filter(point => point[0] >= from && point[0] <= to && point[1] !== null).length === 1), size: 4},
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
