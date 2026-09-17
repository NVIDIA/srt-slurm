/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const catalog = JSON.parse($("dashboard-catalog").textContent);
  const metrics = catalog.metrics || [];
  const components = ["Frontend", "Router", "Workers", "GPU", "Host"];
  const descriptions = {
    Frontend: "Request admission, tokenization, response handling, and frontend runtime activity.",
    Router: "Queueing, routing decisions, prefix-cache estimates, and index activity.",
    Workers: "Engine scheduling, KV cache, iterations, and request latency, alongside worker handler pools, admission, and transport.",
    GPU: "Per-device DCGM measurements. Select an endpoint and GPU to inspect an individual device.",
    Host: "Host resources and process-exporter groups, including CPU, memory, threads, and scheduling."
  };
  const groupOrder = {
    Frontend: ["Requests and latency", "Native HTTP requests", "Native prefill client", "Native decode client", "Tokenization", "Tokenizer cache", "Worker dispatch and streaming", "Detokenization", "Tokio runtime", "Advertised model metadata", "Native HTTP metadata", "Request lifecycle", "First response and transport", "Component lifecycle", "Other captured metrics"],
    Router: ["Routing decisions", "Queue and backpressure", "Worker selection and feedback", "KV matching and cache", "KV index and events", "Other captured metrics"],
    Workers: ["Engine scheduling", "Engine iterations", "Engine KV cache", "Engine KV transfer", "Engine requests and latency", "Engine memory", "Engine speculative decoding", "Engine KV events", "Request lifecycle", "Admission and queues", "Work-handler pool", "First response and transport", "Engine configuration", "Component lifecycle", "Other captured metrics"],
    GPU: ["Utilization", "Framebuffer memory", "Power, clocks and temperature", "NVLink and PCIe", "Hardware events", "Other captured metrics"],
    Host: ["Process CPU and scheduling", "Host scheduling", "Host memory", "Process memory", "NUMA memory and locality", "Processes and threads", "Paging and reclaim", "Network and InfiniBand", "Collection health", "Other captured metrics"]
  };
  const colors = ["#80acff", "#85c9ad", "#efc17b", "#cb9dec", "#fa8e99", "#73c6da", "#d0d17b", "#efac7b", "#92a3e5", "#d995ba", "#a7cfd1", "#bfbaa7"];
  const maxSeries = 12;
  const resolution = Number(catalog.resolution_s) || 1;
  const duration = Math.max(Number(catalog.duration_s) || 0, resolution);
  const commonKeys = ["endpoint", "hostname", "worker_role", "groupname", "gpu"];
  const sourceNames = (catalog.source_files || []).map(x => typeof x === "string" ? x : JSON.stringify(x));
  const hasFinalSource = (catalog.source_files || []).some(source => String(typeof source === "string" ? source : source.path || "").split(/[\\/]/).pop() === "final.parquet");
  let hash = 0;
  for (const char of sourceNames.join("|") + catalog.start_ns + catalog.row_count) hash = (Math.imul(hash, 31) + char.charCodeAt(0)) | 0;
  const storageKey = "tachometer-dashboard-v1-" + hash;
  function record(value) { return value !== null && typeof value === "object" && !Array.isArray(value) ? value : {}; }
  function savedMap(value, valid) { return Object.fromEntries(Object.entries(record(value)).filter(([, item]) => valid(item))); }
  let stored = {};
  try { stored = record(JSON.parse(localStorage.getItem(storageKey) || "{}")); } catch (_) { /* file:// may deny storage */ }
  const state = {
    component: components.includes(stored.component) ? stored.component : "Frontend",
    search: typeof stored.search === "string" ? stored.search : "",
    visibility_version: 2,
    // Legacy `all: false` was the old default, not an explicit request to hide metrics.
    featuredOnly: stored.visibility_version === 2 && stored.featuredOnly === true,
    filters: savedMap(stored.filters, value => typeof value === "string"),
    from: Math.max(0, Number(stored.from) || 0),
    to: Math.min(duration, Number(stored.to) || duration),
    collapsed: savedMap(stored.collapsed, value => typeof value === "boolean"),
    views: savedMap(stored.views, value => typeof value === "string"),
    selections: savedMap(stored.selections, value => Array.isArray(value) && value.every(item => typeof item === "string"))
  };
  if (!(state.to > state.from)) { state.from = 0; state.to = duration; }
  const payloadCache = new Map();
  const pending = new Map();
  const cards = new Map();
  let generation = 0;
  let observer;

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }
  function button(text, handler, cls) {
    const b = el("button", cls, text); b.type = "button";
    b.addEventListener("click", handler); return b;
  }
  function option(value, text) { const node = el("option", "", text); node.value = value; return node; }
  function save() { try { localStorage.setItem(storageKey, JSON.stringify(state)); } catch (_) { /* persistence optional */ } }
  function compact(value) { return Number(value || 0).toLocaleString("en-US"); }
  function valueText(value, unit = "") {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    const n = Number(value), a = Math.abs(n);
    let rendered;
    if (a >= 1e9) rendered = (n / 1e9).toFixed(2) + "G";
    else if (a >= 1e6) rendered = (n / 1e6).toFixed(2) + "M";
    else if (a >= 1e3) rendered = (n / 1e3).toFixed(2) + "k";
    else if (a !== 0 && a < 0.001) rendered = n.toExponential(2);
    else rendered = n.toLocaleString("en-US", { maximumFractionDigits: a < 1 ? 4 : 2 });
    return rendered + (unit ? " " + unit : "");
  }
  function timeText(t) { return Number(t).toLocaleString("en-US", { maximumFractionDigits: 2 }) + " s"; }
  function labelMap(series) {
    const combined = { ...(series.metadata || {}), ...(series.labels || {}), endpoint: String(series.endpoint || ""), scraper_endpoint: String(series.endpoint || "") };
    for (const [key, value] of Object.entries(series.metadata || {})) combined["capture." + key] = value;
    for (const [key, value] of Object.entries(series.labels || {})) combined["label." + key] = value;
    return combined;
  }
  function identity(series) {
    return JSON.stringify([series.endpoint || "", Object.entries(series.metadata || {}).sort(), Object.entries(series.labels || {}).sort()]);
  }
  function seriesLabel(series) {
    const labels = { ...(series.metadata || {}), ...(series.labels || {}), endpoint: String(series.endpoint || "") };
    for (const [key, value] of Object.entries(series.metadata || {})) {
      if (series.labels && key in series.labels && String(value) !== String(series.labels[key])) labels["capture." + key] = value;
    }
    const order = ["endpoint", "hostname", "worker_role", "frontend_index", "worker_index", "worker_process", "groupname", "threadname", "worker", "gpu", "numa_node", "mode", "dp_rank", "status", "stage", "le"];
    const keys = [...order.filter(k => labels[k] !== undefined), ...Object.keys(labels).filter(k => !order.includes(k)).sort()];
    return keys.map(k => `${k}=${labels[k]}`).join(", ") || String(series.id || "Unlabelled series");
  }
  function matches(series) {
    const labels = labelMap(series);
    return Object.entries(state.filters).every(([k, v]) => v === "" || String(labels[k] ?? "") === String(v));
  }
  function warnings(metric) {
    const list = [...(Array.isArray(metric.warnings) ? metric.warnings : [])];
    for (const field of [metric.quality, metric.warning]) if (field) list.push(String(field));
    return [...new Set(list.map(String))];
  }
  function queryMatches(metric, query) {
    return !query || [metric.name, metric.title, metric.group, metric.component, ...(metric.aliases || [])].join(" ").toLowerCase().includes(query.toLowerCase());
  }
  function isVisibleMetric(metric) {
    return queryMatches(metric, state.search) && (!state.featuredOnly || metric.featured === true);
  }
  function unitFor(metric, view) {
    const unit = metric.unit && metric.unit !== "unknown" ? metric.unit : "";
    if (metric.kind === "histogram" && view === "buckets") return "observations/s";
    if (metric.counter && view === "rate") return metric.rate_unit || (unit.endsWith("/s") ? unit : (unit ? unit + "/s" : "/s"));
    return unit;
  }

  async function loadMetric(metric) {
    if (payloadCache.has(metric.id)) {
      const value = payloadCache.get(metric.id); payloadCache.delete(metric.id); payloadCache.set(metric.id, value); return value;
    }
    if (pending.has(metric.id)) return pending.get(metric.id);
    const promise = (async () => {
      if (!window.DecompressionStream) throw new Error("This browser cannot decompress the embedded capture. Open this file in a recent Chrome, Edge, Firefox, or Safari.");
      const embedded = $("payload-" + metric.id);
      if (!embedded) throw new Error("Embedded metric payload is missing: " + metric.id);
      const bytes = Uint8Array.from(atob(embedded.textContent.trim()), c => c.charCodeAt(0));
      const body = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
      const payload = JSON.parse(await new Response(body).text());
      payloadCache.set(metric.id, payload);
      while (payloadCache.size > 8) payloadCache.delete(payloadCache.keys().next().value);
      return payload;
    })();
    pending.set(metric.id, promise);
    try { return await promise; } finally { pending.delete(metric.id); }
  }

  function columns(payload) { return Object.fromEntries(payload.columns.map((name, i) => [name, i])); }
  function getNumber(point, i) { const v = point[i]; return v === null || v === undefined || !Number.isFinite(Number(v)) ? null : Number(v); }
  function pointValue(point, c, view) {
    // A missing scrape may be shorter than a display bin. Keep its bin absent
    // rather than drawing a continuous line through that interval.
    if (Number(point[c.gaps]) > 0) return null;
    if (view === "rate" || view === "buckets") {
      const delta = getNumber(point, c.delta), seconds = getNumber(point, c.observed_s);
      if (Number(point[c.resets]) > 0 || Number(point[c.gaps]) > 0 || delta === null || seconds === null || seconds <= 0) return null;
      return delta / seconds;
    }
    return getNumber(point, c[view] ?? c.mean);
  }
  function histogramSeries(payload, filtered, view) {
    const c = columns(payload), groups = new Map();
    let overflow = 0, incomplete = 0, ambiguous = 0;
    for (const raw of filtered) {
      const bound = raw.labels?.le;
      if (bound === undefined) continue;
      const upper = /^(\+?inf(inity)?)$/i.test(String(bound)) ? Infinity : Number(bound);
      if (Number.isNaN(upper)) continue;
      const labels = { ...(raw.labels || {}) }; delete labels.le;
      const metadata = { ...(raw.metadata || {}) }; delete metadata.le;
      const item = { ...raw, labels, metadata };
      const key = identity(item);
      if (!groups.has(key)) groups.set(key, { ...item, id: "hist:" + key, buckets: new Map(), points: [] });
      groups.get(key).buckets.set(upper, raw.points);
      if (raw.ambiguous) groups.get(key).ambiguous = true;
    }
    const q = Number(view.slice(1)) / 100;
    const result = [];
    for (const group of groups.values()) {
      if (group.ambiguous) { ambiguous++; delete group.buckets; result.push(group); continue; }
      const bounds = [...group.buckets.keys()].sort((a, b) => a - b);
      const bins = new Map();
      for (const upper of bounds) {
        for (const point of group.buckets.get(upper)) {
          const bin = Number(point[c.bin]);
          if (!bins.has(bin)) bins.set(bin, new Map());
          bins.get(bin).set(upper, point);
        }
      }
      for (const [bin, samples] of [...bins.entries()].sort((a, b) => a[0] - b[0])) {
        let value = null, valid = samples.size === bounds.length && bounds.includes(Infinity), previous = 0, observedSeconds = null;
        for (const upper of bounds) {
          const point = samples.get(upper);
          if (!point) { valid = false; break; }
          const count = getNumber(point, c.delta), seconds = getNumber(point, c.observed_s);
          // Bucket deltas must describe the same observation interval.
          if (seconds === null || seconds <= 0 || (observedSeconds !== null && Math.abs(seconds - observedSeconds) > Math.max(1e-6, observedSeconds * 1e-6))) { valid = false; break; }
          observedSeconds = seconds;
          if (count === null || count < previous || Number(point[c.resets]) > 0 || Number(point[c.gaps]) > 0) { valid = false; break; }
          previous = count;
        }
        if (valid) {
          const total = getNumber(samples.get(Infinity), c.delta);
          if (total > 0) {
            const target = q * total;
            let lower = 0, lowerCount = 0;
            for (const upper of bounds) {
              const count = getNumber(samples.get(upper), c.delta);
              if (count >= target) {
                if (upper === Infinity) { overflow++; break; }
                if (upper <= 0 && lower === 0) lower = upper;
                value = count > lowerCount ? lower + (upper - lower) * (target - lowerCount) / (count - lowerCount) : upper;
                break;
              }
              lower = upper; lowerCount = count;
            }
          }
        } else incomplete++;
        group.points.push([bin, value]);
      }
      delete group.buckets; result.push(group);
    }
    return { series: result, overflow, incomplete, ambiguous };
  }
  function buildSeries(metric, payload, view) {
    const filtered = payload.series.filter(matches), c = columns(payload);
    let result;
    if (metric.kind === "histogram" && view !== "buckets") result = histogramSeries(payload, filtered, view);
    else result = {
      series: filtered.map(raw => ({ ...raw, points: raw.points.map(point => [Number(point[c.bin]), pointValue(point, c, view)]) })),
      overflow: 0, incomplete: 0
    };
    for (const series of result.series) {
      series.key = String(series.id || identity(series));
      series.label = seriesLabel(series);
      let peak = -Infinity;
      for (const point of series.points) if (point[1] !== null && Number.isFinite(point[1])) peak = Math.max(peak, point[1]);
      series.peak = peak;
    }
    result.series.sort((a, b) => b.peak - a.peak || a.label.localeCompare(b.label));
    return result;
  }
  function alignedData(series) {
    let first = Infinity, last = -Infinity;
    for (const s of series) for (const point of s.points) { first = Math.min(first, point[0]); last = Math.max(last, point[0]); }
    if (!Number.isFinite(first)) return [[0, duration], ...series.map(() => [null, null])];
    first = Math.max(0, first - 1); last += 1;
    const length = last - first + 1;
    const x = Array.from({ length }, (_, i) => (first + i + 0.5) * resolution);
    return [x, ...series.map(s => {
      const values = Array(length).fill(null);
      for (const [bin, value] of s.points) values[bin - first] = value;
      return values;
    })];
  }

  function updateTimeInputs() {
    $("time-from").value = Number(state.from.toFixed(3)); $("time-to").value = Number(state.to.toFixed(3));
    $("time-summary").textContent = `Elapsed ${timeText(state.from)} to ${timeText(state.to)} · ${timeText(resolution)} display bins`;
  }
  function setTime(from, to) {
    from = Math.max(0, Number(from)); to = Math.min(duration, Number(to));
    if (!Number.isFinite(from) || !Number.isFinite(to) || to <= from) { $("status").textContent = "Choose an end time greater than the start time within this capture."; return; }
    state.from = from; state.to = to; updateTimeInputs(); save();
    for (const card of cards.values()) if (card.plot) { card.plot.setScale("x", { min: from, max: to }); updateLegend(card); }
  }
  function updateLegend(card, index = null) {
    if (!card.drawn) return;
    for (let i = 0; i < card.drawn.length; i++) {
      const s = card.drawn[i]; let value = null;
      if (index !== null && card.plot) value = card.plot.data[i + 1][index];
      else for (const point of s.points) {
        const t = (point[0] + 0.5) * resolution;
        if (t >= state.from && t <= state.to && point[1] !== null) value = point[1];
      }
      if (card.legendValues[i]) card.legendValues[i].textContent = valueText(value, card.unit);
    }
  }
  function destroyPlot(card) {
    if (card.plot) { card.plot.destroy(); card.plot = null; }
    if (card.resize) { card.resize.disconnect(); card.resize = null; }
  }
  function drawCard(card) {
    if (!card.processed || !card.node.isConnected) return;
    destroyPlot(card); card.chart.replaceChildren(); card.legend.replaceChildren(); card.legendValues = [];
    const available = card.processed.series;
    const selected = state.selections[card.metric.id];
    card.drawn = (Array.isArray(selected) ? available.filter(s => selected.includes(s.key)) : available).slice(0, maxSeries);
    const drawn = card.drawn;
    card.count.textContent = `${drawn.length} shown / ${compact(available.length)} matching series`;
    card.count.title = Array.isArray(selected) ? "Explicit series selection; at most 12 drawn." : available.length > maxSeries ? "Showing up to 12 series ranked by their full-capture peak. Use Choose series to select others." : "Every matching series is shown.";
    card.foot.replaceChildren(el("span", "", Array.isArray(selected) ? "Selected series" : available.length > maxSeries ? "Top 12 by full-capture peak" : "All matching series"), el("span", "", card.metric.kind === "histogram" && card.view !== "buckets" ? `Estimated ${card.view} from bucket deltas` : card.view === "rate" ? "Rate = counter delta / observed seconds" : `${card.view === "buckets" ? "Bucket rates" : card.view} per display bin`));
    if (card.unit) card.foot.append(el("span", "", "Unit: " + card.unit));
    const constants = drawn.map(series => {
      let first = null, constant = true;
      for (const point of series.points) if (point[1] !== null) {
        if (first === null) first = point[1]; else if (point[1] !== first) constant = false;
      }
      return { value: first, constant };
    });
    if (constants.length && constants.every(s => s.constant && s.value !== null)) {
      let text;
      if ((card.view === "rate" || card.view === "buckets") && constants.every(s => s.value === 0)) text = "No increments in valid intervals for the shown series.";
      else if (constants.every(s => s.value === constants[0].value)) text = "Full capture: constant at " + valueText(constants[0].value, card.unit) + ".";
      else text = "Full capture: shown series are constant; values differ by label.";
      card.foot.append(el("span", "", text));
    }
    const dynamicWarnings = [];
    if (card.processed.ambiguous) dynamicWarnings.push(`${compact(card.processed.ambiguous)} histogram identities have conflicting raw values; percentile estimates are unavailable.`);
    if (card.processed.overflow) dynamicWarnings.push(`${compact(card.processed.overflow)} interval estimates exceed the largest finite bucket and are shown as gaps.`);
    if (card.processed.incomplete) dynamicWarnings.push(`${compact(card.processed.incomplete)} incomplete, reset, or inconsistent bucket intervals are shown as gaps.`);
    card.warning.textContent = [...warnings(card.metric), ...dynamicWarnings].join(" ");
    if (!drawn.length) {
      card.chart.append(el("div", "chart-message", available.length ? "No series selected. Use Choose series to draw up to 12." : "No series match the selected labels. Clear a filter to see this metric.")); return;
    }
    if (drawn.every(s => s.peak === -Infinity)) {
      card.chart.append(el("div", "chart-message", card.view.startsWith("p") ? "No valid histogram intervals for this selection. Inspect bucket coverage or select Bucket rates." : "No valid values for this view. Inspect the samples or select raw values."));
      return;
    }
    const data = alignedData(drawn);
    const width = Math.max(200, card.chart.clientWidth - 18);
    const opts = {
      width, height: 215, padding: [10, 10, 0, 0],
      scales: { x: { time: false, min: state.from, max: state.to } },
      legend: { show: false },
      cursor: { drag: { x: true, y: false, setScale: false }, sync: { key: "tachometer-time", scales: ["x", null] } },
      select: { show: true },
      series: [{ label: "Elapsed seconds" }, ...drawn.map((s, i) => ({ label: s.label, stroke: colors[i % colors.length], width: 1.25, spanGaps: false, points: { show: false } }))],
      axes: [
        { stroke: "#a9b4c5", grid: { stroke: "#3b424e66" }, ticks: { stroke: "#3b424e" }, font: "11px Segoe UI", size: 28, values: (_, ticks) => ticks.map(t => valueText(t) + "s") },
        { stroke: "#a9b4c5", grid: { stroke: "#3b424e66" }, ticks: { stroke: "#3b424e" }, font: "11px Segoe UI", size: 67, values: (_, ticks) => ticks.map(t => valueText(t)) }
      ],
      hooks: {
        setCursor: [u => updateLegend(card, u.cursor.idx ?? null)],
        setSelect: [u => {
          if (u.select.width < 4) return;
          const from = u.posToVal(u.select.left, "x"), to = u.posToVal(u.select.left + u.select.width, "x");
          u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false); setTime(from, to);
        }]
      }
    };
    card.plot = new uPlot(opts, data, card.chart);
    card.plot.setScale("x", { min: state.from, max: state.to });
    card.chart.setAttribute("role", "img");
    card.chart.setAttribute("aria-label", `${card.metric.title || card.metric.name}, ${card.view}, ${drawn.length} series. Numeric samples are available through Inspect.`);
    drawn.forEach((s, i) => {
      const legendButton = button("", () => {
        const show = !card.plot.series[i + 1].show; card.plot.setSeries(i + 1, { show }); legendButton.setAttribute("aria-pressed", String(show));
      }, "legend-button");
      legendButton.setAttribute("aria-pressed", "true"); legendButton.title = s.label;
      const swatch = el("span", "legend-swatch"); swatch.style.background = colors[i % colors.length];
      legendButton.append(swatch, el("span", "legend-label", s.label));
      const value = el("span", "legend-value"); card.legendValues.push(value); card.legend.append(legendButton, value);
    });
    updateLegend(card);
    card.resize = new ResizeObserver(entries => {
      const next = Math.max(200, entries[0].contentRect.width);
      if (card.plot && Math.abs(next - card.plot.width) > 1) card.plot.setSize({ width: next, height: 215 });
    });
    card.resize.observe(card.chart);
  }
  function chooser(card) {
    const existing = card.node.querySelector(".series-chooser");
    if (existing) { existing.remove(); return; }
    const box = el("div", "series-chooser"), search = el("input"); search.type = "search"; search.placeholder = "Search full series labels"; search.setAttribute("aria-label", "Search series labels");
    const choices = el("div", "series-choices"), note = el("div", "series-chooser-note");
    const top = button("Use top 12", () => { delete state.selections[card.metric.id]; save(); drawCard(card); refresh(); });
    function refresh() {
      const available = card.processed.series.filter(s => s.label.toLowerCase().includes(search.value.toLowerCase()));
      const selected = new Set(state.selections[card.metric.id] || card.drawn.map(s => s.key));
      choices.replaceChildren();
      for (const s of available.slice(0, 100)) {
        const label = el("label", "series-choice"), input = el("input"); input.type = "checkbox"; input.checked = selected.has(s.key); input.disabled = !input.checked && selected.size >= maxSeries;
        input.addEventListener("change", () => {
          if (input.checked) selected.add(s.key); else selected.delete(s.key);
          state.selections[card.metric.id] = [...selected]; save(); drawCard(card); refresh();
        });
        label.append(input, el("span", "", s.label)); choices.append(label);
      }
      note.textContent = `${selected.size} selected; maximum ${maxSeries} on a chart. Showing ${Math.min(100, available.length)} of ${compact(available.length)} label matches. Search to reach additional series.`;
    }
    search.addEventListener("input", refresh); box.append(search, top, choices, note); card.node.insertBefore(box, card.chart); refresh(); search.focus();
  }

  async function activateCard(card) {
    if (card.loaded || card.loading) return;
    card.loading = true;
    try {
      const payload = await loadMetric(card.metric);
      if (card.generation !== generation || !card.node.isConnected) return;
      card.payload = payload; card.loaded = true;
      card.processed = buildSeries(card.metric, payload, card.view); card.unit = unitFor(card.metric, card.view);
      card.choose.disabled = false; card.viewSelect.disabled = false; drawCard(card);
    } catch (error) {
      card.chart.replaceChildren(el("div", "chart-message error", error.message));
    } finally { card.loading = false; }
  }
  function makeCard(metric) {
    const node = el("article", "metric-panel"); node.id = "metric-" + metric.id;
    const heading = el("div", "panel-heading"), titles = el("div");
    titles.append(el("h3", "", metric.title || metric.name), el("div", "metric-name", metric.name));
    heading.append(titles, button("Inspect", () => inspectMetric(metric), "inspect-button"));
    const controls = el("div", "panel-controls"), viewLabel = el("label", "", "View"), viewSelect = el("select");
    viewSelect.setAttribute("aria-label", "View for " + (metric.title || metric.name));
    const views = metric.kind === "histogram" ? [["p50", "p50 estimate"], ["p95", "p95 estimate"], ["p99", "p99 estimate"], ["buckets", "Bucket rates"]] : metric.counter ? [["rate", "Rate / second"], ["last", "Raw cumulative"], ["min", "Raw minimum"], ["max", "Raw maximum"]] : [["mean", "Mean"], ["min", "Minimum"], ["max", "Maximum"], ["last", "Last"]];
    for (const [key, label] of views) viewSelect.append(option(key, label));
    const preferred = state.views[metric.id];
    viewSelect.value = views.some(([key]) => key === preferred) ? preferred : views[0][0];
    viewSelect.disabled = true; viewLabel.append(viewSelect);
    const count = el("span", "series-count", `${compact(metric.series_count)} captured series`);
    const chart = el("div", "chart"), message = el("div", "chart-message", "Loading metric…"); chart.append(message);
    const legend = el("div", "legend"), foot = el("div", "panel-foot"), warning = el("div", "quality-note", warnings(metric).join(" "));
    const card = { metric, node, chart, legend, count, foot, warning, viewSelect, view: viewSelect.value, generation, loaded: false };
    const choose = button("Choose series", () => chooser(card)); choose.disabled = true; card.choose = choose;
    controls.append(viewLabel, choose, count); node.append(heading, controls, chart, legend, foot, warning);
    viewSelect.addEventListener("change", () => {
      card.view = viewSelect.value; state.views[metric.id] = card.view; delete state.selections[metric.id]; save();
      card.node.querySelector(".series-chooser")?.remove();
      card.processed = buildSeries(metric, card.payload, card.view); card.unit = unitFor(metric, card.view); drawCard(card);
    });
    cards.set(metric.id, card); return card;
  }

  function render() {
    generation++; if (observer) observer.disconnect();
    for (const card of cards.values()) destroyPlot(card); cards.clear();
    $("rows").replaceChildren(); $("status").textContent = "";
    $("component-title").textContent = state.component;
    $("component-description").textContent = descriptions[state.component];
    for (const tab of $("tabs").children) {
      const component = tab.dataset.component, active = component === state.component;
      const all = metrics.filter(metric => metric.component === component);
      const shown = all.filter(isVisibleMetric).length;
      tab.setAttribute("aria-selected", String(active)); tab.tabIndex = active ? 0 : -1;
      tab.dataset.visible = String(shown); tab.dataset.total = String(all.length);
      tab.querySelector(".tab-count").textContent = shown === all.length ? String(all.length) : `${shown}/${all.length}`;
      tab.setAttribute("aria-label", `${component}: ${shown} of ${all.length} metric families shown`);
    }
    const componentMetrics = metrics.filter(m => m.component === state.component);
    const visible = componentMetrics.filter(isVisibleMetric)
      .sort((a, b) => Number(a.order ?? 1000000) - Number(b.order ?? 1000000) || (a.title || a.name).localeCompare(b.title || b.name));
    $("panel-count").textContent = visible.length === componentMetrics.length ? `${visible.length} metric families` : `${visible.length} / ${componentMetrics.length} metric families shown`;
    $("panel-count").dataset.visible = String(visible.length); $("panel-count").dataset.total = String(componentMetrics.length);
    if (!visible.length) {
      $("status").textContent = componentMetrics.length ? "No metrics match this view. Clear the search or turn off Featured only." : "No metric families from this component were captured. This does not mean the component was idle.";
    }
    const groups = new Map();
    for (const metric of visible) {
      const group = metric.group || "Other captured metrics";
      if (!groups.has(group)) groups.set(group, []); groups.get(group).push(metric);
    }
    observer = new IntersectionObserver(entries => {
      for (const entry of entries) if (entry.isIntersecting) {
        const card = cards.get(entry.target.dataset.metric);
        if (card && card.node.closest("details").open) { activateCard(card); observer.unobserve(entry.target); }
      }
    }, { rootMargin: "250px" });
    let groupIndex = 0;
    const preferred = groupOrder[state.component] || [];
    const groupRank = name => preferred.includes(name) ? preferred.indexOf(name) : preferred.length;
    const orderedGroups = [...groups.entries()].sort(([a], [b]) => groupRank(a) - groupRank(b) || a.localeCompare(b));
    for (const [name, members] of orderedGroups) {
      const key = state.component + "/" + name;
      const row = el("details", "metric-row"); row.dataset.count = String(members.length);
      row.open = state.collapsed[key] === undefined ? groupIndex < 2 : !state.collapsed[key];
      const summary = el("summary", "", name); summary.append(el("span", "row-count", `${members.length} metric ${members.length === 1 ? "family" : "families"}`));
      const grid = el("div", "panel-grid");
      for (const metric of members) {
        const card = makeCard(metric); card.node.dataset.metric = metric.id; grid.append(card.node); observer.observe(card.node);
      }
      row.append(summary, grid); $("rows").append(row);
      row.addEventListener("toggle", () => {
        state.collapsed[key] = !row.open; save();
        if (row.open) for (const article of grid.children) {
          const card = cards.get(article.dataset.metric);
          if (card.loaded) drawCard(card); else observer.observe(article);
        }
      });
      groupIndex++;
    }
    save();
  }
  function filtersChanged() { renderChips(); save(); render(); }
  function setupFilters() {
    const values = { ...(catalog.label_values || {}) };
    if (values.scraper_endpoint) values.endpoint = values.scraper_endpoint;
    for (const key of commonKeys) {
      const choices = values[key] || [];
      if (!choices.length) continue;
      const label = el("label", "filter", key), select = el("select"); select.setAttribute("aria-label", "Filter " + key); select.dataset.label = key;
      select.append(option("", "All"));
      for (const value of choices) select.append(option(String(value), String(value)));
      select.value = state.filters[key] || "";
      select.addEventListener("change", () => { if (select.value) state.filters[key] = select.value; else delete state.filters[key]; filtersChanged(); });
      label.append(select); $("filters").append(label);
    }
    const keys = Object.keys(values).sort();
    $("custom-key").append(option("", "Select label"));
    for (const key of keys) $("custom-key").append(option(key, key));
    function updateCustomValues() {
      const key = $("custom-key").value; $("custom-value").replaceChildren(option("", "Select value"));
      for (const value of values[key] || []) $("custom-value").append(option(String(value), String(value)));
    }
    $("custom-key").addEventListener("change", updateCustomValues); updateCustomValues();
    $("add-filter").addEventListener("click", () => {
      const key = $("custom-key").value, value = $("custom-value").value;
      if (!key || value === "") return;
      state.filters[key] = value; syncFilterSelects(); filtersChanged();
    });
    $("clear-filters").addEventListener("click", () => { state.filters = {}; syncFilterSelects(); filtersChanged(); });
    renderChips();
  }
  function syncFilterSelects() { for (const select of $("filters").querySelectorAll("select")) select.value = state.filters[select.dataset.label] || ""; }
  function renderChips() {
    $("filter-chips").replaceChildren();
    for (const [key, value] of Object.entries(state.filters)) {
      const chip = button(`${key}=${value} ×`, () => { delete state.filters[key]; syncFilterSelects(); filtersChanged(); }, "filter-chip");
      chip.setAttribute("aria-label", `Remove filter ${key} equals ${value}`); $("filter-chips").append(chip);
    }
  }
  function showExplorer() { $("explorer").hidden = false; renderExplorer(); $("explorer-search").focus(); }
  function renderExplorer() {
    const query = $("explorer-search").value;
    const found = metrics.filter(m => queryMatches(m, query));
    $("explorer-results").replaceChildren(el("p", "", `${found.length} metric families`));
    for (const metric of found) {
      const row = button(metric.title || metric.name, () => {
        state.component = metric.component; state.focus = metric.id; state.search = ""; $("metric-search").value = "";
        state.featuredOnly = false; $("featured-only").checked = false;
        state.collapsed[state.component + "/" + metric.group] = false;
        $("explorer").hidden = true; render();
        const card = cards.get(metric.id);
        if (card) { activateCard(card); card.node.scrollIntoView({ block: "center" }); card.node.classList.add("focus-panel"); card.node.querySelector("button").focus({ preventScroll: true }); }
      }, "explorer-item");
      row.append(el("small", "", metric.name), el("small", "", `${metric.component} / ${metric.group} · ${compact(metric.series_count)} series`));
      $("explorer-results").append(row);
    }
  }

  function showInspector(title) {
    $("inspector-title").textContent = title; $("inspector-body").replaceChildren();
    if (!$("inspector").open) $("inspector").showModal();
    return $("inspector-body");
  }
  function table(headers, rows) {
    const wrap = el("div", "table-wrap"), t = el("table", "data-table"), head = el("thead"), h = el("tr");
    for (const text of headers) h.append(el("th", "", text)); head.append(h); t.append(head);
    const body = el("tbody"); for (const values of rows) { const row = el("tr"); for (const value of values) row.append(el("td", "", value === null || value === undefined ? "—" : typeof value === "number" ? String(value) : value)); body.append(row); }
    t.append(body); wrap.append(t); return wrap;
  }
  async function inspectMetric(metric) {
    const body = showInspector(metric.title || metric.name);
    body.append(el("code", "inline-code", metric.name), el("p", "", metric.description || "Captured metric values and their complete source labels."));
    if (metric.aliases?.length) body.append(el("p", "", "Related inventory entries: " + metric.aliases.join(", ") + ". Bucket observations are preserved; counts derive from the +Inf bucket. Attached sums are omitted because they can be unreliable."));
    const notes = warnings(metric);
    if (metric.kind === "histogram") notes.push("Percentiles are estimates from interval bucket counts. Histogram sum/count columns are not used. Estimates in the unbounded +Inf bucket remain gaps.");
    if (notes.length) { const list = el("ul", "warning-list"); for (const note of notes) list.append(el("li", "", note)); body.append(list); }
    const source = el("p", "", "Loading series…"); body.append(source);
    try {
      const payload = await loadMetric(metric);
      if ($("inspector-title").textContent !== (metric.title || metric.name)) return;
      const all = payload.series, filtered = all.filter(matches);
      source.textContent = `${compact(all.length)} captured series; ${compact(filtered.length)} match the dashboard filters. Values below are the stored display-bin statistics; null means unavailable.`;
      const search = el("input", "inspector-filter"); search.type = "search"; search.placeholder = "Search complete series labels"; search.setAttribute("aria-label", "Search inspected series");
      const select = el("select"); select.setAttribute("aria-label", "Inspected series");
      const count = el("p"), details = el("div");
      let available = filtered;
      function showSeries() {
        const raw = available[Number(select.value)]; details.replaceChildren(); if (!raw) return;
        details.append(el("h3", "", "Source and labels"), el("pre", "", JSON.stringify({ id: raw.id, endpoint: raw.endpoint, metadata: raw.metadata, labels: raw.labels }, null, 2)));
        const c = columns(payload);
        const points = raw.points.filter(p => { const t = Number(p[c.bin]) * resolution; return t + resolution >= state.from && t <= state.to; });
        details.append(el("h3", "", "Samples in the selected time range"), el("p", "", `Showing the first ${Math.min(200, points.length)} of ${compact(points.length)} bins. Elapsed start is bin × ${resolution} seconds.`));
        details.append(table(["elapsed_start_s", ...payload.columns], points.slice(0, 200).map(p => [Number(p[c.bin]) * resolution, ...p])));
        const download = button("Download this series as JSON", () => {
          const blob = new Blob([JSON.stringify({ metric: metric.name, columns: payload.columns, resolution_s: resolution, ...raw }, null, 2)], { type: "application/json" });
          const url = URL.createObjectURL(blob), anchor = el("a"); anchor.href = url; anchor.download = metric.id + "-series.json"; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
        }); details.append(download);
      }
      function searchSeries() {
        available = filtered.filter(s => seriesLabel(s).toLowerCase().includes(search.value.toLowerCase())); select.replaceChildren();
        for (const [i, raw] of available.slice(0, 300).entries()) select.append(option(String(i), seriesLabel(raw)));
        count.textContent = `Selector shows ${Math.min(300, available.length)} of ${compact(available.length)} matching series. Search any label to reach additional series.`;
        showSeries();
      }
      select.addEventListener("change", showSeries); search.addEventListener("input", searchSeries);
      body.append(search, select, count, details); searchSeries();
    } catch (error) { source.textContent = error.message; source.className = "quality-note"; }
  }
  function captureDetails() {
    const body = showInspector("Capture details");
    body.append(el("p", "", "This artifact contains only metrics read from the listed Tachometer Parquet / Arrow files. Frontend, router, worker, GPU, and host labels are retained from that capture. No benchmark logs, traces, client summaries, or resource snapshots supply chart values."));
    body.append(el("p", hasFinalSource ? "" : "quality-note", hasFinalSource ? "The selected files include final.parquet. Its presence alone does not certify capture completeness." : "No final.parquet was selected. This capture may be incomplete; the charts show only the available raw observations."));
    const summary = { schema_version: catalog.schema_version, start_ns: catalog.start_ns, end_ns: catalog.end_ns, duration_s: catalog.duration_s, resolution_s: catalog.resolution_s, bin_count: catalog.bin_count, row_count: catalog.row_count, metric_count: metrics.length, source_files: catalog.source_files, excluded_source_files: catalog.excluded_source_files || catalog.excluded_files || [] };
    body.append(el("pre", "", JSON.stringify(summary, null, 2)), el("h3", "", "Reading the charts"), el("p", "", "Gauges show mean, minimum, maximum, or last value per display bin. Known counters show delta divided by observed seconds. Affected reset and gap intervals remain missing. Histograms show estimated percentiles from cumulative bucket deltas; no attached histogram mean is assumed."), el("p", "", "Charts initially show up to 12 series ranked by full-capture peak. Choose series and Inspect expose every captured label. Series are never summed or averaged across sources implicitly."));
    if (catalog.source_warnings?.length) body.append(el("h3", "", "Source warnings"), el("pre", "", JSON.stringify(catalog.source_warnings, null, 2)));
    if (catalog.warnings?.length) body.append(el("h3", "", "Capture warnings"), el("pre", "", JSON.stringify(catalog.warnings, null, 2)));
  }

  for (const component of components) {
    const count = metrics.filter(m => m.component === component).length;
    const tab = button(component, () => { state.component = component; delete state.focus; render(); }, "tab");
    tab.dataset.component = component; tab.setAttribute("role", "tab"); tab.append(el("span", "tab-count", count)); $("tabs").append(tab);
    tab.addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault(); const current = components.indexOf(state.component);
      const next = event.key === "Home" ? 0 : event.key === "End" ? components.length - 1 : (current + (event.key === "ArrowRight" ? 1 : -1) + components.length) % components.length;
      state.component = components[next]; render(); $("tabs").children[next].focus();
    });
  }
  $("tabs").setAttribute("role", "tablist");
  $("metric-search").value = state.search; $("featured-only").checked = state.featuredOnly;
  let searchTimer;
  $("metric-search").addEventListener("input", event => { state.search = event.target.value; clearTimeout(searchTimer); searchTimer = setTimeout(render, 150); });
  $("featured-only").addEventListener("change", event => { state.featuredOnly = event.target.checked; render(); });
  $("apply-time").addEventListener("click", () => setTime($("time-from").value, $("time-to").value));
  for (const id of ["time-from", "time-to"]) $(id).addEventListener("keydown", event => { if (event.key === "Enter") setTime($("time-from").value, $("time-to").value); });
  $("reset-time").addEventListener("click", () => setTime(0, duration));
  $("collapse-all").addEventListener("click", () => { for (const row of $("rows").children) row.open = false; });
  $("expand-all").addEventListener("click", () => { for (const row of $("rows").children) row.open = true; });
  $("explore-button").addEventListener("click", showExplorer);
  $("close-explorer").addEventListener("click", () => { $("explorer").hidden = true; $("explore-button").focus(); });
  $("explorer-search").addEventListener("input", renderExplorer);
  $("capture-button").addEventListener("click", captureDetails);
  $("close-inspector").addEventListener("click", () => $("inspector").close());
  document.addEventListener("keydown", event => { if (event.key === "Escape" && !$("explorer").hidden) { $("explorer").hidden = true; $("explore-button").focus(); } });
  if (catalog.title) { document.querySelector("h1").textContent = catalog.title; document.title = catalog.title + " — Tachometer"; }
  $("capture-summary").textContent = `${compact(metrics.length)} metric families · ${compact(metrics.reduce((sum, m) => sum + Number(m.series_count || 0), 0))} series · ${compact(catalog.row_count)} raw rows · ${timeText(duration)} captured`;
  const excludedSources = catalog.excluded_source_files || catalog.excluded_files || [];
  if (excludedSources.length || !hasFinalSource) {
    const notices = [];
    if (!hasFinalSource) notices.push("Capture may be incomplete: no final.parquet");
    if (excludedSources.length) notices.push(`${excludedSources.length} excluded file${excludedSources.length === 1 ? "" : "s"}`);
    const warning = button(notices.join(" · ") + " · details", captureDetails, "capture-warning");
    warning.id = "capture-warning"; document.querySelector(".capture-strip").after(warning);
  }
  setupFilters(); updateTimeInputs(); render();
  // Small read-only surface for offline smoke tests and the browser console.
  window.tachometerDashboard = { ready: true, catalog, state, loadMetric, buildSeries, setTime, setTab(component) { if (components.includes(component)) { state.component = component; render(); } }, get charts() { return [...cards.values()].filter(c => c.plot).length; } };
  window.__tachometerDashboard = window.tachometerDashboard;
})();
