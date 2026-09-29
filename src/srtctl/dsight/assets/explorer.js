/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* Offline UI and public API share the same state/actions. */
"use strict";
(async function () {
  const $ = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  const short = (value) => {
    const s = String(value ?? "unknown");
    if (s.startsWith("phase-") && s.includes("/user-")) {
      return s
        .split("/")
        .map((part) => {
          const prefix = part.startsWith("phase-")
            ? "p"
            : part.startsWith("user-")
              ? "u"
              : "c";
          return prefix + part.slice(part.indexOf("-") + 1).slice(0, 8);
        })
        .join(" · ");
    }
    return s.slice(0, 8);
  };
  const fmt = (n, digits = 2) =>
    Number.isFinite(n)
      ? n.toLocaleString(undefined, { maximumFractionDigits: digits })
      : "—";
  const profileLabel = (p) =>
    `${esc(p.worker)}${p.rank === null ? "" : ` / rank ${p.rank}`}${p.engine == null ? "" : ` / engine ${p.engine}`}${p.gpus ? ` / gpu ${esc(p.gpus)}` : ""}`;
  const ms = (s) =>
    s < 0.001
      ? `${fmt(s * 1e6, 1)} µs`
      : s < 1
        ? `${fmt(s * 1000, 2)} ms`
        : `${fmt(s, 3)} s`;
  const clone = (x) => JSON.parse(JSON.stringify(x));
  const safe = (fn) => {
    try {
      $("error").textContent = "";
      const result = fn();
      return result?.catch ? result.catch(e => { $("error").textContent = e.message; }) : result;
    } catch (e) {
      $("error").textContent = e.message;
    }
  };
  let D;
  try {
    const binary = atob($("tracePayload").textContent.trim());
    const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
    const stream = new Blob([bytes])
      .stream()
      .pipeThrough(new DecompressionStream("gzip"));
    D = JSON.parse(await new Response(stream).text());
    $("tracePayload").remove();
  } catch (e) {
    $("tracks").textContent =
      "Could not open embedded trace data: " + e.message;
    throw e;
  }
  const available = D.capabilities;
  const usableProfiles = D.profiles.filter((p) => (p.event_count ?? p.events.length) || (p.cpu?.sample_count ?? p.cpu?.samples?.length));
  const tabs = [
    ...(available.requests ? ["request"] : []),
    ...(available.nsight ? ["nsys"] : []),
    "api",
  ];
  const requests = new Map(D.requests.map((r) => [r.id, r]));
  const sessionRequests = new Map(
    D.sessions.map((s) => [s.id, s.requests.map((id) => requests.get(id))]),
  );
  const profileById = new Map(D.profiles.map((p) => [p.id, p]));
  const sourceById = new Map(D.sources.map((s) => [s.id, s]));
  const detailData = window.DSightDetailData.create(D);
  const metricData = window.DSightMetricData.create(D, {detailData});
  const nsysViews = new Map();
  let detailsReady = Promise.resolve(), detailController, detailKey = "", metricController;
  const metricFamilies = metricData.listFamilies();
  const metricFamilyByName = new Map(metricFamilies.map(family => [family.name, family]));
  const defaultMetric = metricFamilyByName.has("trtllm_num_requests_running")
    ? "trtllm_num_requests_running"
    : (metricFamilies.find(family => family.samples > 0) || metricFamilies[0])?.name || "";
  const history = [];
  let clientDrag = null,
    suppressClientClick = false;
  let metricMounts = [],
    metricCharts = [];
  let metricPanelHeights = new Map();
  let metricGeneration = 0, metricsReady = Promise.resolve(), metricSearch = "";
  const state = {
    from: 0,
    to: D.meta.duration,
    request: null,
    span: null,
    tab: tabs[0],
    page: 0,
    pageSize: 7,
    search: "",
    sort: "start",
    cursor: null,
    expandedSessions: new Set(),
    expandedAgents: new Set(),
    expandedRequests: new Set(),
    nsys: !available.requests && available.nsight,
    profile:
      usableProfiles.find((p) => p.worker === "frontend")?.id ??
      usableProfiles[0]?.id ??
      null,
    metricCharts: {},
    pinnedMetrics: [],
    hardware: false,
    compareNsys: false,
    metric: defaultMetric,
  };
  const overlap = (a, b, lo = state.from, hi = state.to) => a <= hi && b >= lo;
  const selected = () => requests.get(state.request);
  const inRangeRequests = () =>
    D.requests.filter((r) => overlap(r.start, r.end));
  const requestMatches = (r, q) =>
    !q ||
    [
      r.id,
      r.session,
      r.agent,
      r.conversation,
      r.source_trace,
      ...r.server_ids,
      ...r.workers,
    ].some((s) =>
      String(s ?? "")
        .toLowerCase()
        .includes(q),
    );
  const sessionList = () =>
    D.sessions
      .filter(
        (s) =>
          overlap(s.start, s.end) &&
          sessionRequests
            .get(s.id)
            .some(
              (r) =>
                requestMatches(r, state.search.toLowerCase()) &&
                overlap(r.start, r.end),
            ),
      )
      .sort((a, b) =>
        state.sort === "ttft"
          ? Math.max(...sessionRequests.get(b.id).map((r) => r.ttft_ms || 0)) -
            Math.max(...sessionRequests.get(a.id).map((r) => r.ttft_ms || 0))
          : state.sort === "count"
            ? b.requests.length - a.requests.length
            : a.start - b.start,
      );
  const stateJSON = () => ({
    ...state,
    metricCharts: clone(state.metricCharts),
    pinnedMetrics: [...state.pinnedMetrics],
    expandedSessions: [...state.expandedSessions],
    expandedAgents: [...state.expandedAgents],
    expandedRequests: [...state.expandedRequests],
  });
  function validateRange(from, to) {
    if (!Number.isFinite(from) || !Number.isFinite(to))
      throw Error("Enter finite start and end times in seconds.");
    if (from < 0 || to > D.meta.duration + 1e-6 || from >= to)
      throw Error(
        `Range must satisfy 0 ≤ from < to ≤ ${D.meta.duration.toFixed(6)} seconds.`,
      );
  }
  function restore(value) {
    if (!value || typeof value !== "object")
      throw Error("View state must be an object");
    validateRange(value.from ?? state.from, value.to ?? state.to);
    if (value.request && !requests.has(value.request)) value = {...value, request: null, span: null};
    for (const k of [
      "from",
      "to",
      "request",
      "page",
      "search",
      "sort",
      "nsys",
      "hardware",
      "compareNsys",
      "metric",
      "span",
      "cursor",
      "metricCharts",
    ]) {
      if (value[k] !== undefined)
        state[k] = k === "metricCharts"
          ? clone(value[k]) : value[k];
    }
    if (
      tabs.includes(value.tab)
    )
      state.tab = value.tab;
    if (value.profile !== undefined && profileById.has(value.profile))
      state.profile = value.profile;
    for (const k of [
      "expandedSessions",
      "expandedAgents",
      "expandedRequests",
    ]) {
      if (Array.isArray(value[k])) state[k] = new Set(value[k]);
    }
    state.expandedRequests = new Set(
      [...state.expandedRequests].filter((id) => hasLifecycle(requests.get(id))),
    );
    if (state.span && !findInterval(selected(), state.span)) state.span = null;
    if (!metricFamilyByName.has(state.metric)) state.metric = defaultMetric;
    if (Array.isArray(value.pinnedMetrics))
      state.pinnedMetrics = [...new Set(value.pinnedMetrics.filter(name => metricFamilyByName.has(name)))];
    state.page = Math.max(0, Number.isInteger(state.page) ? state.page : 0);
    state.nsys = Boolean(state.nsys && available.nsight);
    state.hardware = Boolean(state.hardware && available.hardware_metrics);
    if (!usableProfiles.some((p) => p.id === state.profile)) state.profile = usableProfiles[0]?.id ?? null;
  }
  function setRange(from, to, remember = true) {
    validateRange(from, to);
    if (remember && (state.from !== from || state.to !== to))
      history.push([state.from, state.to]);
    state.from = from;
    state.to = to;
    state.page = 0;
    const r = selected();
    if (r && overlap(r.start, r.end)) {
      const i = sessionList().findIndex((x) => x.id === r.session);
      if (i >= 0) state.page = Math.floor(i / state.pageSize);
    }
    render();
    return stateJSON();
  }
  function fitRange(a, b) {
    const pad = Math.max((b - a) * 0.06, 0.00001);
    return setRange(
      Math.max(0, a - pad),
      Math.min(D.meta.duration, Math.max(b + pad, a + 0.00001)),
    );
  }
  function zoom(factor) {
    const width = Math.min(D.meta.duration, (state.to - state.from) * factor),
      mid = (state.from + state.to) / 2;
    const from = Math.max(
      0,
      Math.min(D.meta.duration - width, mid - width / 2),
    );
    return setRange(from, from + width);
  }
  function pan(direction) {
    const w = state.to - state.from,
      a = Math.max(
        0,
        Math.min(D.meta.duration - w, state.from + direction * w * 0.5),
      );
    return setRange(a, a + w);
  }
  function selectRequest(id, { expand = false, fit = false } = {}) {
    const r = requests.get(id);
    if (!r) throw Error(`Unknown client request: ${id}`);
    if (!requestMatches(r, state.search.toLowerCase())) state.search = "";
    state.request = id;
    state.span = null;
    state.expandedSessions.add(r.session);
    state.expandedAgents.add(r.agent);
    if (expand && hasLifecycle(r)) state.expandedRequests.add(id);
    if (fit) {
      const pad = Math.max((r.end - r.start) * 0.06, 0.00001);
      state.from = Math.max(0, r.start - pad);
      state.to = Math.min(D.meta.duration, r.end + pad);
    }
    const idx = sessionList().findIndex((s) => s.id === r.session);
    if (idx >= 0) state.page = Math.floor(idx / state.pageSize);
    render();
    return clone(r);
  }
  function expandRequest(id, expanded = true) {
    selectRequest(id);
    if (expanded && hasLifecycle(requests.get(id))) state.expandedRequests.add(id);
    else state.expandedRequests.delete(id);
    render();
    return stateJSON();
  }
  function selectSpan(id, { fit = false, nsys = false } = {}) {
    const r = selected(),
      span = findInterval(r, id);
    if (!span) throw Error("Unknown span for selected request");
    state.span = id;
    if (nsys && span.kind === "progress") {
      if (!span.source_span_id)
        throw Error("This client boundary has no worker attribution.");
      return selectSpan(span.source_span_id, { nsys: true });
    }
    if (nsys) {
      const worker = span.role === "frontend" ? "frontend" : span.worker;
      if (!worker) throw Error("No recorded worker identity for this span");
      fitRange(span.start, span.end);
      return inspectNsys({
        worker,
      });
    }
    if (fit) fitRange(span.start, span.end);
    else render();
    return clone(span);
  }
  function metricStats(series, from = state.from, to = state.to) {
    let samples = 0, min = Infinity, max = -Infinity, sum = 0;
    let first = null, last = null;
    for (const point of series.points) {
      if (point[0] < from || point[0] > to || !Number.isFinite(point[1])) continue;
      if (!first) first = point;
      last = point; samples++; sum += point[1];
      min = Math.min(min, point[1]); max = Math.max(max, point[1]);
    }
    if (!samples)
      return { samples: 0, min: null, max: null, mean: null, last: null };
    return {
      samples, min, max, mean: sum / samples, last: last[1],
      first_time: first[0], last_time: last[0],
    };
  }
  function queryRequests({
    from = state.from,
    to = state.to,
    session,
    agent,
    worker,
    minTTFT = 0,
    search = "",
    offset = 0,
    limit = 50,
  } = {}) {
    validateRange(from, to);
    limit = Math.max(0, Math.min(1000, Math.floor(limit)));
    offset = Math.max(0, Math.floor(offset));
    const rs = D.requests.filter(
      (r) =>
        overlap(r.start, r.end, from, to) &&
        (!session || r.session === session) &&
        (!agent || r.agent === agent) &&
        (!worker || r.workers.includes(worker)) &&
        (r.ttft_ms ?? 0) >= minTTFT &&
        requestMatches(r, search.toLowerCase()),
    );
    return {
      total: rs.length,
      offset,
      limit,
      items: rs
        .slice(offset, offset + limit)
        .map(({ spans, lifecycle, ...r }) => ({
          ...clone(r),
          span_count: spans.length,
        })),
      range: [from, to],
    };
  }
  function queryNsys({
    profile = state.profile,
    from = state.from,
    to = state.to,
    name = "",
    offset = 0,
    limit = 100,
  } = {}) {
    validateRange(from, to);
    const p = profileById.get(profile);
    if (!p) return {available: false, total: 0, items: [], range: [from, to]};
    const formatEvents = ({items, total, ...page}) => ({
      profile: p.id, worker: p.worker, rank: p.rank, attribution: p.attribution,
      capture: p.capture, total, ...page,
      items: items.map((e) => ({
        start: e[0], end: e[1], name: p.names[e[2]], globalTid: e[3],
        pid: p.threads?.find(t => t.global_tid === e[3])?.pid ?? null,
        tid: p.threads?.find(t => t.global_tid === e[3])?.tid ?? null,
        definition: clone(p.name_definitions?.[e[2]] ?? null), rowid: e[4], evidence_source: p.evidence_source,
      })),
    });
    if (p.event_chunks) return detailData.events(p, {from, to, name,
      offset: Math.max(0, Math.floor(offset)), limit: Math.max(0, Math.min(10000, Math.floor(limit))),
    }).then(formatEvents);
    const es = p.events.filter(
      (e) =>
        overlap(e[0], e[1], from, to) &&
        p.names[e[2]].toLowerCase().includes(name.toLowerCase()),
    );
    const n = Math.max(0, Math.min(10000, Math.floor(limit)));
    offset = Math.max(0, Math.floor(offset));
    return {
      profile: p.id,
      worker: p.worker,
      rank: p.rank,
      attribution: p.attribution,
      range: [from, to],
      capture: p.capture,
      total: es.length,
      offset,
      limit: n,
      items: es.slice(offset, offset + n).map((e) => ({
        start: e[0],
        end: e[1],
        name: p.names[e[2]],
        globalTid: e[3],
        pid: p.threads?.find((t) => t.global_tid === e[3])?.pid ?? null,
        tid: p.threads?.find((t) => t.global_tid === e[3])?.tid ?? null,
        definition: clone(p.name_definitions?.[e[2]] ?? null),
        rowid: e[4],
        evidence_source: p.evidence_source,
      })),
    };
  }
  function inspectNsys({
    worker,
    rank,
    profile,
    from,
    to,
    compare = false,
  } = {}) {
    const p =
      profile !== undefined
        ? profileById.get(profile)
        : usableProfiles.find(
            (x) =>
              x.worker === (worker ?? "frontend") &&
              (rank === undefined || x.rank === rank),
          );
    if (!p) throw Error("No usable Nsight report matches this worker/rank.");
    if (from !== undefined || to !== undefined) {
      validateRange(from ?? state.from, to ?? state.to);
      state.from = from ?? state.from;
      state.to = to ?? state.to;
    }
    state.profile = p.id;
    state.nsys = true;
    state.compareNsys = compare;
    state.tab = "nsys";
    render();
    scrollNsys();
    return queryNsys();
  }
  async function queryMetrics({
    from = state.from,
    to = state.to,
    worker,
    host,
    gpu,
    rank,
    name,
    points = false,
  } = {}) {
    validateRange(from, to);
    const selectedSeries = metricData.listSeries({worker, host, gpu, rank, name});
    const ids = new Set(selectedSeries.map(series => String(series.id)));
    const results = new Map();
    // Decode one family at a time, including for all-metric summary exports.
    for (const family of new Set(selectedSeries.map(series => series.name))) {
      for (const series of await metricData.loadFamily(family, {from, to})) {
        if (!ids.has(String(series.id))) continue;
        const {points: rawPoints, ...metadata} = series;
        results.set(String(series.id), {
          ...clone(metadata), ...metricStats(series, from, to),
          ...(points ? {points: rawPoints.filter(p => p[0] >= from && p[0] <= to).map(p => [...p])} : {}),
          ...(series.temporal === "setting" ? {carried_setting: (() => {
            const prior = rawPoints.filter(p => p[0] < from);
            const stamp = prior.length ? prior[prior.length - 1][0] : null;
            return prior.filter(p => p[0] === stamp).map(p => [...p]);
          })()} : {}),
        });
      }
    }
    return selectedSeries.map(series => results.get(String(series.id)));
  }
  function queryCpu({
    profile = state.profile,
    from = state.from,
    to = state.to,
    limit = 30,
  } = {}) {
    validateRange(from, to);
    const p = profileById.get(profile);
    if (!p) return {available: false, total_samples: 0, hotspots: []};
    const cpu = p.cpu;
    if (!(cpu?.sample_count ?? cpu?.samples?.length))
      return {
        total_samples: 0,
        hotspots: [],
        available: false,
        attribution: "CPU samples were not imported for this process",
      };
    if (cpu.chunks) return detailData.cpu(p, from, to).then(({counts, total}) => ({
      available: true, total_samples: total, pid: cpu.pid, attribution: cpu.attribution,
      range: [from, to], evidence_source: p.evidence_source,
      hotspots: [...counts].sort((a, b) => b[1] - a[1]).slice(0, Math.min(200, Math.max(0, limit)))
        .map(([id, n]) => ({symbol: cpu.names[id], samples: n, fraction: total ? n / total : 0})),
    }));
    const samples = cpu.samples.filter((s) => s[0] >= from && s[0] <= to),
      counts = new Map();
    for (const s of samples)
      for (const name of new Set(cpu.stacks[s[2]]))
        counts.set(name, (counts.get(name) || 0) + 1);
    return {
      available: true,
      total_samples: samples.length,
      pid: cpu.pid,
      attribution: cpu.attribution,
      range: [from, to],
      evidence_source: p.evidence_source,
      hotspots: [...counts]
        .sort((a, b) => b[1] - a[1])
        .slice(0, Math.min(200, Math.max(0, limit)))
        .map(([id, n]) => ({
          symbol: cpu.names[id],
          samples: n,
          fraction: samples.length ? n / samples.length : 0,
        })),
    };
  }
  function cpuInspector(p) {
    if (!(p.cpu?.sample_count ?? p.cpu?.samples?.length)) return "";
    const result = nsysViews.get(p.id)?.cpu;
    if (p.cpu?.chunks && !result) return '<p class="help">Zoom in to load CPU sample hotspots, or use await traceExplorer.queryCpu() for the exact window.</p>';
    const q = result ? {
      total_samples: result.total,
      hotspots: [...result.counts].sort((a, b) => b[1] - a[1]).slice(0, 12)
        .map(([id, n]) => ({symbol: p.cpu.names[id], samples: n, fraction: n / result.total})),
    } : queryCpu({ profile: p.id, limit: 12 });
    return `<h3>Frontend CPU sample hotspots</h3><p class="help">${fmt(q.total_samples, 0)} process samples in this window. Inclusive frames; percentages may overlap and do not measure this request’s CPU time.</p><table class="mini-table"><thead><tr><th>Sampled frame</th><th>Samples</th><th>Share</th></tr></thead><tbody>${q.hotspots.map((h) => `<tr><td title="${esc(h.symbol)}">${esc(h.symbol.length > 130 ? h.symbol.slice(0, 127) + "…" : h.symbol)}</td><td>${fmt(h.samples, 0)}</td><td>${fmt(h.fraction * 100, 1)}%</td></tr>`).join("")}</tbody></table>`;
  }
  function queryIterations({
    from = state.from,
    to = state.to,
    worker,
    rank,
    offset = 0,
    limit = 100,
  } = {}) {
    validateRange(from, to);
    if (
      !Number.isInteger(offset) ||
      offset < 0 ||
      !Number.isInteger(limit) ||
      limit < 0 ||
      limit > 1000
    )
      throw Error("Require offset >= 0 and 0 <= limit <= 1000.");
    const rows = D.iterations.filter(
      (r) =>
        (!worker || r.worker === worker) &&
        (rank === undefined || rank === null || r.rank === rank),
    );
    const aligned = rows.filter(
      (r) => r.start !== null && overlap(r.start, r.end, from, to),
    );
    return {
      total: aligned.length,
      unaligned_rows: rows.filter((r) => r.start === null).length,
      range: [from, to],
      offset,
      limit,
      items: clone(aligned.slice(offset, offset + limit)),
      attribution:
        "Shared batch observations. Source timestamp precision and rank scope are preserved; these are not request durations.",
    };
  }
  function queryServerSpans({from = state.from, to = state.to, offset = 0, limit = 100} = {}) {
    validateRange(from, to);
    offset = Math.max(0, Math.floor(offset));
    limit = Math.max(0, Math.min(1000, Math.floor(limit)));
    const rows = (D.server_spans ?? []).filter((r) => overlap(r.start, r.end, from, to));
    return {total: rows.length, offset, limit, items: clone(rows.slice(offset, offset + limit))};
  }
  async function exportSelection() {
    const r = selected();
    const view = stateJSON();
    const result = {
      schema: D.schema,
      job: D.meta.job,
      origin_ns: D.meta.origin_ns,
      view,
      request: r ? clone(r) : null,
      visible_request_count: queryRequests({ limit: 0 }).total,
      server_spans: queryServerSpans(),
      batch_observations: queryIterations({limit: 1000}),
      profile: state.nsys ? await queryNsys({ profile: view.profile, from: view.from, to: view.to, limit: 200 }) : null,
      limitations: D.meta.limitations,
      audit: D.audit,
      sources: D.sources,
    };
    result.metrics = await queryMetrics({from: view.from, to: view.to});
    return result;
  }
  window.traceExplorer = Object.freeze({
    version: "3.1",
    ready: true,
    describe: () => ({
      schema: D.schema,
      meta: clone(D.meta),
      audit: clone(D.audit),
      available: clone(available),
      capabilities: [
        "selectRange",
        "selectRequest",
        "expandRequest",
        "getLifecycle",
        "inspectNsys",
        "queryRequests",
        "queryMetrics",
        "listMetricFamilies",
        "listMetricSeries",
        "whenMetricsReady",
        "queryNsys",
        "exportSelection",
      ],
    }),
    getState: () => stateJSON(),
    setState: (v) => {
      restore(v);
      render();
      return stateJSON();
    },
    selectRange: setRange,
    selectRequest,
    expandRequest,
    selectSpan,
    inspectNsys,
    queryRequests,
    queryMetrics,
    listMetricFamilies: () => metricData.listFamilies(),
    listMetricSeries: filters => metricData.listSeries(filters),
    metricDataStatus: () => metricData.status(),
    detailDataStatus: () => detailData.status(),
    whenDetailsReady: async () => {
      let current;
      do { current = detailsReady; await current; } while (current !== detailsReady);
    },
    whenMetricsReady: async () => {
      let current;
      do { current = metricsReady; await current; } while (current !== metricsReady);
    },
    queryNsys,
    queryCpu,
    queryIterations,
    queryServerSpans,
    listSessions: ({ offset = 0, limit = 100 } = {}) => ({
      total: sessionList().length,
      items: clone(sessionList().slice(offset, offset + Math.min(limit, 1000))),
    }),
    getLifecycle: (id) => {
      const r = requests.get(id);
      if (!r) throw Error("Unknown request");
      return clone(lifecycleModel(r));
    },
    getRequest: (id) => {
      if (!requests.has(id)) throw Error("Unknown request");
      return clone(requests.get(id));
    },
    listProfiles: () =>
      D.profiles.map(({ events, names, cpu, ...p }) => ({
        ...clone(p),
        selected_event_count: p.event_count ?? events.length,
        names: names.length,
        cpu_samples: cpu?.sample_count ?? cpu?.samples.length ?? 0,
      })),
    getSource: (id) => clone(sourceById.get(id) ?? null),
    exportSelection,
  });
  function bar(a, b, label, classes = "", data = "", tooltip = "") {
    if (!overlap(a, b)) return "";
    const lo = Math.max(a, state.from),
      hi = Math.min(b, state.to),
      w = state.to - state.from,
      l = ((lo - state.from) / w) * 100,
      width = ((hi - lo) / w) * 100;
    return `<button class="bar ${classes}" style="left:${l}%;width:max(1px,${width}%)" ${data} aria-label="${esc(tooltip || label)}" data-tooltip="${esc(tooltip || label)}"><span>${width > 2.5 ? esc(label) : ""}</span></button>`;
  }
  function requestBar(r, classes = "") {
    if (!overlap(r.start, r.end)) return "";
    const lo = Math.max(r.start, state.from),
      hi = Math.min(r.end, state.to),
      validFirst = Number.isFinite(r.first) && r.first >= r.start && r.first <= r.end,
      cut = validFirst ? Math.min(100, Math.max(0, ((r.first - lo) / (hi - lo)) * 100)) : null,
      background = validFirst
        ? `linear-gradient(90deg,var(--amber) 0%,var(--amber) ${cut}%,var(--teal) ${cut}%,var(--teal) 100%)`
        : "repeating-linear-gradient(135deg,#b7c2cd 0px,#b7c2cd 5px,#dae1e7 5px,#dae1e7 10px)";
    const text = `Turn ${r.turn} · ${short(r.id)}\nClient TTFT ${fmt(r.ttft_ms)} ms · request ${ms(r.end - r.start)}\n${r.input_tokens ?? "?"} input / ${r.output_tokens ?? "?"} output tokens\nClick to select.${hasLifecycle(r) ? " Expand stages for the full lifecycle." : ""}`;
    return bar(
      r.start,
      r.end,
      `T${r.turn}`,
      `${classes} ${r.id === state.request ? "selected" : ""}`,
      `data-request="${esc(r.id)}"`,
      text,
    ).replace(
      'style="',
      `style="background:${background};`,
    );
  }
  function track(
    label,
    content,
    { classes = "", height = 31, data = "" } = {},
  ) {
    return `<div class="track ${classes}" ${data} style="min-height:${height}px"><div class="track-label">${label}</div><div class="lane" style="min-height:${height - 1}px">${content}</div></div>`;
  }
  const toggle = (kind, id, open, description) =>
    `<button class="toggle" data-toggle="${kind}" data-id="${esc(id)}" aria-expanded="${open}" aria-label="${esc(description)}">${open ? "▾" : "▸"}</button>`;
  const labelText = (s, cls = "") =>
    `<span class="text ${cls}" title="${esc(s)}">${esc(s)}</span>`;
  function hasLifecycle(r) {
    return Boolean(r?.lifecycle?.available ?? r?.lifecycle?.activities?.length);
  }
  function lifecycleModel(r) {
    return r.lifecycle;
  }
  function stageDescription(s) {
    return s.description ?? "Recorded Dynamo OTel interval.";
  }
  function findInterval(r, id) {
    return (
      r?.lifecycle.stages.find((s) => s.id === id) ??
      r?.spans.find((s) => s.id === id)
    );
  }
  function milestoneRows(r) {
    if (!hasLifecycle(r)) return "";
    const model = lifecycleModel(r);
    const html = model.stages
      .map((current, i) => {
        const label = `<button class="stage-row-label ${state.span === current.id && state.request === r.id ? "active" : ""}" data-span="${esc(current.id)}" data-owner-request="${esc(r.id)}" title="${esc(current.label)} · ${ms(current.end - current.start)}"><span class="stage-number">${i + 1}</span><span class="stage-name">${esc(current.label)}</span><small>${ms(current.end - current.start)}</small></button>`;
        const blocks = model.stages
          .slice(0, i + 1)
          .map((s) => {
            const added = s.id === current.id;
            return bar(
              s.start,
              s.end,
              `${s.label} · ${ms(s.end - s.start)}`,
              `phase stage-block ${added ? "stage-current" : "stage-history"} ${s.name.startsWith("kv_router") ? "router" : s.role} ${state.span === s.id && state.request === r.id ? "selected" : ""}`,
              `data-span="${esc(s.id)}" data-stage-span="${esc(s.id)}" data-current-stage="${added}" data-owner-request="${esc(r.id)}"`,
              `${s.label}${added ? "" : " · repeated from an earlier row"}\n${s.name} · ${s.id}\n${s.host} · ${ms(s.end - s.start)}\n${fmt(s.start, 9)}–${fmt(s.end, 9)} s\n${stageDescription(s)}`,
            );
          })
          .join("");
        return track(label, blocks, {
          classes:
            "lifecycle-chain " + (state.request === r.id ? "selected" : ""),
          data: `data-stage-row="${i}" data-owner-request="${esc(r.id)}"`,
        });
      })
      .join("");
    return html + `<div class="row-note">${esc(model.timing)} ${model.issues.map(esc).join(" · ")}</div>`;
  }
  function lifecycleRows(r) {
    if (!hasLifecycle(r)) return "";
    return '<div class="row-note"><strong>Request breakdown · Progress milestones</strong></div>' + milestoneRows(r);
  }
  function clientTracks() {
    if (!available.requests) return "";
    const list = sessionList(),
      pages = Math.max(1, Math.ceil(list.length / state.pageSize));
    state.page = Math.min(state.page, pages - 1);
    let html = `<section id="clientTracks" aria-label="Client sessions &amp; agents"><div class="section-head"><span>Client sessions &amp; agents</span><small>${list.length} sessions · drag timeline to select a range</small></div>`;
    for (const session of list.slice(
      state.page * state.pageSize,
      (state.page + 1) * state.pageSize,
    )) {
      const rs = sessionRequests
          .get(session.id)
          .filter(
            (r) =>
              overlap(r.start, r.end) &&
              requestMatches(r, state.search.toLowerCase()),
          ),
        open = state.expandedSessions.has(session.id);
      html += track(
        toggle("session", session.id, open, "Expand session " + session.id) +
          labelText(`Session ${short(session.id)}`),
        bar(
          session.start,
          session.end,
          `${rs.length} requests`,
          "session",
          `data-toggle="session" data-id="${esc(session.id)}"`,
          `${session.id}\n${rs.length} requests in selected range; grouped by root_correlation_id.`,
        ),
      );
      if (!open) continue;
      const agents = [...new Set(rs.map((r) => r.agent))].sort(
        (a, b) =>
          Math.min(...rs.filter((r) => r.agent === a).map((r) => r.depth)) -
          Math.min(...rs.filter((r) => r.agent === b).map((r) => r.depth)),
      );
      for (const aid of agents) {
        const ar = rs.filter((r) => r.agent === aid),
          aopen = state.expandedAgents.has(aid),
          main = ar[0].depth === 0;
        html += track(
          '<span class="indent"></span>' +
            toggle("agent", aid, aopen, "Expand requests for agent " + aid) +
            labelText(
              `${ar[0].client_kind === "agentperf" ? "Client" : main ? "Main" : "Subagent"} ${short(aid)}`,
            ),
          ar.map((r) => requestBar(r, "agent")).join(""),
        );
        if (!aopen) continue;
        const chosen = ar.includes(selected())
          ? [selected(), ...ar.filter((r) => r !== selected()).slice(0, 7)]
          : ar.slice(0, 8);
        chosen.sort((a, b) => a.start - b.start);
        for (const r of chosen) {
          const ropen = hasLifecycle(r) && state.expandedRequests.has(r.id);
          html += track(
            '<span class="indent2"></span>' +
              (hasLifecycle(r)
                ? toggle(
                    "request",
                    r.id,
                    ropen,
                    "Expand lifecycle stages for request " + r.id,
                  )
                : "") +
              labelText(`T${r.turn}  ${short(r.id)}`),
            requestBar(r),
            { classes: r.id === state.request ? "selected" : "" },
          );
          if (ropen) html += lifecycleRows(r);
        }
        if (ar.length > 8)
          html += `<div class="row-note">Showing ${chosen.length} of ${ar.length} requests for this agent. Zoom or search to narrow the list; all remain queryable through the API.</div>`;
      }
    }
    if (!list.length)
      html +=
        '<div class="row-note">No sessions match this range and search. Clear the search or widen the range.</div>';
    return (
      html +
      `<div class="pager"><span>Sessions ${list.length ? state.page * state.pageSize + 1 : 0}–${Math.min(list.length, (state.page + 1) * state.pageSize)} of ${list.length}</span><div class="buttons"><button data-page="-1" ${state.page === 0 ? "disabled" : ""}>Previous</button><button data-page="1" ${state.page + 1 >= pages ? "disabled" : ""}>Next</button></div></div></section>`
    );
  }
  function metricPanel(key, series, title) {
    const id = `metricChart${metricMounts.length}`;
    metricMounts.push({ id, key, series, title });
    const height = metricPanelHeights.get(key);
    return `<div id="${id}" class="dsight-metric-panel" data-metric-key="${esc(key)}" ${height ? `style="min-height:${height}px"` : ""} aria-busy="true"><div class="ds-metric-loading" role="status">Loading recorded metric samples…</div></div>`;
  }
  function mountMetricCharts() {
    const generation = metricGeneration;
    const from = state.from, to = state.to;
    const mounts = metricMounts.map(mount => ({...mount, host: $(mount.id)}));
    metricsReady = (async () => {
      let firstError;
      for (const {host, key, series, title} of mounts) {
        try {
          const wanted = new Set(series.map(item => String(item.id)));
          const loaded = [];
          for (const name of new Set(series.map(item => item.name))) {
            loaded.push(...(await metricData.loadFamily(name, {from, to, signal: metricController.signal})).filter(item => wanted.has(String(item.id))));
            if (generation !== metricGeneration) return;
          }
          const references = [];
          const referenceIds = new Set(loaded.filter(item => item.reference?.series_id != null).map(item => String(item.reference.series_id)));
          for (const name of new Set(loaded.filter(item => item.reference?.series_id != null).map(item => item.reference.name))) {
            references.push(...(await metricData.loadFamily(name, {from, to, signal: metricController.signal})).filter(item => referenceIds.has(String(item.id))));
            if (generation !== metricGeneration) return;
          }
          if (generation !== metricGeneration || !host.isConnected) return;
          host.replaceChildren();
          metricCharts.push(window.DSightMetricCharts.mount(host, {
            series: loaded, references, title, height: 220, from, to,
            selection: state.metricCharts[key],
            onSelectionChange: (selection) => {
              state.metricCharts[key] = selection;
              window.dispatchEvent(new CustomEvent("trace-explorer:state", {detail: stateJSON()}));
            },
            onRangeChange: (start, end) => safe(() => setRange(
              Math.max(0, start), Math.min(D.meta.duration, end),
            )),
          }));
          host.style.removeProperty("min-height");
          host.setAttribute("aria-busy", "false");
        } catch (error) {
          if (generation !== metricGeneration) return;
          host.replaceChildren();
          const message = document.createElement("div");
          message.className = "ds-metric-empty";
          message.textContent = `Could not load this metric: ${error.message}`;
          host.append(message); host.setAttribute("aria-busy", "false");
          host.style.removeProperty("min-height");
          $("error").textContent = error.message;
          firstError ??= error;
        }
      }
      if (firstError) throw firstError;
      if (generation === metricGeneration)
        window.dispatchEvent(new CustomEvent("trace-explorer:metrics-ready", {detail: {metric: state.metric}}));
    })();
    // Keep the readiness promise rejected for API callers while reporting UI errors above.
    metricsReady.catch(() => {});
  }
  function metricOptions() {
    const query = metricSearch.trim().toLowerCase();
    const matches = metricFamilies.filter(family =>
      [family.name, family.title, family.component, family.group, family.description]
        .some(value => String(value || "").toLowerCase().includes(query)));
    const componentOrder = ["Frontend", "Router", "Workers", "GPU", "Host"];
    for (const family of metricFamilies)
      if (!componentOrder.includes(family.component)) componentOrder.push(family.component);
    matches.sort((a, b) => componentOrder.indexOf(a.component) - componentOrder.indexOf(b.component)
      || (a.group_order ?? 100) - (b.group_order ?? 100)
      || String(a.group).localeCompare(String(b.group))
      || (a.order ?? 100) - (b.order ?? 100) || a.name.localeCompare(b.name));
    const groups = new Map();
    for (const family of matches) {
      const label = [family.component, family.group].filter(Boolean).join(" / ");
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push(family);
    }
    if (!matches.some(family => family.name === state.metric) && metricFamilyByName.has(state.metric))
      groups.set(matches.length ? "Current selection" : "No matches · current selection", [metricFamilyByName.get(state.metric)]);
    const option = family => `<option value="${esc(family.name)}" ${family.name === state.metric ? "selected" : ""}>${esc(family.title && family.title !== family.name ? family.title + " — " : "")}${esc(family.name)}${family.samples === 0 ? " (no samples)" : ""}</option>`;
    return {
      html: [...groups].map(([label, families]) => `<optgroup label="${esc(label)}">${families.map(option).join("")}</optgroup>`).join(""),
      count: `${matches.length} / ${metricFamilies.length} metrics`,
    };
  }
  function metricDescription(family) {
    if (!family) return "No metrics were recorded.";
    const kind = family.value_kind === "histogram"
      ? `Recorded bucket observation counts${family.observation_unit ? `; bounds in ${family.observation_unit}` : ""}.`
      : family.counter || family.value_kind === "counter" ? "Raw cumulative counter values." : "Recorded sample values.";
    return `<code>${esc(family.name)}</code><span>${esc([family.description, kind, family.samples === 0 ? "No samples in the captured trace interval." : ""].filter(Boolean).join(" "))}</span>${family.quality ? `<details class="metric-quality"><summary>Metric notes</summary>${esc(family.quality)}</details>` : ""}`;
  }
  function workerTracks() {
    if (!metricFamilies.length) return "";
    const options = metricOptions();
    const pinned = state.pinnedMetrics.includes(state.metric);
    let html = `<section id="metricsSection" aria-label="Metrics"><div class="section-head metric-section-head"><span>Metrics</span><div class="metric-picker"><input id="metricSearch" type="search" placeholder="Search metrics" aria-label="Search metrics" value="${esc(metricSearch)}"><select id="workerMetric" aria-label="Metric">${options.html}</select><button id="pinMetric" aria-pressed="${pinned}" aria-label="${pinned ? "Unpin" : "Pin"} selected metric" ${state.metric ? "" : "disabled"}>${pinned ? "Unpin" : "Pin"}</button><small id="metricSearchCount" aria-live="polite">${options.count}</small></div></div>`;
    const names = pinned ? state.pinnedMetrics : [...state.pinnedMetrics, state.metric];
    for (const name of names) {
      const family = metricFamilyByName.get(name);
      const pinIndex = state.pinnedMetrics.indexOf(name);
      const isPinned = pinIndex >= 0;
      const choices = D.metrics.filter(series => series.name === name);
      const title = family?.title || choices[0]?.label || name;
      const pinControls = isPinned ? `<div class="metric-pin-controls"><span class="metric-pin-label">Pinned</span>
        <button data-action="move-metric-up" data-metric="${esc(name)}" aria-label="Move ${esc(name)} up" title="Move up" ${pinIndex === 0 ? "disabled" : ""}>↑</button>
        <button data-action="move-metric-down" data-metric="${esc(name)}" aria-label="Move ${esc(name)} down" title="Move down" ${pinIndex === state.pinnedMetrics.length - 1 ? "disabled" : ""}>↓</button>
        <button data-action="unpin-metric" data-metric="${esc(name)}" aria-label="Unpin ${esc(name)}">Unpin</button></div>` : "";
      html += `<section class="metric-card${isPinned ? " metric-card-pinned" : ""}" data-metric-name="${esc(name)}" aria-label="${esc(title || "Metric")}" tabindex="-1"><div class="metric-card-head"><div class="metric-description">${metricDescription(family)}</div>${pinControls}</div>`;
      html += metricPanel(JSON.stringify(["metric", name]), choices, title) + "</section>";
    }
    html += "</section>";
    // Catalog reports expose hardware through the same categorized selector.
    if (D.metric_catalog) return html;
    html += `<div class="section-head"><span>Hardware</span><button id="hardwareToggle" aria-expanded="${state.hardware}">${state.hardware ? "Hide" : "Show"} GPU / host metrics</button></div>`;
    if (state.hardware) {
      const gpuSeries = D.metrics.filter((s) =>
          ["gpu_util", "DCGM_FI_DEV_GPU_UTIL"].includes(s.name),
        ),
        hosts = [...new Set(gpuSeries.map((s) => s.host))];
      for (const host of hosts) {
        const choices = gpuSeries.filter((s) => s.host === host);
        html += metricPanel(JSON.stringify(["hardware", host, "gpu_util"]), choices, `${host} · GPU utilization`);
      }
      const memory = D.metrics.filter((s) => s.name === "memory_MemAvailable_bytes");
      for (const host of new Set(memory.map((s) => s.host)))
        html += metricPanel(JSON.stringify(["hardware", host, "memory_MemAvailable_bytes"]), memory.filter((s) => s.host === host), `${host} · Available RAM`);
      if (!gpuSeries.length)
        html +=
          '<div class="row-note">No GPU utilization series was recorded.</div>';
    } else
      html +=
        '<div class="row-note">GPU and host metrics follow this time range. Individual GPU samples do not identify request ownership.</div>';
    return html;
  }
  function chosenProfiles() {
    const p = profileById.get(state.profile);
    if (!p) return [];
    let chosen = [p];
    if (state.compareNsys && selected()) {
      const ids = new Set(["frontend", ...selected().workers]);
      chosen = usableProfiles
        .filter(
          (x) =>
            ids.has(x.worker) && (p.rank === null || x.rank === null || x.rank === p.rank),
        )
        .sort(
          (a, b) =>
            (a.worker === "frontend"
              ? 0
              : a.worker.startsWith("prefill")
                ? 1
                : 2) -
            (b.worker === "frontend"
              ? 0
              : b.worker.startsWith("prefill")
                ? 1
                : 2),
        );
    }
    return chosen;
  }
  function nsysTracks() {
    if (!state.nsys) return "";
    return chosenProfiles().map(nsysTracksFor).join("");
  }
  function densityTrack(p) {
    const bins = p.event_density, width = D.meta.duration / bins.length;
    const peak = Math.max(1, ...bins);
    const rects = bins.map((n, i) => {
      if (!n || (i + 1) * width < state.from || i * width > state.to) return "";
      const x = Math.max(0, (i * width - state.from) / (state.to - state.from) * 1000);
      const right = Math.min(1000, ((i + 1) * width - state.from) / (state.to - state.from) * 1000);
      return `<rect x="${x}" y="${32 - 28 * n / peak}" width="${Math.max(0, right - x)}" height="${28 * n / peak}" fill="#8067b4"/>`;
    }).join("");
    return `<div class="section-head" data-nsys-heading="${p.id}"><span>Nsight · ${profileLabel(p)}</span><small>Density overview</small></div>`
      + `<div class="row-note">${fmt(p.event_count, 0)} imported NVTX ranges in this report. Counts per ${ms(width)} bin; overlapping ranges are not CPU utilization. Zoom in for exact ranges. The API returns exact paginated rows.</div>`
      + track(labelText("NVTX interval density"), `<svg class="metric-svg" viewBox="0 0 1000 35" preserveAspectRatio="none">${rects}</svg>`, {classes: "nsys-track", height: 35});
  }
  function refreshDetails() {
    if (!D.delivery) return;
    const chosen = state.nsys ? chosenProfiles() : state.tab === "nsys" ? [profileById.get(state.profile)].filter(Boolean) : [];
    const from = state.from, to = state.to;
    const key = JSON.stringify([from, to, chosen.map(p => p.id)]);
    if (key === detailKey) return;
    detailKey = key;
    detailController?.abort();
    detailController = new AbortController();
    const signal = detailController.signal;
    nsysViews.clear();
    for (const p of D.profiles) if (p.event_chunks) p.events = [];
    for (const p of chosen) nsysViews.set(p.id, {from, to, loading: true});
    detailsReady = (async () => {
      for (const p of chosen) {
        const sampleCount = (p.cpu?.chunks || []).filter(c => c.bounds[0] <= to && c.bounds[3] >= from).reduce((n, c) => n + c.count, 0);
        if (detailData.estimate(p, from, to) > 20000 || sampleCount > 20000) {
          nsysViews.set(p.id, {from, to, summary: true});
          continue;
        }
        const result = await detailData.events(p, {from, to, limit: 20000, signal});
        const cpu = p.cpu?.chunks ? await detailData.cpu(p, from, to, signal) : null;
        signal.throwIfAborted();
        p.events = result.items;
        nsysViews.set(p.id, {from, to, total: result.total, cpu});
      }
      signal.throwIfAborted();
      if ($("nsysTracks")) $("nsysTracks").innerHTML = nsysTracks();
      renderInspector();
      window.dispatchEvent(new Event("trace-explorer:details-ready"));
    })().catch(error => {
      if (error.name === "AbortError") return;
      if (!signal.aborted) {
        for (const p of chosen) nsysViews.set(p.id, {from, to, error: error.message});
        if ($("nsysTracks")) $("nsysTracks").innerHTML = nsysTracks();
        renderInspector();
        $("error").textContent = error.message;
      }
      throw error;
    });
    detailsReady.catch(() => {});
  }
  function nsysTracksFor(p) {
    if (p.event_chunks) {
      const view = nsysViews.get(p.id);
      if (!view || view.from !== state.from || view.to !== state.to || view.loading)
        return `<div class="section-head" data-nsys-heading="${p.id}"><span>Nsight · ${profileLabel(p)}</span></div><div class="row-note">Loading selected ranges…</div>`;
      if (view.error) return `<div class="row-note">Could not load Nsight ranges: ${esc(view.error)}</div>`;
      if (view.summary) return densityTrack(p);
    }
    const es = p.events.filter((e) => overlap(e[0], e[1]));
    let html = `<div class="section-head" data-nsys-heading="${p.id}"><span>Nsight · ${profileLabel(p)}</span><small>${fmt(es.length, 0)} selected NVTX ranges</small></div><div class="row-note">Shared CPU/NVTX activity. Shows up to 8 threads and 5 overlap lanes each; all imported events remain queryable. ${p.cuda ? "The source contains a CUDA kernel table; this view displays host NVTX and CPU samples." : ""}</div>`;
    const r = selected();
    if (r) {
      html += track(labelText("Selected request"), requestBar(r));
      for (const s of r.spans.filter(
        (s) =>
          s.name === "request.preprocessing" ||
          s.name.startsWith("worker.operation"),
      ))
        html += track(
          labelText(s.name),
          bar(
            s.start,
            s.end,
            ms(s.end - s.start),
            `phase ${s.role}`,
            `data-span="${s.id}"`,
          ),
        );
    }
    const perThread = new Map();
    for (const e of es) {
      if (!perThread.has(e[3])) perThread.set(e[3], []);
      perThread.get(e[3]).push(e);
    }
    for (const [tid, events] of [...perThread]
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, 8)) {
      const levels = [];
      for (const e of events) {
        let level = levels.findIndex((xs) => xs.at(-1)[1] <= e[0]);
        if (level < 0) {
          level = levels.length;
          levels.push([]);
        }
        levels[level].push(e);
      }
      for (const [i, level] of levels.slice(0, 5).entries()) {
        const label =
          i === 0
            ? `PID ${p.threads?.find((t) => t.global_tid === tid)?.pid ?? "?"} / TID ${p.threads?.find((t) => t.global_tid === tid)?.tid ?? tid}`
            : `overlap lane ${i + 1}`;
        let content;
        if (level.length > 1800) {
          const bins = Array(500).fill(0),
            width = state.to - state.from;
          for (const e of level) {
            const start = Math.max(
                0,
                Math.floor(((e[0] - state.from) / width) * 500),
              ),
              end = Math.min(
                499,
                Math.floor(((e[1] - state.from) / width) * 500),
              );
            for (let j = start; j <= end; j++) bins[j]++;
          }
          const max = Math.max(1, ...bins);
          content = `<svg class="metric-svg" viewBox="0 0 500 35" preserveAspectRatio="none">${bins.map((n, j) => (n ? `<rect x="${j}" y="${32 - (28 * n) / max}" width="1" height="${(28 * n) / max}" fill="#8067b4"/>` : "")).join("")}</svg>`;
        } else
          content = level
            .map((e) =>
              bar(
                e[0],
                e[1],
                p.names[e[2]],
                "phase",
                `data-nvtx="${e[4]}" data-profile="${p.id}"`,
                `${p.names[e[2]]}\n${ms(e[1] - e[0])}\n${profileLabel(p)}\nShared activity, row ${e[4]}`,
              ),
            )
            .join("");
        html += track(labelText(label), content, {
          classes: "nsys-track",
          height: 35,
        });
      }
    }
    if (!es.length)
      html +=
        '<div class="row-note">No selected NVTX ranges overlap this window. Check capture coverage.</div>';
    if (es.length > 1800)
      html +=
        '<div class="row-note">Dense tracks show event density. Zoom in for named ranges; the API returns exact intervals.</div>';
    return html;
  }
  function evidence(ref, label = "Source") {
    if (!ref) return "";
    const s = sourceById.get(ref[0]);
    return `<div class="evidence-item"><span class="tag">${esc(label)}</span><br><code>${esc(s?.path ?? "Unknown source")}${ref[1] !== undefined ? ":" + ref[1] : ""}</code>${ref[2] !== undefined ? `<br>span index ${ref[2]}` : ""}</div>`;
  }
  function pathNode(id, title, host, active) {
    const hasProfile = usableProfiles.some((p) => p.worker === id);
    return `<button class="path-node ${active ? "active" : ""}" data-path-worker="${esc(id)}" ${hasProfile ? "" : "disabled"}><strong>${esc(title)}</strong><small>${esc(host || "host not mapped")}</small><small>${hasProfile ? "Inspect Nsight" : "No Nsight report"}</small></button>`;
  }

  function spanHasProfile(r,span) {
    const source = span?.kind === "progress" ? r.spans.find((s) => s.id === span.source_span_id) : span;
    const worker = source?.role === "frontend" ? "frontend" : source?.worker;
    return worker && usableProfiles.some((p) => p.worker === worker);
  }
  function requestInspector() {
    const r = selected();
    if (!r)
      return "<p>Select a request to follow its frontend, prefill, and decode path.</p>";
    const pathWorkers = D.workers.filter((w) => r.workers.includes(w.id)),
      sp = findInterval(r, state.span),
      front = r.spans.find((s) => s.role === "frontend");
    let html = `<div class="help">Session ${esc(short(r.session))} / ${r.client_kind === "agentperf" ? "client" : r.depth ? "subagent" : "main agent"} / turn ${esc(r.turn)}</div><div class="request-id mono">${esc(r.id)}</div><div class="stats"><div class="stat">${fmt(r.ttft_ms)}<small>Client TTFT · ms</small></div><div class="stat">${fmt(r.end - r.start, 3)}<small>Request duration · s</small></div><div class="stat">${fmt(r.input_tokens, 0)}<small>Input tokens</small></div><div class="stat">${fmt(r.output_tokens, 0)}<small>Output tokens</small></div></div><div class="actions"><button id="fitSession" title="Fit all recorded requests in this session, including subagents">Fit Session</button><button id="fitRequest">Fit request</button><button id="fitTTFT" ${r.first === null ? "disabled" : ""}>Fit TTFT</button>${hasLifecycle(r) ? `<button id="expandTTFT" aria-expanded="${state.expandedRequests.has(r.id)}">${state.expandedRequests.has(r.id) ? "Collapse" : "Expand"} lifecycle</button>` : ""}</div>`;
    if (sp) {
      const stage = [
          ...lifecycleModel(r).stages,
          ...lifecycleModel(r).activities,
        ].find((s) => s.id === sp.id),
        context = sp.routing_context,
        route = context
          ? r.spans.find((s) => s.id === context.route_span_id)
          : null;
      html += `<div class="phase-focus"><strong>${esc(stage?.label ?? sp.name)}</strong><br><span class="mono">${esc(sp.name)} · ${esc(sp.id)}</span><br>${ms(sp.end - sp.start)} ${sp.kind === "progress" ? "between milestones" : "recorded elapsed"} · ${esc(sp.role)} · ${esc(sp.host)}${stage && !context ? `<p class="help">${esc(stageDescription(stage))}</p>` : ""}${route ? `<br>Route: ${esc(route.routing.phase)} · DP rank ${esc(route.routing.dp_rank)} · attempt ${esc(route.routing["request.attempt"])}<p class="help">${esc(context.basis)}.</p>` : ""}<div class="actions"><button id="fitSpan">Fit phase</button>${spanHasProfile(r,sp) ? '<button id="inspectSpan">Inspect phase in Nsight</button>' : ""}</div></div>`;
    }
    if (sp?.kind === "progress")
      html +=
        evidence(
          sp.from_boundary.evidence,
          "Interval start: " + sp.from_boundary.label,
        ) +
        evidence(
          sp.to_boundary.evidence,
          "Interval end: " + sp.to_boundary.label,
        );
    if (sp && sp.kind !== "progress") {
      if (sp.worker_basis) html += `<p class="help">Worker association: ${esc(sp.worker_basis)}.</p>`;
      html += evidence(sp.evidence, "Dynamo OTel span");
    }
    if (hasLifecycle(r) && state.expandedRequests.has(r.id)) {
      html += `<h3>Progress milestones</h3><div class="stage-list">${lifecycleModel(
        r,
      )
        .stages.map(
          (s) =>
            `<button data-span="${esc(s.id)}" data-owner-request="${esc(r.id)}" class="stage-choice ${s.id === state.span ? "active" : ""}" title="${esc(s.name)} · ${esc(s.id)}"><span>${esc(s.label)}</span><small>${ms(s.end - s.start)}</small></button>`,
        )
        .join(
          "",
        )}</div><p class="help">Each delta begins at the preceding milestone.</p>`;
      if (r.lifecycle.issues.length)
        html += `<p class="notice">${r.lifecycle.issues.map(esc).join("<br>")}</p>`;
    }
    if (front || pathWorkers.length) {
      html += "<h3>Recorded request path</h3>";
      if (front) {
        html += pathNode("frontend", "Frontend + router", front.host, true);
        const prefill = D.workers.filter((w) => w.role === "prefill");
        if (prefill.length)
          html += `<div class="path-arrow">↓</div><div class="prefill-options">${prefill
            .map(
              (w) =>
                `<button data-path-worker="${esc(w.id)}" class="${r.workers.includes(w.id) ? "active" : ""}" title="Inspect Nsight for ${esc(w.id)}" ${usableProfiles.some((p) => p.worker === w.id) ? "" : "disabled"}>${esc(w.id)}</button>`,
            )
            .join("")}</div>`;
      }
      let hasPathNode = Boolean(front);
      for (const role of ["prefill", "decode", "agg"]) {
        const recorded = pathWorkers.filter((e) => e.role === role);
        if (recorded.length) {
          if (hasPathNode && role !== "prefill") html += '<div class="path-arrow">↓</div>';
          html += recorded
            .map((w) => pathNode(w.id, w.id, w.host, true))
            .join("");
          hasPathNode = true;
        }
      }
      if (pathWorkers.length > 2)
        html +=
          '<p class="help">All recorded workers are shown, including repeated routing attempts.</p>';
      if (pathWorkers.length)
        html += '<p class="help" style="margin-top:9px">Click a worker to inspect its Nsight report. Use the metric chart legend to show or hide worker lines.</p>';
    }
    if (r.server_ids.length || r.engine.length) {
      html +=
        '<div class="detail-heading">Identity bridge</div><dl class="facts"><dt>Client</dt><dd class="mono">' +
        esc(r.id) +
        '</dd><dt>Dynamo</dt><dd class="mono">' +
        r.server_ids.map(esc).join("<br>") +
        "</dd>";
      for (const b of r.worker_bindings ?? [])
        html += `<dt>${esc(b.worker)} binding</dt><dd>${b.ambiguous ? '<strong>Ambiguous</strong><br>' : ''}<span class="mono">${esc(b.process ?? "process not recorded")}</span><br>${esc(b.basis)}${r.server_ids.length > 1 ? `<br>Dynamo <span class="mono">${esc(b.server_id)}</span>` : ''}</dd>`;
      for (const e of r.engine)
        html += `<dt>${esc(e.worker)} engine</dt><dd>engine client ${esc(e.client_id)}<br><span class="mono">disagg ${esc(e.disagg_id)}</span>${e.identity_ambiguous ? '<br>Process-local ID scope unresolved' : ''}</dd>`;
      html += '</dl>';
      if (r.engine.length)
        html += '<p class="help">Engine client IDs are process-local. Router DP rank is not assumed to match a Nsight process rank; rank selection shows shared activity.</p>';
    }
    if ((r.worker_bindings ?? []).some((e) => e.ambiguous))
      html += '<p class="warn">Conflicting worker bindings are retained as evidence and omitted from the recorded request path.</p>';
    if (r.issues.length)
      html += `<p class="warn">${r.issues.map(esc).join("; ")}</p>`;
    return html;
  }
  function nsysInspector() {
    const p = profileById.get(state.profile);
    if (!p) return "<p>No Nsight SQLite exports were imported.</p>";
    const view = p.event_chunks ? nsysViews.get(p.id) : null;
    const pending = p.event_chunks && (!view || view.from !== state.from || view.to !== state.to || view.loading);
    const detailNote = pending ? "Loading selected ranges…" : view?.error ? `Could not load ranges: ${esc(view.error)}`
      : view?.summary ? "Density overview. Zoom in for exact range counts and elapsed totals, or use await traceExplorer.queryNsys()." : "";
    const es = p.events.filter((e) => overlap(e[0], e[1])),
      sums = new Map();
    for (const e of es) {
      const name = p.names[e[2]],
        q = sums.get(name) || { name, count: 0, time: 0 };
      q.count++;
      q.time += Math.max(
        0,
        Math.min(e[1], state.to) - Math.max(e[0], state.from),
      );
      sums.set(name, q);
    }
    const groups = [...sums.values()]
      .sort((a, b) => b.time - a.time)
      .slice(0, 18);
    return `<p>Inspect activity beside the selected request. Worker and time joins identify context; they do not assign shared work to one request.</p><div class="nsys-controls"><label>Report<select id="profileSelect">${usableProfiles.map((x) => `<option value="${x.id}" ${p.id === x.id ? "selected" : ""}>${profileLabel(x)}</option>`).join("")}</select></label><button id="showNsys">${state.nsys ? "Hide" : "Show"} Nsight tracks</button>${selected()?.workers.length ? `<button id="compareNsys" aria-pressed="${state.compareNsys}">${state.compareNsys ? "Show one report" : "Compare frontend + request workers"}</button>` : ""}</div><dl class="facts"><dt>Coverage</dt><dd>${fmt(p.capture[0], 3)} to ${fmt(p.capture[1], 3)} s</dd>${p.truncated ? `<dt>Partial import</dt><dd>Events included through ${fmt(p.imported_range?.[1],3)} s; later source events are omitted.</dd>` : ""}<dt>Matching ranges</dt><dd>${detailNote ? "Not yet queried" : fmt(es.length, 0)}</dd><dt>Recorded host</dt><dd>${esc(p.host || "not in exported metadata")}</dd>${p.cuda ? "<dt>CUDA source</dt><dd>Kernel table retained in original export</dd>" : ""}<dt>Excluded ranges</dt><dd>${p.invalid_or_boundary_ranges} malformed / boundary</dd></dl><p class="help">Selected NVTX categories: frontend preprocessing/routing and engine iteration/scheduling/forward preparation. Detokenize ranges below 100 µs remain in the original report, along with other excluded categories.</p>${detailNote ? `<p class="help">${detailNote}</p>` : ""}${cpuInspector(p)}<h3>Ranges in selected window</h3><p class="help">Inclusive, clipped elapsed time; nested and parallel ranges overlap. Totals are not CPU utilization or additive TTFT.</p><table class="mini-table"><thead><tr><th>Range</th><th>Count</th><th>Elapsed</th></tr></thead><tbody>${groups.map((g) => `<tr><td>${esc(g.name)}</td><td>${fmt(g.count, 0)}</td><td>${ms(g.time)}</td></tr>`).join("")}</tbody></table>${evidence([p.evidence_source], "Nsight SQLite")}<div class="detail-heading">Collection notes</div><ul class="quality-list">${(p.diagnostics || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
  }
  function apiInspector() {
    const r = selected() ?? D.requests[0], p = usableProfiles[0];
    const id = JSON.stringify(r?.id);
    const code = [
      "const x = window.traceExplorer;",
      `x.selectRange(0, ${D.meta.duration});`,
      ...(r ? ["x.queryRequests({limit: 10});", `x.selectRequest(${id});`, `x.getRequest(${id});`] : []),
      ...(hasLifecycle(r) ? [`x.getLifecycle(${id});`, `x.expandRequest(${id});`] : []),
      ...(p ? [`await x.inspectNsys({profile: ${p.id}});`, "await x.whenDetailsReady();", "await x.queryNsys({limit: 20});"] : []),
      ...(available.metrics ? ["x.listMetricFamilies();", "x.listMetricSeries({name: x.getState().metric});", "await x.queryMetrics({name: x.getState().metric});"] : []),
      ...(available.iterations ? ["x.queryIterations({limit: 10});"] : []),
      ...(available.server_activity ? ["x.queryServerSpans({offset: 0, limit: 10});"] : []),
      "await x.exportSelection();",
    ].join("\n");
    return `<p>The API controls the same inputs and expansions you see here, and returns structured evidence.</p><div class="code">${esc(code)}</div><div class="actions"><button id="downloadState">Export evidence JSON</button><button id="copyState">Copy view state</button></div><h3>Current state</h3><div class="code">${esc(JSON.stringify({ range_seconds: [state.from, state.to], request: state.request, ttft_expanded: state.expandedRequests.has(state.request), nsys_profile: state.nsys ? state.profile : null }, null, 2))}</div><p class="help">Times are seconds from origin_ns. Queries support offset / limit and report total counts. View links preserve range, selection, and expansion.</p>`;
  }
  function renderInspector() {
    const scroll = $("inspectorBody").scrollTop;
    $("inspectorBody").innerHTML =
      state.tab === "request"
        ? requestInspector()
        : state.tab === "nsys"
          ? nsysInspector()
          : apiInspector();
    $("inspectorBody").scrollTop = scroll;
    document.querySelectorAll("[data-tab]").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === state.tab);
      b.setAttribute("aria-pressed", String(b.dataset.tab === state.tab));
    });
    $("selectionTag").textContent = selected()
      ? selected().status
      : "No selection";
    refreshDetails();
  }
  function overview() {
    const bins = Array(300).fill(0);
    const observations = available.requests ? D.requests : available.server_activity ? D.server_spans :
      D.metrics.length ? D.metrics.flatMap((m) => (m.points ?? []).map((p) => ({start:p[0]}))) :
      D.profiles.map((p) => ({start: Math.max(0,p.capture[0])}));
    for (const r of observations)
      bins[Math.min(299, Math.floor((r.start / D.meta.duration) * 300))]++;
    const max = Math.max(...bins, 1),
      start = (state.from / D.meta.duration) * 1200,
      end = (state.to / D.meta.duration) * 1200;
    $("overview").setAttribute("viewBox", "0 0 1200 57");
    $("overview").setAttribute("preserveAspectRatio", "none");
    $("overview").innerHTML =
      bins
        .map(
          (n, i) =>
            `<rect x="${i * 4}" y="${48 - (n / max) * 38}" width="3.2" height="${(n / max) * 38}" fill="#91acc8"/>`,
        )
        .join("") +
      `<rect x="${start}" y="1" width="${Math.max(1, end - start)}" height="54" fill="#2f72aa12" stroke="#397bac" stroke-width="2"/><path d="M${start},0v56M${end},0v56" stroke="#397bac" stroke-width="3"/>`;
    $("overviewLabel").textContent =
      `${available.requests ? fmt(D.requests.length, 0)+" client requests" : "Recorded source coverage"} · ${fmt(D.meta.duration, 2)} s`;
  }
  function alignRuler() {
    document.querySelector(".axis-row").style.paddingRight =
      $("tracks").offsetWidth - $("tracks").clientWidth + "px";
  }
  window.addEventListener("resize", alignRuler);
  function scrollNsys() {
    const e = document.querySelector("[data-nsys-heading]");
    if (e)
      $("tracks").scrollTop +=
        e.getBoundingClientRect().top - $("tracks").getBoundingClientRect().top;
  }
  function render() {
    clearClientDrag();
    metricGeneration++;
    metricController?.abort();
    metricController = new AbortController();
    // Reserve existing chart heights during async reloads so lower pins stay in view.
    const scroll = $("tracks").scrollTop;
    metricPanelHeights = new Map([...document.querySelectorAll(".dsight-metric-panel")]
      .map(host => [host.dataset.metricKey, host.getBoundingClientRect().height]));
    for (const chart of metricCharts) chart.destroy();
    metricCharts = [];
    metricMounts = [];
    $("rangeFrom").value = Number(state.from.toFixed(6));
    $("rangeTo").value = Number(state.to.toFixed(6));
    $("rangeFrom").max = D.meta.duration;
    $("rangeTo").max = D.meta.duration;
    $("rangeFrom").min = 0;
    $("rangeTo").min = 0;
    $("rangeSummary").textContent =
      `${ms(state.to - state.from)} selected · all tracks aligned`;
    $("rangeBack").disabled = !history.length;
    $("search").value = state.search;
    $("sessionSort").value = state.sort;
    $("axis").innerHTML = Array.from(
      { length: 6 },
      (_, i) =>
        `<span class="tick" style="left:${i * 20}%">${fmt(state.from + ((state.to - state.from) * i) / 5, state.to - state.from < 1 ? 6 : 3)} s</span>`,
    ).join("");
    $("tracks").innerHTML = clientTracks() + workerTracks() + (state.nsys ? `<div id="nsysTracks">${nsysTracks()}</div>` : "") ||
      '<div class="loading">No timeline tracks are available. Imported data and source coverage are available in Agent API.</div>';
    mountMetricCharts();
    $("tracks").scrollTop = scroll;
    if ($("workerMetric")) $("workerMetric").value = state.metric;
    $("visibleCount").textContent =
      `${fmt(inRangeRequests().length, 0)} requests in range`;
    $("visibleCount").hidden = !available.requests;
    renderInspector();
    overview();
    alignRuler();
    $("statusText").textContent =
      `Origin ${D.meta.start_utc} · ${D.sessions.length} sessions · ${D.workers.length} workers · ${D.profiles.length} Nsight reports`;
    window.dispatchEvent(
      new CustomEvent("trace-explorer:state", { detail: stateJSON() }),
    );
  }
  function download(value, name) {
    const a = document.createElement("a"),
      url = URL.createObjectURL(
        new Blob([JSON.stringify(value, null, 2)], {
          type: "application/json",
        }),
      );
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function copy(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch (_) {
      const area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      const ok = document.execCommand("copy");
      area.remove();
      if (!ok) throw Error("Clipboard unavailable; use Export selection.");
    }
  }
  document.addEventListener("click", (event) =>
    safe(() => {
      const b = event.target.closest("button");
      if (!b) return;
      if (["move-metric-up", "move-metric-down"].includes(b.dataset.action)) {
        const index = state.pinnedMetrics.indexOf(b.dataset.metric);
        const direction = b.dataset.action === "move-metric-up" ? -1 : 1;
        const next = index + direction;
        if (index < 0 || next < 0 || next >= state.pinnedMetrics.length) return;
        const cards = [...document.querySelectorAll(".metric-card-pinned")];
        const panel = cards[index], neighbor = cards[next];
        const pins = [...state.pinnedMetrics];
        [pins[index], pins[next]] = [pins[next], pins[index]];
        state.pinnedMetrics = pins;
        // Keep mounted charts and pending loads attached to their existing hosts.
        if (direction < 0) neighbor.before(panel);
        else neighbor.after(panel);
        document.querySelectorAll(".metric-card-pinned").forEach((card, i) => {
          card.querySelector('[data-action="move-metric-up"]').disabled = i === 0;
          card.querySelector('[data-action="move-metric-down"]').disabled = i === pins.length - 1;
        });
        const focus = b.disabled ? panel.querySelector('[data-action^="move-metric-"]:not(:disabled)') : b;
        focus.focus({preventScroll: true});
        focus.scrollIntoView({block: "nearest"});
        renderInspector();
        window.dispatchEvent(new CustomEvent("trace-explorer:state", {detail: stateJSON()}));
        return;
      }
      if (b.id === "pinMetric" || b.dataset.action === "unpin-metric") {
        const name = b.id === "pinMetric" ? state.metric : b.dataset.metric;
        if (!metricFamilyByName.has(name)) return;
        const cardIndex = [...document.querySelectorAll(".metric-card")].indexOf(b.closest(".metric-card"));
        state.pinnedMetrics = state.pinnedMetrics.includes(name)
          ? state.pinnedMetrics.filter(metric => metric !== name)
          : [...state.pinnedMetrics, name];
        render();
        if (cardIndex < 0) $("pinMetric").focus({preventScroll: true});
        else {
          const cards = document.querySelectorAll(".metric-card");
          const nearby = cards[Math.min(cardIndex, cards.length - 1)];
          (nearby?.querySelector('[data-action="unpin-metric"]') || nearby || $("pinMetric")).focus();
        }
        return;
      }
      if (b.dataset.request) {
        selectRequest(b.dataset.request);
        return;
      }
      if (b.dataset.span) {
        if (b.dataset.ownerRequest && state.request !== b.dataset.ownerRequest)
          selectRequest(b.dataset.ownerRequest);
        state.tab = "request";
        selectSpan(b.dataset.span);
        return;
      }
      if (b.dataset.toggle) {
        const kind = b.dataset.toggle,
          id = b.dataset.id,
          key = {
            session: "expandedSessions",
            agent: "expandedAgents",
            request: "expandedRequests",
          }[kind];
        if (kind === "request" && !state.expandedRequests.has(id)) {
          expandRequest(id);
          return;
        }
        state[key].has(id) ? state[key].delete(id) : state[key].add(id);
        render();
        return;
      }
      if (b.dataset.page) {
        state.page += Number(b.dataset.page);
        render();
        $("tracks").scrollTop = 0;
        return;
      }
      if (b.dataset.tab) {
        state.tab = b.dataset.tab;
        renderInspector();
        $("inspectorBody").scrollTop = 0;
        return;
      }
      if (b.dataset.pathWorker) {
        inspectNsys({ worker: b.dataset.pathWorker });
        return;
      }
      if (b.dataset.nvtx) {
        const p = profileById.get(Number(b.dataset.profile)),
          e = p.events.find((e) => String(e[4]) === b.dataset.nvtx);
        if (e) fitRange(e[0], e[1]);
        return;
      }
      const r = selected();
      const actions = {
        applyRange: () =>
          setRange(Number($("rangeFrom").value), Number($("rangeTo").value)),
        fullRun: () => setRange(0, D.meta.duration),
        zoomIn: () => zoom(0.5),
        zoomOut: () => zoom(2),
        panLeft: () => pan(-1),
        panRight: () => pan(1),
        rangeBack: () => {
          const range = history.pop();
          if (range) setRange(...range, false);
        },
        fitSpan: () => selectSpan(state.span, { fit: true }),
        inspectSpan: () => selectSpan(state.span, { nsys: true }),
        collapseAll: () => {
          state.expandedSessions.clear();
          state.expandedAgents.clear();
          state.expandedRequests.clear();
          render();
        },
        fitSession: () => {
          const session = D.sessions.find((s) => s.id === r?.session);
          if (session) fitRange(session.start, session.end);
        },
        fitRequest: () => r && fitRange(r.start, r.end),
        fitTTFT: () => {
          if (r && r.first !== null) {
            if (hasLifecycle(r)) state.expandedRequests.add(r.id);
            fitRange(r.start, r.first);
          }
        },
        expandTTFT: () =>
          r && expandRequest(r.id, !state.expandedRequests.has(r.id)),
        compareNsys: () => {
          state.compareNsys = !state.compareNsys;
          state.nsys = true;
          render();
          scrollNsys();
        },
        hardwareToggle: () => {
          state.hardware = !state.hardware;
          render();
        },
        showNsys: () => {
          state.nsys = !state.nsys;
          render();
          if (state.nsys) scrollNsys();
        },
        saveSelection: async () =>
          download(await exportSelection(), `trace-${D.meta.job}-selection.json`),
        downloadState: async () =>
          download(await exportSelection(), `trace-${D.meta.job}-selection.json`),
        copyState: () =>
          copy(JSON.stringify(stateJSON(), null, 2)).catch(
            (e) => ($("error").textContent = e.message),
          ),
        shareView: () => {
          const url = new URL(location.href);
          url.hash = "view=" + encodeURIComponent(JSON.stringify(stateJSON()));
          location.hash = url.hash;
          copy(url.href).catch((e) => ($("error").textContent = e.message));
        },
      };
      return actions[b.id]?.();
    }),
  );
  document.addEventListener("change", (event) =>
    safe(() => {
      const t = event.target;
      if (t.id === "profileSelect") {
        state.profile = Number(t.value);
        state.nsys = true;
        render();
        scrollNsys();
      }
      if (t.id === "workerMetric") {
        state.metric = t.value;
        render();
        $("workerMetric").focus({preventScroll: true});
      }
      if (t.id === "sessionSort") {
        state.sort = t.value;
        state.page = 0;
        render();
      }
    }),
  );
  let searchTimer;
  document.addEventListener("input", event => {
    if (event.target.id !== "metricSearch") return;
    metricSearch = event.target.value;
    const options = metricOptions();
    $("workerMetric").innerHTML = options.html;
    $("metricSearchCount").textContent = options.count;
  });
  $("search").addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.search = $("search").value;
      state.page = 0;
      render();
    }, 180);
  });
  for (const id of ["rangeFrom", "rangeTo"]) {
    $(id).addEventListener("keydown", (e) => {
      if (e.key === "Enter")
        safe(() =>
          setRange(Number($("rangeFrom").value), Number($("rangeTo").value)),
        );
    });
  }
  // Commit the same shared range action after a horizontal client-lane drag.
  const clientDragTime = (g, x) =>
    g.from + Math.max(0, Math.min(1, (x - g.left) / g.width)) * (g.to - g.from);
  function clearClientDrag() {
    const g = clientDrag;
    clientDrag = null;
    if (!g) return null;
    g.preview?.remove();
    $("tracks").classList.remove("selecting-client-range");
    if (g.active) suppressClientClick = true;
    if ($("tracks").hasPointerCapture(g.pointerId))
      $("tracks").releasePointerCapture(g.pointerId);
    return g;
  }
  function previewClientRange(x) {
    const g = clientDrag;
    if (!g?.active) return;
    const panel = g.panel.getBoundingClientRect(),
      viewport = $("tracks").getBoundingClientRect();
    const first = g.panel.querySelector(".track").getBoundingClientRect(),
      pager = g.panel.querySelector(".pager").getBoundingClientRect();
    const a = clientDragTime(g, g.startX),
      b = clientDragTime(g, x),
      lo = Math.min(a, b),
      hi = Math.max(a, b);
    const left = g.left + ((lo - g.from) / (g.to - g.from)) * g.width,
      top = Math.max(first.top, viewport.top);
    Object.assign(g.preview.style, {
      left: left - panel.left + "px",
      top: top - panel.top + "px",
      width: ((hi - lo) / (g.to - g.from)) * g.width + "px",
      height: Math.max(0, Math.min(pager.top, viewport.bottom) - top) + "px",
    });
    const label = g.preview.firstElementChild;
    label.textContent = `${fmt(lo, 6)}–${fmt(hi, 6)} s · ${ms(hi - lo)}`;
    const alignRight = left > g.left + g.width / 2;
    label.style.left = alignRight ? "auto" : "0";
    label.style.right = alignRight ? "0" : "auto";
  }
  document.addEventListener(
    "pointerdown",
    () => {
      clearClientDrag();
      suppressClientClick = false;
    },
    true,
  );
  document.addEventListener(
    "click",
    (event) => {
      if (suppressClientClick && event.detail) {
        suppressClientClick = false;
        event.preventDefault();
        event.stopImmediatePropagation();
      }
    },
    true,
  );
  $("tracks").addEventListener("pointerdown", (event) => {
    const lane = event.target.closest("#clientTracks .lane");
    if (!lane || event.button !== 0 || !event.isPrimary) return;
    const rect = lane.getBoundingClientRect();
    if (!rect.width) return;
    clientDrag = {
      pointerId: event.pointerId,
      startX: event.clientX,
      left: rect.left,
      width: rect.width,
      from: state.from,
      to: state.to,
      panel: $("clientTracks"),
      active: false,
      preview: null,
    };
  });
  document.addEventListener("pointermove", (event) => {
    const g = clientDrag;
    if (!g || event.pointerId !== g.pointerId) return;
    if (!g.active && Math.abs(event.clientX - g.startX) >= 5) {
      g.active = true;
      g.preview = document.createElement("div");
      g.preview.className = "client-range-preview";
      g.preview.innerHTML = "<span></span>";
      g.preview.setAttribute("aria-hidden", "true");
      g.panel.append(g.preview);
      $("tracks").setPointerCapture(event.pointerId);
      $("tracks").classList.add("selecting-client-range");
      $("tooltip").style.display = "none";
    }
    if (g.active) {
      event.preventDefault();
      previewClientRange(event.clientX);
    }
  });
  document.addEventListener("pointerup", (event) =>
    safe(() => {
      if (!clientDrag || event.pointerId !== clientDrag.pointerId) return;
      const g = clearClientDrag();
      if (!g.active) return;
      const a = clientDragTime(g, g.startX),
        b = clientDragTime(g, event.clientX);
      if ((Math.abs(b - a) / (g.to - g.from)) * g.width >= 5)
        setRange(Math.min(a, b), Math.max(a, b));
    }),
  );
  document.addEventListener("pointercancel", (event) => {
    if (clientDrag?.pointerId === event.pointerId) clearClientDrag();
  });
  $("tracks").addEventListener("lostpointercapture", (event) => {
    if (clientDrag?.pointerId === event.pointerId) clearClientDrag();
  });
  $("tracks").addEventListener("scroll", clearClientDrag, { passive: true });
  $("tracks").addEventListener("dragstart", (event) => {
    if (event.target.closest("#clientTracks .lane")) event.preventDefault();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && clientDrag) {
      clearClientDrag();
      event.preventDefault();
    }
  });
  window.addEventListener("blur", clearClientDrag);
  window.addEventListener("resize", clearClientDrag);
  let brush = null;
  const overviewTime = (e) =>
    Math.max(
      0,
      Math.min(
        D.meta.duration,
        ((e.clientX - $("overview").getBoundingClientRect().left) /
          $("overview").getBoundingClientRect().width) *
          D.meta.duration,
      ),
    );
  $("overview").addEventListener("pointerdown", (e) => {
    brush = overviewTime(e);
    $("overview").setPointerCapture(e.pointerId);
  });
  $("overview").addEventListener("pointerup", (e) =>
    safe(() => {
      if (brush === null) return;
      const end = overviewTime(e),
        start = brush;
      brush = null;
      if (Math.abs(end - start) > D.meta.duration / 1200)
        setRange(Math.min(start, end), Math.max(start, end));
    }),
  );
  $("overview").addEventListener("pointercancel", () => {
    brush = null;
  });
  document.addEventListener("pointermove", (event) => {
    if (clientDrag?.active) return;
    const lane = event.target.closest(".lane");
    if (lane) {
      const rect = lane.getBoundingClientRect(),
        f = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
      state.cursor = state.from + f * (state.to - state.from);
      document.documentElement.style.setProperty("--cursor", `${f * 100}%`);
      $("cursorLabel").textContent = `Cursor ${fmt(state.cursor, 6)} s`;
    }
    const el = event.target.closest("[data-tooltip]"),
      tip = $("tooltip");
    if (!el) {
      tip.style.display = "none";
      return;
    }
    tip.textContent = el.dataset.tooltip;
    tip.style.display = "block";
    tip.style.left =
      Math.max(8, Math.min(innerWidth - 400, event.clientX + 13)) + "px";
    tip.style.top =
      Math.max(8, Math.min(innerHeight - 150, event.clientY + 15)) + "px";
  });
  document.querySelectorAll(".tabs [data-tab]").forEach((button) => {
    button.hidden = !tabs.includes(button.dataset.tab);
  });
  document.querySelector(".timeline-controls").hidden = !available.requests;
  document.querySelector(".legend").hidden = !available.requests && !available.nsight;
  document.querySelector(".legend").innerHTML = [
    ...(available.requests ? ['<span><i class="dot" style="background:var(--amber)"></i>Client TTFT</span>',
      '<span><i class="dot" style="background:var(--teal)"></i>Output reception</span>'] : []),
    ...(D.requests.some((r) => !Number.isFinite(r.first) || r.first < r.start || r.first > r.end)
      ? ['<span><i class="dot" style="background:#b7c2cd"></i>First-token timing unavailable</span>'] : []),
    ...(available.request_breakdown ? ['<span><i class="dot" style="background:var(--violet)"></i>OTel activity</span>'] : []),
    ...(available.nsight ? ['<span><i class="dot" style="background:var(--violet)"></i>Host NVTX</span>'] : []),
    '<span class="muted">Shared time axis</span>',
  ].join("");
  if (!available.requests) {
    document.querySelector(".inspector-title h2").textContent = "Captured evidence";
    $("selectionTag").hidden = true;
  }
  const workerRoles = [...new Set(D.workers.map((w) => w.role))]
    .map((role) => `${D.workers.filter((w) => w.role === role).length} ${role}`).join(" / ");
  $("runSubtitle").textContent =
    `Run ${D.meta.job}${workerRoles ? " · " + workerRoles + " workers" : ""} · ${available.requests ? D.meta.phase+" client phase" : "recorded source window"}`;
  $("joinBadge").textContent =
    `${fmt(D.audit.clients_with_server_identity, 0)} / ${fmt(D.requests.length, 0)} client → server`;
  $("joinBadge").style.display = D.audit.clients_with_server_identity ? "" : "none";
  $("joinBadge").classList.add(
    D.audit.clients_with_server_identity === D.requests.length ? "ok" : "warn",
  );
  const imported = [
    [D.requests.length, "clients"], [D.audit.joined_spans, "joined OTel spans"],
    [D.server_spans?.length, "unjoined server spans"], [D.metrics.length, "metric series"],
    [usableProfiles.length, "Nsight reports"], [D.iterations.length, "batch observations"],
  ].filter(([count]) => count).map(([count, label]) => `${fmt(count,0)} ${label}`);
  $("coverageNotice").classList.toggle("has-warning", D.meta.warnings.length > 0);
  $("coverageNotice").innerHTML =
    `<strong>Imported:</strong> ${imported.join(" · ") || "No supported observations"}. ${D.meta.warnings.map(esc).join(" ")}`;
  if (!available.requests) $("cursorLabel").textContent = "Time since source window start";
  if (D.meta.qualification?.passed)
    $("runSubtitle").textContent += " · capture qualified";
  const candidates = [...D.requests]
    .filter((r) => r.spans.length && r.workers.length && r.first !== null)
    .sort((a, b) => b.ttft_ms - a.ttft_ms);
  const preferred =
    candidates[Math.min(20, candidates.length - 1)] || D.requests[0];
  if (preferred) {
    state.request = preferred.id;
    state.expandedSessions.add(preferred.session);
    state.expandedAgents.add(preferred.agent);
  }
  if (location.hash.startsWith("#view=")) {
    try {
      restore(JSON.parse(decodeURIComponent(location.hash.slice(6))));
    } catch (e) {
      $("error").textContent = "Saved view could not be restored: " + e.message;
    }
  } else {
    state.sort = "ttft";
    const idx = sessionList().findIndex((s) => s.id === preferred?.session);
    state.page = Math.floor(Math.max(0, idx) / state.pageSize);
  }
  render();
  window.dispatchEvent(new Event("trace-explorer:ready"));
})().catch((e) => {
  document.getElementById("error").textContent = e.stack || String(e);
  window.traceExplorerError = String(e);
});
