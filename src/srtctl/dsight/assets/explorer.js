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
      return fn();
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
  const requests = new Map(D.requests.map((r) => [r.id, r]));
  const sessionRequests = new Map(
    D.sessions.map((s) => [s.id, s.requests.map((id) => requests.get(id))]),
  );
  const profileById = new Map(D.profiles.map((p) => [p.id, p]));
  const sourceById = new Map(D.sources.map((s) => [s.id, s]));
  const history = [];
  let clientDrag = null,
    suppressClientClick = false;
  const state = {
    from: 0,
    to: D.meta.duration,
    request: null,
    span: null,
    tab: "request",
    page: 0,
    pageSize: 7,
    search: "",
    sort: "start",
    cursor: null,
    expandedSessions: new Set(),
    expandedAgents: new Set(),
    expandedRequests: new Set(),
    nsys: false,
    profile:
      D.profiles.find((p) => p.worker === "frontend")?.id ??
      D.profiles[0]?.id ??
      null,
    metricSeries: {},
    iterationWorker: null,
    iterationRank: 0,
    expandedWorkers: new Set(),
    hardware: false,
    compareNsys: false,
    metric: "trtllm_num_requests_running",
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
    expandedSessions: [...state.expandedSessions],
    expandedAgents: [...state.expandedAgents],
    expandedRequests: [...state.expandedRequests],
    expandedWorkers: [...state.expandedWorkers],
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
    if (value.request && !requests.has(value.request))
      throw Error("Unknown request in saved view");
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
      "metricSeries",
      "iterationWorker",
      "iterationRank",
    ]) {
      if (value[k] !== undefined) state[k] = value[k];
    }
    if (
      ["request", "nsys", "iterations", "evidence", "api"].includes(value.tab)
    )
      state.tab = value.tab;
    if (value.profile !== undefined && profileById.has(value.profile))
      state.profile = value.profile;
    for (const k of [
      "expandedSessions",
      "expandedAgents",
      "expandedRequests",
      "expandedWorkers",
    ]) {
      if (Array.isArray(value[k])) state[k] = new Set(value[k]);
    }
    state.expandedRequests = new Set(
      [...state.expandedRequests].filter((id) => hasLifecycle(requests.get(id))),
    );
    if (state.span && !findInterval(selected(), state.span)) state.span = null;
    state.page = Math.max(0, Number.isInteger(state.page) ? state.page : 0);
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
    for (const worker of r.workers) state.expandedWorkers.add(worker);
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
        rank: worker === "frontend" ? undefined : 0,
      });
    }
    if (fit) fitRange(span.start, span.end);
    else render();
    return clone(span);
  }
  function metricStats(series, from = state.from, to = state.to) {
    const points = series.points.filter((p) => p[0] >= from && p[0] <= to);
    if (!points.length)
      return { samples: 0, min: null, max: null, mean: null, last: null };
    const values = points.map((p) => p[1]);
    return {
      samples: values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      mean: values.reduce((a, b) => a + b, 0) / values.length,
      last: values.at(-1),
      first_time: points[0][0],
      last_time: points.at(-1)[0],
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
    if (!p) throw Error("Unknown Nsight profile");
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
        : D.profiles.find(
            (x) =>
              x.worker === (worker ?? "frontend") &&
              (rank === undefined || x.rank === rank),
          );
    if (!p) throw Error("No captured Nsight report matches this worker/rank.");
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
  function queryMetrics({
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
    return D.metrics
      .filter(
        (s) =>
          (!worker || s.worker === worker) &&
          (!host || s.host === host) &&
          (gpu === undefined || s.gpu === String(gpu)) &&
          (rank === undefined || String(s.rank) === String(rank)) &&
          (!name || s.name === name),
      )
      .map((s) => ({
        id: s.id,
        name: s.name,
        raw_name: s.raw_name,
        labels: clone(s.labels),
        endpoint: s.endpoint,
        worker: s.worker,
        host: s.host,
        gpu: s.gpu,
        rank: s.rank,
        worker_process: s.worker_process,
        host_source: s.host_source,
        host_evidence: s.host_evidence,
        unit: s.unit,
        ...metricStats(s, from, to),
        ...(points
          ? { points: s.points.filter((p) => p[0] >= from && p[0] <= to) }
          : {}),
      }));
  }
  function queryCpu({
    profile = state.profile,
    from = state.from,
    to = state.to,
    limit = 30,
  } = {}) {
    validateRange(from, to);
    const p = profileById.get(profile);
    if (!p) throw Error("Unknown profile");
    const cpu = p.cpu;
    if (!cpu)
      return {
        total_samples: 0,
        hotspots: [],
        available: false,
        attribution: "CPU samples were not imported for this process",
      };
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
    if (!p.cpu) return "";
    const q = queryCpu({ profile: p.id, limit: 12 });
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
        (rank === undefined || r.global_rank === rank),
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
        "Shared batch context; whole-second log timestamps. No request ownership or universal NVTX counter shift.",
    };
  }
  function iterationInspector() {
    const worker =
      state.iterationWorker ??
      selected()?.engine.find((e) => e.role === "decode")?.worker ??
      D.workers[0]?.id;
    const rank = state.iterationRank,
      q = queryIterations({ worker, rank, limit: 50 });
    return `<h3>TRT-LLM iteration context</h3><p class="help">${esc(q.attribution)} Host step time covers a completed host loop; previous-device time is delayed. Neither is a request stage duration.</p><div class="nsys-controls"><label>Worker<select id="iterationWorker">${D.workers.map((w) => `<option value="${esc(w.id)}" ${w.id === worker ? "selected" : ""}>${esc(w.id)}</option>`).join("")}</select></label><label>Global rank<input id="iterationRank" type="number" min="0" value="${rank}" aria-label="Iteration global rank"></label></div><p class="help">${fmt(q.total)} rows overlap this window; first 50 shown. ${q.unaligned_rows ? `${q.unaligned_rows} rows have no timezone; rebuild with --iteration-timezone.` : "Timestamps are aligned to the configured log timezone with one-second resolution."}</p><table class="mini-table"><thead><tr><th>Iteration</th><th>Batch</th><th>Host ms</th><th>Prev. GPU ms</th></tr></thead><tbody>${q.items.map((r) => `<tr><td>${r.iteration}</td><td>${r.batch_requests}</td><td>${fmt(r.host_step_ms)}</td><td>${fmt(r.previous_device_step_ms)}</td></tr>`).join("")}</tbody></table>`;
  }
  function exportSelection() {
    const r = selected();
    return {
      schema: D.schema,
      job: D.meta.job,
      origin_ns: D.meta.origin_ns,
      view: stateJSON(),
      request: r ? clone(r) : null,
      visible_request_count: queryRequests({ limit: 0 }).total,
      metrics: queryMetrics(),
      profile: state.nsys ? queryNsys({ limit: 200 }) : null,
      limitations: D.meta.limitations,
      audit: D.audit,
      sources: D.sources,
    };
  }
  window.traceExplorer = Object.freeze({
    version: "2.0",
    ready: true,
    describe: () => ({
      schema: D.schema,
      meta: clone(D.meta),
      audit: clone(D.audit),
      capabilities: [
        "selectRange",
        "selectRequest",
        "expandRequest",
        "getLifecycle",
        "inspectNsys",
        "queryRequests",
        "queryMetrics",
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
    queryNsys,
    queryCpu,
    queryIterations,
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
        selected_event_count: events.length,
        names: names.length,
        cpu_samples: cpu?.samples.length ?? 0,
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
      cut =
        r.first === null
          ? 0
          : Math.min(100, Math.max(0, ((r.first - lo) / (hi - lo)) * 100));
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
      `style="background:linear-gradient(90deg,var(--amber) 0%,var(--amber) ${cut}%,var(--teal) ${cut}%,var(--teal) 100%);`,
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
  function lifecycleRows(r) {
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
    const streams = model.activities
      .filter((s) => s.kind === "concurrent")
      .map((s) =>
        track(
          labelText(s.label + " · concurrent"),
          bar(
            s.start,
            s.end,
            s.label,
            "phase frontend",
            `data-span="${esc(s.id)}" data-owner-request="${esc(r.id)}"`,
            s.description,
          ),
        ),
      )
      .join("");
    return (
      html +
      streams +
      `<div class="row-note">${esc(model.timing)} ${model.issues.map(esc).join(" · ")}</div>`
    );
  }
  function clientTracks() {
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
  function plot(series, color = "#087f8c") {
    if (!series) return '<div class="empty-lane">No recorded series</div>';
    const points = series.points.filter(
      (p) => p[0] >= state.from && p[0] <= state.to,
    );
    if (!points.length)
      return '<div class="empty-lane">No metric sample in this window · widen range</div>';
    const max =
        series.unit === "%" ? 100 : Math.max(1, ...points.map((p) => p[1])),
      w = state.to - state.from,
      xy = points.map((p) => [
        ((p[0] - state.from) / w) * 1000,
        33 - (p[1] / max) * 29,
      ]);
    let path = "";
    for (let i = 0; i < xy.length; i++) {
      const [x, y] = xy[i];
      path += `${i ? " L" : "M"}${x.toFixed(3)},${y.toFixed(3)}`;
    }
    return `<svg class="metric-svg" viewBox="0 0 1000 36" preserveAspectRatio="none" role="img" aria-label="${esc(series.label)} sampled values"><path d="${path}" fill="none" stroke="${color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/>${xy.length === 1 ? `<circle cx="${xy[0][0]}" cy="${xy[0][1]}" r="2" fill="${color}"/>` : ""}</svg>`;
  }
  function workerTracks() {
    const r = selected();
    let html =
      '<div class="section-head"><span>Server workers</span><select id="workerMetric" aria-label="Worker metric"><option value="trtllm_num_requests_running">Running requests</option><option value="trtllm_num_requests_waiting">Waiting requests</option><option value="dynamo_component_inflight_requests">In-flight requests</option><option value="trtllm_kv_cache_utilization">KV cache utilization</option></select></div>';
    for (const w of D.workers) {
      const choices = D.metrics.filter(
          (s) => s.worker === w.id && s.name === state.metric,
        ),
        ss =
          choices.find((s) => s.id === state.metricSeries[w.id]) ?? choices[0],
        stat = ss ? metricStats(ss) : null;
      html += track(
        toggle(
          "worker",
          w.id,
          state.expandedWorkers.has(w.id),
          "Expand worker " + w.id,
        ) +
          labelText(w.id) +
          `<span class="metric-last">${stat?.samples ? fmt(stat.mean, 1) : "—"} avg</span>`,
        plot(ss, w.role === "prefill" ? "#b77512" : "#087f8c"),
        { classes: r?.workers.includes(w.id) ? "selected" : "", height: 37 },
      );
      if (state.expandedWorkers.has(w.id)) {
        if (choices.length)
          html += `<div class="row-note"><label>Metric series <select data-worker-series="${esc(w.id)}" aria-label="Metric series for ${esc(w.id)}">${choices.map((s) => `<option value="${s.id}" ${s.id === ss.id ? "selected" : ""}>${esc(s.raw_name)} · ${esc(s.endpoint)} · rank ${esc(s.rank ?? "unrecorded")}</option>`).join("")}</select></label></div>`;
        const related =
          r?.lifecycle.activities.filter((s) => s.worker === w.id) ?? [];
        for (const activity of related)
          html += track(
            labelText(
              (activity.depth ? "↳ " : "") +
                activity.label +
                (activity.kind === "envelope" ? " · inclusive" : ""),
            ),
            bar(
              activity.start,
              activity.end,
              activity.label,
              `phase ${w.role}`,
              `data-span="${esc(activity.id)}"`,
              activity.description,
            ),
          );
        html += `<div class="row-note">${esc(w.host)} · ${w.profiles.length} rank reports. ${w.profiles.length ? `<button data-worker-nsys="${esc(w.id)}">Inspect Nsight</button>` : "No Nsight export for this worker."}</div>`;
      }
    }
    html += `<div class="section-head"><span>Hardware</span><button id="hardwareToggle" aria-expanded="${state.hardware}">${state.hardware ? "Hide" : "Show"} GPU / host metrics</button></div>`;
    if (state.hardware) {
      const gpuSeries = D.metrics.filter((s) =>
          ["gpu_util", "DCGM_FI_DEV_GPU_UTIL"].includes(s.name),
        ),
        hosts = [...new Set(gpuSeries.map((s) => s.host))];
      for (const host of hosts) {
        for (const series of gpuSeries.filter((s) => s.host === host))
          html += track(
            labelText(`${host} / GPU ${series.gpu}`),
            plot(series, "#6879b1"),
            { height: 37 },
          );
      }
      for (const series of D.metrics.filter(
        (s) => s.name === "memory_MemAvailable_bytes",
      ))
        html += track(
          labelText(`${series.host} / free RAM`),
          plot(series, "#607b91"),
          { height: 37 },
        );
      if (!gpuSeries.length)
        html +=
          '<div class="row-note">No GPU utilization series was recorded.</div>';
    } else
      html +=
        '<div class="row-note">GPU and host metrics follow this time range. Individual GPU samples do not identify request ownership.</div>';
    return html;
  }
  function nsysTracks() {
    if (!state.nsys) return "";
    const p = profileById.get(state.profile);
    if (!p) return "";
    let chosen = [p];
    if (state.compareNsys && selected()) {
      const ids = new Set(["frontend", ...selected().workers]);
      chosen = D.profiles
        .filter(
          (x) =>
            ids.has(x.worker) && (x.rank === null || x.rank === (p.rank ?? 0)),
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
    return chosen.map(nsysTracksFor).join("");
  }
  function nsysTracksFor(p) {
    const es = p.events.filter((e) => overlap(e[0], e[1]));
    let html = `<div class="section-head" data-nsys-heading="${p.id}"><span>Nsight · ${esc(p.worker)}${p.rank === null ? "" : ` / rank ${p.rank}`}</span><small>${fmt(es.length, 0)} selected NVTX ranges</small></div><div class="row-note">Shared CPU/NVTX activity. Shows up to 8 threads and 5 overlap lanes each; all imported events remain queryable. ${p.cuda ? "CUDA kernels exist in the source export; this view imports NVTX and CPU only." : "No CUDA kernel table in this export."}</div>`;
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
            ? `TID ${Number(BigInt(tid) & 0xffffffn)}`
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
                `${p.names[e[2]]}\n${ms(e[1] - e[0])}\n${p.worker} / rank ${p.rank ?? "frontend"}\nShared activity, row ${e[4]}`,
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
    return `<button class="path-node ${active ? "active" : ""}" data-path-worker="${esc(id)}"><strong>${esc(title)}</strong><small>${esc(host || "host not mapped")}</small></button>`;
  }

  function requestInspector() {
    const r = selected();
    if (!r)
      return "<p>Select a request to follow its frontend, prefill, and decode path.</p>";
    const pathWorkers = [
        ...new Map(r.engine.map((e) => [e.worker, e])).values(),
      ],
      sp = findInterval(r, state.span),
      front = r.spans.find((s) => s.role === "frontend");
    let html = `<div class="help">Session ${esc(short(r.session))} / ${r.client_kind === "agentperf" ? "client" : r.depth ? "subagent" : "main agent"} / turn ${esc(r.turn)}</div><div class="request-id mono">${esc(r.id)}</div><div class="stats"><div class="stat">${fmt(r.ttft_ms)}<small>Client TTFT · ms</small></div><div class="stat">${fmt(r.end - r.start, 3)}<small>Request duration · s</small></div><div class="stat">${fmt(r.input_tokens, 0)}<small>Input tokens</small></div><div class="stat">${fmt(r.output_tokens, 0)}<small>Output tokens</small></div></div><div class="actions"><button id="fitRequest">Fit request</button><button id="fitTTFT" ${r.first === null ? "disabled" : ""}>Fit TTFT</button>${hasLifecycle(r) ? `<button id="expandTTFT" aria-expanded="${state.expandedRequests.has(r.id)}">${state.expandedRequests.has(r.id) ? "Collapse" : "Expand"} lifecycle</button>` : ""}</div>`;
    if (sp) {
      const stage = [
          ...lifecycleModel(r).stages,
          ...lifecycleModel(r).activities,
        ].find((s) => s.id === sp.id),
        context = sp.routing_context,
        route = context
          ? r.spans.find((s) => s.id === context.route_span_id)
          : null;
      html += `<div class="phase-focus"><strong>${esc(stage?.label ?? sp.name)}</strong><br><span class="mono">${esc(sp.name)} · ${esc(sp.id)}</span><br>${ms(sp.end - sp.start)} ${sp.kind === "progress" ? "between milestones" : "recorded elapsed"} · ${esc(sp.role)} · ${esc(sp.host)}${stage && !context ? `<p class="help">${esc(stageDescription(stage))}</p>` : ""}${route ? `<br>Route: ${esc(route.routing.phase)} · DP rank ${esc(route.routing.dp_rank)} · attempt ${esc(route.routing["request.attempt"])}<p class="help">${esc(context.basis)}.</p>` : ""}<div class="actions"><button id="fitSpan">Fit phase</button><button id="inspectSpan">Inspect phase in Nsight</button></div></div>`;
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
    if (sp && sp.kind !== "progress")
      html += evidence(sp.evidence, "Dynamo OTel span");
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
        )}</div><p class="help">Each delta begins at the preceding milestone. Raw spans keep their original inclusive durations in the source table and worker tracks.</p>`;
    }
    if (hasLifecycle(r) && state.expandedRequests.has(r.id)) {
      html += `<h3>Source measurements</h3><p class="help">Dynamo OTel spans below are inclusive. The operation contains backend stream creation and response pumping. Frontend streaming runs concurrently.</p><table class="mini-table"><thead><tr><th>Runtime activity</th><th>Elapsed</th><th>Source</th></tr></thead><tbody>${r.lifecycle.activities.map((a) => `<tr><td><button data-span="${esc(a.id)}" title="${esc(a.description)}">${esc((a.depth ? "↳ " : "") + a.label)}</button></td><td>${ms(a.end - a.start)}</td><td title="${esc(a.name)}">OTel</td></tr>`).join("")}</tbody></table>`;
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
                `<button data-path-worker="${w.id}" class="${r.workers.includes(w.id) ? "active" : ""}">${esc(w.id)}</button>`,
            )
            .join("")}</div>`;
      }
      let hasPathNode = Boolean(front);
      for (const role of ["prefill", "decode", "aggregated"]) {
        const recorded = pathWorkers.filter((e) => e.role === role);
        if (recorded.length) {
          if (hasPathNode && role !== "prefill") html += '<div class="path-arrow">↓</div>';
          html += recorded
            .map((e) => pathNode(e.worker, e.worker, e.host, true))
            .join("");
          hasPathNode = true;
        }
      }
      if (pathWorkers.length > 2)
        html +=
          '<p class="help">All recorded workers are shown, including repeated routing attempts.</p>';
      if (pathWorkers.length)
        html += '<p class="help" style="margin-top:9px">Click a worker to expand its aligned measurements. Metric values are sample means for the explicitly selected series.</p>';
    }
    if (r.server_ids.length || r.engine.length) {
      html +=
        '<div class="detail-heading">Identity bridge</div><dl class="facts"><dt>Client</dt><dd class="mono">' +
        esc(r.id) +
        '</dd><dt>Dynamo</dt><dd class="mono">' +
        r.server_ids.map(esc).join("<br>") +
        "</dd>";
      for (const e of r.engine)
        html += `<dt>${esc(e.worker)}</dt><dd>engine client ${esc(e.client_id)}<br><span class="mono">disagg ${esc(e.disagg_id)}</span></dd>`;
      html += '</dl>';
      if (r.engine.length)
        html += '<p class="help">Engine client IDs are process-local. Router DP rank is not assumed to match a Nsight process rank; rank selection shows shared activity.</p>';
    }
    if (r.issues.length)
      html += `<p class="warn">${r.issues.map(esc).join("; ")}</p>`;
    return html;
  }
  function nsysInspector() {
    const p = profileById.get(state.profile);
    if (!p) return "<p>No Nsight SQLite exports are embedded.</p>";
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
    return `<p>Inspect activity beside the selected request. Worker and time joins identify context; they do not assign shared work to one request.</p><div class="nsys-controls"><label>Report<select id="profileSelect">${D.profiles.map((x) => `<option value="${x.id}" ${p.id === x.id ? "selected" : ""}>${esc(x.worker)}${x.rank === null ? "" : ` / rank ${x.rank}`}</option>`).join("")}</select></label><button id="showNsys">${state.nsys ? "Hide" : "Show"} Nsight tracks</button><button id="compareNsys" aria-pressed="${state.compareNsys}">${state.compareNsys ? "Show one report" : "Compare frontend + request workers"}</button></div><dl class="facts"><dt>Coverage</dt><dd>${fmt(p.capture[0], 3)} to ${fmt(p.capture[1], 3)} s</dd><dt>Matching ranges</dt><dd>${fmt(es.length, 0)}</dd><dt>Recorded host</dt><dd>${esc(p.host || "not in exported metadata")}</dd><dt>CUDA kernels</dt><dd>${p.cuda ? "Present in source; not imported" : "Not captured"}</dd><dt>Excluded ranges</dt><dd>${p.invalid_or_boundary_ranges} malformed / boundary</dd></dl><p class="help">Selected NVTX categories: frontend preprocessing/routing and engine iteration/scheduling/forward preparation. Detokenize ranges below 100 µs remain in the original report, along with other excluded categories.</p>${cpuInspector(p)}<h3>Ranges in selected window</h3><p class="help">Inclusive, clipped elapsed time; nested and parallel ranges overlap. Totals are not CPU utilization or additive TTFT.</p><table class="mini-table"><thead><tr><th>Range</th><th>Count</th><th>Elapsed</th></tr></thead><tbody>${groups.map((g) => `<tr><td>${esc(g.name)}</td><td>${fmt(g.count, 0)}</td><td>${ms(g.time)}</td></tr>`).join("")}</tbody></table>${evidence([p.evidence_source], "Nsight SQLite")}<div class="detail-heading">Collection notes</div><ul class="quality-list">${(p.diagnostics || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
  }
  function evidenceInspector() {
    const r = selected();
    return `<h3>Join coverage</h3><dl class="facts"><dt>Client requests</dt><dd>${fmt(D.audit.client_requests, 0)}</dd><dt>Client → Dynamo</dt><dd>${fmt(D.audit.clients_with_server_identity, 0)}</dd>${D.audit.clients_with_lifecycle ? `<dt>With lifecycle</dt><dd>${fmt(D.audit.clients_with_lifecycle, 0)}</dd>` : ""}<dt>Both engine maps</dt><dd>${fmt(D.audit.clients_with_both_engine_maps, 0)}</dd><dt>Ambiguous engine IDs</dt><dd>${D.audit.ambiguous_engine_ids}</dd></dl><h3>Timing and coverage</h3><ul class="quality-list">${[...D.meta.warnings, ...D.meta.limitations].map((x) => `<li>${esc(x)}</li>`).join("")}</ul><p class="help">No negative residual is relabeled as execution time. Clock anchors remain uncorrected; GPU metrics preserve host / GPU labels.</p>${
      r
        ? evidence(r.evidence, "Client record") +
          r.bridge_evidence
            .map((e) => evidence(e, "Client → Dynamo"))
            .join("") +
          r.engine
            .map((e) =>
              evidence(
                e.evidence,
                `${e.worker} → engine client ${e.client_id}`,
              ),
            )
            .join("") +
          r.spans
            .filter(
              (s) => s.name === "request.lifecycle" || s.id === state.span,
            )
            .map((s) => evidence(s.evidence, s.name))
            .join("")
        : ""
    }`;
  }
  function apiInspector() {
    const id = selected()?.id ?? "client-request-id";
    const code = [
      "const x = window.traceExplorer;",
      "x.selectRange(10, 20);",
      "x.queryRequests({limit: 10});",
      `x.selectRequest("${id}");`,
      `x.getRequest("${id}");`,
      ...(hasLifecycle(selected())
        ? [`x.getLifecycle("${id}");`, `x.expandRequest("${id}");`]
        : []),
      'x.inspectNsys({worker: "prefill-0", rank: 0});',
      "x.queryNsys({limit: 20});",
      "x.queryMetrics();",
      "x.queryIterations({worker: 'decode-0', rank: 0});",
      "x.exportSelection();",
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
          : state.tab === "iterations"
            ? iterationInspector()
            : state.tab === "evidence"
              ? evidenceInspector()
              : apiInspector();
    $("inspectorBody").scrollTop = scroll;
    document.querySelectorAll("[data-tab]").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === state.tab);
      b.setAttribute("aria-pressed", String(b.dataset.tab === state.tab));
    });
    $("selectionTag").textContent = selected()
      ? selected().status
      : "No selection";
  }
  function overview() {
    const bins = Array(300).fill(0);
    for (const r of D.requests)
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
      `${fmt(D.requests.length, 0)} selected client requests · ${fmt(D.meta.duration, 2)} s`;
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
    const scroll = $("tracks").scrollTop;
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
    $("tracks").innerHTML = clientTracks() + workerTracks() + nsysTracks();
    $("tracks").scrollTop = scroll;
    if ($("workerMetric")) $("workerMetric").value = state.metric;
    $("visibleCount").textContent =
      `${fmt(inRangeRequests().length, 0)} requests in range`;
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
            worker: "expandedWorkers",
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
        state.expandedWorkers.add(b.dataset.pathWorker);
        render();
        return;
      }
      if (b.dataset.workerNsys) {
        inspectNsys({ worker: b.dataset.workerNsys, rank: 0 });
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
        saveSelection: () =>
          download(exportSelection(), `trace-${D.meta.job}-selection.json`),
        downloadState: () =>
          download(exportSelection(), `trace-${D.meta.job}-selection.json`),
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
      actions[b.id]?.();
    }),
  );
  document.addEventListener("change", (event) =>
    safe(() => {
      const t = event.target;
      if (t.dataset.workerSeries) {
        state.metricSeries[t.dataset.workerSeries] = Number(t.value);
        render();
      }
      if (t.id === "iterationWorker") {
        state.iterationWorker = t.value;
        renderInspector();
      }
      if (t.id === "iterationRank") {
        state.iterationRank = Number(t.value);
        renderInspector();
      }
      if (t.id === "profileSelect") {
        state.profile = Number(t.value);
        state.nsys = true;
        render();
        scrollNsys();
      }
      if (t.id === "workerMetric") {
        state.metric = t.value;
        render();
      }
      if (t.id === "sessionSort") {
        state.sort = t.value;
        state.page = 0;
        render();
      }
    }),
  );
  let searchTimer;
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
  $("runSubtitle").textContent =
    `Run ${D.meta.job} · ${D.workers.filter((w) => w.role === "prefill").length} prefill / ${D.workers.filter((w) => w.role === "decode").length} decode workers · ${D.meta.phase} client phase`;
  $("joinBadge").textContent =
    `${fmt(D.audit.clients_with_server_identity, 0)} / ${fmt(D.requests.length, 0)} client → server`;
  $("joinBadge").style.display = D.audit.clients_with_server_identity ? "" : "none";
  $("joinBadge").classList.add(
    D.audit.clients_with_server_identity === D.requests.length ? "ok" : "warn",
  );
  $("coverageNotice").innerHTML =
    `<strong>Imported:</strong> ${D.requests.length} clients${D.audit.joined_spans ? ` · ${D.audit.joined_spans} joined OTel spans` : ""} · ${D.metrics.length} metric series · ${D.profiles.length} Nsight exports. ${D.meta.warnings.map(esc).join(" ")} <button data-tab="evidence">View evidence</button>`;
  if (D.meta.qualification?.passed)
    $("runSubtitle").textContent += " · capture qualified";
  const candidates = [...D.requests]
    .filter((r) => r.spans.length && r.engine.length === 2 && r.first !== null)
    .sort((a, b) => b.ttft_ms - a.ttft_ms);
  const preferred =
    candidates[Math.min(20, candidates.length - 1)] || D.requests[0];
  state.request = preferred.id;
  for (const worker of preferred.workers) state.expandedWorkers.add(worker);
  state.expandedSessions.add(preferred.session);
  state.expandedAgents.add(preferred.agent);
  if (location.hash.startsWith("#view=")) {
    try {
      restore(JSON.parse(decodeURIComponent(location.hash.slice(6))));
    } catch (e) {
      $("error").textContent = "Saved view could not be restored: " + e.message;
    }
  } else {
    state.sort = "ttft";
    const idx = sessionList().findIndex((s) => s.id === preferred.session);
    state.page = Math.floor(Math.max(0, idx) / state.pageSize);
  }
  render();
  window.dispatchEvent(new Event("trace-explorer:ready"));
})().catch((e) => {
  document.getElementById("error").textContent = e.stack || String(e);
  window.traceExplorerError = String(e);
});
