const get = async (path) => {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${await response.text()}`);
  return response.json();
};

const number = (value, digits = 0) => value == null ? "—" : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
const fixed = (value, digits = 1) => value == null ? "—" : Number(value).toFixed(digits);
const percent = (value, digits = 0) => value == null ? "—" : `${(Number(value) * 100).toFixed(digits)}%`;
const html = (value) => String(value ?? "—").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character]));

let routes = [];
let selectedDecisionId = null;
let selectionVersion = 0;

function setupBeaverAudio() {
  const audio = document.querySelector("#beaver-audio");
  document.querySelector("#beaver-badge").addEventListener("click", () => {
    audio.currentTime = 0;
    audio.play().catch(() => {});
  });
}

function populateSummary(summary) {
  document.querySelector("#traces").textContent = number(summary.requestTraces);
  document.querySelector("#kv-hit").textContent = summary.avgKvHitRate == null ? "—" : summary.avgKvHitRate.toFixed(3);
  document.querySelector("#ttft").textContent = summary.avgTtftMs == null ? "—" : `${number(summary.avgTtftMs)} ms`;
  document.querySelector("#workers").textContent = summary.workerAliases.join(" ");
  const settings = summary.routerSettings;
  const strip = document.querySelector("#router-settings");
  if (!settings) return;
  const fields = [
    ["mode", settings.router_mode],
    ["cache credit", settings.overlap_score_credit],
    ["credit decay", settings.overlap_score_credit_decay],
    ["prefill scale", settings.prefill_load_scale],
    ["active cost", settings.decode_active_request_weight],
    ["temperature", settings.router_temperature],
  ].filter(([, value]) => value != null);
  strip.innerHTML = `<span class="eyebrow">router settings</span>${fields.map(([label, value]) => `<span>${html(label)} <b>${html(value)}</b></span>`).join("")}`;
  strip.classList.toggle("visible", fields.length > 0);
}

function renderChart(timeline) {
  const traces = timeline.traces;
  const customdata = traces.map((row) => [row.prefillDecisionId, row.prefillWorkerAlias, row.decodeWorkerAlias]);
  const shared = { mode: "lines+markers", marker: { size: 4 }, line: { width: 1.25 }, customdata };
  const data = [
    { ...shared, x: traces.map(r => r.benchS), y: traces.map(r => r.kvHitRate), name: "KV hit rate", line: { color: "#6d9eff", width: 1.4 }, hovertemplate: "bench +%{x:.2f}s<br>KV hit %{y:.3f}<br>path %{customdata[1]} → %{customdata[2]}<extra></extra>" },
  ];
  const lowerPrefixSelections = traces.filter((row) => row.lowerPrefixSelected);
  if (lowerPrefixSelections.length) {
    data.push({
      x: lowerPrefixSelections.map((row) => row.benchS), y: lowerPrefixSelections.map((row) => row.kvHitRate),
      mode: "markers", marker: { size: 7, color: "#ff4d4f", line: { color: "#141f2b", width: 1 } },
      customdata: lowerPrefixSelections.map((row) => [row.prefillDecisionId, row.prefillWorkerAlias, row.decodeWorkerAlias]),
      hovertemplate: "bench +%{x:.2f}s<br>lower-prefix prefill choice<br>path %{customdata[1]} → %{customdata[2]}<extra></extra>",
    });
  }
  const axis = { showgrid: true, gridcolor: "#253346", zeroline: false, tickfont: { color: "#6f8398" }, titlefont: { color: "#aebdca", size: 11 } };
  Plotly.newPlot("chart", data, {
    paper_bgcolor: "#162231", plot_bgcolor: "#162231", font: { color: "#e6e0d5", family: "ui-monospace, SFMono-Regular, Menlo, monospace", size: 11 },
    margin: { l: 60, r: 22, t: 28, b: 44 }, hovermode: "closest", dragmode: "zoom", showlegend: false,
    xaxis: { ...axis, title: "benchmark elapsed seconds" }, yaxis: { ...axis, title: "KV hit rate", range: [0, 1] },
    annotations: [
      { text: "KV hit / request", x: 0, xref: "paper", y: 1.04, yref: "paper", showarrow: false, font: { color: "#6d9eff", size: 12 } },
    ],
  }, { displaylogo: false, responsive: true });
  document.querySelector("#chart").on("plotly_click", (event) => {
    const decisionId = event.points?.[0]?.customdata?.[0];
    if (decisionId) selectDecision(decisionId);
  });
}

function renderTable() {
  const selected = routes.find((row) => row.prefillDecisionId === selectedDecisionId || row.decodeDecisionId === selectedDecisionId);
  const selectionLabel = document.querySelector("#route-log-selection");
  if (selectionLabel) selectionLabel.textContent = selected ? `selected +${number(selected.benchS, 2)}s` : "select a row for the full scorecard";
  document.querySelector("#decision-rows").innerHTML = routes.map((row) => {
    const rate = row.overlapBlocks != null && row.totalBlocks ? row.overlapBlocks / row.totalBlocks : null;
    const active = row.prefillDecisionId === selectedDecisionId || row.decodeDecisionId === selectedDecisionId ? " selected" : "";
    const marker = active ? '<span class="route-log-selected">selected</span>' : "";
    const prefix = row.prefillDecisionId ? `${number(row.overlapBlocks)} / ${number(row.totalBlocks)} · ${percent(rate, 0)}` : "—";
    const stageScore = (stage, decisionId, score) => decisionId ? `<button class="stage-link${decisionId === selectedDecisionId ? " selected" : ""}" type="button" data-decision-id="${html(decisionId)}"><span>${stage}</span>${fixed(score)}</button>` : "—";
    const path = [row.prefillWorkerAlias, row.decodeWorkerAlias].filter(Boolean).join(" → ");
    const defaultId = row.prefillDecisionId || row.decodeDecisionId;
    return `<tr class="decision-row${active}" data-default-decision-id="${html(defaultId)}"><td>+${number(row.benchS, 2)}s${marker}</td><td class="worker">${html(path)}</td><td>${prefix}</td><td>${stageScore("P", row.prefillDecisionId, row.prefillScoreBlocks)}</td><td>${stageScore("D", row.decodeDecisionId, row.decodeScoreBlocks)}</td></tr>`;
  }).join("");
}

function costBar(candidate, maxCost) {
  const prefill = (candidate.prefillLoadScale ?? 0) * (candidate.adjustedPrefillBlocks ?? 0);
  const decode = candidate.decodeBlocks ?? 0;
  const active = candidate.activeRequestCostBlocks ?? 0;
  const denominator = Math.max(maxCost || 0, prefill + decode + active, 1);
  return `<div class="cost-bar" title="prefill ${fixed(prefill)} + decode ${fixed(decode)} + active ${fixed(active)} blocks"><i class="prefill" style="width:${(prefill / denominator) * 100}%"></i><i class="decode" style="width:${(decode / denominator) * 100}%"></i><i class="active" style="width:${(active / denominator) * 100}%"></i></div><div class="terms"><span>P ${fixed(prefill)}</span><span>D ${fixed(decode)}</span><span>A ${fixed(active)}</span></div>`;
}

function renderInspector(data) {
  const panel = document.querySelector("#inspector");
  if (!data.found) {
    panel.classList.add("empty");
    document.querySelector("#inspector-title").textContent = "No score record";
    document.querySelector("#inspector-reason").textContent = "No matching DYN_LOG=debug formula.";
    document.querySelector("#request-facts").innerHTML = "";
    document.querySelector("#route-verdict").innerHTML = "";
    document.querySelector("#candidate-rows").innerHTML = "";
    return;
  }
  panel.classList.remove("empty");
  const candidates = data.candidates ?? [];
  const isDecode = data.stage === "decode";
  const stageName = isDecode ? "Decode" : data.stage === "prefill" ? "Prefill" : "Route";
  const path = [data.requestPath?.prefillWorkerAlias, data.requestPath?.decodeWorkerAlias].filter(Boolean).join(" → ");
  const selected = candidates.find((candidate) => candidate.selected);
  const tied = selected?.costBlocks == null ? [] : candidates.filter((candidate) => Math.abs((candidate.costBlocks ?? Infinity) - selected.costBlocks) < 0.000001);
  const next = selected?.costBlocks == null ? null : candidates.find((candidate) => (candidate.costBlocks ?? Infinity) > selected.costBlocks + 0.000001);
  const delta = selected?.costBlocks != null && next?.costBlocks != null ? next.costBlocks - selected.costBlocks : null;
  document.querySelector("#inspector-title").textContent = `${stageName} route at +${number(data.benchS, 2)}s`;
  document.querySelector("#inspector-reason").textContent = selected ? (tied.length > 1 ? `${selected.workerAlias} tied at ${fixed(selected.costBlocks)} with ${tied.filter((candidate) => !candidate.selected).map((candidate) => candidate.workerAlias).join(", ")}.` : `${selected.workerAlias} wins: ${fixed(selected.costBlocks)} blocks${delta == null ? "" : ` · ${fixed(delta)} below ${next.workerAlias}`}.`) : "No selected score was materialized.";
  document.querySelector("#route-verdict").innerHTML = selected ? `<div class="route-destination"><span>chosen ${stageName.toLowerCase()} worker</span><strong>${html(selected.workerAlias)}</strong></div><div class="route-score"><span>router score</span><b>${fixed(selected.costBlocks)}</b><small>blocks</small></div><div class="route-margin"><span>request path</span><b>${html(path || "—")}</b><p>${tied.length > 1 ? "Equal minimum scores; selector broke the tie." : next ? `${selected.workerAlias} has the lowest recorded score; ${next.workerAlias} is next.` : "No second score was recorded."}</p></div>` : "";
  const overlapRate = data.overlapBlocks != null && data.totalBlocks ? data.overlapBlocks / data.totalBlocks : null;
  document.querySelector("#request-facts").innerHTML = [
    ["path", path || data.selectedWorkerAlias],
    ...(isDecode ? [] : [["cache match", `${number(data.overlapBlocks)} / ${number(data.totalBlocks)} · ${percent(overlapRate, 0)}`]]),
    ["KV hit", percent(data.kvHitRate, 1)],
    ["TTFT / E2E", `${fixed(data.ttftMs)} / ${fixed(data.e2eMs)} ms`],
  ].map(([label, value]) => `<div><span>${label}</span><strong>${html(value)}</strong></div>`).join("");
  const maxCost = Math.max(...candidates.map((candidate) => candidate.costBlocks ?? 0), 1);
  const maxPrefix = Math.max(...candidates.map((candidate) => candidate.effectiveCachedBlocks ?? 0), 0);
  document.querySelector("#candidate-rows").innerHTML = `<div class="candidate-columns${isDecode ? " decode" : ""}"><span>worker</span>${isDecode ? "" : "<span>prefix overlap</span>"}<span>score <em>P prefill · D decode · A active</em></span><span>worker state</span></div>${candidates.map((candidate) => {
    const selectedClass = candidate.selected ? " selected" : "";
    const state = `${number(candidate.runningReqs)} requests running · ${number(candidate.queuedReqs)} queued`;
    const cacheUse = `KV cache: ${percent(candidate.gpuCacheUsageFraction, 0)} used`;
    const prefixOverlap = candidate.effectiveCachedBlocks;
    const prefixRate = prefixOverlap != null && data.totalBlocks ? prefixOverlap / data.totalBlocks : null;
    const hasMostPrefix = maxPrefix > 0 && Math.abs((prefixOverlap ?? 0) - maxPrefix) < 0.000001;
    const prefixNote = hasMostPrefix ? '<span class="best-prefix">highest</span>' : "";
    return `<article class="candidate${selectedClass}${isDecode ? " decode" : ""}"><div class="candidate-worker"><strong>${html(candidate.workerAlias)}</strong><span class="selection">${candidate.selected ? "chosen" : "candidate"}</span></div>${isDecode ? "" : `<div class="candidate-prefix"><p>${fixed(prefixOverlap)} blocks${prefixNote}</p><small>${percent(prefixRate, 0)} of prompt</small></div>`}<div class="candidate-score"><b>${fixed(candidate.costBlocks)}</b><span>blocks</span>${costBar(candidate, maxCost)}</div><div class="candidate-state"><p>${html(state)}</p><small>${html(cacheUse)}</small></div></article>`;
  }).join("")}`;
}

async function selectDecision(decisionId) {
  if (!decisionId || decisionId === selectedDecisionId) return;
  selectedDecisionId = decisionId;
  renderTable();
  const version = ++selectionVersion;
  document.querySelector("#inspector").classList.add("loading");
  try {
    const data = await get(`/api/decision?id=${encodeURIComponent(decisionId)}`);
    if (version === selectionVersion) renderInspector(data);
  } catch (error) {
    if (version === selectionVersion) {
      document.querySelector("#inspector-title").textContent = "Decision unavailable";
      document.querySelector("#inspector-reason").textContent = error.message;
    }
  } finally {
    if (version === selectionVersion) document.querySelector("#inspector").classList.remove("loading");
  }
}

document.querySelector("#decision-rows").addEventListener("click", (event) => {
  const decision = event.target.closest("[data-decision-id]");
  if (decision) {
    selectDecision(decision.dataset.decisionId);
    return;
  }
  const row = event.target.closest("[data-default-decision-id]");
  if (row) selectDecision(row.dataset.defaultDecisionId);
});

(async () => {
  try {
    setupBeaverAudio();
    const [summary, timeline, loadedRoutes] = await Promise.all([get("/api/summary"), get("/api/timeline"), get("/api/decisions")]);
    routes = loadedRoutes;
    populateSummary(summary);
    renderChart(timeline);
    renderTable();
    const first = routes.find((row) => row.prefillDecisionId || row.decodeDecisionId);
    if (first) selectDecision(first.prefillDecisionId || first.decodeDecisionId);
  } catch (error) {
    const node = document.querySelector("#error"); node.textContent = error.message; node.style.display = "block";
  }
})();
