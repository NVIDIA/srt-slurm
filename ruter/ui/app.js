const get = async (path) => {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${await response.text()}`);
  return response.json();
};

const number = (value, digits = 0) => value == null ? "—" : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
const fixed = (value, digits = 1) => value == null ? "—" : Number(value).toFixed(digits);
const percent = (value, digits = 0) => value == null ? "—" : `${(Number(value) * 100).toFixed(digits)}%`;
const html = (value) => String(value ?? "—").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character]));

let decisions = [];
let selectedRequestId = null;
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
  ].filter(([, value]) => value != null);
  strip.innerHTML = `<span class="eyebrow">router settings</span>${fields.map(([label, value]) => `<span>${html(label)} <b>${html(value)}</b></span>`).join("")}`;
  strip.classList.toggle("visible", fields.length > 0);
}

function renderChart(timeline) {
  const traces = timeline.traces;
  const customdata = traces.map((row) => [row.dynamoRequestId, row.prefillWorkerAlias]);
  const data = [{
    x: traces.map((row) => row.benchS), y: traces.map((row) => row.kvHitRate), customdata,
    mode: "lines+markers", marker: { size: 4 }, line: { color: "#6d9eff", width: 1.4 },
    hovertemplate: "bench +%{x:.2f}s<br>KV hit %{y:.3f}<br>prefill worker %{customdata[1]}<extra></extra>",
  }];
  const lowerPrefixSelections = traces.filter((row) => row.lowerPrefixSelected);
  if (lowerPrefixSelections.length) data.push({
    x: lowerPrefixSelections.map((row) => row.benchS), y: lowerPrefixSelections.map((row) => row.kvHitRate),
    mode: "markers", marker: { size: 7, color: "#ff4d4f", line: { color: "#141f2b", width: 1 } },
    customdata: lowerPrefixSelections.map((row) => [row.dynamoRequestId, row.prefillWorkerAlias]),
    hovertemplate: "bench +%{x:.2f}s<br>prefill chose less cache overlap<br>selected worker %{customdata[1]}<extra></extra>",
  });
  const axis = { showgrid: true, gridcolor: "#253346", zeroline: false, tickfont: { color: "#6f8398" }, titlefont: { color: "#aebdca", size: 11 } };
  Plotly.newPlot("chart", data, {
    paper_bgcolor: "#162231", plot_bgcolor: "#162231", font: { color: "#e6e0d5", family: "ui-monospace, SFMono-Regular, Menlo, monospace", size: 11 },
    margin: { l: 60, r: 22, t: 28, b: 44 }, hovermode: "closest", dragmode: "zoom", showlegend: false,
    xaxis: { ...axis, title: "benchmark elapsed seconds" }, yaxis: { ...axis, title: "KV hit rate", range: [0, 1] },
    annotations: [{ text: "KV hit / request", x: 0, xref: "paper", y: 1.04, yref: "paper", showarrow: false, font: { color: "#6d9eff", size: 12 } }],
  }, { displaylogo: false, responsive: true });
  document.querySelector("#chart").on("plotly_click", (event) => {
    const requestId = event.points?.[0]?.customdata?.[0];
    if (requestId) selectDecision(requestId);
  });
}

function renderTable() {
  const selected = decisions.find((row) => row.dynamoRequestId === selectedRequestId);
  document.querySelector("#route-log-selection").textContent = selected ? `selected +${number(selected.benchS, 2)}s` : "select a row for the scorecard";
  document.querySelector("#decision-rows").innerHTML = decisions.map((row) => {
    const rate = row.overlapBlocks != null && row.totalBlocks ? row.overlapBlocks / row.totalBlocks : null;
    const active = row.dynamoRequestId === selectedRequestId ? " selected" : "";
    const marker = row.dynamoRequestId === selectedRequestId ? '<span class="route-log-selected">selected</span>' : "";
    const matched = row.phase === "decode" ? "load-only" : `${number(row.overlapBlocks)} / ${number(row.totalBlocks)}`;
    return `<tr class="decision-row${active}" data-request-id="${html(row.dynamoRequestId)}"><td>+${number(row.benchS, 2)}s${marker}</td><td class="worker">${html(row.workerAlias)}</td><td>${matched}</td><td>${row.phase === "decode" ? "—" : percent(rate, 0)}</td><td>${fixed(row.costBlocks)} blocks</td></tr>`;
  }).join("");
}

function costBar(candidate, maxCost) {
  const prefill = (candidate.prefillLoadScale ?? 0) * (candidate.adjustedPrefillBlocks ?? 0);
  const decode = candidate.decodeBlocks ?? 0;
  const active = candidate.activeRequestCostBlocks ?? 0;
  const denominator = Math.max(maxCost || 0, prefill + decode + active, 1);
  return `<div class="cost-bar" title="prefill ${fixed(prefill)} + decode ${fixed(decode)} + active ${fixed(active)} blocks"><i class="prefill" style="width:${(prefill / denominator) * 100}%"></i><i class="decode" style="width:${(decode / denominator) * 100}%"></i><i class="active" style="width:${(active / denominator) * 100}%"></i></div><div class="terms"><span>P ${fixed(prefill)}</span><span>D ${fixed(decode)}</span><span>A ${fixed(active)}</span></div>`;
}

function phaseVerdict(phase) {
  const candidates = phase.candidates ?? [];
  const selected = candidates.find((candidate) => candidate.selected);
  const next = selected?.costBlocks == null ? null : candidates.find((candidate) => (candidate.costBlocks ?? Infinity) > selected.costBlocks + 0.000001);
  const margin = selected?.costBlocks != null && next?.costBlocks != null ? next.costBlocks - selected.costBlocks : null;
  const label = phase.phase === "prefill" ? "prefill" : phase.phase === "decode" ? "decode" : "route";
  return `<div class="route-destination"><span>${html(label)} worker</span><strong>${html(phase.selectedWorkerAlias)}</strong></div><div class="route-score"><span>lowest score</span><b>${fixed(selected?.costBlocks)}</b><small>blocks</small></div><div class="route-margin"><span>margin to next</span><b>${margin == null ? "—" : `${fixed(margin)} blocks`}</b><p>${phase.phase === "decode" ? "Decode selection is load-only." : phase.lowerPrefixSelected ? "Chosen despite a higher prefix-overlap candidate." : "Cache overlap and load were scored together."}</p></div>`;
}

function candidateRows(phase) {
  const candidates = phase.candidates ?? [];
  const maxCost = Math.max(...candidates.map((candidate) => candidate.costBlocks ?? 0), 1);
  const maxPrefix = Math.max(...candidates.map((candidate) => candidate.effectiveCachedBlocks ?? 0), 0);
  const role = phase.phase === "prefill" ? "Prefill" : phase.phase === "decode" ? "Decode" : "Workers";
  return `<section class="phase-candidates"><div class="contender-intro"><h3>${role} candidates</h3><p>${phase.phase === "decode" ? "load-only selection" : "prefix overlap and load"}</p></div><div class="candidate-columns"><span>worker</span><span>prefix overlap</span><span>score <em>P prefill · D decode · A active</em></span><span>worker state</span></div>${candidates.map((candidate) => {
    const prefixOverlap = candidate.effectiveCachedBlocks;
    const prefixRate = prefixOverlap != null && phase.totalBlocks ? prefixOverlap / phase.totalBlocks : null;
    const hasMostPrefix = phase.phase === "prefill" && maxPrefix > 0 && Math.abs((prefixOverlap ?? 0) - maxPrefix) < 0.000001;
    const state = `${number(candidate.runningReqs)} running · ${number(candidate.queuedReqs)} queued`;
    return `<article class="candidate${candidate.selected ? " selected" : ""}"><div class="candidate-worker"><strong>${html(candidate.workerAlias)}</strong><span class="selection">${candidate.selected ? "chosen" : "candidate"}</span></div><div class="candidate-prefix"><p>${phase.phase === "decode" ? "—" : `${fixed(prefixOverlap)} blocks${hasMostPrefix ? '<span class="best-prefix">highest</span>' : ""}`}</p><small>${phase.phase === "decode" ? "not used" : `${percent(prefixRate, 0)} of prompt`}</small></div><div class="candidate-score"><b>${fixed(candidate.costBlocks)}</b><span>blocks</span>${costBar(candidate, maxCost)}</div><div class="candidate-state"><p>${html(state)}</p><small>KV cache: ${percent(candidate.gpuCacheUsageFraction, 0)} used</small></div></article>`;
  }).join("") || '<p class="chart-note">No DYN_LOG=debug formula was captured for this decision.</p>'}</section>`;
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
  const phases = data.phases ?? [];
  document.querySelector("#inspector-title").textContent = phases.map((phase) => `${phase.phase === "prefill" ? "P" : phase.phase === "decode" ? "D" : ""}-${phase.selectedWorkerAlias}`).join(" → ");
  document.querySelector("#inspector-reason").textContent = "Each panel is the exact candidate set and last scrape before that routing decision.";
  document.querySelector("#route-verdict").innerHTML = phases.map(phaseVerdict).join("");
  const facts = data.facts ?? {};
  document.querySelector("#request-facts").innerHTML = [
    ["KV hit", percent(facts.kvHitRate, 1)],
    ["TTFT / E2E", `${fixed(facts.ttftMs)} / ${fixed(facts.e2eMs)} ms`],
    ["input / cached", `${number(facts.inputTokens)} / ${number(facts.cachedTokens)} tokens`],
  ].map(([label, value]) => `<div><span>${label}</span><strong>${html(value)}</strong></div>`).join("");
  document.querySelector("#candidate-rows").innerHTML = phases.map(candidateRows).join("");
}

async function selectDecision(requestId) {
  if (!requestId || requestId === selectedRequestId) return;
  selectedRequestId = requestId;
  renderTable();
  const version = ++selectionVersion;
  document.querySelector("#inspector").classList.add("loading");
  try {
    const data = await get(`/api/decision?id=${encodeURIComponent(requestId)}`);
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
  const row = event.target.closest("[data-request-id]");
  if (row) selectDecision(row.dataset.requestId);
});

(async () => {
  try {
    setupBeaverAudio();
    const [summary, timeline, loadedDecisions] = await Promise.all([get("/api/summary"), get("/api/timeline"), get("/api/decisions")]);
    decisions = loadedDecisions;
    populateSummary(summary);
    renderChart(timeline);
    renderTable();
    const first = decisions.find((row) => row.dynamoRequestId);
    if (first) selectDecision(first.dynamoRequestId);
  } catch (error) {
    const node = document.querySelector("#error"); node.textContent = error.message; node.style.display = "block";
  }
})();
