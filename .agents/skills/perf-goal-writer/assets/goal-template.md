# Performance benchmark goal template

Replace bracketed fields with verified or explicitly user-supplied values. Adapt
the scope to the task and remove inapplicable clauses. Leave unknowns visibly
unresolved in a provisional draft; do not invent targets, permissions or results.

```text
/goal

[Improve the named behavior or answer the benchmark question] for [model,
workload and load points], using [identifiable baseline]. Success means [exact
metrics, percentiles, units, directions and thresholds at each load point],
while preserving [correctness and error/unfinished-request criteria].

Controls and scope: Keep [workload, hardware/topology, placement, engine knobs,
baseline revisions and measurement method] fixed except for [experimental
variables]. Start from [recorded baseline/arm]. Changes are permitted in
[repositories/code/configuration]; do not change [explicit exclusions].
Use [repeat/sample plan and variability reporting] within [supplied budget].

1. Analyze: Compare matched baseline and candidate runs with diagnostic
instrumentation enabled. Correlate [relevant available sources] over the same
window by request, frontend, worker and rank. Verify coverage, clock alignment,
effective configuration and metric definitions. Identify missing evidence and
collect it before making causal claims.

2. Hypothesize: Rank falsifiable explanations with supporting and conflicting
evidence. For each, state the predicted signal change and smallest experiment
that distinguishes it from alternatives.

3. Verify: Apply justified changes within the allowed scope, rebuild as needed
and rerun controlled comparisons. Test each fix separately and then together,
checking predicted causal signals alongside the performance targets.
Use matched observability-on runs for diagnosis and matched observability-off
runs for final performance [adapt if monitoring overhead is the target]. Check
errors, incomplete work and request mix as well as latency and throughput.

Preserve baseline and candidate experiments in separate worktrees/run folders.
Record exact source revisions, images, configurations and evidence locations.
Maintain root_cause.md with measured findings, rejected hypotheses, reproduction
steps and before/after results across resumptions.

Complete when [all target/validity criteria and requested deliverables] are
verified, the before/after table reports every required run/load point, and
[required causal-evidence review] finds no unsupported conclusions. Include
[source-linked evidence and dashboard views, if applicable]. Iterate within the
agreed scope and budget; if blocked, report the unmet criteria and evidence.
```
