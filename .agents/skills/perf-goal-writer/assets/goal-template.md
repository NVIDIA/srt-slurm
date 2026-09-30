# Performance benchmark goal template

Replace bracketed fields with verified or explicitly user-supplied values. Adapt
the scope and remove inapplicable clauses. Keep the filled goal compact: prefer
2,000 characters or fewer; never exceed 4,000 for Claude Code or a lower applicable
harness/user limit. Count the complete payload after substitution, including
`/goal`, spaces and newlines. Put the verified count outside the copyable block.
Leave unknowns explicit in provisional drafts; do not invent targets or authority.

```text
/goal

[Objective] for [workload/load points] versus [baseline/run/revisions].
Pass: [metric, percentile, units, direction and bounds at every load point],
with [correctness/error/unfinished-request criteria].

Fix [controls]; vary only [experimental variables]. May change [scope];
do not change [exclusions]. Use [repeat/sample/aggregation plan] within [budget].

1. Analyze: Compare matched instrumented runs. Correlate [relevant sources]
by request, frontend, worker/rank and time. Check capture coverage, clocks,
effective config and metric definitions; collect missing evidence within scope.

2. Hypothesize: Rank falsifiable causes with supporting/conflicting evidence,
predicted signals and the smallest discriminating experiments.

3. Verify: Test fixes separately, then together; check predicted signals and
targets. Match instrumentation between arms: observability on for diagnosis,
off for final performance [adapt to the requested measurement policy].
Preserve functional settings across both. Check request mix, errors and
unfinished work; report every required load point/repetition and variability.

Keep experiment worktrees/run folders and baselines separate. Record revisions,
images, config, evidence and rejected hypotheses in root_cause.md with
reproduction steps. Complete when [acceptance criteria/deliverables/review]
are verified. Iterate within scope/budget; if blocked, report unmet criteria.
```
