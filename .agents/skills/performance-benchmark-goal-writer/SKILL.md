---
name: performance-benchmark-goal-writer
description: Write or refine a ready-to-use /goal prompt for a performance benchmark, regression investigation or optimization campaign. Define measurable targets, a matched baseline, change boundaries, diagnostic evidence and completion criteria. Use for drafting the goal, not executing the campaign.
---

# Write a performance benchmark goal

Turn the user's performance question into an actionable goal using an
**Analyze → Hypothesize → Verify** loop. Drafting alone does not authorize starting
a goal, launching jobs, changing code or publishing results. Honor execution
already authorized by the user when a request includes both drafting and action.

Use [the goal template](assets/goal-template.md) as a starting point. Scale it to
the task: a fixed benchmark comparison does not need an optimization campaign,
and an observability feature needs evidence-coverage criteria in addition to any
performance target.

## Establish the experiment

Read the supplied context and baseline artifacts that are already accessible.
Separate facts inspected directly from user-supplied values and unresolved
inputs. Capture the information that changes the experiment:

| Input | What the goal must make explicit |
| --- | --- |
| Objective | The behavior to improve or comparison to answer, target workload, and reason for the comparison. |
| Acceptance | Exact metric, percentile, unit, direction, threshold or baseline tolerance, and the concurrency/load points where it must hold. Include correctness and error guardrails. |
| Baseline | Identifiable run/recipe, source revisions, image versions or digests, and measured window. Distinguish an existing reference from a baseline that still needs to be collected. |
| Controls | Model/precision, dataset and request mix, input/output lengths, cache/warmup state, hardware, topology, placement, worker counts, parallelism and measurement method. Hold them fixed except for named experimental variables. |
| Change scope | Repositories, code paths and configuration knobs that may change; knobs and revisions that must remain fixed; any preferred starting arm. |
| Evidence and resources | Existing captures, missing instrumentation, repeat/sample plan, compute/time limits and required deliverables or review. Preserve user-specified budgets; do not invent unlimited resources. |

Resolve missing objective, baseline identity, acceptance or change authority with
a concise bundled question when needed. Continue drafting independent parts.
If the user wants a provisional draft, label unresolved fields and proposed
choices explicitly. Do not present it as ready to execute. For an exploratory
benchmark with no performance target, define the comparison and evidence to
deliver rather than inventing a required speedup.

Keep example-specific choices conditional. A routing mode, disabled queue,
engine setting, preferred workload, numeric target or permission to modify
another repository is a requirement only when the current task supplies it.
Do not replace a pinned experimental base with latest main; related upstream
changes can be proposed as separate, recorded experiments.

## Make success unambiguous

- Translate "on par" or "better" into an explicit acceptance relation. For a
  positive baseline value B, candidate C and allowed fractional regression r,
  lower-is-better parity means `C <= B * (1 + r)`; higher-is-better parity means
  `C >= B * (1 - r)`. Preserve strict versus inclusive comparisons. A zero
  baseline needs an absolute tolerance or another agreed definition.
- Preserve the requested percentile and units. An absolute latency difference
  and a percentage difference are different targets. Require every named target
  at every specified load point; do not substitute an average across points.
- Define the request population and aggregation method. Include failed,
  cancelled, empty and unfinished requests in the validity accounting; latency
  over successful requests alone must not hide dropped or censored work. State
  the numerator/denominator for an error-rate comparison.
- Name the repetition/sample plan and how variability will be reported. If that
  plan is unresolved, mark it as such. Do not make a single best run or a small
  tail sample the completion criterion for a noisy comparison.

## Write the investigation loop

**Analyze.** For an investigation, require matched baseline and candidate runs
with diagnostic instrumentation enabled. Correlate client records, lifecycle
traces, engine statistics/scheduler logs, frontend and engine metrics, host/GPU telemetry and
available profiler evidence over the same window. Select sources relevant to
the hypothesis; these sources are not guaranteed to exist. Verify capture
coverage, clock alignment, effective configuration and metric definitions.
Break down the relevant work by request, frontend, worker and rank over time.
For event/routing questions, follow the actual producer, transport, consumer and
decision path rather than treating an enabled flag as proof of working events.

**Hypothesize.** Ask for ranked, falsifiable explanations with supporting and
conflicting evidence. Each hypothesis needs a predicted signal change and the
smallest experiment that distinguishes it from alternatives. When evidence is
missing, specify what to instrument and collect before attributing a cause.
Keep proposed causes labeled as hypotheses until the evidence supports them.

**Verify.** Restrict changes to the permitted code/configuration scope. Rebuild
when necessary, record the resulting revisions/images, and run controlled
comparisons. Test fixes individually before their combination, checking the
predicted causal signals alongside the performance targets. Preserve the
controls while measuring errors, unfinished requests and request mix alongside
latency and throughput. A benchmark-only goal stops at the defined comparison;
it does not silently authorize implementation changes.

Compare observability-on runs for diagnosis and matched observability-off runs
for final performance, unless the intended target explicitly includes monitoring
overhead. Never compare an instrumented candidate with an uninstrumented baseline
as evidence of a fix. Treat additional debug logging and profiling as measured
conditions too. In srt-slurm, consult the
[observability configuration](../../../docs/config-reference.md#observability)
for the supported controls; enabling observability alone does not establish
that every requested trace or metric was captured. Check the resolved recipes
in both modes: observability can also enable engine settings or event emission.
Pin the functional settings required by the experiment, and disclose any
remaining differences that prevent isolating instrumentation overhead.

If DSight is part of the analysis, read [the DSight guide](../../../docs/dsight.md)
and load its applicable skills before using it. Include exact metric identities,
source provenance and aligned query windows in the evidence plan. A saved view
supports an explanation; its existence alone does not establish a root cause.

For a dashboard/observability goal, define which known issue each source or view
must make diagnosable. Prefer reusable time series, request lifecycle and session
views with source attribution and graceful missing-data behavior. Verify the
normalized evidence as well as the UI; avoid run-specific panels or descriptions
that encode the desired conclusion.

## Deliver the goal

Return one copyable `/goal` prompt with the objective first, the three-step loop,
controls, allowed changes and concrete completion criteria. Put unresolved inputs
after the draft so they are easy to answer. Use the user's requested output path
when supplied; do not install a goal or run it merely because it was drafted.

Require separate experiment worktrees/run directories and preserved baselines.
Keep a durable root-cause record with evidence, rejected hypotheses, exact
revisions/configurations, reproduction steps and results across resumptions.
Before declaring success, check every acceptance criterion and challenge causal
claims against conflicting evidence. Include independent adversarial review
when the task calls for it; do not imply that a self-check is independent review.
If a budget or external blocker prevents completion, report partial results and
the unmet criteria rather than declaring the goal achieved.
