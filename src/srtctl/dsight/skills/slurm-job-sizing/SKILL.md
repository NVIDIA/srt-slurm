---
name: slurm-job-sizing
description: Size srt-slurm allocation time, warmup and benchmark traffic before preparing or submitting jobs. Prefer short, parallel hypothesis tests for debugging, target validation jobs within 30 minutes, and derive full-run budgets from InferenceMAX run history and phase timing distributions.
---

# Size and submit Slurm jobs

Request the shortest allocation that can complete the task and preserve its
evidence. **Use non-preemptible resources.** Keep the required GPU type/count,
worker layout, parallelism, topology and experimental controls fixed. Reduce
unneeded traffic and allocated idle time instead of changing the experiment to
fit the queue. Only an explicit user instruction may relax those constraints.

With these resources fixed, requested walltime is the main adjustable scheduling
lever. A shorter time limit can fit a scheduling gap without delaying a
higher-priority job; it does not guarantee an earlier start. This
[backfill scheduling](https://slurm.schedmd.com/sched_config.html) works with
non-preemptible jobs and does not require a preemptible partition. Do not claim
that walltime always changes the numeric priority score: that depends on the
site's [priority configuration](https://slurm.schedmd.com/priority_multifactor.html).

Use the site's approved login and submission workflow. Cluster-specific account,
partition, QoS and resource directives belong in `srtslurm.yaml` and the resolved
recipe; do not copy them from an unrelated run.

## Choose the smallest useful experiment

| Task | Work to perform | Allocation target |
| --- | --- | --- |
| Existing-data analysis, dashboard generation, recipe validation | Reuse preserved captures; use DSight, dry-run or local tests. | No GPU allocation. Use CPU resources if needed. |
| Build, smoke test, debug or verify without a full benchmark | Split the investigation into small hypothesis tests; run independent tests in parallel with minimum useful warmup and traffic. | **At most `00:30:00` per whole job**, shorter when the budget supports it. |
| Full benchmark or sustained-load investigation | Preserve the required warmup/cache state, measured duration, repetitions and concurrency points. | Search InferenceMAX run history and derive the budget from comparable runs' phase timing distributions. No fixed full-run duration or overhead multiplier. |

State what completion proves before selecting the duration. A startup check may
need just readiness and one completed request. A scheduler, cache or transfer
check must actually exercise those paths; a short single-request test cannot
establish behavior under the target concurrency. For a delayed failure, inspect
prior request/metric timelines to find its onset and allow time to observe the
outcome, not just start the triggering request.

**Prefer short jobs in parallel for debugging.** Break independent hypotheses or
verification tasks into separate jobs, each with a predicted signal, controlled
change, completion criterion and evidence output. Submit independent jobs
together when the authorized aggregate GPU/budget limits allow; avoid putting
all experiments in one long serial allocation. Each job retains the resources
its hypothesis needs. Give each job its own allocation, mutable state and output
paths so experiments do not interfere. Keep matched controls and record revisions for
each arm. Serialize actual dependencies, such as building an image before tests
that consume it, or comparisons that require the same physical nodes/cache state.
Do not add dependencies between otherwise independent tests.

For short validation, use a separate recipe/override: reduce warmup and measured
traffic together, and select only the necessary load point. A few representative
requests or 1–3 minutes of traffic can be a starting proposal, not a universal
minimum. Retain the context/output lengths, concurrency and cache conditions
needed to reproduce the issue. Label reduced runs as diagnostic; they do not
replace the full comparison or support tail-percentile claims by themselves.

Prepare source, dependencies, images and data before the measured job wherever
possible. Separate a build from a benchmark when that avoids repeating setup;
use CPUs for work that does not require GPUs. Verify caches on the assigned nodes
rather than assuming an earlier allocation's local files survived. If necessary
startup or the symptom's onset cannot fit within 30 minutes, first split/offload
preparation and reduce unnecessary traffic. Document any remaining unavoidable
exception with timings; do not silently request a multi-hour debugging job or
pretend an impossible 30-minute budget is safe.

## Research full-run history before choosing walltime

**For every full load point, search the InferenceMAX GitHub repository's run
history first.** Resolve the repository from the user's task or existing CI
integration, following any move/rename; do not substitute an unrelated fork.
Keep private repository/run links in the task's private evidence record.

Discover benchmark workflows, then search their runs by model, hardware,
engine and workload. Inspect multiple comparable attempts, including their
recipes at the recorded commit, matrix jobs, timestamped logs and available
artifacts. A workflow title, timeout setting or result summary alone is not a
duration estimate. Example discovery commands, with recorded identifiers:

```bash
history_repo="<owner/repository>"
gh workflow list --repo "$history_repo"
gh run list --repo "$history_repo" --workflow "<benchmark-workflow>" --limit 100 \
  --json databaseId,displayTitle,headSha,status,conclusion,createdAt,url
gh run view "<run-id>" --repo "$history_repo" --attempt "<attempt>" --json jobs
gh run view "<run-id>" --repo "$history_repo" --job "<job-id>" --log
```

Page back or narrow by workflow/date when recent runs are not comparable. Follow
linked run artifacts to the actual allocation and client logs; record the run,
attempt, load point and source boundaries used for every timing. A workflow can
contain queue waits, multiple allocations or a concurrency sweep, and CPU work
after GPU release. Its elapsed time is not one load point's allocated runtime.
DSight's request window also excludes startup and teardown. When using DSight,
read [the guide](../../../../../docs/dsight.md) and
[dsight-query](../dsight-query/SKILL.md).

For each candidate, reconstruct preparation/build, model/service startup,
collector readiness, warmup, measured traffic, drain, capture finalization and
durable preservation. Show a row per run/load point with these timings and the
allocation total. Mark missing boundaries as unknown, not zero. Check whether
phases overlap and which work actually held the allocation.

Group comparable runs by model/precision, hardware/worker topology, engine/image,
client/workload, concurrency, cache state and instrumentation. Examine both each
run's phase breakdown and variation across runs: report sample count, median and
observed range for phases and totals, with upper percentiles only when supported
by enough observations. Separate cold staging/JIT from cached startup and explain
slow outliers. Use completed measurements to estimate successful duration;
retain failed, cancelled and timed-out attempts separately, with their causes
when known, to assess budget or reliability risks. Do not treat their short
elapsed times as successful runs.

Choose walltime from this distribution and explicit uncertainty, not the fastest
run, a fixed allowance or an assumed measurement-to-allocation ratio. Use observed
total/overhead variability to justify the margin; summing phase percentiles does
not establish a percentile of total runtime. Do not transplant a lower-concurrency
warmup budget into a higher-concurrency run. If history is unavailable, expired or
poorly matched, record what was searched and the gaps; use available matched local
runs or a bounded pilot to establish the missing timings, and label the estimate
provisional instead of silently falling back to a fixed duration.

## Budget the complete allocation

Budget the critical path from allocation start, excluding time waiting in queue:

```text
walltime = in-job preparation/build + model/service startup + collector readiness
         + sum(warmup + measured traffic + drain + reset/restart for each phase)
         + collector/profiler finalization + durable artifact preservation
         + explicit uncertainty margin
```

Count repeated work for every concurrency, arm and repetition sharing the
allocation. For work that actually overlaps, use the observed critical path
rather than adding all workers' durations. For count-based clients, estimate
time from comparable request completions; a request count is not a time limit.
A timed sending window also needs drain/grace time and unfinished-request
accounting. Keep safety margin separate from measured stage costs.

Present the budget as phase, source run(s), observed distribution, proposed
allowance and uncertainty. Derive every allowance from the task and history.
Multiple load points need a new total; choose separate jobs or a sequential
allocation based on startup cost, available resources and the experiment's
cache/reset requirements.

Finalize and flush captures, then preserve node-local logs, traces and client
results before releasing GPUs. Move Nsight SQLite export, DSight HTML/JSON/SQLite
generation and other offline analysis to CPU resources after preservation.
DSight is CLI-only and does not require a live deployment.

## Change the controls the runner actually consumes

There is no universal srt-slurm warmup or measured-duration field. Inspect the
selected runner and pinned client's configuration before editing the recipe.

| Scope / runner | Current controls |
| --- | --- |
| Allocation | `slurm.time_limit` renders `#SBATCH --time`. Set it explicitly so a long `default_time_limit` is not inherited. Check `sbatch_directives` and any outer allocation/wrapper for conflicting limits. |
| `sa-bench` | `benchmark.concurrencies`, `benchmark.num_warmup_mult`, `benchmark.num_prompts_mult`; each count is concurrency × multiplier, repeated at every load point. Preserve `benchmark.isl`/`benchmark.osl` or the custom dataset. These are counts, not seconds. |
| `agentperf` | Workload settings live in the file named by `benchmark.agentperf_config`; srt-slurm injects endpoint/model/concurrencies. Inspect that client's settling time, phase timeout and stop criteria. A timeout is not necessarily a measured duration. |
| `custom` (including external AgentX/AIPerf wrappers) | Inspect `benchmark.command`, `benchmark.env` and the referenced script. Change the actual client warmup and traffic arguments there. Do not assume unrelated `BenchmarkConfig` fields affect a custom command. |

See the current [configuration reference](../../../../../docs/config-reference.md),
[benchmark schema](../../../../../src/srtctl/core/schema.py), and runners:
[SA-Bench](../../../../../src/srtctl/benchmarks/sa_bench.py),
[AgentPerf](../../../../../src/srtctl/benchmarks/agentperf.py),
[custom](../../../../../src/srtctl/benchmarks/custom.py).

**Shorten work, not just its timeout.** Keep enough warmup for the required
steady state/cache condition and allow its requests to finish. Preserve the
instrumentation needed for the question. Keep profiling delay/capture windows
inside the revised workload and verify all required sources actually recorded.

Recompute every dependent deadline when changing walltime or traffic: readiness
and launch reserves, warmup watchdog, client timeout, drain/grace, profiler stop,
copy budget, and validators' expected measurement duration. An old completion
deadline must not fall inside the new measured window. Before starting a phase,
read the scheduler's actual allocation end time and check that the remaining time
covers the work plus safe finalization. Editing a recipe does not resize an
existing allocation. Do not rely on a time extension or Slurm's final kill to stop and
flush profilers. If the budget no longer fits, preserve the failed attempt and
revise the next job; do not silently truncate a full measurement.

## Validate, submit and record

Prepare a compact rationale: purpose and success evidence; fixed resources;
parallel hypothesis jobs and dependencies; InferenceMAX search/comparable run
references for full runs; phase distributions; warmup/traffic settings; stage
budget and margin; total walltime and aggregate resources; output location.
When submission is already authorized, this is a progress record, not an
additional approval gate.

From the checkout, with `recipe` pointing to the prepared recipe or named override:

```bash
recipe="<recipe.yaml>"
uv run --no-dev srtctl dry-run -f "$recipe"
# After checking the resolved recipe and generated sbatch script:
uv run --no-dev srtctl apply -f "$recipe" -y
```

Verify the rendered time, non-preemptible partition/QoS, GPU layout and actual
client arguments. Keep required run tags/provenance from the site's workflow.
Submit directly once ready; an idle-node snapshot or estimated start is not a
prerequisite. Record the returned job ID and use one background monitor through
completion/failure, inspecting actual state and reason. Do not repeatedly cancel
and resubmit a valid pending job merely because its estimated start changes.

Afterward, record requested versus actual elapsed time and each stage's duration,
completion/coverage, errors and unfinished work. Release resources promptly after
preservation. Use these observations to size the next job; a timeout or failed
warmup is a failed attempt, not evidence that the shorter benchmark passed.
