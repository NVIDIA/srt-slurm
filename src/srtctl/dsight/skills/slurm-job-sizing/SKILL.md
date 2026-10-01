---
name: slurm-job-sizing
description: Size srt-slurm allocation time, warmup and benchmark traffic to the task before preparing or submitting Slurm jobs. Use for builds, tests, debugging, verification and full performance runs; keep validation jobs within 30 minutes when feasible and estimate full runs from comparable evidence.
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
| Build, smoke test, debug or verify without a full benchmark | Build only what changed; run the minimum warmup and traffic that exercise the required path and produce the evidence. | **At most `00:30:00` for the whole job**, shorter when the budget supports it. |
| Full benchmark or sustained-load investigation | Preserve the required warmup/cache state, measured duration, repetitions and concurrency points. | Derive from comparable runs; **`01:30:00` is a candidate for one 60-minute measurement**, subject to overhead. |

State what completion proves before selecting the duration. A startup check may
need just readiness and one completed request. A scheduler, cache or transfer
check must actually exercise those paths; a short single-request test cannot
establish behavior under the target concurrency. For a delayed failure, inspect
prior request/metric timelines to find its onset and allow time to observe the
outcome, not just start the triggering request.

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

## Estimate the complete allocation

Find recent comparable runs: same model/precision, hardware and worker topology,
engine/image, client/workload, concurrency, cache state and instrumentation.
Read the resolved recipe, Slurm accounting and timestamped startup/client/cleanup
logs. DSight's request window is not the allocation's total duration. When using
DSight, read [the guide](../../../../../docs/dsight.md) and
[dsight-query](../dsight-query/SKILL.md).

Separate observations from estimates. Use successful runs to size completion;
failed/timed-out runs reveal missing budget, not a successful short duration.
Prefer several comparable timings over the fastest run. With sparse history,
state uncertainty and run a bounded pilot if it can resolve the unknowns.
Do not transplant a lower-concurrency warmup budget into a higher-concurrency
run: request counts, queueing and completion times can all change.

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

Illustrative arithmetic, **not measured defaults**:

| Stage | Small validation | One full load point |
| --- | ---: | ---: |
| Prepared-image startup and collector readiness | 10 min | 12 min |
| Warmup | 2 min | 8 min |
| Measured traffic | 3 min | 60 min |
| Drain, capture finalization and durable copy | 5 min | 5 min |
| Uncertainty margin | 5 min | 5 min |
| Requested allocation | **25 min** | **90 min** |

Replace each allowance with evidence for the actual task. A 60-minute run fits
90 minutes only if all other work fits the remaining 30 minutes. Multiple load
points need a new total; choose separate jobs or a sequential allocation based
on startup cost and the experiment's cache/reset requirements.

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
comparable run references; warmup/traffic settings; stage budget and margin;
total walltime; output location. When submission is already authorized, this is
a progress record, not an additional approval gate.

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
