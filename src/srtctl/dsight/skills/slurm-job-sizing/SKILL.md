---
name: slurm-job-sizing
description: Size Slurm allocations, warmup and traffic for short debugging jobs or full srt-slurm benchmarks.
---

# Size Slurm jobs to the task

Use **non-preemptible resources** and preserve the required GPU count/type,
topology and experiment controls. Requested walltime is the main adjustable
scheduling lever: shorter jobs can fit scheduling gaps, though numeric priority
depends on the site.

- **Define the evidence needed.** Reuse existing captures or local/CPU checks
  when sufficient; otherwise choose the smallest experiment that answers the
  question.
- **Debug, build, test or verify:** prefer short, independent hypothesis tests
  **in parallel**, within authorized aggregate resources. Target **at most
  30 minutes per whole job**. Reduce warmup and measured traffic to the minimum
  that exercises the relevant path; preserve the concurrency and request shape
  needed to reproduce it. Isolate each job's state and outputs; serialize actual
  dependencies. Split preparation first if the budget does not fit, and justify
  unavoidable exceptions. Label reduced runs as diagnostic.
- **Full load point:** search the task's **InferenceMAX GitHub run history**
  before choosing walltime. Find multiple comparable runs by model, hardware,
  engine, workload, concurrency, cache state and instrumentation. Inspect their
  logs/artifacts and break down each run into startup, warmup, measurement, drain
  and preservation. Compare phase and total timing distributions; separate queue
  time and failed/incomplete attempts. Base the allowance and margin on those
  observations, not a fixed duration or the fastest run. If evidence is missing,
  state the gaps and use a bounded pilot or matched local history.
- **Set the whole-job budget.** Include preparation, startup, all load points,
  drain, capture finalization, preservation and uncertainty; account for overlap.
  Set `slurm.time_limit` explicitly and adjust the selected runner's actual
  warmup/traffic settings, using the [configuration reference](../../../../../docs/config-reference.md).
  Update dependent timeouts and profiling windows together; shortening a timeout
  alone does not shorten the work. Preserve full-run measurement requirements.
- **Validate and submit.** Dry-run the recipe; check rendered walltime, resources
  and client arguments. Record a short budget rationale, submit through the site's
  approved workflow when authorized, and record/monitor the job ID.
- **Preserve and learn.** Save logs, traces and client results before releasing
  resources. Run offline analysis on CPUs. Record actual phase durations and
  completion status to improve the next estimate.
