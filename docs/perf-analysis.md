# Per-run Performance Analysis

The agent writes this report from the run's saved artifacts; srtctl does not
generate it automatically.

For **every Slurm run you launch or analyze**, write and maintain
`<run_dir>/perf-analysis.md` on the cluster. Use the actual job output directory
reported by submission: normally `outputs/<job_id>/`, the parent of
`RuntimeContext.log_dir`. Honor custom output paths. Keep the authoritative
report beside the run's configuration, metadata and `logs/`; local copies and
campaign summaries may link to it.

## Metrics for this run

Record the job/config, model/engine and benchmark versions, hardware/GPU count,
workload, concurrency and measurement window. Give each metric's **value, unit
and source file/field**:

- **General:** TTFT p50/p95/p99 (ms), ITL p50/p99 (ms/token), and output throughput
  (tokens/s). Label input and total throughput separately when available.
- **Agentic benchmarks:** include the metrics relevant to this run's benchmark.

## Contribution to a performance investigation

If the run supports a performance diagnosis, debugging task or improvement,
include the goal/hypothesis, target metric, baseline and change tested. Explain
**what this run taught us and how it advances that goal**, even if the result is
negative or inconclusive.

For each finding, connect the observed metric/data to the evidence and decision:

- Link the relevant result files, log lines, traces, profiles or exported queries;
  give the metric/field, request or worker identity and time range needed to
  reproduce the observation.
- For DSight UI or dashboard evidence, include the report path and a saved view
  link with the relevant range, request and pinned metrics/panels, plus a short
  explanation of what to inspect. Follow the
  [DSight guide and skills](dsight.md#agent-skills) as applicable.
- Name/link each skill used to collect or analyze that evidence, its relevant
  command/query and resulting artifact, and explain how it helped test the
  hypothesis, locate a bottleneck, rule out a cause or validate an improvement.
