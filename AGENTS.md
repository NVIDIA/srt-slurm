Read `CLAUDE.md` for the development guide: commands, PR descriptions, style, and the Design Rules.
Subsystem rules are in the `AGENTS.md` file of each directory you touch; review criteria are in `REVIEW.md`; recurring procedures are skills in `.agents/skills/`.

For every Slurm run you launch or analyze, write `<run_dir>/perf-analysis.md` in
the actual run output directory on the cluster, including failed or partial
runs. Follow [Per-run Performance Analysis](CLAUDE.md#per-run-performance-analysis)
for the required TTFT/ITL percentiles, throughput, agentic Pareto/SLO metrics,
sources and missing-data handling. For performance diagnosis, debugging or
improvement runs, explain how the collected metrics, files, DSight/dashboard
views and skills helped answer the goal. Verify the report before declaring the
analysis complete.

Before using DSight, read [docs/dsight.md](docs/dsight.md) and load the applicable
DSight skills listed there by reading the linked skill files. This applies to
building or querying reports, analyzing existing results, preparing dashboard
views, and changing DSight code.
