---
name: srtctl
description: Run srt-slurm (srtctl) inference benchmarks on a Slurm cluster. Use when setting up a checkout, writing or migrating a recipe, submitting a job, or reading its results.
---

# srtctl

`srtctl` turns a YAML recipe into a Slurm job that launches inference workers, a frontend, the services they need, a metrics scraper and a benchmark client, then writes logs and results to `outputs/<job_id>/`. Work from the srt-slurm checkout that holds the cluster's `srtslurm.yaml`; the docs below are paths in that checkout.

## Invariants

- `srtctl dry-run -f <recipe>` before `srtctl apply`. A recipe that fails dry-run fails on the cluster.
- Recipes are `schema: 2`. Start from `examples/`, never from scratch. Block-style YAML only.
- Read `srtslurm.yaml` for model aliases and cluster defaults instead of guessing. Ask the user where model weights live before adding `model_paths`; never download weights unless asked.
- A benchmark that "succeeded" with an implausible `Total generated tokens` did not succeed.

## Where to look

| Task | Read |
|---|---|
| Fresh checkout: install, `make setup`, first `srtslurm.yaml` | `docs/installation.md` (Run Setup, Configure srtslurm.yaml) |
| Every `srtslurm.yaml` key, including `preflight`, `default_mounts`, aliases | `docs/config-reference.md` (Cluster Config Fields) |
| Pick a starting recipe by engine, frontend and topology | `examples/README.md` (Matrix) |
| Any recipe field | `docs/config-reference.md`, or the MCP `explain_field` tool |
| Prefill/decode split, `nodes: colocate`, GPU fit | `docs/config-reference.md` (roles, Colocating decode on the prefill nodes) |
| Dynamo install, `dynamo.source`, sidecar mode | `docs/config-reference.md` (dynamo, Native sidecar mode) |
| etcd, NATS, exporters, Mooncake, declared sidecars | `docs/services.md` (Implicit Services) |
| Move a v1 recipe (`backend:`, `infra:`, `resources.*_nodes`) to schema 2 | `docs/cli.md` (srtctl migrate), `docs/legacy-v1.md` for the key mapping |
| `apply` flags: `--set`, `--serve-only`, `--tags`, `--json`, `--no-preflight` | `docs/cli.md` (srtctl apply) |
| Sweeps and override files | `docs/sweeps.md`, `docs/overrides.md` |
| What a job wrote and how to read it | `docs/monitoring.md` (Log Structure, benchmark.out) |
| Build/query the offline client, worker and hardware timeline | `docs/dsight.md`; `srtctl dsight build` then `srtctl dsight query` or MCP `query_trace` |
| Metrics and the per-run dashboard | `docs/component-dashboard.md` |

## MCP

`srtctl-mcp` offers the schema tools (`schema_summary`, `explain_field`, `validate_config`, `resolve_config`, `get_config_reference`) anywhere, and the job tools (`submit_job`, `dry_run`, `job_status`, `job_logs`, `list_jobs`, `cancel_job`) when it runs on a login node inside the checkout. See `docs/README.md`.

## Trace analysis

Generate DSight explicitly from preserved artifacts with `srtctl dsight build`;
it is not part of job execution. Read `docs/dsight.md` before interpreting its
data. Start with `query_trace(kind="summary", dataset=...)` to check coverage,
then query requests and lifecycle in a bounded time range. Worker operation
spans are inclusive; frontend streaming is concurrent. Iteration/Nsight overlap
is shared context, not request ownership. Use saved view links for human review.
Never infer unrecorded queue/compute/KV timing from a residual.
