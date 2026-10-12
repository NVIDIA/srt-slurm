# Introduction

`srtctl` is a command-line tool for running distributed LLM inference benchmarks on SLURM clusters. It replaces complex shell scripts and 50+ CLI flags with one declarative `schema: 2` YAML recipe: `engine:` names the inference engine (SGLang, vLLM, TRT-LLM, TileRT, ATOM), `roles:` describes each worker role (prefill, decode, agg) with its node and GPU counts, `env`, and `args`, `frontend:` picks the router, `benchmark:` the load, and `services:` anything else launched next to the job.

New here: [Installation](installation.md) sets up a checkout and submits a first job.

## Why srtctl?

Running large language models across multiple GPUs and nodes requires orchestrating many moving parts: SLURM job scripts, container mounts, engine configuration, worker coordination, and benchmark execution. Traditionally, this meant maintaining brittle bash scripts with hardcoded parameters.

`srtctl` solves this by providing:

- **Declarative configuration** - Define your entire job in a single YAML file
- **Validation** - Catch configuration errors before submitting to SLURM
- **Reproducibility** - Every job saves its full configuration for later reference
- **Parameter sweeps** - Run grid searches across configurations with a single command
- **Profiling support** - Built-in torch/nsys profiling modes

## How It Works

When you run `srtctl apply -f config.yaml`, the tool:

1. Resolves aliases from your cluster config (`srtslurm.yaml`) and normalizes `engine:`, `roles:`, `placement:`, and `services:` into the internal config
2. Validates the result against the schema (a colocated decode split that does not fit, a benchmark field the type does not use, and a moving `dynamo.source.rev` are all rejected here)
3. Generates a SLURM batch script and the per-role engine configuration
4. Submits to SLURM

Once allocated, workers launch inside containers, discover each other through etcd (NATS only when a recipe selects a NATS request or event plane), and begin serving. If you've configured a benchmark, it runs automatically against the serving endpoint and saves results to the log directory.

## How these docs are organized

- **Get Started**: installation, `srtslurm.yaml`, and the CLI workflows.
- **Write a Recipe**: the [Recipe Guide](config-reference.md) and one page per recipe block. These pages explain behavior, how blocks interact, and worked examples.
- **Run and Operate** and **Analyze**: monitoring, troubleshooting, profiling, telemetry, and trace tools.
- **Reference**: generated from the code, so it is never stale. [Schema Reference](schema-reference.md) lists every field with its type, default, and allowed values; [CLI Reference](cli-reference.md) lists every command and flag.

When a guide page and a generated page disagree about a field, the generated page is right.

## For agents and editors

- **llms.txt**: [`llms.txt`](https://nvidia.github.io/srt-slurm/llms.txt) lists every page on this site with a one-line summary; it is rebuilt with the site.
- **JSON Schema**: [`schema/recipe.schema.json`](https://nvidia.github.io/srt-slurm/schema/recipe.schema.json) (recipes and override files) and [`schema/cluster.schema.json`](https://nvidia.github.io/srt-slurm/schema/cluster.schema.json) (`srtslurm.yaml`), the same output as `srtctl schema [--cluster]`. Put `# yaml-language-server: $schema=https://nvidia.github.io/srt-slurm/schema/recipe.schema.json` on a recipe's first line for live validation in editors. It checks shape only; `srtctl dry-run` checks cross-field rules.
- **Agent skill**: `srtctl skill --target claude|codex|cursor` installs the in-package skill (how to author, validate, submit, and read back a run) into a project.
- **MCP server**: `srtctl-mcp` has two halves. The schema tools (`schema_summary`, `explain_field`, `validate_config`, `preflight_config`, `resolve_config`, `get_config_reference`) are recipe-authoring helpers that work anywhere and never read host-side `srtslurm.yaml`. `schema_summary` and `explain_field` answer from the schema dataclasses (the same data as the Schema Reference), with prose from the Write a Recipe pages added as context. The job tools (`submit_job`, `dry_run`, `job_status`, `job_logs`, `list_jobs`, `cancel_job`) drive `srtctl apply`, `sacct`, `squeue`, and `scancel` and read the job's output directory, so they only do anything when the server runs on a login node of the cluster, inside the checkout that has its `srtslurm.yaml`.
