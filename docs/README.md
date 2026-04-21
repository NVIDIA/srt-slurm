# Introduction

`srtctl` is a command-line tool for running distributed LLM inference benchmarks on SLURM clusters. It replaces complex shell scripts and 50+ CLI flags with clean, declarative YAML configuration files.

## Table of Contents

- [Why srtctl?](#why-srtctl)
- [How It Works](#how-it-works)
- [Commands](#commands)
- [Next Steps](#next-steps)

## Why srtctl?

Running large language models across multiple GPUs and nodes requires orchestrating many moving parts: SLURM job scripts, container mounts, SGLang configuration, worker coordination, and benchmark execution. Traditionally, this meant maintaining brittle bash scripts with hardcoded parameters.

`srtctl` solves this by providing:

- **Declarative configuration** - Define your entire job in a single YAML file
- **Validation** - Catch configuration errors before submitting to SLURM
- **Reproducibility** - Every job saves its full configuration for later reference
- **Parameter sweeps** - Run grid searches across configurations with a single command
- **Profiling support** - Built-in torch/nsys profiling modes

## How It Works

When you run `srtctl apply -f config.yaml`, the tool:

1. Validates your configuration against the schema
2. Resolves any aliases from your cluster config (`srtslurm.yaml`)
3. Generates a SLURM batch script and SGLang configuration files
4. Submits to SLURM

The `srtctl-mcp` server is different from `srtctl apply`: it is a schema and
recipe-authoring helper. It does not use host-side `srtslurm.yaml` for cluster
defaults, aliases, containers, model paths, filesystem checks, or dry-run
behavior. Run `srtctl` on the compute side, or use an orchestrator remote
preflight path, for those checks.

Once allocated, workers launch inside containers, discover each other through ETCD and NATS, and begin serving. If you've configured a benchmark, it runs automatically against the serving endpoint and saves results to the log directory.

## Commands

| Command                                            | Description                             |
| -------------------------------------------------- | --------------------------------------- |
| `srtctl apply -f <config>`                         | Submit job(s) to SLURM                  |
| `srtctl apply -f <config> --setup-script <script>` | Submit with custom setup script         |
| `srtctl apply -f <config> --tags tag1,tag2`        | Submit with tags for filtering          |
| `srtctl dry-run -f <config>`                       | Validate and preview without submitting |
| `srtctl validate -f <config>`                      | Alias for dry-run                       |

## Next Steps

- [Installation](installation.md) - Set up `srtctl` and submit your first job
- [Monitoring](monitoring.md) - Understanding job logs and debugging
- [Parameter Sweeps](sweeps.md) - Run grid searches across configurations
- [Config Overrides](overrides.md) - Multi-variant jobs from a single file
- [Profiling](profiling.md) - Performance analysis with torch/nsys
- [Analyzing Results](analyzing.md) - Dashboard and visualization
- [SGLang Router](sglang-router.md) - Alternative to Dynamo for PD disaggregation
