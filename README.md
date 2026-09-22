# srtctl

Command-line tool for distributed LLM inference benchmarks on SLURM clusters using SGLang, vLLM, TensorRT LLM and AMD's ATOM. Replace complex shell scripts and 50+ CLI flags with a declarative `schema: 2` YAML recipe: `engine:` names the engine, `roles:` describes each worker role, and `services:` covers everything launched next to the workers.

## Quick Start

```bash
git clone https://github.com/NVIDIA/srt-slurm.git
cd srt-slurm
uv sync --no-dev

# One-time setup: downloads etcd, NATS, uv and the tachometer binaries for the
# compute nodes' architecture and writes srtslurm.yaml (account, partition, GPUs per node)
make setup ARCH=aarch64  # or ARCH=x86_64

uv run srtctl dry-run -f examples/sglang/dynamo-agg.yaml   # render without submitting
uv run srtctl apply   -f examples/sglang/dynamo-agg.yaml   # submit
```

## Let an agent drive it

srtctl ships a skill that teaches Claude Code, Codex or Cursor how to set up a checkout, write `srtslurm.yaml`, author and validate recipes, submit them, and read the results. Install it into the checkout and hand the agent the checkout:

```bash
uv run srtctl skill --target claude    # .claude/skills/srtctl/SKILL.md
uv run srtctl skill --target codex     # .codex/skills/srtctl/SKILL.md
uv run srtctl skill --target cursor    # .cursor/rules/srtctl.mdc
```

`srtctl-mcp` is the matching MCP server: schema tools (`schema_summary`, `explain_field`, `validate_config`, `resolve_config`) anywhere, and job tools (`submit_job`, `dry_run`, `job_status`, `job_logs`, `list_jobs`, `cancel_job`) when it runs on a login node inside the checkout. See [docs/README.md](docs/README.md).

## Documentation

**Full documentation:** https://nvidia.github.io/srt-slurm/

- [Installation](docs/installation.md) - Setup and configuration
- [Examples](examples/README.md) - Runnable 2.0 recipes, one per frontend and topology
- [Configuration Reference](docs/config-reference.md) - Every recipe section
- [CLI](docs/cli.md) - Every `srtctl` subcommand and flag
- [Legacy (v1) layout](docs/legacy-v1.md) - The old `backend:` recipe layout; `srtctl migrate` rewrites it
- [Monitoring](docs/monitoring.md) - Job logs and debugging
- [Parameter Sweeps](docs/sweeps.md) - Grid searches
- [Profiling](docs/profiling.md) - Torch/nsys profiling
- [DSight trace explorer](docs/dsight.md) - explicitly build an offline client/worker/hardware timeline; query it through CLI or MCP
- [Component Performance Dashboard](docs/component-dashboard.md) - the per-run HTML dashboard built from the tachometer parquet
- [ruter](docs/ruter.md) - Dynamo router post-processing

## Commands

```bash
# Submit job(s)
srtctl apply -f config.yaml

# Deploy an inference endpoint without running a benchmark
srtctl apply -f config.yaml --serve-only

# Override a recipe value without editing the file
srtctl apply -f config.yaml --set roles.agg.gpus=2

# Submit with tags for filtering
srtctl apply -f config.yaml --tags experiment,baseline

# Dry-run (validate without submitting)
srtctl dry-run -f config.yaml

# Rewrite a v1 recipe into the 2.0 layout
srtctl migrate -f config.yaml

# Install the agent skill into this checkout
srtctl skill --target claude
```
