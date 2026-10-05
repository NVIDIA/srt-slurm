# srtctl

Command-line tool for distributed LLM inference benchmarks on SLURM clusters using SGLang, vLLM, TensorRT LLM, TileRT and AMD's ATOM. Replace complex shell scripts and 50+ CLI flags with a declarative `schema: 2` YAML recipe: `engine:` names the engine, `roles:` describes each worker role, and `services:` covers everything launched next to the workers.

**Documentation:** https://nvidia.github.io/srt-slurm/ (source in [docs/](docs/README.md); agents: [llms.txt](https://nvidia.github.io/srt-slurm/llms.txt))

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

[Installation](docs/installation.md) covers cluster-specific setup; [examples/](examples/README.md) has runnable recipes, one per frontend and topology.

## Let an agent drive it

srtctl ships a skill that teaches Claude Code, Codex or Cursor how to set up a checkout, write `srtslurm.yaml`, author and validate recipes, submit them, and read the results. Install it into the checkout and hand the agent the checkout:

```bash
uv run srtctl skill --target claude    # .claude/skills/srtctl/SKILL.md
uv run srtctl skill --target codex     # .codex/skills/srtctl/SKILL.md
uv run srtctl skill --target cursor    # .cursor/rules/srtctl.mdc
```

The `srtctl-mcp` server, the recipe JSON Schema for editors, and `llms.txt` are described in the [documentation home](docs/README.md#for-agents-and-editors).

## Where things are documented

- Write a recipe: [Recipe Guide](docs/config-reference.md), then the topic pages it links (engines, topology, frontends, benchmarks, services, ...)
- Every field, type, default and allowed value: [Schema Reference](docs/schema-reference.md) (generated from the code)
- Every command and flag: [CLI Guide](docs/cli.md) and [CLI Reference](docs/cli-reference.md) (generated)
- Run and debug: [Monitoring](docs/monitoring.md), [SLURM FAQ](docs/slurm-faq.md)
- Analyze: [Profiling](docs/profiling.md), [DSight](docs/dsight.md), [Component Performance Dashboard](docs/component-dashboard.md)
- Old recipes: [Legacy (v1) layout](docs/legacy-v1.md); `srtctl migrate` rewrites them
