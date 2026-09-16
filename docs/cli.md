# CLI Reference

`srtctl` is the main command-line interface for submitting benchmark jobs to SLURM.

## Table of Contents

- [Quick Start](#quick-start)
- [Interactive Mode](#interactive-mode)
  - [Example Browser](#example-browser)
  - [Configuration Summary](#configuration-summary)
  - [Interactive Actions Menu](#interactive-actions-menu)
  - [sbatch Preview](#sbatch-preview)
  - [Parameter Modification](#parameter-modification)
  - [Sweep Preview](#sweep-preview)
  - [Submission Confirmation](#submission-confirmation)
  - [Workflow Examples](#workflow-examples)
- [Commands](#commands)
  - [srtctl apply](#srtctl-apply)
  - [srtctl dry-run](#srtctl-dry-run)
  - [srtctl resolve-override](#srtctl-resolve-override)
  - [srtctl migrate](#srtctl-migrate)
  - [srtctl monitor](#srtctl-monitor)
  - [srtctl status-server](#srtctl-status-server)
  - [srtctl skill](#srtctl-skill)
- [Output](#output)
- [Sweep Support](#sweep-support)
- [Config Override Support](#config-override-support)
- [Tips](#tips)

---

## Quick Start

```bash
# Interactive mode - browse examples, preview, and submit
srtctl

# Submit a job directly
srtctl apply -f examples/sglang/sglang-router-disagg.yaml

# Deploy the recipe and keep its inference endpoint available until cancellation
srtctl apply -f examples/sglang/sglang-router-disagg.yaml --serve-only

# Preview without submitting
srtctl dry-run -f config.yaml

# Live dashboard - monitor all jobs in one place
srtctl monitor
```

## Interactive Mode

Running `srtctl` with no arguments launches an interactive TUI (Text User Interface) powered by Rich and Questionary:

```bash
srtctl
# or explicitly:
srtctl -i
```

Interactive mode is ideal for:
- Exploring curated examples without memorizing paths
- Previewing and tweaking configurations before submission
- Understanding what a sweep will expand to
- Quick experimentation and validation

### Example Browser

On launch, interactive mode scans the `examples/` directory and presents curated configurations organized by subdirectory:

```
? Select an example:
  ── examples/sglang ──
    dynamo-agg.yaml
    dynamo-disagg.yaml
    sglang-router-agg.yaml
    sglang-router-disagg.yaml
  ── examples/vllm ──
    dynamo-agg.yaml
    ...
  ──────────────
  📁 Browse for file...
```

**Features:**
- Examples grouped by parent directory for easy navigation
- Arrow keys to navigate, Enter to select
- "Browse for file..." option for configs outside `examples/`
- If no examples are found, prompts for manual path entry

### Configuration Summary

After selecting an example, you'll see a tree-style summary:

```
📋 Configuration
┌─────────────────────────────────────────────────────────────┐
│ deepseek-r1-1p4d                                            │
└─────────────────────────────────────────────────────────────┘

deepseek-r1-1p4d
├── 📦 Model
│   ├── path: deepseek-r1
│   ├── container: latest
│   └── precision: fp8
├── 🖥️  Resources
│   ├── gpu_type: gb200
│   ├── prefill: 1 workers
│   ├── decode: 4 workers
│   └── gpus_per_node: 4
├── 📊 Benchmark
│   ├── type: sa-bench
│   ├── isl: 1024, osl: 1024
│   └── concurrencies: [128, 256, 512]
└── 🔄 Sweep Parameters (if present)
    ├── chunked_prefill_size: [4096, 8192]
    └── max_total_tokens: [8192, 16384]
```

### Interactive Actions Menu

After viewing the config summary, you'll see an action menu:

```
? What would you like to do?
  🚀 Submit job(s)          - Submit to SLURM cluster
  👁️  Preview sbatch script  - View generated SLURM script with syntax highlighting
  ✏️  Modify parameters      - Interactively change values before submission
  🔍 Dry-run                - Full dry-run preview without submission
  📁 Select different config - Choose a different example
  ❌ Exit                   - Exit interactive mode
```

### sbatch Preview

The "Preview sbatch script" option shows the exact SLURM script that will be submitted:

```bash
┌─ Generated sbatch Script ────────────────────────────────────────────────────┐
│  1 │ #!/bin/bash                                                             │
│  2 │ #SBATCH --job-name=deepseek-r1-1p4d                                     │
│  3 │ #SBATCH --nodes=5                                                       │
│  4 │ #SBATCH --gpus-per-node=4                                               │
│  5 │ #SBATCH --time=04:00:00                                                 │
│  6 │ #SBATCH --partition=batch                                               │
│  7 │ ...                                                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

The script is syntax-highlighted with line numbers for easy reading.

### Parameter Modification

The "Modify parameters" option lets you interactively change key settings:

```
Modify Configuration
Press Enter to keep current value, or type new value

? Job name [deepseek-r1-1p4d]: my-experiment
? Prefill workers [1]:
? Decode workers [4]: 8
? Input sequence length [1024]: 2048
? Output sequence length [1024]: 2048
```

**Modifiable fields:**
- `name` - Job name
- Prefill workers - Number of prefill workers (`roles.prefill.workers` in the recipe)
- Decode workers - Number of decode workers (`roles.decode.workers` in the recipe)
- `benchmark.isl` - Input sequence length
- `benchmark.osl` - Output sequence length

Modified configs are saved to a temporary file and used for submission.

### Sweep Preview

For configs with a `sweep:` section, interactive mode shows an expansion table:

```
┌─ Sweep Jobs ────────────────────────────────────────────────────────────────┐
│ #  │ Job Name                           │ Parameters                        │
├────┼────────────────────────────────────┼───────────────────────────────────┤
│ 1  │ deepseek-r1-1p4d_cps4096_mtt8192   │ chunked_prefill_size=4096,        │
│    │                                    │ max_total_tokens=8192              │
│ 2  │ deepseek-r1-1p4d_cps4096_mtt16384  │ chunked_prefill_size=4096,        │
│    │                                    │ max_total_tokens=16384             │
│ 3  │ deepseek-r1-1p4d_cps8192_mtt8192   │ chunked_prefill_size=8192,        │
│    │                                    │ max_total_tokens=8192              │
│ 4  │ deepseek-r1-1p4d_cps8192_mtt16384  │ chunked_prefill_size=8192,        │
│    │                                    │ max_total_tokens=16384             │
└─────────────────────────────────────────────────────────────────────────────┘

Total jobs: 4
```

### Submission Confirmation

Before submitting, you'll be asked to confirm:

```
? Submit to SLURM? (y/N)
```

For sweeps, the confirmation shows:
- Full configuration summary
- Sweep expansion table
- Total job count

### Workflow Examples

**Exploring a curated example:**
```
$ srtctl
> Select: examples/sglang/sglang-router-disagg.yaml
> Action: 👁️  Preview sbatch script  (review generated script)
> Action: 🔍 Dry-run                 (full dry-run)
> Action: 📁 Select different config (try another)
```

**Quick experiment with modifications:**
```
$ srtctl
> Select: examples/vllm/dynamo-agg.yaml
> Action: ✏️  Modify parameters
  > Change decode workers: 8
  > Change isl: 2048
> Action: 🚀 Submit job(s)
> Confirm: y
```

**Sweep validation:**
```
$ srtctl
> Select: configs/my-sweep.yaml
> View: Sweep table showing 16 jobs
> Action: 🔍 Dry-run (saves all expanded configs to dry-runs/)
> Review generated configs
> Action: 🚀 Submit job(s)
```

## Commands

### `srtctl apply`

Submit a job or sweep to SLURM.

```bash
srtctl apply -f <config.yaml> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `-f, --file` | Path to YAML config file, directory, or `file:selector` for overrides (required) |
| `--sweep` | Force sweep mode (usually auto-detected) |
| `--setup-script` | Custom setup script from `configs/` |
| `--tags` | Comma-separated tags for the run |
| `--serve-only` | Deploy the endpoint without running a benchmark; serve until cancellation |
| `--set KEY=VALUE` | Override one recipe value by dotted path before validation (repeatable). Also on `dry-run`, `preflight`, `resolve-override` |
| `--unset KEY` | Remove one recipe key by dotted path before validation (repeatable) |
| `-y, --yes` | Skip confirmation prompts |
| `--no-preflight` | Skip the pre-submit `model.path` / `model.container` / telemetry filesystem checks for this run. `preflight: false` in `srtslurm.yaml` does the same for every run on a cluster whose paths exist only on compute nodes |

`--set` and `--unset` are the supported way to tweak a recipe from a script instead of editing the YAML. Paths are dotted, `[N]` indexes a list, and quotes protect a segment that contains dots (`container_mounts."/a/b.c"`). Values parse as YAML: `720` is an int, `"720"` a string, `[4, 8]` a list; a mapping such as `{"rope_type": "yarn"}` stays a literal string because that is how engine flags take JSON. Overrides are applied to the raw document before cluster defaults, sweep expansion, and validation, so an explicit `--set` always wins and `{placeholder}` values still expand. On an override file the value is written into `base` and every `override_*` / `zip_override_*` variant, so no variant can shadow it. The applied overrides are listed in each `--json` record as `applied_overrides`, and the `config.yaml` copied into the job directory reflects them. The source file is never modified.

**Examples:**

```bash
# Submit single job
srtctl apply -f examples/sglang/sglang-router-disagg.yaml

# Tweak a recipe from a script without editing it
srtctl apply -f config.yaml --set health_check.max_attempts=720 --unset sbatch_directives.exclude
srtctl apply -f config.yaml --set 'roles.decode.args.speculative-config={"method": "eagle"}'
srtctl dry-run -f config.yaml --set benchmark.concurrencies=[4,8]

# Serve the same recipe without running its configured benchmark
srtctl apply -f examples/sglang/sglang-router-disagg.yaml --serve-only

# Submit sweep (auto-detected from sweep: section)
srtctl apply -f configs/my-sweep.yaml

# Submit all override variants (base + overrides)
srtctl apply -f config.yaml

# Submit only a specific override variant
srtctl apply -f config.yaml:override_tp64

# Submit only the base config (ignore overrides)
srtctl apply -f config.yaml:base

# With tags
srtctl apply -f config.yaml --tags "experiment-1,baseline"
```

`--serve-only` submits the recipe normally, waits until the configured workers and frontend are healthy, prints
the frontend URL in the sweep log, and keeps the service running until the job is cancelled or reaches its Slurm
time limit. It ignores the recipe's configured benchmark for that submission. Use `scancel <job-id>` to stop the
service; srtctl then cleans up the processes it started.

### `srtctl dry-run`

Preview what would be submitted without actually submitting.

```bash
srtctl dry-run -f <config.yaml> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `-f, --file` | Path to YAML config file, directory, or `file:selector` for overrides (required) |
| `--sweep` | Force sweep mode |

**Examples:**

```bash
# Preview single job - shows sbatch script
srtctl dry-run -f config.yaml

# Preview sweep - shows job table and saves configs
srtctl dry-run -f sweep-config.yaml

# Preview all override variants
srtctl dry-run -f override-config.yaml

# Preview a specific override variant
srtctl dry-run -f override-config.yaml:override_tp64
```

Dry-run output includes:
- Syntax-highlighted sbatch script
- Container mounts table (labeled by source: built-in, srtslurm.yaml, configuration)
- Environment variables table (grouped by scope: global, prefill, decode, aggregated)
- srun options (if configured)
- For sweeps: table of all jobs with parameters
- Generated configs saved to `dry-runs/` folder

### `srtctl resolve-override`

Expand an override config and write the specialised YAML file(s) without submitting.

```bash
srtctl resolve-override -f <config.yaml> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `-f, --file` | Override YAML file, or `file:selector` to resolve a specific variant (required) |
| `--stdout` | Print resolved YAML to stdout instead of writing files |

**Examples:**

```bash
# Write all variants next to the source file
srtctl resolve-override -f config.yaml

# Write a single override variant
srtctl resolve-override -f config.yaml:override_lowmem

# Print to stdout
srtctl resolve-override -f config.yaml:override_lowmem --stdout

# Inspect a single zip variant
srtctl resolve-override -f config.yaml:zip_override_tp_sweep[0] --stdout
```

The resolved YAML preserves the field order and comments from the source file. Base fields appear first in their original order; override-only fields are appended at the end. Output files follow the same `{stem}_{suffix}.yaml` naming convention used by `apply`.

See [Config Overrides — Resolving Without Submitting](overrides.md#resolving-overrides-without-submitting) for details.

### `srtctl migrate`

Rewrites a v1 recipe (no `schema: 2`; `backend:`, `backend.<mode>_environment`, `infra:`, `resources.<role>_nodes` / `_workers` / `gpus_per_<role>`, `dynamo.version` / `hash` / `wheel`) into the 2.0 layout. The rewrite is deterministic and keeps comments and key order; do not translate by hand.

```bash
srtctl migrate -f old.yaml                 # print the schema-2 document, file untouched
srtctl migrate -f old.yaml --in-place      # rewrite it; a directory is walked recursively
srtctl migrate -f old.yaml --output new.yaml
srtctl migrate -f old.yaml --verify        # migrate in memory and prove v1 and v2 resolve identically
```

The key-by-key mapping is in [legacy-v1.md](legacy-v1.md). Notable rewrites: `decode_nodes: 0` becomes `roles.decode.nodes: colocate` with an explicit `gpus` on both roles; v1 `frontend.type: sglang` (the router) becomes `sglang-router`; `infra` becomes `services:` entries; benchmark fields the recipe's type never reads are removed because schema 2 rejects them. The migrator prints a note for each change and for what it deliberately leaves to you: `dynamo.top_of_tree` (pin a commit in `source.rev`), a dedicated etcd node under a frontend that runs no etcd, and a v1 recipe that never named a Dynamo to install (v1 pip-installed PyPI 0.8.0 implicitly; choose `dynamo.source` or `dynamo.install: false`). Finish with `--verify` and a `dry-run`.

### `srtctl monitor`

Live terminal dashboard for all your jobs. See [Monitoring](monitoring.md) for full documentation.

```bash
srtctl monitor                          # Active + recently completed jobs
srtctl monitor --all                    # Include older jobs from outputs/
srtctl monitor --interval 10            # Refresh every 10s (default: 5)
srtctl monitor --once                   # Snapshot and exit
srtctl monitor --resume KEY             # Resume a previous session
```

### `srtctl status-server`

Run the native status collector. Point `reporting.status.endpoint` in `srtslurm.yaml` or a recipe at it and every `srtctl apply` shows up as a job row with an ordered event feed (`submitted`, `starting`, `workers`, `frontend`, `benchmark`, then `completed` or `failed`, each with its stage and message). Jobs and events persist in one SQLite file and the process prints one line per transition. The endpoints are in [Status API](status-api-spec.md).

```bash
srtctl status-server                                        # 127.0.0.1:8080, ~/.local/state/srtctl/status.db, no token needed on loopback
srtctl status-server --host 0.0.0.0 --allow-unauthenticated # Open on a trusted network such as a login node
SRTCTL_STATUS_TOKEN=... SRTCTL_STATUS_READ_TOKEN=... \
  srtctl status-server --host 0.0.0.0                       # Bearer tokens required (write token; optional read-only token)
srtctl status-server --port 9000 --db /lustre/shared/status.db
srtctl status-server --host 0.0.0.0 --cors-origin https://ui.example  # UI hosted elsewhere may call the API (read-only)
curl http://login-node:8080/api/jobs                        # Newest jobs first
curl "http://login-node:8080/api/events?after=0"            # Global event feed; pass next_cursor back as after
```

Open `http://<host>:8080/` in a browser for the built-in UI (jobs table, per-job event timeline, live event feed); paste the read token once and the page keeps it in `localStorage`. Run the server where both the submitting host (the POST at apply time) and the allocation's head node (the PUTs during the run) can reach it, typically a login node. Listening beyond loopback without a token is refused unless `--allow-unauthenticated` is passed. With a token set on the server, export the same `SRTCTL_STATUS_TOKEN` in the shell that runs `srtctl apply`; the reporter sends it as a bearer token and never puts it in a recipe. See [Status API](status-api-spec.md#authentication).

### `srtctl skill`

Install the in-package agent skill, one document that teaches a coding agent how to drive srtctl (the 2.0 recipe shape, dry-run before apply, where a run's logs and artifacts live, the MCP tools):

```bash
srtctl skill --target claude            # .claude/skills/srtctl/SKILL.md
srtctl skill --target codex             # .codex/skills/srtctl/SKILL.md
srtctl skill --target cursor            # .cursor/rules/srtctl.mdc
srtctl skill --target claude --root /path/to/project
srtctl skill --target claude --print    # to stdout
```

### `srtctl-mcp`

The MCP server (`srtctl-mcp`, stdio by default, `SRTCTL_MCP_TRANSPORT=streamable-http` with `SRTCTL_MCP_HOST` / `SRTCTL_MCP_PORT` for HTTP) exposes the schema tools anywhere and the job lifecycle tools (`submit_job`, `dry_run`, `job_status`, `job_logs`, `list_jobs`, `cancel_job`) when it runs on a Slurm login node in a checkout with `srtslurm.yaml`. `job_status` returns the Slurm accounting row, the job metadata, the orchestrator's current stage, any `[ERROR]` lines, the benchmark rollup, and the sweep-log tail; `job_logs` lists `outputs/<job_id>/logs` or tails one file.

## Output

When you submit a job, `srtctl` creates an output directory:

```
outputs/<job_id>/
├── config.yaml         # Copy of submitted config
├── sbatch_script.sh    # Generated SLURM script
└── <job_id>.json       # Job metadata
```

## Sweep Support

Configs with a `sweep:` section are automatically detected and expanded:

```yaml
sweep:
  chunked_prefill_size: [4096, 8192]
  max_total_tokens: [8192, 16384]
```

This creates 4 jobs (2 × 2 Cartesian product). See [Parameter Sweeps](sweeps.md) for details.

## Config Override Support

Configs with a `base` top-level key are automatically detected as override configs. Each `override_<suffix>` section is deep-merged with base and submitted as a separate job.

```bash
# Submit all variants (base + all overrides)
srtctl apply -f override-config.yaml

# Submit only the tp64 override variant
srtctl apply -f override-config.yaml:override_tp64

# Submit only the base (ignoring overrides)
srtctl apply -f override-config.yaml:base
```

The `:selector` syntax works with `apply`, `dry-run`, and `resolve-override`. If the selector is used on a non-override config, a warning is logged and the config is processed normally.

Override configs also work with directory submission — override files in the directory are auto-detected and expanded.

To inspect the resolved YAML before submitting (preserving field order and comments), use `resolve-override`:

```bash
srtctl resolve-override -f override-config.yaml --stdout
```

See [Config Overrides](overrides.md) for full YAML syntax, merge semantics, and field-order / comment-preservation behaviour.

## Debugging Running Jobs

The full srun command (with all container mounts, environment variables, and flags) is logged at INFO level in the sweep log:

```bash
# Find the full srun commands for a running job
grep "srun command" outputs/<job_id>/logs/sweep_<job_id>.log

# Per-worker env vars and inner commands are also logged
grep -E "Env:|Command:" outputs/<job_id>/logs/sweep_<job_id>.log
```

## Tips

- Use `srtctl` (no args) for exploring curated examples interactively
- Use `srtctl apply -f` for scripting and CI pipelines
- Always `dry-run` first for sweeps to check job count
- Check `outputs/<job_id>/` for submitted configs and metadata
