# CLI Reference

<!-- GENERATED FILE. Do not edit by hand. Rendered from the argparse parser in src/srtctl/cli/submit.py by `srtctl schema-docs`; CI fails when this file is stale. -->

Every `srtctl` subcommand and argument, generated from the parser itself. Workflows, examples, and what each command does are in the [CLI Guide](cli.md). Running `srtctl` with no arguments starts the interactive mode.

```text
Examples:
  srtctl                                         # Interactive mode
  srtctl apply -f config.yaml                    # Submit job
  srtctl apply -f config.yaml --serve-only       # Serve until cancelled; do not benchmark
  srtctl apply -f ./configs/                     # Submit all YAMLs in directory
  srtctl apply -f config.yaml --sweep            # Submit sweep
  srtctl preflight -f config.yaml                # Check model/container availability
  srtctl dry-run -f config.yaml                  # Dry run
  srtctl render -f config.yaml --to ./rendered   # Write the sbatch script for someone else to submit
  srtctl resolve-override -f config.yaml         # Resolve override YAML (no submit)
  srtctl resolve-override -f config.yaml --stdout  # Print to stdout
  srtctl monitor                                 # Live job dashboard
  srtctl monitor --outputs /path/to/outputs      # Dashboard with custom outputs dir
  srtctl status-server --host 0.0.0.0            # Local status collector for reporting.status.endpoint
  srtctl schema-docs [--check]                   # Regenerate (or verify) the generated docs
  srtctl schema [--cluster]                      # JSON Schema for recipes (or srtslurm.yaml)
  srtctl migrate -f config.yaml --in-place       # Rewrite a pre-2.0 recipe into the current schema (dir: recursive)
  srtctl skill --target claude                   # Install the srtctl agent skill into this project
  srtctl --version                               # Version (from the git tag), commit, schema and lockfile versions
```

## `srtctl dsight`

Build and query an offline inference trace dashboard

Aliases: `dashboard`

### `srtctl dsight build`

Build an offline HTML from existing run artifacts

| Argument | Default | Description |
| --- | --- | --- |
| `logs` | required | Run directory or its logs/ directory |
| `--output, -o OUTPUT` | required | New or previously generated dashboard directory |
| `--client CLIENT` |  | Select one AIPerf profile_export.jsonl or AgentPerf requests.jsonl |
| `--metrics METRICS` |  | One raw Tachometer capture leaf or file (default: logs/tachometer/local) |
| `--nsys-sqlite SQLITES` |  | Existing Nsight SQLite exports; does not enable profiling |
| `--no-otel` |  | Skip OTel import and request lifecycle breakdowns |
| `--single-file` |  | Embed all data for file:// viewing (larger HTML) |
| `--job JOB` |  | Display identifier (default: parent directory of logs) |
| `--phase PHASE` | `profiling` | Client benchmark_phase to include; 'all' includes warmup explicitly |
| `--iteration-timezone ITERATION_TIMEZONE` |  | IANA timezone of iteration/batch logs without offsets, e.g. America/Los_Angeles |
| `--max-profile-events MAX_PROFILE_EVENTS` | `250000` | Maximum imported NVTX events per report; truncation is reported |

### `srtctl dsight query`

Query the generated dataset as JSON without a browser

| Argument | Default | Description |
| --- | --- | --- |
| `dataset` | required | Dashboard directory, trace-data.sqlite or legacy trace-data.json.gz |
| `kind` | `summary` | One of: `summary`, `requests`, `request`, `lifecycle`, `metrics`, `profiles`, `nsys`, `cpu`, `iterations`, `sources`, `server_spans`. |
| `--from START` |  |  |
| `--to END` |  |  |
| `--request REQUEST_ID` |  |  |
| `--session SESSION` |  |  |
| `--agent AGENT` |  |  |
| `--worker WORKER` |  |  |
| `--name NAME` |  |  |
| `--search SEARCH` |  |  |
| `--rank RANK` |  |  |
| `--profile PROFILE` |  |  |
| `--min-ttft-ms MIN_TTFT_MS` | `0` |  |
| `--offset OFFSET` | `0` |  |
| `--limit LIMIT` | `100` |  |
| `--points` |  | Include up to 1000 raw points per metric series |

## `srtctl apply`

Submit job(s) to SLURM

| Argument | Default | Description |
| --- | --- | --- |
| `-f, --file CONFIG` | required | YAML config file, directory, or file:selector for overrides |
| `-o, --output OUTPUT_DIR` |  | Custom output directory for job logs |
| `--sweep` |  | Force sweep mode |
| `-y, --yes` |  | Skip confirmation prompts |
| `--set KEY=VALUE` |  | Override a recipe value by dotted path before validation (repeatable), e.g. --set health_check.max_attempts=720 or --set 'roles.prefill.args.dist-timeout=1800'. Values parse as YAML scalars or lists; mappings stay literal strings. On override files the value is written into base and every variant. |
| `--unset KEY` |  | Remove a recipe key by dotted path before validation (repeatable), e.g. --unset health_check |
| `--setup-script SETUP_SCRIPT` |  | Custom setup script in configs/ |
| `--tags TAGS` |  | Comma-separated tags |
| `--serve-only` |  | Deploy the inference endpoint without running a benchmark; keep serving until the job is cancelled. |
| `--json` |  | Emit one JSON line per submission on stdout; prose output goes to stderr. |
| `--mock` |  | Stub sbatch and spawn a detached mock worker that runs the full SweepOrchestrator locally. For testing external harnesses without cluster access. |
| `--mock-tick-s MOCK_TICK_S` | `0.2` | Per-phase wall time used by the detached mock worker. |
| `--no-preflight` |  | Skip the pre-submit model.path / model.container / telemetry filesystem checks. Useful when those paths only exist on compute nodes (e.g. node-local NVMe like /scratch/models/...) and not on the node invoking srtctl. The framework itself will still fail loudly at runtime if a path is genuinely missing on the compute node. |

## `srtctl dry-run`

Validate without submitting

| Argument | Default | Description |
| --- | --- | --- |
| `-f, --file CONFIG` | required | YAML config file, directory, or file:selector for overrides |
| `-o, --output OUTPUT_DIR` |  | Custom output directory for job logs |
| `--sweep` |  | Force sweep mode |
| `-y, --yes` |  | Skip confirmation prompts |
| `--set KEY=VALUE` |  | Override a recipe value by dotted path before validation (repeatable), e.g. --set health_check.max_attempts=720 or --set 'roles.prefill.args.dist-timeout=1800'. Values parse as YAML scalars or lists; mappings stay literal strings. On override files the value is written into base and every variant. |
| `--unset KEY` |  | Remove a recipe key by dotted path before validation (repeatable), e.g. --unset health_check |

## `srtctl render`

Render the exact sbatch script `srtctl apply` would submit, plus the staged recipe, into --to DIR, and print the script path as the last line of stdout. For launchers that must own the sbatch call themselves (e.g. a harness whose contract is `exec sbatch --parsable ...`).

| Argument | Default | Description |
| --- | --- | --- |
| `-f, --file CONFIG` | required | YAML config file, directory, or file:selector for overrides |
| `-o, --output OUTPUT_DIR` |  | Custom output directory for job logs |
| `--sweep` |  | Force sweep mode |
| `-y, --yes` |  | Skip confirmation prompts |
| `--set KEY=VALUE` |  | Override a recipe value by dotted path before validation (repeatable), e.g. --set health_check.max_attempts=720 or --set 'roles.prefill.args.dist-timeout=1800'. Values parse as YAML scalars or lists; mappings stay literal strings. On override files the value is written into base and every variant. |
| `--unset KEY` |  | Remove a recipe key by dotted path before validation (repeatable), e.g. --unset health_check |
| `--to RENDER_DIR` | required | Directory to render into |
| `--setup-script SETUP_SCRIPT` |  | Custom setup script in configs/ |
| `--serve-only` |  | Render a serve-only job: deploy the endpoint and keep serving until the job is cancelled. |
| `--no-preflight` |  | Skip the pre-render model.path / model.container / telemetry filesystem checks. |

## `srtctl preflight`

Check model and container availability without submitting

| Argument | Default | Description |
| --- | --- | --- |
| `-f, --file CONFIG` | required | YAML config file, or file:selector for overrides |
| `--set KEY=VALUE` |  | Override a recipe value by dotted path before validation (repeatable), e.g. --set health_check.max_attempts=720 or --set 'roles.prefill.args.dist-timeout=1800'. Values parse as YAML scalars or lists; mappings stay literal strings. On override files the value is written into base and every variant. |
| `--unset KEY` |  | Remove a recipe key by dotted path before validation (repeatable), e.g. --unset health_check |

## `srtctl monitor`

Live dashboard for srt-slurm jobs

| Argument | Default | Description |
| --- | --- | --- |
| `args` |  |  |

## `srtctl status-server`

Run the native status collector that reporting.status.endpoint can point at

| Argument | Default | Description |
| --- | --- | --- |
| `--host HOST` | `127.0.0.1` | Bind address (default: 127.0.0.1; use 0.0.0.0 so compute nodes can reach it) |
| `--port PORT` | `8080` | Listen port (default: 8080) |
| `--db DB` |  | SQLite file for jobs and events (default: ~/.local/state/srtctl/status.db) |
| `--token-env VAR` | `SRTCTL_STATUS_TOKEN` | Environment variable holding the write token; grants every route (default: SRTCTL_STATUS_TOKEN) |
| `--read-token-env VAR` | `SRTCTL_STATUS_READ_TOKEN` | Environment variable holding a read-only token for GET routes (default: SRTCTL_STATUS_READ_TOKEN) |
| `--allow-unauthenticated` |  | Listen beyond loopback with no token set; only for a network that is trusted end to end |
| `--cors-origin ORIGIN` |  | Let the UI served from this origin call the API from a browser (repeatable; '*' allows any origin, including a page opened from a file). Read-only routes only. Off by default |

## `srtctl resolve-override`

Resolve override YAML into specialised files without submitting

| Argument | Default | Description |
| --- | --- | --- |
| `-f, --file CONFIG` | required | Override YAML file, or file:selector |
| `--stdout` |  | Print resolved YAML to stdout instead of writing files |
| `--set KEY=VALUE` |  | Override a recipe value by dotted path before validation (repeatable), e.g. --set health_check.max_attempts=720 or --set 'roles.prefill.args.dist-timeout=1800'. Values parse as YAML scalars or lists; mappings stay literal strings. On override files the value is written into base and every variant. |
| `--unset KEY` |  | Remove a recipe key by dotted path before validation (repeatable), e.g. --unset health_check |

## `srtctl diff`

Compare fingerprints from two runs

| Argument | Default | Description |
| --- | --- | --- |
| `path_a` | required | First output dir or lockfile |
| `path_b` | required | Second output dir or lockfile |
| `--verbose` |  | Show all package changes |

## `srtctl check`

Check environment against a fingerprint

| Argument | Default | Description |
| --- | --- | --- |
| `path` | required | Lockfile or output dir to check against |
| `--json` |  | Output as JSON |

## `srtctl schema-docs`

Regenerate the generated docs (schema reference, JSON Schemas, CLI reference) from the code

| Argument | Default | Description |
| --- | --- | --- |
| `--check` |  | Exit 1 if a checked-in generated file is stale instead of rewriting it (used by CI) |
| `--docs-dir DOCS_DIR` |  | Docs directory to write into (default: the checkout's docs/) |

## `srtctl schema`

Print the JSON Schema for recipes (or srtslurm.yaml with --cluster)

| Argument | Default | Description |
| --- | --- | --- |
| `--cluster` |  | Emit the schema for srtslurm.yaml instead of a recipe |
| `--output OUTPUT` |  | Write the JSON Schema to this path instead of stdout |

## `srtctl skill`

Install the in-package agent skill (how to drive srtctl) for Claude Code, Codex, or Cursor

| Argument | Default | Description |
| --- | --- | --- |
| `--target TARGET` | required | Which agent's project skill layout to write One of: `claude`, `codex`, `cursor`. |
| `--root ROOT` |  | Project root to install under (default: the current directory) |
| `--print` |  | Print the skill instead of writing it |

## `srtctl migrate`

Rewrite a pre-2.0 recipe (plain, override, sweep, or lock file) into the current schema version

| Argument | Default | Description |
| --- | --- | --- |
| `-f, --file MIGRATE_FILES` | required | Recipe YAML to migrate; a directory is walked recursively (repeatable) |
| `--in-place` |  | Rewrite the file(s) instead of printing |
| `--output OUTPUT` |  | Write the migrated recipe to this path (single file only; default: print to stdout) |
