# CLAUDE.md

Development guide for working on this codebase.

## Quick Reference

```bash
# Run lint + tests (recommended)
make check

# Launch-plan snapshots for every example recipe (regenerate after an intended change)
make snapshots

# Just lint
make lint

# Just tests
make test

# Run single test file
uv run pytest tests/test_e2e.py -v

# Run single test
uv run pytest tests/test_e2e.py::TestH100Cluster::test_endpoint_allocation -v

# Auto-fix lint issues
uv run ruff check --fix src/srtctl/
uv run ruff format src/srtctl/
```

## Per-run Performance Analysis

For **every Slurm run you launch or analyze**, write and maintain
`<run_dir>/perf-analysis.md` in the actual job output directory on the cluster.
Read and follow [the report requirements](docs/perf-analysis.md).

Include TTFT p50/p95/p99, ITL p50/p99, throughput and the applicable agentic
Pareto/SLO metrics, with units, sources and missing-data explanations. For a
performance diagnosis, debugging task or improvement, explain how the metrics,
files, DSight/dashboard views and skills advance the goal.

## Using DSight

Before using DSight, read [docs/dsight.md](docs/dsight.md) and load the applicable
[DSight skills](docs/dsight.md#agent-skills) by reading the linked skill files.
This includes building or querying reports, analyzing existing results,
preparing dashboard views, and changing DSight code. The `dsight-query` skill
lives under `.agents/skills/` with the other repository agent skills.

Before preparing or submitting Slurm jobs, use
[slurm-job-sizing](src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md)
to size allocation time, warmup and measured traffic to the task. It covers
short, parallel hypothesis tests and full benchmarks sized from history and phase
timing distributions, while preserving non-preemptible resources.

## Pull Request Descriptions

Write for a reviewer who knows the repository but has no access to the author's
chat, agent session, or corporate environment.

- Explain the problem, the changes introduced by this PR, and their impact on
  users or behavior. Include a concrete before/after example when useful.
- Keep the description scoped to the actual diff. Do not attribute existing
  features, unrelated work, or planned follow-ups to this PR. Explain necessary
  dependencies and link to public repository issues or PRs.
- Do not include corporate-internal links, private dashboards, internal hostnames,
  local or cluster artifact paths, session IDs, or references to earlier chat.
  Replace session-specific shorthand with enough plain-language context for an
  independent reviewer. Summarize relevant evidence directly; use public code,
  tests, documentation, or reproducible examples as supporting references.
- State the validation actually performed and relevant limitations. Keep the
  description concise, and update it when the PR's scope changes.

## Code Style

- **Python 3.10+** - use modern syntax (`|` unions, `match` statements)
- **Ruff** for linting and formatting (config in `pyproject.toml`)
- **Type hints** everywhere - use `ty` for type checking
- **Frozen dataclasses** for configs (`@dataclass(frozen=True)`)
- **Line length**: 120 characters

## Python Patterns

Follow these patterns when extending the codebase:

- **Frozen dataclasses for config** - Use `@dataclass(frozen=True)` for all configuration objects. Immutability prevents accidental mutation and makes code easier to reason about.
- **Protocol over ABC** - Prefer `typing.Protocol` for interface definitions (see `BackendProtocol`). Enables duck typing without inheritance coupling.
- **marshmallow_dataclass for validation** - Combine dataclasses with marshmallow schemas for type-safe config loading with validation. Custom fields (e.g., `BackendConfigField`) handle polymorphic deserialization.
- **Factory classmethods** - Use `@classmethod` named `from_*` for construction (e.g., `RuntimeContext.from_config()`, `SrtConfig.from_yaml()`). Keep `__init__` simple.
- **TYPE_CHECKING guard** - Import type-only dependencies under `if TYPE_CHECKING:` to avoid circular imports. Use string annotations for forward refs.
- **Computed properties** - Use `@property` for derived values instead of storing computed state. See `Topology.gpus_per_prefill`, `RuntimeContext.container_log_dir`.
- **Registry pattern** - Use decorators for extensible registration (`@register_benchmark("sa-bench")`). New implementations just decorate and import.
- **TypedDict for external data** - Use `TypedDict` for typing dicts from JSON/external sources where you can't control the structure.
- **Single source of truth** - Create context objects (like `RuntimeContext`) that compute all derived paths/values once at startup rather than recomputing.
- **testing** - when we make a new significant feature change, we should always add a new test
- **Unused parameters stay untouched** - A hook that ignores an argument just ignores it. Ruff's unused-argument rules are off, so `del name` lines add nothing but noise.

## Design Rules

Read these before adding a feature. Each rule names the existing pattern to reuse. The most common review finding in this repo is a new mechanism where one already exists.

- **Names go in tables, never in branches.** A connector, vendor, router, exporter, or engine name is compared as a string in exactly one place: the table or registry that owns it (`_CONNECTOR_MAP` in `backends/vllm.py`, `@register_service`, `@register_benchmark`, `get_frontend`). Consumers read attributes of the row, not the name. If a change adds `if x == "<name>"` in two files, or repeats a `Literal["a", "b"]` across modules, it is a missing table row or a missing config field.
- **Cluster differences live in `srtslurm.yaml`.** Anything that varies by cluster or hardware (NIC, visible-devices env var, default GPU exporter, sbatch directives, mounts, host setup) is a `ClusterConfig` field in `core/schema.py` following `network_interface` and the `default_*` blocks, read once into `RuntimeContext`. A vendor enum in Python is the wrong tool for this. Never call `load_cluster_config()` from a schema property or a stage: it is uncached and creates a second source of truth.
- **One resolver per overridable setting.** A setting the recipe can set at engine level and override per role (`roles.<role>.args.connector`, DP size) has one accessor on the backend in the `get_config_for_mode` style (`VLLMProtocol.connector_for_mode`, with `kv_connector_for_mode` for the table row and `kv_transfer_config(mode)` for the flag), and every consumer uses it: command builder, process env, frontend, and schema validator. Two readers of the raw fields disagree the moment a role override appears.
- **Frontends own readiness; backends own worker commands and ports.** `core/health.py` and the stage mixins contain no `frontend_type == "..."` checks and no `getattr(frontend, "hook", fallback)` probing. The frontend implements the protocol hook; if a hook is missing, add it to `FrontendProtocol`. A frontend asks a backend a question through a method (`backend.is_grpc_mode(mode)`), never by reading its fields by name.
- **Backends answer through `BackendProtocol`, never through `getattr`.** The stage mixins, schema validators, services, and dry-run ask `backend.mooncake_kv_store`, `backend.failover`, `backend.get_environment_for_mode(mode)`, `backend.get_srun_config().sequential_node_start`; a backend without the feature returns `None` or `{}`. `getattr(backend, "x", default)` and `hasattr(backend, "f")` do not appear in `src/`: they pass on every backend, so a typo or a rename fails silently at runtime. Logic that is genuinely one engine's narrows with `isinstance(backend, VLLMProtocol)` and reads typed fields. When a consumer needs a new answer, add the member to `BackendProtocol` and implement it on every backend, including the neutral default.
- **Every listener a process opens comes from the allocator.** Two processes can share a node in this repo (`nodes: colocate`, DP endpoints), so any port a worker binds (HTTP, bootstrap, side channel, handshake, notify, metrics) is allocated by `NodePortAllocator` and carried on `Process`. An upstream default port left in a generated config is a collision on the first colocated recipe. See Ports in `src/srtctl/core/AGENTS.md`.
- **Modes are not types.** A new `frontend.type`, `services[].type`, or `engine.type` is for a different process with its own launch, health API, and registration model. A different CLI shape, transport, or discovery mode of the same binary is an override inside the existing class: `trtllm_serve` handles aggregate and disaggregated in one type, `sglang-router` picks http or grpc per mode. A new frontend type is one registered module; the only remaining name checks are for Dynamo- and sglang-router-specific features (request tracing, the gateway's own metrics listener, `slow_down`).
- **Check upstream before working around it.** When a change encodes an upstream behavior (what a health endpoint returns, which keys a connector reads, what a flag does), read the upstream source at the version the container ships and cite the commit in the PR. Do not add a probe, shim, or port-scan workaround for something upstream already handles.
- **Reuse the machinery before adding a mechanism.** Services plus `placement` before a bespoke launcher, `roles.<role>.restart` before a wrapper loop, `host_setup` before a setup script that needs the host. The smallest diff that rides existing machinery beats a self-contained new module.
- **A user-visible feature ships complete.** A `tests/` case (dry-run for visible config, mock orchestrator for behavior), a `docs/` page or section, an example recipe under `examples/`, and regenerated `docs/schema-reference.md`. In a stacked PR, a test lives in the layer that introduces the behavior it asserts.

## Where to look

This file holds the rules for every change. Subsystem rules live next to the code in `AGENTS.md` files (read the one for each directory you touch), and the explanations live in `docs/`, which is the source of truth for behavior. Link to a doc page instead of copying it here.

| Area | Rules | Reference |
| --- | --- | --- |
| Frontends, routers, readiness | `src/srtctl/frontends/AGENTS.md` | `docs/architecture.md` |
| Backends, Mooncake, shadow engine recovery | `src/srtctl/backends/AGENTS.md` | `docs/mooncake-kv-store.md`, `docs/shadow-engine-recovery.md` |
| Services, pools, services-only jobs | `src/srtctl/services/AGENTS.md` | `docs/services.md`, `docs/pools.md` |
| Ports, liveness, cleanup, resources | `src/srtctl/core/AGENTS.md` | `docs/architecture.md` |
| Orchestrator stages, host setup | `src/srtctl/cli/AGENTS.md` | `docs/cli.md` |
| Status reporting | `src/srtctl/status_server/AGENTS.md` | `docs/monitoring.md` |
| Benchmarks | `src/srtctl/benchmarks/AGENTS.md` | `docs/config-reference.md` |
| DSight reports, queries and analysis | `src/srtctl/dsight/AGENTS.md` | `docs/dsight.md`, `docs/dsight-storage.md` |
| Tests, mock orchestrator, snapshots | `tests/AGENTS.md` | `tests/README.md` |
| Documentation | `docs/AGENTS.md` | `docs/README.md` |
| Recipe fields | `docs/schema-reference.md` (generated) | `docs/config-reference.md` |

Procedures that recur are skills under `.agents/skills/` (`add-config-field`, `validate-without-cluster`, `design-rule-sweep`). Review criteria are in `REVIEW.md`.

When asked to write or refine a performance benchmark or optimization goal, use
[perf-goal-writer](.agents/skills/perf-goal-writer/SKILL.md)
to define its targets, baseline, change boundaries and evidence requirements.

Several Design Rules are enforced by `tests/test_design_rules.py`. Its baselines list code that predates a rule and may only shrink: fix a baselined site and delete its entry, never add one to make a change pass.

## Key Concepts

### RuntimeContext

Single source of truth for computed paths. Created once at job start:

```python
runtime = RuntimeContext.from_config(config, job_id)
runtime.log_dir          # /path/to/logs/12345_1P_4D_...
runtime.head_node_ip     # 10.0.0.1
runtime.container_mounts # List of mount strings
```

### Endpoint Allocation

Maps logical workers to physical nodes/GPUs:

```python
endpoints = allocate_endpoints(
    num_prefill=2, num_decode=4, num_agg=0,
    gpus_per_prefill=8, gpus_per_decode=4, gpus_per_agg=0,
    gpus_per_node=8,
    available_nodes=("node0", "node1", "node2"),
)
# Returns List[Endpoint] with node assignments and GPU indices
```

## Validating Without a Cluster

Nothing here needs SLURM. From cheapest to most complete:

1. `uv run srtctl dry-run -f <recipe>`: resolved config, mounts, env, srun options, and the sbatch script.
2. `tests/test_dry_run.py`, `tests/test_render.py`: assertions on that output.
3. `src/srtctl/mock.py` (`run_mock_sweep`): the real `SweepOrchestrator` with srun, health checks and hostnames faked; `tests/test_mock_sweep.py` and `tests/test_pools.py` use it.
4. `tests/test_launch_snapshots.py`: every example recipe through the mock orchestrator, with each srun call (placement, env, mounts, command) compared to `tests/snapshots/launch/`. A change that alters what runs on the cluster must regenerate them with `make snapshots`, so the PR diff shows the launch change.

See `.agents/skills/validate-without-cluster/SKILL.md`.

## Common Tasks

### Adding or Changing Any Config Field

`docs/schema-reference.md` is generated from the dataclasses in `core/schema.py` and `backends/`. After adding, renaming, or re-typing a field, run `uv run srtctl schema-docs` and commit the result; CI and `tests/test_schema_docs.py` fail when the file is stale. Put the field's description in the class docstring `Attributes:` block or in a `#` comment directly above the field so it lands in the generated table.

### Adding Config That Affects srun (Mounts, Env Vars, Options)

When adding new config fields that affect what gets passed to srun (environment variables, container mounts, srun options), you must also update:

1. `show_config_details()` in `src/srtctl/cli/submit.py` -- this renders all mounts/env/options in `srtctl dry-run` output so users can verify config before submitting
2. `tests/test_dry_run.py` -- add test cases verifying the new config appears in dry-run output

Config sources that feed into dry-run display:
- **Mounts**: `config.extra_mount`, `config.container_mounts`, `default_mounts` from srtslurm.yaml
- **Env vars**: `config.environment` (global), `roles.<role>.env` (read by the engine from the roles bound onto it at load)
- **srun options**: `config.srun_options`
- **Host setup**: `config.host_setup`, `default_host_setup` from srtslurm.yaml

Adding a backend, frontend or router mode, service kind, or benchmark: follow the checklist in that package's `AGENTS.md`.

## Testing

Tests are in `tests/`; `make check` runs lint (Ruff and a blocking `ty check`), the schema-doc check, and all tests. See `tests/AGENTS.md` for which suite covers what.

## Debugging

### Check Generated Commands

`srtctl dry-run` shows the sbatch script, all container mounts (with source labels), environment variables (global and per-mode), and srun options:

```bash
srtctl dry-run -f config.yaml
```

### Find Full srun Commands at Runtime

The full srun command (with all mounts, env vars, and flags) is logged at INFO level in the sweep log:

```bash
tail -f outputs/<job_id>/logs/sweep_<job_id>.log | grep "srun command"
```

Per-worker env vars and commands are also logged individually (search for `Env:` and `Command:` lines).
