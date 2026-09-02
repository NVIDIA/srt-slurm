# Auxiliary Services

A generic top-level `auxiliary_services` block for declaring bolt-on sidecar processes -- persistent
background services launched alongside the main inference workers -- without needing a core srtctl
code change every time. The same role Kubernetes' `podTemplate.spec.containers` plays: an arbitrary
component you can wire in from the recipe.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Discovery Environment](#discovery-environment)
- [Launch Order](#launch-order)
- [Building From Source](#building-from-source)
- [Example: dynamo.thunderagent_router](#example-dynamothunderagent_router)
- [Validation](#validation)
- [Limitations](#limitations)

---

## Overview

Before this feature, adding a persistent sidecar to a job meant hand-writing a bespoke stage --
that's how Tachometer telemetry collection is implemented today (`TelemetryStageMixin`).
`auxiliary_services` generalizes that shape into config: name a service, give it a command (and,
optionally, a container image, environment variables, and a git source to build first), and srtctl
launches it as a tracked process next to the rest of the job.

Typical use: an experimental component that isn't part of any released Dynamo image yet -- built
from a specific PR ref, launched with its own CLI flags, but still needing to talk to the same
etcd/nats the rest of the job uses.

`auxiliary_services` works on both srtctl execution paths, with two separate implementations
behind the same config block:

- **Real multi-node `sbatch`/SLURM path** (`srtctl apply`, no `--bash`): implemented by
  `AuxiliaryServiceStageMixin` in `src/srtctl/cli/mixins/auxiliary_stage.py`, used by
  `SweepOrchestrator`. Every service launches via `srun` on the **head node only** (the same
  placement as Tachometer) -- there is exactly one instance regardless of how many worker nodes
  the job has, so `source`/`build_command` build-once semantics are trivial. `command` and
  `build_command` run inside the service's container (`container_image`, defaulting to the job's
  main container); the git clone itself runs on the bare head-node host (not containerized), since
  git/network access is a host concern and the job container isn't guaranteed to have git.
  Crash detection and cleanup are handled by the same `ProcessRegistry` that watches every other
  process in the job (`start_process_monitor`, `registry.cleanup()`) -- no extra bookkeeping.
- **`--bash` dev-mode path** (`srtctl apply --bash`): implemented by `AuxiliaryServiceStageMixin`
  in `src/srtctl/render/direct_stages/auxiliary_stage.py`, used by `DirectRunner`. Every service
  runs as a plain local subprocess (no `srun`, no containers) on the machine running `--bash`, and
  a service that exits within ~2 seconds of launch fails the whole run (`self._die(...)`) --
  appropriate for a single foreground process where a dead sidecar usually means a real
  misconfiguration. The real path instead treats a dead sidecar as best-effort (`critical=False`),
  matching how Tachometer and the DCGM/node exporters already behave there: a telemetry-adjacent
  sidecar must never tear down a multi-node benchmark run.

Both paths launch services in the order they're declared in the YAML list and share the same
discovery-env injection (`ETCD_ENDPOINTS`/`NATS_SERVER`) -- see [Launch Order](#launch-order) and
[Discovery Environment](#discovery-environment) below. The two implementations are kept separate
rather than shared because they operate on different shapes (plain dicts off a JSON plan for the
dev-mode stage's stdlib-only package vs. `AuxiliaryServiceConfig` dataclasses for the real path)
and launch through entirely different mechanisms (local `Popen` vs. `srun`).

## Quick Start

```yaml
auxiliary_services:
  - name: my-sidecar
    command: ["python3", "-m", "my_package.my_sidecar", "--port", "9000"]
```

That's the whole contract: `name` (unique, used for logs) and `command` (argv, not
shell-interpreted) are the only required fields. Everything else defaults to something
reasonable -- the service runs in the job's main container, inherits `ETCD_ENDPOINTS`/`NATS_SERVER`
discovery env, and starts after workers and the frontend are healthy.

## Configuration Reference

```yaml
auxiliary_services:
  - name: my-sidecar                    # required, unique across the list
    command: ["python3", "-m", "pkg"]   # required, non-empty argv
    container_image: my-image.sqsh      # optional, default: job container
    env:                                # optional, default: {}
      MY_FLAG: "1"
    source:                             # optional: build from git before launching
      git: https://github.com/org/repo
      rev: <commit-sha-or-refs/pull/N/head>   # required, must be immutable
      path: subdir                      # optional, subdirectory to build/launch from
    build_command: ["bash", "-lc", "pip install -e ."]  # required if source is set
    inherit_discovery_env: true         # optional, default: true
```

| Field                    | Required | Default          | Notes                                                                 |
| ------------------------ | -------- | ---------------- | ---------------------------------------------------------------------- |
| `name`                   | yes      | --                | Must be unique across `auxiliary_services`. Used for `<name>.log`. |
| `command`                | yes      | --                | Argv passed directly to the process (no shell). |
| `container_image`        | no       | job container     | Same alias resolution as the job's main container. |
| `env`                    | no       | `{}`              | Merged on top of discovery env (below). |
| `source`                 | no       | none              | Git repo to clone before building. See [Building From Source](#building-from-source). |
| `build_command`          | no       | none              | Argv run once, from the cloned `source`, before `command` is launched. srtctl warns if `source` is set without `build_command` -- almost always a mistake. |
| `inherit_discovery_env`  | no       | `true`            | Inject `ETCD_ENDPOINTS`/`NATS_SERVER`. |

## Discovery Environment

When `inherit_discovery_env: true` (the default), srtctl injects the same two environment variables
the `dynamo` frontend type already receives:

```
ETCD_ENDPOINTS=http://<infra_node_ip>:2379
NATS_SERVER=nats://<infra_node_ip>:4222
```

This lets an auxiliary service register itself against the same etcd/nats instance the rest of the
job's Dynamo components use, without you having to compute the infra node address yourself. Set
`inherit_discovery_env: false` for services that don't talk to Dynamo's discovery plane at all.

## Launch Order

`auxiliary_services` are started once workers and the frontend are confirmed healthy, in the order
they're declared in the YAML list -- no dependency graph, no separate ordering config. If one
service needs to be up before another starts, just declare it first.

The two paths differ in what "started" means before moving to the next service in the list:

- **`--bash` dev-mode**: a short startup-poll -- the process must still be alive ~2 seconds after
  launch -- before the next service starts.
- **Real `sbatch`/SLURM path**: the `srun` launch call returning is enough to move on; ongoing
  health is left to the shared `ProcessRegistry` background monitor, the same as every other
  process in the job.

## Building From Source

`source` + `build_command` clone and build a service once before it's launched -- the same
git-clone-then-build shape `DynamoConfig`'s `hash`/`top_of_tree` source install uses for Dynamo
itself, generalized to arbitrary auxiliary services:

```yaml
source:
  git: https://github.com/ai-dynamo/dynamo
  rev: refs/pull/14000/head   # commit SHA, tag, or refs/pull/<n>/head -- never a branch name
  path: lib/bindings/python   # optional: subdirectory to build/launch from
build_command: ["bash", "-lc", "maturin develop --uv && pip install -e ."]
```

`rev` must be an immutable ref -- srtctl rejects `main`, `master`, and `HEAD` at load time, since a
moving branch makes the build non-reproducible and defeats any caching a future revision of this
feature might add. Use a commit SHA once you have one, or `refs/pull/<n>/head` while iterating
against an open, unmerged PR.

## Example: dynamo.thunderagent_router

The motivating use case: running Dynamo's experimental `dynamo.thunderagent_router` (a
program-aware admission-control router) as a sidecar next to the normal `dynamo` frontend/backend
workers, built from an unmerged PR:

```yaml
frontend:
  type: dynamo

auxiliary_services:
  - name: thunderagent-router
    source:
      git: https://github.com/ai-dynamo/dynamo
      # NOTE: switch this to a merged commit SHA once the upstream PR lands.
      rev: refs/pull/14000/head
    build_command:
      - bash
      - -lc
      - "cd lib/bindings/python && maturin develop --uv && cd ../../.. && pip install -e ."
    command:
      - python3
      - -m
      - dynamo.thunderagent_router
      - --endpoint
      - dyn://namespace.component.endpoint
      - --model-name
      - my-model
      - --router-block-size
      - "64"
      - --pause-threshold
      - "0.9"
      - --soft-demote-threshold
      - "0.75"
      - --pause-target
      - "0.5"
      - --resume-hysteresis
      - "0.1"
    # inherit_discovery_env: true (default) -- the router registers against
    # the same etcd/nats the dynamo frontend and workers already use.
```

`build_command` runs once, from the cloned checkout, before `command` starts. Because
`inherit_discovery_env` defaults to `true`, the router picks up `ETCD_ENDPOINTS`/`NATS_SERVER`
automatically and needs no extra env configuration to see the same Dynamo namespace as the rest of
the job.

## Validation

srtctl rejects the following at load time (dry-run), before submitting:

- Empty or duplicate `name` across the list.
- Empty `command`, or a `command` containing a blank argument.
- `build_command` explicitly set to `[]` (omit the field entirely instead).
- `source.git` or `source.rev` empty, or `source.rev` set to a moving branch name (`main`, `master`, `HEAD`).

`source` set without `build_command` is a warning, not a hard error -- there are edge cases (a
prebuilt binary checked into the repo) where nothing needs building.

## Limitations

- `srtctl dry-run` shows each service's command/env/source (see `show_config_details()` in
  `src/srtctl/cli/submit.py`) so you can verify the config before submitting. On the real
  `sbatch`/SLURM path it also notes that every service launches on the head node only.
- Declared order is launch order, not a readiness gate. If your service needs another auxiliary
  service to be *ready* (not just started) before it can boot, build that wait into your own
  `command` (e.g. poll a health endpoint before doing real work).
- On the real `sbatch`/SLURM path, every auxiliary service runs as a single instance on the head
  node -- there's no way to fan a service out across worker nodes today. If you need a per-worker
  sidecar, this feature doesn't cover that yet.
