# Config Overrides

Config overrides let you define a single YAML file with a shared `base` configuration and multiple named variants. Each variant is submitted as an independent SLURM job.

## Table of Contents

- [Overview](#overview)
- [base + override\_\*](#base--override_)
- [zip\_override\_\*](#zip_override_)
- [Selector Syntax](#selector-syntax)
- [Combining All Modes](#combining-all-modes)
- [Resolving Overrides Without Submitting](#resolving-overrides-without-submitting)
- [Output Files](#output-files)
- [Tips](#tips)

---

## Overview

An override config file has a `base` top-level key plus one or more variant keys:

| Key prefix | Variants generated | Combination |
|---|---|---|
| `override_<name>` | 1 | Manual — you write the values |
| `zip_override_<name>` | N | Zip — list leaves zipped in parallel |

Override files are auto-detected by the presence of a `base` key:

```bash
srtctl apply -f config.yaml       # submit all override_* and zip_override_* variants (base excluded)
srtctl apply -f config.yaml:base  # submit base only
srtctl dry-run -f config.yaml     # preview without submitting
```

> **Note:** Running without a selector submits every `override_*` and `zip_override_*` variant but **not** the `base` config. Use `:base` to submit the base config explicitly.

---

## base + override\_\*

Each `override_*` section is **deep-merged** with `base`, producing one job per section.

```yaml
schema: 2
base:
  name: "my-job"
  model:
    path: "deepseek-r1"
    container: "sglang:latest"
    precision: "fp8"
  resources:
    gpu_type: "gb200"
    gpus_per_node: 8
  engine: sglang
  roles:
    prefill:
      nodes: 1
      workers: 1
    decode:
      nodes: 1
      workers: 1
  benchmark:
    type: "sa-bench"
    isl: 1024
    osl: 1024
    concurrencies: [4, 8]

# Variant 1: lower memory usage
override_lowmem:
  roles:
    prefill:
      args:
        mem-fraction-static: 0.75
    decode:
      args:
        mem-fraction-static: 0.75

# Variant 2: higher concurrency
override_highconc:
  benchmark:
    concurrencies: [16, 32, 64]
```

Running `srtctl apply -f config.yaml` submits **2 jobs**: `my-job_lowmem`, `my-job_highconc` (base excluded).
To also submit the base config: `srtctl apply -f config.yaml:base`.

### Naming

If an `override_*` section includes a `name:` field, it is used as the job name. Otherwise, the name is auto-generated as `{base_name}_{suffix}`:

```yaml
override_maxtpt:
  name: "my-job-max-throughput"   # used as-is
  ...

override_lowmem:                  # no name → auto-generates "my-job_lowmem"
  ...
```

### Deep merge rules

- **dict** → merged recursively (unmentioned keys are kept from base)
- **list** → fully replaced (not appended)
- **scalar** → override replaces base
- **`null`** → deletes the key from the result

---

## zip\_override\_\*

`zip_override_*` sections generate **N jobs** by zipping list-valued leaves in parallel — like Python's `zip()`. Use this to sweep a set of parameters that belong together.

```yaml
base:
  name: "my-job"
  ...

zip_override_tp_sweep:
  name: ["my-job-tp4", "my-job-tp8", "my-job-tp16"]
  roles:
    prefill:
      args:
        tensor-parallel-size: [4, 8, 16]
        mem-fraction-static: [0.85]      # length-1 → broadcast to all 3
    decode:
      args:
        tensor-parallel-size: [4, 8, 16]
  benchmark:
    concurrencies: [[4, 8], [4, 8], [4]] # list-of-list → literal list per variant
```

This generates **3 jobs**:

| Variant | tensor-parallel-size | mem-fraction-static | concurrencies |
|---|---|---|---|
| `tp_sweep_0` | 4 | 0.85 | `[4, 8]` |
| `tp_sweep_1` | 8 | 0.85 | `[4, 8]` |
| `tp_sweep_2` | 16 | 0.85 | `[4]` |

### Zip rules

**List leaf → zip dimension.** Every leaf whose value is a list becomes a zip dimension:
```yaml
tensor-parallel-size: [4, 8, 16]   # zip dimension, N=3
```

**Length-1 list → broadcast.** A list with a single element is repeated for all N variants:
```yaml
mem-fraction-static: [0.85]        # broadcast to [0.85, 0.85, 0.85]
```

**Scalar → broadcast.** A plain scalar value applies unchanged to all variants:
```yaml
trust-remote-code: true            # same for every variant
```

**List-of-list → literal list.** Wrap a list in another list to pass it as a literal value:
```yaml
concurrencies: [[4, 8], [4, 8, 16]]   # variant 0 gets [4,8], variant 1 gets [4,8,16]
```

**Incompatible lengths raise an error.** All non-broadcast lists must have the same length:
```yaml
# ERROR: lengths 2 and 3 are incompatible
tensor-parallel-size: [4, 8]
nodes: [1, 2, 3]
```

### Auto-naming

If `name` is not a zip dimension, variant names are auto-generated as `{base_name}_{group}_{i}`:
```yaml
zip_override_tp_sweep:
  roles:
    prefill:
      args:
        tensor-parallel-size: [4, 8]
# generates: my-job_tp_sweep_0, my-job_tp_sweep_1
```

Provide a `name` list to set names explicitly:
```yaml
zip_override_tp_sweep:
  name: ["job-tp4", "job-tp8"]
  roles:
    prefill:
      args:
        tensor-parallel-size: [4, 8]
```

---

## Selector Syntax

Submit a specific variant instead of all of them using `-f config.yaml:<selector>`:

```bash
# base only
srtctl apply -f config.yaml:base

# single override_ variant
srtctl apply -f config.yaml:override_lowmem

# all variants in a zip group
srtctl apply -f config.yaml:zip_override_tp_sweep

# single variant by 0-based index
srtctl apply -f config.yaml:zip_override_tp_sweep[0]
```

### Glob patterns

Selectors support shell-style glob patterns (`*`, `?`). The pattern is matched against all `override_*` and `zip_override_*` key names — `base` is always excluded regardless of the pattern.

```bash
# all keys containing "maxtpt" (both override_* and zip_override_*)
srtctl apply -f config.yaml:*maxtpt*

# all override_maxtpt_* keys only
srtctl apply -f config.yaml:override_maxtpt*

# all zip groups
srtctl apply -f config.yaml:zip_override_*
```

A pattern that matches nothing raises a clear error listing the available keys.

Always preview first with `dry-run`:
```bash
srtctl dry-run -f config.yaml:*maxtpt*
```

---

## Combining All Modes

You can mix `override_*` and `zip_override_*` in a single file:

```yaml
base:
  name: "my-job"
  ...

override_lowmem:
  roles:
    prefill:
      args:
        mem-fraction-static: 0.75
    decode:
      args:
        mem-fraction-static: 0.75

zip_override_tp_sweep:
  roles:
    prefill:
      args:
        tensor-parallel-size: [4, 8]
    decode:
      args:
        tensor-parallel-size: [4, 8]
```

`srtctl apply -f config.yaml` submits **3 jobs**: `lowmem` + `tp_sweep_0` + `tp_sweep_1` (base is excluded unless you pass `:base`).

---

## Resolving Overrides Without Submitting

`srtctl resolve-override` expands an override file and writes the specialised YAML for each variant — without submitting any jobs. It is useful for inspecting exactly what config will be submitted, committing expanded configs to version control, or feeding them into other tooling.

```bash
# Write all variants next to the source file
srtctl resolve-override -f config.yaml

# Write a single variant
srtctl resolve-override -f config.yaml:override_lowmem

# Print to stdout instead of writing files
srtctl resolve-override -f config.yaml --stdout

# Preview a specific zip variant
srtctl resolve-override -f config.yaml:zip_override_tp_sweep[0] --stdout
```

The resolved YAML preserves field order and comments from the source file:

- **Field order** — base fields appear first, in their original order. Fields that are only present in an `override_*` section are appended at the end.
- **Comments** — inline and block comments from both `base` and `override_*` sections are kept in the output. For `zip_override_*` variants, base comments are preserved; the zip section's list comments are not carried over (they reference list elements that no longer exist after slicing).

Output files are written next to the source file using the same naming convention as `apply`:

```text
config.yaml                 # source
config_lowmem.yaml          # override_lowmem resolved
config_tp_sweep_0.yaml      # zip_override_tp_sweep variant 0
config_tp_sweep_1.yaml      # zip_override_tp_sweep variant 1
```

---

## Output Files

Each submitted job gets its own directory under `outputs/<job_id>/`:

```text
outputs/6717/
├── config.yaml            # original override YAML (for reference)
├── config_tp_sweep_0.yaml # resolved config for this specific variant
├── sbatch_script.sh       # generated SLURM script
├── 6717.json              # job metadata
└── logs/
    └── sweep_6717.log
```

---

## Tips

- Use `dry-run` before any real submission to verify expansion
- Put shared defaults in `base` to keep variants minimal
- Use `zip_override_*` when parameters belong together (e.g. tp-size + node count)
- Override files are `schema: 2` documents like any other recipe; the v1 layout is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it, variants included
- Use `override_*` for one-off named configurations
- Broadcast (`[value]`) avoids repeating the same value across all list entries
- Use `override_<glob>*` to run a named subset without listing each variant explicitly
