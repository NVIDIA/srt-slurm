# Shared Recipe Files (`extends`)

A recipe can list shared partial recipes in a top-level `extends`. srtctl merges them under the
recipe before anything else reads it, so recipes that differ in a few keys keep everything else in
one shared file. The merge rules are the ones [config overrides](overrides.md) use.

## Example

```text
recipes/qwen3/
├── qwen3-0.6b-agg.yaml   # shared: model, engine, worker shape
├── sa-bench-128.yaml     # shared: benchmark settings
├── tp1.yaml              # a recipe built from both
└── variants.yaml         # an override file built from both
```

A shared file is a partial recipe: any recipe keys, none of them required.

```yaml
# sa-bench-128.yaml
schema: 2
slurm:
  time_limit: "00:30:00"
benchmark:
  type: "sa-bench"
  isl: 128
  osl: 128
  concurrencies: [4, 8]
```

The recipe names the files and sets what differs:

```yaml
# tp1.yaml
schema: 2
extends: [qwen3-0.6b-agg.yaml, sa-bench-128.yaml]
name: "qwen3-0.6b-extends-tp1"
benchmark:
  concurrencies: [4, 8, 16]   # replaces the shared list; type, isl and osl stay
```

```bash
srtctl dry-run -f recipes/qwen3/tp1.yaml
srtctl apply -f recipes/qwen3/tp1.yaml
srtctl resolve-override -f recipes/qwen3/tp1.yaml --stdout   # print the merged recipe
```

[`examples/features/extends/`](https://github.com/NVIDIA/srt-slurm/tree/main/examples/features/extends)
holds this layout.

## Rules

- **Order.** `extends` is one path or a list of paths. The files merge left to right, then the
  recipe on top: a later file wins over an earlier one. A file may be listed only once.
- **Merge.** As for overrides: mappings merge key by key, lists and scalars replace the whole value
  (to add one mount, write out the full list), and `null` deletes a key an earlier file set. A
  `null` that deletes nothing stays a `null`, as it would in the file alone
  (`telemetry.dcgm_exporter: null` still keeps the cluster's default GPU exporter out).
- **Only the recipe uses `extends`.** It is a top-level key of the file passed to `-f`, not a key
  inside `base:` or a variant, and a file in `extends` cannot have its own `extends`: every file a
  recipe is built from is listed in that one place.
- **Only the recipe defines jobs.** A file in `extends` cannot hold `base`, `override_*`,
  `zip_override_*` or `sweep`.
- **Paths** are relative to the recipe's directory (to the working directory when the recipe comes
  from stdin, `-f -`). Absolute paths work; environment variables, `~` and globs are not expanded.
- **Schema.** The recipe declares `schema: 2`. A file in `extends` may repeat it with the same value.
- **Validation.** Files in `extends` are partial, so only the merged recipe is validated. A missing
  file, a file that is not a YAML mapping, or a file in `extends` that uses `extends` or defines jobs
  stops srtctl before anything is submitted.

## Override files

In an [override file](overrides.md), the files in `extends` merge under `base`, and each variant is
merged over the result. An override file with `extends` may leave `base` out: the merged files are
then its base, and `:base` selects them.

```yaml
# variants.yaml
schema: 2
extends: [qwen3-0.6b-agg.yaml, sa-bench-128.yaml]

override_tp2:
  roles:
    agg:
      workers: 1
      gpus: 2
      args:
        tensor-parallel-size: 2
```

```bash
srtctl apply -f recipes/qwen3/variants.yaml                 # every variant
srtctl apply -f recipes/qwen3/variants.yaml:override_tp2    # one variant
srtctl apply -f recipes/qwen3/variants.yaml:base            # the shared files alone
```

Variants still do not stack: each one is the base plus that one section.

## Precedence

From lowest to highest; a later step wins where both set a key:

1. the files in `extends`, left to right
2. the recipe itself (its `base:` in an override file)
3. the selected variant (`override_*` or one `zip_override_*` slice)
4. `--set` and `--unset`

Cluster defaults from `srtslurm.yaml` are applied after that and only fill keys that are still
unset, as for any recipe ([Cluster Configuration](cluster-config.md)). A value the recipe or a
variant sets cannot be changed by a file in `extends`; to change it for one run, use `--set`.

## What the job records

srtctl submits the merged recipe, not the files it came from. The `config.yaml` in the job's output
directory is the merged recipe without `extends`, headed by the files it was built from, so the run
can be reproduced without them:

```text
# Composed by srtctl from these files (later files win):
#   /home/me/recipes/qwen3/qwen3-0.6b-agg.yaml
#   /home/me/recipes/qwen3/sa-bench-128.yaml
#   /home/me/recipes/qwen3/tp1.yaml
schema: 2
name: "qwen3-0.6b-extends-tp1"
model:
  ...
```

For an override file, `config.yaml` is the merged override file and `config_<suffix>.yaml` the
variant the job runs ([Output Files](overrides.md#output-files)). Each job of a sweep keeps the same
header.

## Directories

`srtctl apply -f dir/` and `srtctl dry-run -f dir/` skip any file that another file in the directory
lists in `extends`: shared files sit next to the recipes that use them without being submitted.
Every other file is merged with its own `extends` before it is submitted.

## Inspecting the merged recipe

`srtctl resolve-override` writes what each job runs, with `extends` merged and `--set` applied, and
submits nothing. For a recipe it writes `<name>_resolved.yaml` next to the file; for an override
file, one `<name>_<suffix>.yaml` per variant:

```bash
srtctl resolve-override -f tp1.yaml --stdout
srtctl resolve-override -f variants.yaml:override_tp2 --stdout
```

In Python, `load_config`, `SrtConfig.from_yaml` and `validate_config_file` (the check CI runs over
`examples/`) merge `extends` the same way. A tool that chooses shared files at launch can pass them
to `srtctl.core.compose.compose_recipe(document, base_dir=..., layers=[...])`, which merges `layers`
below the recipe's own `extends` under the same rules.
