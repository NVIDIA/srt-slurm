## Summary

Building the `random` dataset tokenizes and decodes every prompt, which costs minutes per run at long ISL and repeats at every concurrency point of a sweep. This adds an opt-in `benchmark.dataset_cache_dir` that does the work once and reuses it on later runs.

- The host directory is created if missing and mounted into the benchmark container at `/sa-bench-dataset-cache`, so cache files written by one job are reused by the next. Applies to `dataset_name: random` only; leaving the field unset preserves today's regenerate-every-run behavior.
- Cache files are named for what you need when deciding what to delete by hand:

```
<served model name>_isl<ISL>_osl<OSL>_n<prompt count>_<digest>.pkl
my-model-fp8_isl8192_osl1_n32_3f9ac2b1.pkl
```

  `my-model-fp8` is `model.path` as served, `n32` is how many prompts the dataset holds (`concurrency × num_prompts_mult`, so warmup and measured runs get separate files), and `3f9ac2b1` is the first 8 hex chars of a SHA-256 over the rest of the cache key: seed, range ratio, prefix length, chat-template flag and tokenizer. Without the digest, datasets differing only in those would silently overwrite each other. A separate file per prompt count is expected rather than wasteful: `num_prompts` changes how many draws come out of the seeded RNG, so a dataset built for one concurrency is not a subset of another.
- Each file stores its full parameter set plus a SHA-256 fingerprint of the tokenizer files, verified on load. Swapping a checkpoint in at the same path invalidates the cache instead of silently reusing prompts built for the old tokenizer. Weights are not hashed; they never affect prompt generation.
- Caching is best effort: a corrupt file, read-only directory or full filesystem logs a warning and falls back to generating. Writes are atomic so concurrent sweeps sharing a directory cannot read a partial file.

```yaml
benchmark:
  type: "sa-bench"
  isl: 8192
  osl: 1
  concurrencies: "1x2x4x8x16"
  dataset_cache_dir: "/lustre/shared/sa-bench-cache"
```

## Filling the cache up front: `srtctl cache-inputs`

A cache still has to be built once, and by default the first job that needs a dataset pays for it while holding GPUs. The new `srtctl cache-inputs` command moves that cost off the allocation: it reads a recipe, works out every dataset its benchmark loop would build, and generates them into `benchmark.dataset_cache_dir` before the job is ever submitted.

```bash
srtctl cache-inputs -f config.yaml            # build everything the recipe needs
srtctl cache-inputs -f config.yaml --dry-run  # list the datasets and print the srun command
```

- One dataset per prompt count the run asks for. A recipe with `concurrencies: "64x128"`, `num_prompts_mult: 8` and `num_warmup_mult: 1` gives four files — `n=64`, `n=512`, `n=128`, `n=1024` — because warmup and measured runs draw separately from the seeded RNG. Counts that repeat across concurrency levels are built once.
- Only prompt generation runs: one node, no GPUs, no server, no frontend. The prewarm can therefore run while the cluster is busy with the very jobs it is preparing for. Started from inside an existing allocation it attaches to that instead of queueing again.
- Re-running is cheap and safe. Datasets already in the cache are verified by reading them back and skipped, and only the missing ones are generated, so this is also how you resume after a time limit cut a long prewarm short.
- `--time`, `--account` and `--partition` default to the recipe's `slurm` block, and `--num-workers` sets generation parallelism. A recipe with no `dataset_cache_dir`, a non-`random` dataset or a different `benchmark.type` is rejected up front with a message naming the field to fix.

The prewarm is assembled from the pieces a real run uses, so the files it writes are exactly the files the benchmark then looks for: the container command comes from `SABenchRunner.build_command`, bench.sh derives the prompt counts from the same variables as its benchmark loop, and the new `--prewarm-dataset-cache` flag builds the dataset through the same loader and returns before the first request. Any divergence would surface as a cache miss and a regenerated dataset, never as wrong prompts.

Tests cover the plan (which datasets, which mounts, which srun flags, and attaching to an existing allocation) plus a run of the real bench.sh in prewarm mode against a recording stand-in for `python3`, asserting it asks for exactly the prompt counts the planner computed and never contacts a server.

## Reading the worker logs: stage markers

A worker log is one long stream with nothing in it to say which concurrency level or which phase produced a given line, so attributing a stall or an OOM to a specific run means cross-referencing timestamps by hand. SA-Bench now marks its phases directly in the live worker logs:

```
======== [12:34:10] cc=1024 warmup begin ========
...
======== [12:35:58] cc=1024 warmup end ========

======== [12:35:58] cc=1024 benchmark begin ========
...
======== [12:44:14] cc=1024 benchmark end ========
```

- Every prefill, decode and aggregated worker log gets the same four markers per concurrency level, so `grep -n '^========' *_w*.out` prints the timeline of a run and the lines between two markers are exactly what that phase produced.
- The marker is written by `lib/stage_banner.sh`, a small helper next to the existing `lib/profiling.sh`. If a run has no worker logs to annotate the marker goes to the benchmark log instead of being dropped.
- srun now opens `--output` files with `--open-mode=append`. Without it srun keeps writing at its own offset and would overwrite the injected lines. Log paths are unique per job, so for the first open append and truncate are equivalent; the visible difference is that a process restarted against the same path no longer erases what came before it.