# benchmarks/

Launchers and clients that are not part of srt-slurm's core: scripts a recipe runs as a `custom` benchmark command. The folder is mounted into every job container at `/benchmarks`, the same way `configs/` is mounted at `/configs`, so a recipe references a script by its container path:

```yaml
benchmark:
  type: custom
  command: /benchmarks/rl/miles/launch.sh
  env: { ... }
```

Core stays generic. What a custom command can rely on is the environment srt-slurm injects (see `docs/config-reference.md`, section `custom`): the frontend and engine endpoints when the job has them, `SRT_SERVICE_<NAME>_NODES` / `_IPS` / `_NODE_COUNT` for every launched service, `SRT_GPUS_PER_NODE`, `SRT_WORKER_NODES`, and whatever the recipe sets in `benchmark.env`.

| Folder | Contents |
| --- | --- |
| `rl/` | RL post-training framework launchers, one folder per framework. `rl/miles/launch.sh` drives [Miles](https://github.com/radixark/miles) against a `services[].type: ray` cluster. See `rl/README.md` and `docs/miles.md`. |

The built-in benchmark types (`sa-bench`, `gsm8k`, ...) keep their scripts inside the package under `src/srtctl/benchmarks/scripts/`, mounted at `/srtctl-benchmarks`; those ship with the wheel. This folder is for what lives with the checkout.
