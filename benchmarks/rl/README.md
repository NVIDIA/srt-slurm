# RL framework launchers

One folder per framework, each with a `launch.sh` that srt-slurm runs as a `custom` benchmark command. The repo-root `benchmarks/` folder is mounted in every job container at `/benchmarks`, so a recipe points at `/benchmarks/rl/<framework>/launch.sh`.

The contract a launcher works against is the environment every custom benchmark command receives (see `docs/config-reference.md`, section `custom`):

- `SRT_SERVICE_<NAME>_NODES`, `SRT_SERVICE_<NAME>_IPS`, `SRT_SERVICE_<NAME>_NODE_COUNT` for every service the job launched, `<NAME>` being the service name upper-cased with non-alphanumerics as `_`. The first IP of a `ray` service is its head.
- `SRT_GPUS_PER_NODE`, `SRT_WORKER_NODES`.
- Whatever the recipe sets in `benchmark.env`, applied last.

A launcher owns nothing but the translation from that environment to the framework's own launch command. The framework owns placement, the training loop, and its engines; srt-slurm owns the allocation, the container, the Ray service (or whatever the framework needs brought up), telemetry, and teardown.

| Folder | Framework | Drives |
| --- | --- | --- |
| `miles/` | [Miles](https://github.com/radixark/miles) | a `services[].type: ray` cluster; the Miles launch script's own `MILES_SCRIPT_*` options pass through |

Keep a launcher small and explicit: required inputs fail fast with a message naming the recipe key to set, and `*_DRY_RUN=1` prints the resolved command so `tests/` can check it without the framework installed.
