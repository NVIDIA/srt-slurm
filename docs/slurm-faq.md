# SLURM FAQ

## Cluster Compatibility Settings

Some SLURM clusters don't support certain SBATCH directives. If you encounter errors during job submission, you may need to adjust these settings in your `srtslurm.yaml`.

## GPU Resource Specification

If you see this error when submitting jobs:

```
sbatch: error: Invalid generic resource (gres) specification
```

Your cluster doesn't support the `--gpus-per-node` directive. Disable it:

```yaml
use_gpus_per_node_directive: false
```

This will omit the `#SBATCH --gpus-per-node` directive from generated job scripts while keeping all other functionality intact.

## Segment-Based Scheduling

If you see this error when submitting jobs:

```
sbatch: error: Invalid --segment specification
```

Your cluster doesn't support the `--segment` directive for topology-aware scheduling. Disable it:

```yaml
use_segment_sbatch_directive: false
```

The `--segment` directive ensures all allocated nodes are within the same network segment/switch for optimal interconnect performance between prefill and decode workers. If your cluster doesn't support it, SLURM will still allocate nodes but may scatter them across the cluster.

## Exclusive Node Access

Some clusters require jobs to explicitly request exclusive access to nodes. If your cluster requires this, enable the `--exclusive` directive:

```yaml
use_exclusive_sbatch_directive: true
```

This adds `#SBATCH --exclusive` to the job script, ensuring your job has sole access to the allocated nodes. This is often required on clusters where GPU jobs must not share nodes with other jobs.

## One Container Per Image Per Node

Every `srun` that srtctl launches with a container image also passes
`--container-name=srtctl_<image hash>_<job id>`. The first step that lands on a
node creates the container; later steps of the same job that run the same image
on that node attach to it instead of extracting the image again.

Pyxis/enroot extracts each unnamed container into `ENROOT_RUNTIME_PATH`, which
is tmpfs (host RAM) on most clusters. A disaggregated recipe can put a prefill
worker, a decode worker, the frontend, the benchmark client and the telemetry
exporters on one node from the same image, so an unnamed 40 GiB image costs
150-200 GiB of host RAM per node -- RAM the engines' host KV cache
(`host_cache_size`, `KV_OFFLOADING=dram`) needs, and the reason a context worker
can be OOM-killed once its host cache fills.

- The name includes the job id, so two jobs sharing a node never attach to each
  other's container, and Pyxis removes job-scoped containers at job end.
- Steps launched with `ENROOT_REMAP_ROOT=yes` (the dynamo cold install: workers
  and the dynamo frontend) get a separate container, because a rootfs created
  without the remap cannot serve them. On such recipes a node therefore holds
  two containers (remapped workers/frontend, plain client/services/stage steps),
  not one; recipes whose container already has dynamo installed hold one.
- Steps that write into the rootfs now share it with the engines on that node:
  `setup_script` runs once per step against the same files (make it
  idempotent), and benchmark scripts that `pip install` at run time install
  into the live worker rootfs. The dynamo install lock and sentinel are
  designed for this (see `_serialize_node_install` in `core/schema.py`).
- `--container-mounts` is still passed on every step; Pyxis applies it to the
  attaching step and may warn that the container already exists, which is
  harmless.
- The `Connection Commands` printed at start-up include the name, so an
  interactive `srun ... --pty bash` lands in the workers' rootfs.
