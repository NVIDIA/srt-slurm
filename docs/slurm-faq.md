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

## Host Memory Used by Container Filesystems

Each Pyxis container step normally extracts its own root filesystem. On clusters
where Enroot stores those filesystems in RAM, colocated workers, frontends and
sidecars using a large image can exhaust host memory before model loading finishes.

Enable job-scoped root filesystem reuse in the recipe:

```yaml
reuse_container: true
```

The default is `false`. A cluster can enable reuse for recipes that omit the
setting by adding `default_reuse_container: true` to `srtslurm.yaml`.
An explicit `reuse_container: false` in a recipe overrides that default.
Reuse requires Pyxis support for
`--container-name=NAME:no_exec`. With reuse enabled, srt-slurm gives containerized steps
using the same image and root-remapping mode the same Pyxis container name.
Steps on a node reuse that node's extracted filesystem; filesystems are never
shared across nodes or jobs. Different images and root-remapping modes receive
different names. The first worker launch prepares its image across all nodes of
that role together, with one task per node; later launches reuse that preparation.
Other steps prepare their own target nodes on first use. Only overlapping
container names and nodes wait for each other, so independent preparations can
run concurrently. Preparation and lock waiting have a shared 10-minute deadline.
A failure names the launching step, image, nodes and log file, and stops the launch.
This prevents concurrent first launches from attaching to a partially extracted image.
Workers and frontends using the built-in Dynamo installer run with root remapping,
so they use a separate filesystem from plain service and benchmark steps.
An explicitly configured `srun_options.container-name` takes precedence over the generated name and leaves
preparation and namespace-sharing behavior under the user's control.

The shared filesystem also shares writes: package installations and changes
inside the image become visible to other steps using it. Setup scripts must be
idempotent, and concurrent package installations must be serialized. Step-specific
commands, mounts and environment variables still apply independently: generated
names use Pyxis's `:no_exec` mode to reuse the filesystem while creating separate
namespaces for each step. Reuse reduces duplicate filesystem memory;
model weights, KV caches, Mooncake pools and process
allocations still need their own host memory budget.

Names contain the Slurm job ID to keep jobs separate. Named container cleanup
depends on cluster configuration: Pyxis removes them at job exit when configured
with `container_scope=job`; otherwise the site's Enroot cleanup policy applies.
See the [Pyxis configuration](https://github.com/NVIDIA/pyxis/blob/107519944221822ea1dace4db8e7234b2eaa4cd5/config.c)
and [job cleanup implementation](https://github.com/NVIDIA/pyxis/blob/107519944221822ea1dace4db8e7234b2eaa4cd5/pyxis_slurmd.c).
The reuse and `no_exec` behavior is implemented in
[`pyxis_slurmstepd.c`](https://github.com/NVIDIA/pyxis/blob/107519944221822ea1dace4db8e7234b2eaa4cd5/pyxis_slurmstepd.c#L1305-L1337).

This is an opt-in adaptation of
[PR #484](https://github.com/NVIDIA/srt-slurm/pull/484)
([source commit](https://github.com/NVIDIA/srt-slurm/commit/318c460f7846e396a03f701acfd97f67e24ded45)).
Try the two-service
[reuse example](https://github.com/NVIDIA/srt-slurm/blob/main/examples/features/reuse-container.yaml)
and inspect its resolved configuration with `srtctl dry-run -f examples/features/reuse-container.yaml`.
