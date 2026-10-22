# Diagnose Mooncake EGM host-memory allocation failures

Submit from the repository root on RapLab:

```bash
sbatch --account=root --partition=batch-r41 --nodelist=vr-r41-cn15 \
  tools/check_mooncake_egm_reclaim.sbatch
```

The job requests one exclusive node, four GPUs, all allocatable host memory
(`--mem=0`), and 20 minutes. It uses one container, a local Mooncake master,
and native EGM donors. It keeps the first 250GiB donor alive while starting a
second donor, reproducing the concurrency that the sequential allocation checks
in [the two-node smoke test](check_mooncake_egm.md) do not exercise.
No inference workers or transfer workload are loaded. All donor payload storage
is Fabric-exportable HOST_NUMA memory, with `protocol=nvlink`, EGM enabled,
and `local_buffer_size=0`. CUDA contexts and driver bookkeeping can still consume
GPU memory; no GPU payload pool is requested.

## Prerequisites and overrides

The RapLab defaults are:

```text
MC_EGM_IMAGE=/home/_shared/tensorrt-llm-release-arm64-f1bdd25-user_imant_glm5-rubin-30428.sqsh
MC_EGM_WHEEL_DIR=/home/rihuo/mooncake_egm_wheels
```

The wheel directory must contain exactly one EGM-enabled Linux ARM64 CUDA 13
Mooncake wheel matching the image's Python ABI. The job mounts this directory
read-only and installs the wheel only inside its container, using
`configs/setup_mooncake_egm.sh --mooncake-only`. See the
[existing wheel requirements](check_mooncake_egm.md#prerequisites) for the
required upstream commits and build options. The launcher records the wheel
name and SHA256, copied scripts, Slurm allocation and driver information.
Override paths with `MC_EGM_IMAGE` and `MC_EGM_WHEEL_DIR` when needed.

## Experiments

By default, the job runs these cases serially. Both donors remain alive until
the second allocation has finished or failed, then they are stopped before the
next case.

| Case | First donor | Second donor | Allocation chunk setting |
| --- | --- | --- | --- |
| `same-numa` | GPU 0, host NUMA 0, 250GiB | GPU 1, host NUMA 0, 250GiB | Upstream default (`MC_MAX_MR_SIZE` unset) |
| `cross-numa` | GPU 0, host NUMA 0, 250GiB | GPU 2, host NUMA 1, 250GiB | Upstream default (`MC_MAX_MR_SIZE` unset) |
| `reverse-gpu` | GPU 1, host NUMA 0, 250GiB | GPU 0, host NUMA 0, 250GiB | Upstream default (`MC_MAX_MR_SIZE` unset) |
| `chunked` | GPU 0, host NUMA 0, 250GiB | GPU 1, host NUMA 0, 250GiB | `MC_MAX_MR_SIZE=17179869184` (16GiB) for both donors |

Each donor sees only its selected physical GPU through `CUDA_VISIBLE_DEVICES`.
Host NUMA placement is explicit; the inventory records CUDA GPU identities and
host NUMA attributes so the assumed GPU/NUMA mapping can be checked.
The inherited `MC_MAX_MR_SIZE` value is recorded, then unset for baseline
cases. `MC_MAX_MR_SIZE` controls each allocation chunk, not the donor's total capacity.
A smaller chunk passing would suggest allocation-size or reclamation behavior;
it does not establish that aggregate serving memory fits.

Cases in one job are not independent: completed allocations can warm driver
pools or change page-cache and fragmentation state. For a cleaner comparison,
run one case per independent job, for example:

```bash
MC_EGM_RECLAIM_CASES=same-numa \
  sbatch --account=root --partition=batch-r41 --nodelist=vr-r41-cn15 \
  tools/check_mooncake_egm_reclaim.sbatch

MC_EGM_RECLAIM_CASES=chunked \
  sbatch --account=root --partition=batch-r41 --nodelist=vr-r41-cn15 \
  tools/check_mooncake_egm_reclaim.sbatch
```

Even separate jobs can inherit host driver pools and caches. Compare the
recorded baseline snapshots and repeat in a reversed order before attributing
a difference to the chunk setting.

## Evidence to inspect

Start with `outputs/mooncake-egm-reclaim-<job_id>/logs/perf-analysis.md`.
The report distinguishes completed allocation failures from missing evidence,
timeouts and incomplete cleanup. A successful diagnostic means its selected
experiments finished with recorded outcomes; it does not require every
allocation to succeed.

The job collects global and per-NUMA memory counters, NUMA VM statistics,
buddy information, process limits, actual process cgroup membership, visible
ancestor memory limits/events, NVIDIA driver parameters and read-only shrinker
counters where accessible. Timed samples and snapshots around each donor's
start help establish whether `Inactive(file)` or `KReclaimable` changed during
allocation. Missing paths and permission failures are retained as evidence gaps.
The job does not run `sudo`, mount debugfs, evict caches, invoke shrinker scans,
reset GPUs or change driver settings. Debugfs may be hidden inside the container;
host snapshots are best-effort and preserve access errors.

Useful comparisons are:

- If whichever donor starts second fails on NUMA 0, while the cross-NUMA case
  succeeds, investigate cumulative NUMA-local capacity and reclaim behavior.
- If failure follows a particular GPU after startup order is reversed, inspect
  its CUDA identity, topology and native error before assuming memory pressure.
- If smaller chunks succeed at the same total capacity, compare buddy counters,
  reclaim activity and driver logs. This is evidence for size-dependent behavior,
  not proof of a specific driver defect.
- Rising cgroup limit/OOM events support a job memory-limit explanation. A
  visible leaf with `memory.max=max` does not rule out a hidden ancestor limit.
- A high `KReclaimable` count alone does not identify NVIDIA-owned memory or
  guarantee that it can be reclaimed for this particular request.

Linux documents these counters in
[/proc memory accounting](https://docs.kernel.org/filesystems/proc.html) and
[cgroup memory accounting](https://docs.kernel.org/admin-guide/cgroup-v2.html).
CUDA's [EGM guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html)
describes HOST_NUMA and Fabric allocations. Mooncake's native allocation behavior
is checked against commit `042ab8a18b0e6ccfe87782261a5556a3dd4446e4`.
The public NVIDIA driver implementation can help interpret reclaimable pools,
but it does not establish the exact allocation/retry behavior of the installed
620.47 driver; the job records that version rather than assuming a source match.

## Configuration and budget

| Variable | Default | Meaning |
| --- | --- | --- |
| `MC_EGM_RECLAIM_POOL_SIZE` | `250GiB` | Capacity of each HOST_NUMA donor |
| `MC_EGM_RECLAIM_CASES` | `same-numa,cross-numa,reverse-gpu,chunked` | Ordered experiment selection |
| `MC_EGM_RECLAIM_TIMEOUT` | `60` | Per-donor allocation deadline, seconds |
| `MC_EGM_RECLAIM_BUDGET` | `600` | Whole experiment budget, seconds |
| `MC_EGM_RECLAIM_SAMPLE_INTERVAL` | `1` | Memory sampling interval, seconds |
| `MC_EGM_RECLAIM_HOLD_SECONDS` | `2` | Hold after allocation for sampling, seconds |
| `MC_EGM_RECLAIM_CHUNK_SIZE` | `16GiB` | `MC_MAX_MR_SIZE` used by `chunked` |
| `MC_EGM_SETUP_TIMEOUT` | `240` | Container wheel setup deadline, seconds |
| `MC_EGM_READY_TIMEOUT` | `60` | Master readiness deadline, seconds |
| `MC_EGM_STEP_TIMEOUT` | `960` | Whole container-step deadline, seconds |
| `MC_EGM_OUTPUT_ROOT` | `$SLURM_SUBMIT_DIR/outputs` | Artifact parent directory |

The [slurm-job-sizing skill](../src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md)
keeps this a short allocation experiment: up to four minutes for wheel setup,
one minute for master readiness and ten minutes for experiments, under a
16-minute step deadline. The 20-minute allocation leaves a cleanup and report
reserve. Increase walltime and dependent deadlines together for longer tests.
The whole-step deadline takes precedence over individual phase deadlines.

Per-run artifacts include:

- `logs/perf-analysis.md`: generated starting point, outcomes, evidence gaps and
  interpretation limits, generated in the actual run's logs directory.
- `logs/reclaim/summary.json`: selected cases, timing and completion/cleanup
  status; per-case JSON, native logs, snapshots and `samples.jsonl` capture
  allocation and reclamation evidence.
- `logs/host-before.json` and `logs/host-after.json`: batch-host snapshots, useful
  when container mount namespaces hide cgroup ancestors or debugfs.
- `logs/master.log`, setup and step logs: native errors and process lifecycle.
- `logs/launch.json`, Slurm snapshots and copied source: reproducibility.

TTFT p50/p95/p99, ITL p50/p99, token throughput, agentic Pareto and SLO metrics
are N/A because no inference traffic runs. No DSight capture is required for
this allocation-only diagnosis; memory samples and native logs answer the goal.
This test does not validate remote NVLink transfers or full TRT-LLM serving
capacity. Use the two-node smoke test for host-transfer correctness.

Share the artifacts after completion:

```bash
JOBID=12345

tar -czf "mooncake-egm-reclaim-${JOBID}.tar.gz" \
  -C outputs "mooncake-egm-reclaim-${JOBID}"
```
