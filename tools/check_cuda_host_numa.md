# CUDA host-NUMA allocation reproducer

This one-node diagnostic reproduces concurrent host-memory allocation directly
through the CUDA Driver API. The C++ program does not import or link Mooncake,
TensorRT-LLM, a CUDA runtime, or any serving framework. A Python standard-library
controller starts separate processes, retains their allocations, samples Linux
memory counters and writes the final report. No wheel installation or network
service is required.

Submit from the repository root, selecting the affected node:

```bash
sbatch --account=root --partition=batch-r41 --nodelist=vr-r41-cn15 \
  tools/check_cuda_host_numa.sbatch
```

The allocation is one exclusive node, four GPUs and all host memory, for 20
minutes. It uses one job step with all allocated CPUs and no CPU binding, so the
Slurm step does not accidentally constrain the requested NUMA nodes. It creates
no model or device-memory payload and performs no copies. CUDA initialization
can still consume driver bookkeeping memory; no CUDA context is explicitly created.

The default container is the existing RapLab ARM64 CUDA development image. To
use a different CUDA development image, set `CUDA_EGM_IMAGE`. Administrators can
run directly on the host instead, provided a C++17 compiler, CUDA 12.9 or newer
driver headers (CUDA 13 preferred), Python 3.10+, and the NVIDIA driver are
installed:

```bash
CUDA_EGM_IMAGE=host sbatch --account=root --partition=batch-r41 \
  --nodelist=vr-r41-cn15 tools/check_cuda_host_numa.sbatch
```

No `nvcc` is required. The script uses `g++` or `c++`, `cuda.h` and `libcuda`.
Override `CUDA_EGM_CXX`, `CUDA_EGM_CUDA_HOME`, `CUDA_EGM_INCLUDE` (a directory
containing `cuda.h`), or `CUDA_EGM_LIBCUDA` (a full library path) if needed.
When a CUDA stub is used at link time, the running program still requires the
actual NVIDIA driver library; the script never adds the stubs directory to the
runtime library path. Build failures are preserved in `logs/build.log` and
produce an incomplete report rather than a false allocation result.

## What it tests

Each process selects its single visible GPU and requests **250GiB of host
memory** with `CU_MEM_ALLOCATION_TYPE_PINNED`,
`CU_MEM_LOCATION_TYPE_HOST_NUMA` and a Fabric handle. The important call is
`cuMemCreate`. The reproducer does not reserve virtual addresses, map the
allocation or allocate GPU memory: the observed failure occurs during
`cuMemCreate`, before mapping in the original application.

| Case | First process | Second process | Allocation requests |
| --- | --- | --- | --- |
| `same-numa` | GPU 0, NUMA 0 | GPU 1, NUMA 0 | One 250GiB request each |
| `cross-numa` | GPU 0, NUMA 0 | GPU 2, NUMA 1 | One 250GiB request each |
| `reverse-gpu` | GPU 1, NUMA 0 | GPU 0, NUMA 0 | One 250GiB request each |
| `chunked` | GPU 0, NUMA 0 | GPU 1, NUMA 0 | Same 250GiB total per process, with requests capped at 16GiB |

Chunks are balanced to respect the reported CUDA allocation granularity. For
250GiB with a 16GiB cap and 16MiB granularity, this is 16 requests of 15.625GiB.

The first process must remain alive while the second allocates. A failed second
process retains any successfully created chunks until the controller captures
the result and requests cleanup. Handles are released in reverse order with
`cuMemRelease`. Unexpected process death, a timeout or unconfirmed cleanup stops
further cases, because later results could otherwise reflect an unknown state.

Cases run serially and may reuse driver-cached pages from earlier cases. Each
case records a fresh baseline. Separate jobs can also inherit cached driver
memory; a new job does not guarantee a cold node. Successful `cuMemRelease`
confirms that the process released its handle, but does not guarantee that the
driver returned every backing page immediately to the Linux free list.

Use a single case or another size when needed:

```bash
CUDA_EGM_CASES=same-numa CUDA_EGM_SIZE=224GiB \
  sbatch --account=root --partition=batch-r41 --nodelist=vr-r41-cn15 \
  tools/check_cuda_host_numa.sbatch
```

For live administrator inspection, run one case and retain the result for a
minute. Both successful and partially allocated handles remain resident during
this hold; `logs/probe/chunked/rank-*.json` records the process IDs and live sizes:

```bash
CUDA_EGM_CASES=chunked CUDA_EGM_HOLD_SECONDS=60 \
  sbatch --account=root --partition=batch-r41 --nodelist=vr-r41-cn15 \
  tools/check_cuda_host_numa.sbatch
```

Other controls are `CUDA_EGM_ALLOCATION_TIMEOUT=60` seconds per allocation,
`CUDA_EGM_BUDGET=600` seconds for the experiment,
`CUDA_EGM_SAMPLE_INTERVAL=1` second, `CUDA_EGM_HOLD_SECONDS=5`,
`CUDA_EGM_CHUNK_SIZE=16GiB`, `CUDA_EGM_BUILD_TIMEOUT=120` seconds and
`CUDA_EGM_STEP_TIMEOUT=960` seconds. Keep overrides within the allocation's
20-minute walltime and preserve cleanup time. The 600-second experiment budget
starts after compilation and the initial container inventories. The 960-second
step bounds compilation, inventories, experiments and step cleanup; the whole
20-minute allocation also reserves final host inventory and report generation.
This follows the repository's
[short diagnostic job-sizing guidance](../src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md).

## Evidence for administrators

The run directory is `outputs/cuda-host-numa-<jobid>` under the submission
directory, or under `CUDA_EGM_OUTPUT_ROOT` when provided. The job writes
`logs/perf-analysis.md` there even when compilation or setup fails.

Start with these files:

| File | Purpose |
| --- | --- |
| `logs/perf-analysis.md` | Generated result table, completion status, counter evidence and limitations |
| `logs/probe/summary.json` | Machine-readable allocation outcomes and cleanup evidence |
| `logs/probe/<case>/` | CUDA call logs/results and snapshots while both processes are retained |
| `logs/host-before.json`, `logs/host-after.json` | Host-visible NUMA, zone, buddy, reclaim/compaction, cgroup and driver evidence |
| `logs/container-before.json`, `logs/container-after.json` | Evidence from the job-step/container view |
| `logs/container-hardware.txt` | GPU identities, driver version, topology, NUMA layout and IMEX device visibility |
| `logs/kernel-before.txt`, `logs/kernel-after.txt` | Best-effort read-only `dmesg` capture, including permission errors |
| `logs/build.log`, `logs/build.json`, `logs/launch.json` | Compiler invocation, source hashes, job settings and library resolution |
| `logs/controller.log`, `logs/probe.log`, `logs/rank-0.err` | Launch, supervisor and step errors |
| `tools/cuda_host_numa_repro.cpp`, `tools/cuda_host_numa_probe.py`, `submitted.sbatch` | Exact source and batch script used |

The inventories read existing NVIDIA shrinker **count** interfaces when
accessible, but never write their scan interfaces. Permission-denied and missing
files are reported explicitly. The script does not use `sudo`, mount debugfs,
drop caches, reset GPUs or change the driver. If shrinker counts, `pagetypeinfo`
or kernel logs are unavailable to the job, an administrator can capture those
read-only views during the reproduced failure.

An allocation returning `CUDA_ERROR_OUT_OF_MEMORY` is a completed observation,
not a harness failure. A zero batch exit code means that the requested trials
and cleanup completed; inspect the report to see which allocations failed.
Incomplete evidence, unexpected child death and unsafe cleanup return nonzero.
The report labels inference latency, throughput and agentic SLO metrics N/A
because no inference traffic is involved. No DSight report is needed for this
allocation-only reproducer.

Package the complete run to hand to an administrator:

```bash
tar -czf cuda-host-numa-<jobid>.tar.gz -C outputs cuda-host-numa-<jobid>
```

The archive includes the sources, batch script, build command and README, so an
administrator can inspect the exact API calls without access to a serving
framework. To resubmit an unpacked bundle, run `sbatch submitted.sbatch` from its
root directory, passing the appropriate account, partition, node and image or
`CUDA_EGM_IMAGE=host` as above.
