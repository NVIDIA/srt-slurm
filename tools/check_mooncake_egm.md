# Two-node Mooncake EGM/NVLink smoke test

Submit from the repository root on RapLab:

```bash
MC_EGM_IMAGE=/home/_shared/tensorrt-llm-release-arm64-f1bdd25-user_imant_glm5-rubin-30428.sqsh \
MC_EGM_WHEEL_DIR=/home/rihuo/mooncake_egm_wheels \
sbatch --account=root --partition=batch-r41 tools/check_mooncake_egm.sbatch
```

This requests two exclusive nodes, all allocatable host memory, one Slurm
topology segment, four GPUs per node, and 15 minutes. Both `sbatch` and `srun`
request `--gpus-per-node=4` so the container step receives GPU devices for
NVLink copy submission and EGM NUMA discovery. Hecate's full-node allocation
policy rejects this GPU resource request; this launcher now targets clusters
that support explicit GPU allocation.
It runs one persistent container per node. Rank 0 starts the master, checks
small and recipe-sized host allocations, then starts a native EGM provider;
rank 1 writes pinned host data to that provider's host DRAM and reads it back
into a separate pinned host buffer. Both payload endpoints use host memory.
No model or Dynamo server is loaded.

## Prerequisites

Place exactly one EGM-enabled Linux ARM64 CUDA 13 Mooncake wheel matching the
image's Python ABI in:

```text
/lustre/fsw/coreai_comparch_inferencex/rihuo/mooncake_egm_wheels/
```

The default image is the same as the Dynamo EGM recipe:

```text
/lustre/fsw/coreai_comparch_inferencex/rihuo/tensorrt-llm-arm64-rubin-glm5-20261008-kernels-round7-30434.sqsh
```

Override these paths when submitting to another cluster:

```bash
MC_EGM_IMAGE=/shared/images/trtllm-arm64-cuda13.sqsh \
MC_EGM_WHEEL_DIR=/shared/mooncake_egm_wheels \
  sbatch --account=root --partition=batch-r42 tools/check_mooncake_egm.sbatch
```

The test reuses `configs/setup_mooncake_egm.sh --mooncake-only` and
`configs/mooncake_egm_donor.py`, installing the wheel in each container without
installing Dynamo or changing the host Python environment. Use a wheel
containing EGM pooling and the direct-read stack checked against upstream
`042ab8a18b0e6ccfe87782261a5556a3dd4446e4`. This includes the EGM pool commit
`a2c6c5b7` and `9c0528d9`, the upstream squash of `db195f7` with an identical
patch. Build with `WITH_STORE=ON`, `USE_CUDA=ON`, `USE_MNNVL=ON` and
`USE_VRAM_SEGMENT=OFF` (the upstream default). The existing stock `0.3.13`
image wheel does not suffice. See the
[Dynamo recipe requirements](../recipes/trtllm/vrnvl72/agentic/dynamo/README.md#egm-over-nvlink).

Both nodes need HOST_NUMA/Fabric allocation support, working NVLink fabric
membership and IMEX access inside the container. `--segment=2` requests
placement in one scheduler segment; successful transfers establish the actual
path. The test records GPU, topology, Fabric and IMEX inventory from each node.

## Allocation failure diagnostics

To investigate a second donor failing while the first donor remains alive, use
the [one-node EGM reclamation diagnostic](check_mooncake_egm_reclaim.md). It
compares concurrent host pools, NUMA placement, GPU startup order and allocation
chunks, with memory/reclaim counters sampled during setup.

Allocation checks run by default before the original small transfer test. On
the provider node, four fresh processes try this matrix sequentially:

| Visible physical GPU | Host pool | NUMA selection |
| --- | --- | --- |
| 0 | 4GiB | explicit `0` |
| 0 | 4GiB | `auto` |
| 0 | 260GiB | explicit `0` |
| 0 | 260GiB | `auto` |

Each process sees only that one GPU and uses the native EGM Store configuration
with `local_buffer_size=0`. It allocates Fabric-exportable HOST_NUMA memory,
closes the Store, and exits before the next process starts. No GPU payload
tensor or ordinary host-memory fallback is used. With one visible GPU, `auto`
selects that GPU's host NUMA node; the inventory queries CUDA's
`CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` and records its PCI identity. See the
[CUDA EGM allocation guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/extended-gpu-memory.html).

The helper captures `/proc/self/status` allowed CPU/memory-node masks, process
limits, global and per-NUMA memory counters, buddy information, and cgroup
membership/mounts. It reads visible cgroup ancestors' memory limits, usage and
OOM events, including v1 and v2 layouts. A container may hide higher ancestors;
missing files and permission errors are saved as missing evidence. A leaf
`memory.max=max` alone does not establish an unlimited job allocation. See the
[kernel cgroup memory documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html).
Each case retains before/after snapshots; the regular node inventory also
captures full `nvidia-smi -q` output and best-effort kernel messages.

Start with `logs/perf-analysis.md`, then inspect
`logs/allocation-diagnostics/summary.json` and each linked case's JSON/native log:

- Small allocations pass and 260GiB fails: investigate size-dependent capacity
  or driver resource limits. The exact limiting resource is still unproven.
- Explicit NUMA 0 fails and `auto` passes: compare the discovered host NUMA ID
  and per-node memory counters before changing the recipe's placement.
- A process is killed, times out or cannot confirm cleanup: remaining probes
  and the small transfer test stop. Inspect cgroup events, limits and native logs.
- All sequential probes pass but serving fails: these checks do not establish
  that four concurrent donors plus inference workers fit. The four 260GiB
  donors need 1040GiB per prefill node, with 520GiB on each selected NUMA node.

Controlled allocation failures with confirmed cleanup are recorded and followed
by the original `4GiB` remote transfer test. Its host-transfer status and the
allocation diagnostic status are separate: a small transfer PASS does not
validate a 260GiB pool. Incomplete diagnostics return a failing job status.

Use `MC_EGM_ALLOC_GPU=1` to test a different physical GPU. For DEP16 or DEP32
pool sizes, use `MC_EGM_ALLOC_SIZES=4GiB,256GiB` or `4GiB,224GiB`.
Set `MC_EGM_ALLOC_DIAGNOSTICS=0` to run only the original transfer test.
An inherited `MC_MAX_MR_SIZE` is recorded and honored; for example,
`MC_MAX_MR_SIZE=17179869184` requests 16GiB allocation chunks while retaining
the full pool capacity. Smaller chunks are a further diagnostic, not an
established remedy for insufficient aggregate RAM.

## Traffic and evidence

The provider allocates a small `4GiB` EGM pool with
`enable_egm_store_pool=true`, `egm_numa_nodes=auto`, `protocol=nvlink`, and zero
local buffer. All four GPUs are visible so auto discovery includes both
supported host NUMA nodes. Native EGM setup validates the installed transport
and allocates Fabric-exportable HOST_NUMA memory.

The remote client has zero global and local Store capacity, with explicit
`nvlink` transport and automatic discovery disabled. TENT is disabled and
`MC_USE_NVLINK_IPC` is unset. The only capacity belongs to the provider on the
other node. Small JSON files on shared storage coordinate readiness; payloads
go through Mooncake, not these files.

By default, each of the four client GPUs is selected in turn to submit copies
for 64 KiB, 1 MiB, 16 MiB and 64 MiB payloads, with two warmup iterations and ten
measured iterations per case. Source and destination tensors are allocated with
`device="cpu", pin_memory=True`; payload generation and comparisons run on the
CPU. The helper checks that every buffer is a pinned CPU tensor and that
`cudaPointerGetAttributes` reports `cudaMemoryTypeHost`. Device, managed and
unpinned buffers fail the test. The client has no GPU payload tensors.
The helper uses the same batched multi-buffer Put/Get APIs as tekit's Store
connector and validates the complete returned payload on every iteration.
It records remote memory replica descriptors, transfer latency percentiles and
effective payload throughput. Correctness checks and payload preparation are
outside the reported transfer timing.

Successful completion requires host-buffer allocation evidence for every case,
all payload checks, complete replicas owned by the remote provider, and an actual
`NvlinkTransport: using cudaMemcpyBatchAsync` message in the client log.
The provider's EGM configuration and capacity log establish host DRAM
allocation; the client log establishes use of the native NVLink transfer path.
There is no RDMA/TCP fallback or throughput threshold. Low measured bandwidth
does not by itself mean the transport failed.

This is a functional diagnostic for one provider and one remote client. It
does not validate full-serving capacity, inference SLOs, every physical fabric
link, or the serving image's V2 KV allocator. Pinned host client buffers are
initiator-only endpoints here; the provider owns the Fabric-exported memory.
Classic Mooncake NVLink submits these pointers directly to CUDA batched copies;
TENT is not required. GPUs still provide CUDA contexts, streams and copy
submission, so host-only payloads do not imply zero total GPU memory use or zero
interference with other GPU work.

## Configuration and results

The principal environment overrides are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MC_EGM_POOL_SIZE` | `4GiB` | Total provider host DRAM |
| `MC_EGM_ALLOC_DIAGNOSTICS` | `1` | Run allocation probes before the small transfer test; `0` skips them |
| `MC_EGM_ALLOC_SIZES` | `4GiB,260GiB` | Host-pool sizes tested sequentially |
| `MC_EGM_ALLOC_GPU` | `0` | One physical CUDA GPU index or UUID visible to each allocation probe |
| `MC_EGM_ALLOC_NUMA_NODES` | `0,auto` | Independent NUMA selections tested for each size |
| `MC_EGM_ALLOC_TIMEOUT` | `30` | Per-case setup/cleanup deadline in seconds |
| `MC_EGM_ALLOC_BUDGET` | `150` | Whole allocation-probe budget in seconds, plus bounded teardown |
| `MC_EGM_GPU_IDS` | `0,1,2,3` | CUDA copy submission ordinals tested sequentially |
| `MC_EGM_SIZES` | `65536,1048576,16777216,67108864` | Payload sizes in bytes |
| `MC_EGM_ITERATIONS` | `10` | Measured iterations per submission device and size |
| `MC_EGM_WARMUP` | `2` | Warmup iterations per case |
| `MC_EGM_SETUP_TIMEOUT` | `240` | Wheel setup deadline in seconds per node |
| `MC_EGM_READY_TIMEOUT` | `180` | Master/provider readiness deadline in seconds |
| `MC_EGM_CLIENT_TIMEOUT` | `180` | Whole client deadline in seconds |
| `MC_EGM_TIMEOUT` | `720` | Whole Slurm step deadline in seconds |
| `MC_EGM_OUTPUT_ROOT` | `$SLURM_SUBMIT_DIR/outputs` | Shared artifact parent directory |

The [slurm-job-sizing skill](../src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md)
keeps this diagnostic short: the step deadline includes extraction, setup,
allocation probes and traffic, leaving three minutes for shutdown and reporting in
the 15-minute allocation. Increase the traffic deadline, whole-step deadline
and submitted walltime together when extending the workload. Preserve a cleanup
reserve. The 720-second whole-step deadline takes precedence over individual
phase deadlines. An outer timeout also bounds allocation inventory/driver calls
to the probe budget plus ten seconds, followed by at most ten seconds to kill
the diagnostic process. No cluster run was launched while preparing the scripts.

Artifacts are saved under `outputs/mooncake-egm-<job_id>/`:

- `logs/perf-analysis.md`: authoritative pass/fail report, measured transfer
  metrics, separate allocation findings, evidence sources and missing-data explanations.
- `logs/allocation-diagnostics/summary.json`, `inventory.json`,
  `inventory-after.json`, and per-case JSON/log files: allocation outcome,
  single-GPU CUDA host NUMA mapping, memory/limit snapshots, native CUDA errors
  and cleanup status. `logs/allocation-diagnostics.log` captures the helper itself.
- `logs/client-results.json` and `logs/client.log`: pinned host allocation
  evidence, correctness, per-case measurements, replica descriptors and native
  NVLink transfer evidence.
- `logs/provider-config.json`, `logs/provider.log` and `logs/master.log`:
  allocated host-pool configuration, capacity and lifecycle evidence.
- `logs/setup-*.log`, `logs/inventory-*.txt`, `logs/rank-*.out` and
  `logs/rank-*.err`: environment setup, hardware and failure details.
- `logs/launch.json`, Slurm snapshots and copied scripts: wheel SHA256,
  workload configuration and allocation provenance.

Inference TTFT/ITL percentiles, token throughput and agentic Pareto/SLO metrics
are explicitly N/A because no inference requests are served. No DSight capture
is needed to answer this connectivity and correctness question.
