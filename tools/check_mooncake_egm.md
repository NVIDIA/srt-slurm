# Two-node Mooncake EGM/NVLink smoke test

Submit from the repository root on Hecate:

```bash
sbatch --account=root --partition=batch-xdr tools/check_mooncake_egm.sbatch
```

This requests two exclusive nodes, all allocatable host memory, one Slurm
topology segment, and 15 minutes. It expects four visible GPUs per node.
Both `sbatch` and `srun` omit GPU/GRES resource requests for Hecate's full-node
allocation policy; `--gpus-per-node` causes `Invalid generic resource (gres)
specification` there. CUDA devices 0–3 remain visible inside each container
for NVLink copy submission and EGM NUMA discovery.
It runs one persistent container per node. Rank 0 hosts the Mooncake master and a native EGM provider;
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
allocation and traffic, leaving three minutes for shutdown and reporting in
the 15-minute allocation. Increase the traffic deadline, whole-step deadline
and submitted walltime together when extending the workload. Preserve a cleanup
reserve. No cluster run was launched while preparing the scripts.

Artifacts are saved under `outputs/mooncake-egm-<job_id>/`:

- `logs/perf-analysis.md`: authoritative pass/fail report, measured transfer
  metrics, evidence sources and missing-data explanations.
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
