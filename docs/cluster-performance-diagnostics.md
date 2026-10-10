# Comparing cluster performance

Use the same model weights, container build, workload, context lengths, concurrency,
speculative-acceptance policy, and stream interval on both clusters. Different image
filenames or model names require verification before attributing latency to hardware.

`configs/cluster-perf-diagnostics.py` collects serving placement and loaded-library
paths, GPU clocks/power/utilization over time, readable RDMA counter deltas, package
versions, selected TRT-LLM source hashes, and optional model metadata hashes and weight
sizes. It does not change clocks, drivers, routing, or host settings. Missing commands,
permissions, and counters remain explicit. A zero readable-counter count means unknown
transport health, rather than zero transport errors.

## Run the same batch diagnostics on either cluster

From the repository root on shared storage, submit `configs/cluster-diagnostics.sbatch`
with a cluster argument. Each preset selects the supplied f1bdd25 image; allocation
settings come from your sbatch options and site defaults:

```bash
sbatch --account=<account> --partition=<partition> configs/cluster-diagnostics.sbatch raplab
sbatch --account=<account> --partition=<partition> configs/cluster-diagnostics.sbatch hecate
```

For eight-node MPI and NCCL testing, no master address or port needs to be assigned:

```bash
sbatch --nodes=8 --account=<account> --partition=<partition> --export=ALL,RUN_NCCL=1 \
  configs/cluster-diagnostics.sbatch raplab
```

Use `hecate` for the other cluster. Inside the communication step, MPI rank zero
selects its IPv4 source address for a route to another allocated node. It starts
a TCPStore on an OS-assigned port and broadcasts the endpoint through MPI while
keeping the listener open. The endpoint appears in `communication.log` and the
`nccl_rendezvous` field in `communication.json`. Node hostnames must resolve inside
the container and the selected interface must allow TCP connections between nodes.
Manual overrides remain available for sites that need a different route.

The default allocation is two exclusive nodes. Add `--nodes=8` to measure the
32-rank decode layout. Each node runs the existing host/container diagnostics and
sequential copy tests, then a structured idle snapshot. The job subsequently runs
MPI communication with four GPU-local CPU/memory-bound ranks per node. It stages
helpers into shared `cluster-diagnostics-<jobid>/`, records phase exit statuses, and
creates a `.tar.gz` bundle even when a collection phase fails. Failed phases make
the batch exit nonzero; individual unavailable checks remain in their own files.
This job does not inspect a running server's environment or reproduce serving ITL.

Export overrides before submission (sbatch's normal `--export=ALL` inheritance):

| Variable | Default / purpose |
| --- | --- |
| `CONTAINER_IMAGE` | Override the cluster preset image |
| `DIAG_SOURCE_DIR` | `$SLURM_SUBMIT_DIR/configs`; shared directory containing helpers |
| `LOG_DIR` | `$SLURM_SUBMIT_DIR/cluster-diagnostics-$SLURM_JOB_ID` |
| `RUN_COPY_BENCH` | `1`; set `0` to skip sequential GPU copies |
| `COPY_MIB`, `COPY_ITERATIONS` | `64`, `5` |
| `INSTALL_TOOLS` | `0`; optional `1` installs missing OS utilities only in the writable container |
| `SAMPLE_DURATION` | `10` seconds per node, after copy tests |
| `RUN_COMM_BENCH` | `1`; set `0` for hardware/copy diagnostics only |
| `RANKS_PER_NODE` | `4`; must match the GPUs visible on each node |
| `MPI_PLUGIN` | `pmix`; override with the site's supported Slurm MPI plugin |
| `COMM_ITERATIONS`, `COMM_WARMUP` | `200`, `20` |
| `RUN_NCCL` | `0`; set `1` to include NCCL GPU all-gather |
| `MASTER_ADDR`, `MASTER_PORT` | Optional overrides; otherwise address and port are detected inside the communication step |

Raplab communication tests apply all `MPI_UCX_*`, `UCX_*`, `NCCL_*`, and
`OMPI_MCA_*` settings from the c560 Raplab recipe's decode environment. These
overrides are applied only to communication ranks, before MPI initialization,
and recorded in `settings.txt` and per-rank metadata. They include the eight
requested NICs, NCCL's exact-match rail/plane suffixes, MPI's UCX PML and
collective exclusions, and the separate MPI/UCX transport settings. The NUMA
wrapper narrows `UCX_NET_DEVICES` by GPU locality and retains explicit `eth0`;
MPI and NCCL retain all eight devices. Hardware/copy collection continues to
use the submission environment. Hecate communication tests also inherit their
settings from the submission environment; export any intended overrides there.
Every rank must see all four node GPUs for the default mapping. Packages are not
installed by default; missing MPI/PyTorch dependencies fail the communication phase
explicitly. These commands submit jobs from your cluster environment and use no SSH.

## Collect during matched serving runs

Run in the serving container or a namespace where the worker PIDs are visible. Use
one collector per serving node, and record every worker PID on that node. Collecting
in a separate container can hide workers or describe different packages/libraries.

```bash
python3 configs/cluster-perf-diagnostics.py collect \
  --pid 1234 --pid 1235 --pid 1236 --pid 1237 \
  --model /model --duration 30 --interval 1 --output raplab-node.json
```

Repeat on the other cluster with its PIDs and output filename. The environment capture
uses an explicit list of transport, rank, and performance variables; it does not dump
the whole worker environment. Check all thread CPU masks, memory masks, `numa_maps`,
and cgroup limits. The memory mask shows allowed nodes; `numa_maps` describes actual
resident-page placement and memory policy, but does not identify which mappings are
CUDA-pinned staging buffers. Library paths likewise identify mappings, not binary
content equivalence. Model metadata hashes and weight sizes do not prove identical
weight contents; use full shard checksums if their provenance is uncertain.

```bash
python3 configs/cluster-perf-diagnostics.py compare raplab-node.json hecate-node.json \
  --section packages --section code_sha256 --section model
```

Without `--section`, comparison reports all differences. Hostnames, timestamps, PIDs,
GPU UUIDs, paths, and package metadata can legitimately differ; it is a field diff,
not an automatic root-cause diagnosis.

## Separate MPI scheduling latency from GPU collectives

Use an idle allocation matching the decode endpoint: the same number of nodes and
GPU ranks, container, exported transport settings, CPU binding, and memory binding.
The script requires NumPy and mpi4py already installed in the image. It benchmarks
MPI buffer all-gather at 64 B, 1 KiB, and 8 KiB per rank, plus Python-object all-gather.
Results report p50/p99/max of the slowest rank per iteration, with warmups excluded.
Compare these outputs with `compare --section results --section world_size`.
Calls run back to back; these are microbenchmarks, not serving-iteration timings or
the private runtime's chunked object-transfer implementation.

For four GPUs and 352 allocated logical CPUs per node, an example step **inside an
existing eight-node allocation** is:

```bash
# Set CONTAINER_IMAGE to the matched serving image on this cluster.
# Export the intended MPI_UCX_*, UCX_*, NCCL_*, and OMPI_MCA_* settings first.
srun --mpi=pmix --nodes=8 --ntasks-per-node=4 --cpus-per-task=88 \
  --cpu-bind=none --kill-on-bad-exit=1 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$PWD:/diag,/dev/infiniband:/dev/infiniband" \
  bash /diag/configs/numa_cpu_bind.sh --bind-memory \
  python3 /diag/configs/cluster-perf-diagnostics.py communication \
  --iterations 200 --warmup 20 --output /diag/mpi-communication.json
```

Adjust allocated CPUs and launcher options to your cluster. Do not constrain the
whole step to NUMA0. Ensure every rank sees the node's four GPUs, or check the wrapper's
`CUDA_VISIBLE_DEVICES`/`SLURM_LOCALID` mapping when using per-task GPU assignment.

Add `--nccl` to the Python command for GPU all-gather at 32 B, 64 KiB, and 1 MiB per
rank, with automatic rendezvous address/port selection. Optional `--master-addr` and
`--master-port` override either value. Rank zero publishes the endpoint through MPI.
The TCPStore holds its listener open, using PyTorch's documented
[ephemeral-port support](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.TCPStore).
Each timed GPU call includes host dispatch and CUDA synchronization. Correctness is
checked outside timing. All ranks must use distinct GPUs; the script uses the local
MPI rank when multiple GPUs are visible.

Set `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH` for the microbenchmark to
record NCCL transport choices. Successful NCCL results do not establish the performance
of MegaMoE's fused direct-NVLink dispatch/combine. Use a serving GPU/CPU trace to measure
those kernels, graph replay, DSA attention, and scheduler gaps.

For raw link bandwidth/reachability, the existing `configs/raplab-rdma-bandwidth.sbatch`
tests RDMA; override its cluster-specific allocation/image settings. Pair NICs by the
actual subnet and measure both directions. For host-to-GPU copies, the existing
`configs/raplab-cluster-diagnostics.sbatch` supplies a sequential idle-node control.
Its unrestricted versus fully bound tests change CPU and memory policy together; to
isolate that difference, run fresh per-GPU processes, fix memory locality, vary CPU
binding only, and randomize/repeat the test order.
