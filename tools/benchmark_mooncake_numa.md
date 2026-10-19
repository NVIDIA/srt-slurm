<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Mooncake NUMA cache GET diagnostic

This two-node diagnostic measures real Mooncake Store cache GETs with controlled
host-memory placement and destination-device affinity. It loads no model and
requests no GPUs. Compare affinity ON/OFF ratios within a cluster before comparing
absolute results across machines.

Submit from the repository root. Choose the partition on the command line; the
script contains no partition or node list.

```bash
MC_NUMA_CLUSTER=raplab \
  sbatch -p batch-r42 tools/benchmark_mooncake_numa.sbatch
```

The default runs affinity ON only. It leaves affinity OFF comparisons explicitly
unrun because cross-group RDMA paths have failed on RapLab. An optional Slurm
`--nodelist=NODE_A,NODE_B` can select two allocated RapLab nodes.

On Hecate, explicitly opt into the ON/OFF comparison:

```bash
MC_NUMA_CLUSTER=hecate MC_NUMA_ALLOW_CROSS_RAIL=1 \
MC_NUMA_IMAGE=/lustre/fsw/coreai_comparch_inferencex/rihuo/tensorrt-llm-arm64-rubin-glm5-20261008-kernels-round7-30434.sqsh \
  sbatch -A root -p batch-xdr --segment=2 tools/benchmark_mooncake_numa.sbatch
```

`--segment=2` requests the two-node allocation in one Hecate topology segment
for rack-local placement, while leaving both node names to Slurm. This follows
the repository's segment convention: segment size equals the allocation's node
count. The physical rack boundary follows the site's configured topology.
The launcher records `SLURM_JOB_SEGMENT_SIZE` in `logs/controller.log`; the
initial job and node snapshots preserve the allocation for verification.
Pass `--segment` to `sbatch` when submitting; adding it to `srun` inside an
existing allocation cannot repair the allocation's placement. See the
[Slurm segment option](https://slurm.schedmd.com/sbatch.html#OPT_segment).

The default image for RapLab is
`/home/_shared/tensorrt-llm-arm64-rubin-glm5-20261008-kernels-round7-30434.sqsh`.
The Hecate command explicitly selects the corresponding image under
`/lustre/fsw/coreai_comparch_inferencex/rihuo/` using `MC_NUMA_IMAGE`.
If it is unavailable or does not match the target architecture, set
`MC_NUMA_IMAGE` to a suitable image containing Mooncake 0.3.13 before submitting.
The cluster label does not change the image, network configuration or resources.
The controller checks the installed Mooncake version and required APIs. The
launcher never installs or upgrades its wheel.

## Experiment and controls

The first Slurm rank is the donor/controller; the second is the reader/agent.
One persistent two-node Slurm step avoids repeated container launches. Each task
gets the full CPU allocation from `SLURM_JOB_CPUS_PER_NODE`; the launcher requires
two homogeneous nodes and does not assume a machine-specific CPU count.

The donor starts its Store with zero automatic global-segment allocation and
mounts a prefaulted memfd-backed host pool with an explicit NUMA location. This
avoids Store's automatic multi-NUMA segment allocation overriding the experiment.
Both endpoints query physical page residency; a location label alone is not
accepted as proof. Mooncake may bind its internal worker threads per HCA, so
the launcher's CPU binding does not prove every native thread remains on that
socket; buffer residency and NIC counters provide the placement controls.
The master uses dynamically allocated ports. Data moves
through registered host buffers and real Store PUT/GET APIs.

The plan covers both donor-memory NUMA nodes and both reader-buffer NUMA nodes.
By default, four placement combinations times three rounds produce 12 affinity
ON samples. With `MC_NUMA_ALLOW_CROSS_RAIL=1`, matched ON/OFF samples produce
24 samples. Ordering is reproducibly shuffled, with reversed donor order on odd
rounds; ON/OFF ordering is not guaranteed to be counterbalanced. The reader
retains its selected NUMA group's NIC set across an ON/OFF comparison; the donor
retains the union of the two NIC groups. Context count is therefore held constant
within the comparison.

Discovery uses ACTIVE InfiniBand devices common to both nodes, requiring matching
NUMA placement. Optional group overrides must name devices present on both nodes.
The benchmark records actual topology, GIDs, physical placement and NIC evidence.
It does not infer routing from NIC names or the cluster label.

A small two-key affinity trace runs in a separate process outside the measured
window. Timed processes disable per-slice tracing. Donor per-HCA counters, when
available, cover the whole reader process, including startup, validation and
warmup. They provide NIC-group evidence rather than isolated timed-window
bandwidth. A separate trace is sampled configuration evidence and cannot prove
every timed transfer used the same NIC. Warmup precedes each measurement.
Full-payload validation is separate from the GET API timer; reports distinguish
API bandwidth/latency from any end-to-end verified goodput.

## Settings

All settings are inherited by both container tasks.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MC_NUMA_IMAGE` | ARM64 image above | Container image; override for the target architecture and installed wheel |
| `MC_NUMA_CLUSTER` | `raplab` | Run label; does not select a partition or nodes |
| `MC_NUMA_ALLOW_CROSS_RAIL` | `0` | `1` explicitly enables affinity OFF comparisons and their potentially unreachable paths |
| `MC_NUMA_ROUNDS` | `3` | Repetitions of the placement/affinity comparisons |
| `MC_NUMA_SECONDS` | `10` | Measured GET duration per sample, seconds |
| `MC_NUMA_WARMUP` | `2` | Warmup duration per sample, seconds |
| `MC_NUMA_BATCH_SIZE` | `512` | Pages per cache GET batch |
| `MC_NUMA_PAGE_BYTES` | `2006016` | Bytes per cached page |
| `MC_NUMA_POOL_GIB` | `4` | Donor memory pool, GiB |
| `MC_NUMA_WORKING_BATCHES` | `2` | Working-set batches in the donor |
| `MC_NUMA_GROUP0`, `MC_NUMA_GROUP1` | discovered | Optional comma-separated NIC names for NUMA0 and NUMA1 |
| `MC_NUMA_BUDGET` | `1320` | Positive controller body budget, at most 1320 seconds |
| `MC_NUMA_OUTPUT_ROOT` | `$SLURM_SUBMIT_DIR/outputs` | Parent directory of the new run directory |
| `MC_NUMA_TOOL_DIR` | `$SLURM_SUBMIT_DIR/tools` | Source directory containing the four captured Python helpers |
| `ENROOT_MAX_PROCESSORS` | `8` | Host-side Enroot extraction workers; exported before container creation and independent of benchmark CPU binding |

The default reader batch is 1,027,080,192 B (about 0.957 GiB); two working
batches contain about 1.913 GiB of cache payload. The allocation uses
`--exclusive --mem=0` to request complete nodes and all allocatable memory on
each node. Slurm's `--exclusive` alone still allocates only the requested or
default memory; see the [Slurm allocation options](https://slurm.schedmd.com/sbatch.html#OPT_exclusive).
This provides room for image extraction, the 4 GiB donor pool, registered reader
buffers and process overhead without a fixed memory cap. Passing `--mem=SIZE`
to `sbatch` overrides this default if a smaller allocation is desired.

The launcher also limits Enroot extraction to eight workers instead of scaling
decompression parallelism to the full CPU allocation. This setting is applied
on the host before Pyxis starts the container; setting it inside
`launch-node.sh` would be too late. See [Enroot create configuration](https://github.com/NVIDIA/enroot/blob/main/doc/cmd/create.md).
The launcher does not check available-memory or cgroup headroom before
allocation; allocation and NUMA-residency failures are reported by the workload.

The [slurm-job-sizing skill](../src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md)
sets a 25-minute allocation, a 22-minute controller budget and three-minute
signal/cleanup reserve. The default 12 samples need about 144 seconds of configured
warmup plus measurement; 24 samples need about 288 seconds, before preparation,
traces, validation and cleanup. Increased rounds, durations or timeouts must fit
the whole-job budget; a shortened timeout alone does not shorten valid traffic.

## Setup, artifacts and interpretation

Both tasks run writable containers with remapped root. If numactl is absent,
the wrapper installs only numactl using container-local apt with a 240-second
bound and records the operation in `logs/agents/NODE.install.log`. It changes
neither host packages nor the Mooncake installation. Container setup failure is
a preflight failure, not a GET bandwidth result.

A fresh run is created at `outputs/mooncake-numa-JOB_ID/`, or under the selected
output root. Existing run directories are refused. The run captures its launcher
and four Python helpers under `tools/`. Agent stdout/stderr, inventories, traces,
sample records and cleanup evidence live under `logs/`; the controller writes
`logs/perf-analysis.md` using the
[per-run analysis requirements](../docs/perf-analysis.md). A bootstrap report
exists even if setup fails. The file named `slurm-job-final.txt` is a snapshot
taken during cleanup, and can still say RUNNING; it is not final accounting.

Treat GET errors, retries, missing placement evidence, failed validation and
incomplete samples explicitly. A failed batch has no valid successful-transfer
latency or bandwidth. Do not count attempted bytes as successfully retrieved
bytes. Inspect the number of valid samples behind latency percentiles.

This is a host-cache diagnostic. It does not reproduce a large, long-lived,
GPU-serving cache or establish a model speedup. TTFT p50/p95/p99, ITL p50/p99,
token throughput and agentic Pareto/SLO metrics are N/A. No DSight report or
dashboard is needed to inspect Store return values, placement, affinity traces
and sample measurements. An ON/OFF ratio is meaningful only with matched NICs,
working set, payload, API, wheel, CPU binding, warmup and instrumentation.
Absolute cross-cluster differences also include hardware and fabric differences;
they cannot be attributed to affinity alone.
