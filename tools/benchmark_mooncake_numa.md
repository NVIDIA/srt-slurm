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
  sbatch -p batch-xdr tools/benchmark_mooncake_numa.sbatch
```

This command leaves node selection to Slurm. The default image is
`/home/_shared/tensorrt-llm-release-arm64-f1bdd25-user_imant_glm5-rubin-30428.sqsh`.
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

The default reader batch is 1,027,080,192 B (about 0.957 GiB); two working
batches contain about 1.913 GiB of cache payload. The allocation reserves
16 GiB/node by default, allowing a 4 GiB donor pool, registered reader buffers
and process overhead. For larger pools or batches, users must request a higher
Slurm `--mem` limit with room for registered buffers and process overhead. The
launcher does not check available-memory or cgroup headroom before allocation;
allocation and NUMA-residency failures are reported by the workload.

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
