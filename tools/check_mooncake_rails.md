# Check Mooncake cross-rail connectivity

For Hecate, submit from the repository root:

```bash
sbatch tools/check_hecate_rails.sbatch
```

This requests account `root`, partition `batch-xdr`, two scheduler-selected
exclusive nodes, all allocatable host memory (`--mem=0`), and `--segment=2` so
the allocation stays within one Slurm topology segment. There is no fixed
nodelist and no GPU request. The segment constraint records the scheduler's
placement decision; it does not independently identify a physical tray.
The Hecate image defaults to:

```text
/lustre/fsw/coreai_comparch_inferencex/rihuo/tensorrt-llm-arm64-rubin-glm5-20261008-kernels-round7-30434.sqsh
```

The Hecate wrapper runs the same 128-case READ/WRITE matrix described below,
using three seconds per case, an eight-second case timeout, a 27-minute
controller budget and a 30-minute allocation. This keeps blocked cross-rail
pairs from consuming the whole allocation before other pairs are checked.
Container extraction uses at most eight workers by default
(`ENROOT_MAX_PROCESSORS=8`), independently of the test's full CPU allocation.
`submitted.sbatch` preserves the Hecate wrapper alongside the common launcher.

The result answers whether every one of the eight selected NICs can exchange
RDMA data with every selected NIC on the other node, and records each pair's
average speed. Compare the matrices and NIC/subnet inventory for separate
connectivity groups. All 64 pairs passing supports all-to-all reachability on
these two nodes; it does not prove that the physical network has one plane,
because interconnected planes can also provide that reachability. Failures
must be distinguished from setup, TCP bootstrap and NUMA-binding errors.

For a matched RapLab comparison, use the same transfer settings and container
build, changing only the image's filesystem path and allocation placement:

```bash
MC_RAIL_CLUSTER=raplab \
MC_RAIL_IMAGE=/home/_shared/tensorrt-llm-arm64-rubin-glm5-20261008-kernels-round7-30434.sqsh \
MC_RAIL_DURATION=3 MC_RAIL_CASE_TIMEOUT=8 MC_RAIL_BUDGET=1620 \
  sbatch --time=00:30:00 tools/check_mooncake_rails.sbatch
```

Compare `logs/summary.csv` and the READ/WRITE matrices in
`logs/perf-analysis.md`, including failures and unrun cells. CSV speeds include
Gbit/s and decimal GB/s (Gbit/s divided by eight); the report also summarizes
same-name and cross-name pair medians. Each case uses one NIC at each end, so
this does not measure eight-rail aggregate bandwidth.

## Common launcher and RapLab defaults

Submit from the repository root on the cluster:

```bash
sbatch tools/check_mooncake_rails.sbatch
```

Keep `check_mooncake_rails.py`, `mooncake_rail_worker.py`, and
`setup_mooncake_rails.sh` beside the batch script. No SSH is used. Package
installation runs inside the container; no host package installation is performed.

Defaults are account `root`, partition `batch-r42`, nodes `vr-r42-cn03` and
`vr-r42-cn05`, exclusive allocation, all allocatable host memory, and 25 minutes.
The launcher discovers the full CPU count from `SLURM_JOB_CPUS_PER_NODE` and
requires two homogeneous nodes. The CPUs cover both sockets so the agent can
bind each test to the selected NIC's NUMA node. No GPUs or model are requested. Two persistent
container agents avoid repeatedly creating Slurm steps for every NIC pair.
Requeue is disabled and existing run directories are rejected, so old mailbox
results cannot be mistaken for a new measurement.

The container defaults to:

```text
/home/_shared/tensorrt-llm-release-arm64-f1bdd25-user_imant_glm5-rubin-30428.sqsh
```

Each node starts a writable container with `--container-remap-root` and
`--container-writable`. If tools are missing, its setup script runs
`apt-get update` and `apt-get install -y --no-install-recommends perftest numactl`
inside that container, without sudo. It then executes the worker in the same
container so the installed tools remain available. Already installed tools are
reused. The source SquashFS image and host OS are not modified; package changes
belong to the temporary container filesystem.

Setup has a four-minute timeout and saves `logs/agents/<node>.install.log`.
APT repositories must be reachable from the container. Setup failure is published
as a failed inventory so the controller stops with a report rather than waiting
for the full test budget. Preflight then records tool paths, versions/help and
NUMA/NIC inventory; invalid binding or NIC state remains a clear preflight failure.
Ubuntu perftest 6.20 exits 1 after printing `Version: 6.20`. Setup accepts a
version probe returning 0 or 1 only when it prints a valid version line; the
actual RDMA tests still require successful exits and positive measured bandwidth.
Inventory also saves OS/kernel information, current affinity, actual NUMA-bind
checks and visible process cgroup limits; ancestor cgroup limits are not surveyed.
The worker also captures `findmnt` output for the shared run directory when the
command is available, to help identify filesystem visibility problems.

Setup separately attempts to install `ibverbs-utils` and `infiniband-diags`
inside the container, with a one-minute timeout. These tools are optional;
missing packages or management permissions do not fail the RDMA tests.
The inventory's `optional_diagnostics` section saves bounded `ibstat`,
`ibv_devinfo`, `rdma link` and per-NIC `iblinkinfo` results. The latter query
examines only the local port and first-hop switch, avoiding a full fabric scan.
NIC sysfs evidence includes advertised link rate, LID, GUID, GID and PKey
information. Successful topology output can help identify switch/subnet groups;
missing output remains an explicit evidence gap.

Before any perftest process starts, the controller requires a ping/reply round
trip with each agent and records its duration in `logs/mailbox-checks.json`.
Replies are discovered through directory scans, matching command discovery,
instead of repeatedly checking a filename before it exists. This avoids
priming negative filename lookups; shared filesystems can still delay visibility.
A failed round trip stops the run as a coordination failure. It cannot establish
whether an RDMA NIC pair works.

The job runs 128 directed cases: eight reader NICs times eight donor NICs,
for RDMA reads and writes. The selected NICs are `mlx5_0`, `mlx5_1`, `mlx5_2`,
`mlx5_5`, `mlx5_6`, `mlx5_7`, `mlx5_8`, and `mlx5_11`. Each endpoint binds
its CPU and host allocation to its own NIC's NUMA node. Thus a reader NUMA-1
NIC can reach a donor NUMA-0 NIC without introducing an intentional local
NIC-to-remote-socket memory access at either end.

Perftest uses explicit devices, port 1, GID index 0, two QPs, a 2,006,016-byte
message and five-second duration with one-second margins. Reads use depth 16
per QP; the read-only option is omitted from writes. TCP exchanges connection
information; `-R` and `-z` are absent so the bootstrap IP does not choose the
RDMA data device. Server TCP ports are allocated dynamically. Readiness checks
the child process's listening socket through procfs without opening a probe
connection, because perftest accepts only one client connection.

The default traffic duration totals about 11 minutes, plus process setup and
cleanup. The existing 22-minute controller budget includes container apt setup
(up to four minutes, plus one minute for optional tools, concurrently on the
two nodes). Inventory waits up to seven minutes for container startup, apt
setup and collection. Per-case timeouts bound unhealthy tests; any remaining
cases are explicitly NOT RUN. Three minutes remain for
cleanup and reporting, following the repository's
[job-sizing skill](../src/srtctl/dsight/skills/slurm-job-sizing/SKILL.md).

Inspect:

```text
outputs/mooncake-rails-<job_id>/
  metadata.json
  plan.json
  logs/perf-analysis.md
  logs/summary.csv
  logs/results.jsonl
  logs/mailbox-checks.json
  logs/cases/<operation>-<reader_nic>-<donor_nic>.<server|client>.log
  logs/agents/<node>.inventory.json
  logs/agents/<node>.install.log
```

A matrix PASS requires successful exits at both endpoints and a finite positive
client average Gbit/s result with completed iterations for the requested size.
Missing output is never treated as a successful zero-bandwidth test. A failure
may arise from bootstrap TCP, NUMA permissions, memory registration or tool
configuration as well as the RDMA fabric; use the retained per-case commands
and raw output to distinguish them. CPU frequency warnings alone do not
invalidate an otherwise completed measurement.

If a child cannot be reaped after bounded TERM/KILL waits, its agent reports a
teardown error. The controller retains that attempted case as INCOMPLETE and
stops before starting another case; it never counts that attempt as NOT RUN.

All 64 NIC pairs must pass for each operation before this run supports complete
cross-rail connectivity on these two nodes. Other worker/donor nodes require
their own pair tests. Single-pair bandwidth is not aggregate multi-rail
bandwidth, and these tests do not establish a serving TTFT improvement.
TTFT/ITL/token throughput and agentic SLO metrics are N/A because no model is
served. Perftest transfers a whole cache page per operation; Mooncake currently
splits pages into smaller work requests.

For a quick four-pair read/write check:

```bash
MC_RAIL_NICS=mlx5_0,mlx5_6 sbatch tools/check_mooncake_rails.sbatch
```

Change nodes or the image without editing the script:

```bash
sbatch --nodelist=vr-r42-cn[17-18] tools/check_mooncake_rails.sbatch
MC_RAIL_IMAGE=/path/to/image-with-perftest.sqsh sbatch tools/check_mooncake_rails.sbatch
```

Optional overrides: `MC_RAIL_READER`, `MC_RAIL_DONOR`, `MC_RAIL_NICS`,
`MC_RAIL_IMAGE`, `MC_RAIL_OUTPUT_ROOT`, `MC_RAIL_TOOL_DIR`,
`MC_RAIL_DURATION`, `MC_RAIL_CASE_TIMEOUT`, `MC_RAIL_BUDGET`,
`MC_RAIL_INVENTORY_TIMEOUT`, `MC_RAIL_CLUSTER`, and `ENROOT_MAX_PROCESSORS`.
Keep the controller budget below the submitted walltime with three minutes reserved for cleanup; increase
the walltime and controller budget together if extending traffic. The selected
reader/donor must belong to the two-node allocation.
