# Monitoring

## Table of Contents

- [Live Dashboard (srtctl monitor)](#live-dashboard-srtctl-monitor)
- [Checking Job Status](#checking-job-status)
- [Log Directory](#log-directory)
- [Log Structure](#log-structure)
- [Key Files](#key-files)
- [Common Commands](#common-commands)
- [Connecting to Running Jobs](#connecting-to-running-jobs)

---

## Live Dashboard (srtctl monitor)

`srtctl monitor` is a live terminal dashboard that brings everything into one place: SLURM queue state, job lifecycle stage, worker readiness, and benchmark metrics — all auto-refreshing without juggling `squeue` and `tail -f`.

```bash
srtctl monitor                          # Active + recently completed jobs
srtctl monitor --all                    # Also include older jobs from outputs/
srtctl monitor --outputs /path/to/out   # Override outputs directory
srtctl monitor --interval 10            # Refresh interval in seconds (default: 5)
srtctl monitor --once                   # Print snapshot and exit
srtctl monitor --resume KEY             # Resume a previous session
```

The outputs directory is auto-detected from `./outputs/` or `../outputs/`.

### Columns

| Column | Description |
|--------|-------------|
| Job ID | SLURM job ID (`▶` marks the selected row) |
| Slurm | Queue state: RUNNING / PENDING / ENDED … |
| Stage | Lifecycle stage inferred from the sweep log |
| Workers | Live readiness, e.g. `2/4P  4/4D` |
| Time | Elapsed wall time |
| Config | GPU type, topology, benchmark type, ISL/OSL |
| Metrics | Throughput (tok/s), TTFT, TPOT |

**Lifecycle stages:** Starting → Starting Infra → Head Ready → Starting Workers → Awaiting Workers → Starting Frontend → Benchmarking → Completed / Failed / Killed / Timed Out

### Keybindings

**Main view**

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate jobs |
| `↵` | Open detail view |
| `y` | Open `config.yaml` in vim |
| `d` | Delete output dir (finished) or cancel job (active) — prompts to confirm |
| `c` | Toggle last vs all concurrencies in Metrics |
| `a` | Toggle active-only vs all jobs |
| `q` | Quit |

**Detail view** (`↵` on a job — sweep log left, worker + benchmark logs right)

| Key | Action |
|-----|--------|
| `↑` / `↓` | Cycle panels (sweep / worker / benchmark) |
| `←` / `→` | Cycle worker files or benchmark concurrency sections |
| `↵` | Open current log in vim |
| `r` | Toggle auto-refresh |
| `ESC` | Back to job list |

### Session Resume

On exit, a session key is printed:

```
To resume this session, use  srtctl monitor --resume abc123def456
```

Sessions are saved to `/tmp/srt-dash-<user>.json` and restore the full set of tracked job IDs, including completed jobs.

---

## Checking Job Status

```bash
# List your running jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# Cancel a job
scancel <job_id>
```

## Log Directory

After submission, `srtctl` tells you where logs are stored:

```
Submitted batch job 4459
Logs: logs/4459_4P_1D_20251122_041341/
```

The directory name follows the pattern: `{job_id}_{prefill}P_{decode}D_{timestamp}`

## Log Structure

```
logs/4459_4P_1D_20251122_041341/
│
├── config.yaml                              # Resolved job configuration
├── sglang_config.yaml                       # SGLang worker configuration
├── sbatch_script.sh                         # Generated SLURM script
├── nginx.conf                               # Load balancer configuration
├── 4459.json                                # Job metadata
│
├── log.out                                  # Main orchestration stdout
├── log.err                                  # Main orchestration stderr
├── benchmark.out                            # Benchmark results
├── benchmark.err                            # Benchmark errors
│
├── {node}_prefill_w{n}.out                  # Prefill worker stdout
├── {node}_prefill_w{n}.err                  # Prefill worker stderr (SGLang logs)
├── {node}_decode_w{n}.out                   # Decode worker stdout
├── {node}_decode_w{n}.err                   # Decode worker stderr (SGLang logs)
├── {node}_frontend_{n}.out                  # Frontend stdout
├── {node}_frontend_{n}.err                  # Frontend stderr
├── {node}_nginx.out                         # Nginx stdout
├── {node}_nginx.err                         # Nginx stderr
├── {node}_config.json                       # Per-node SGLang config dump
│
├── hwinfo/                                  # NVLink/MNNVL snapshots
│   ├── before.out                           # Fabric state before the run
│   └── after.out                            # Fabric state when the run ended
│
├── cached_assets/                           # Cached model assets
└── sa-bench_isl_1024_osl_1024/              # Benchmark results
    ├── isl_1024_osl_1024_concurrency_128_req_rate_inf.json
    ├── isl_1024_osl_1024_concurrency_512_req_rate_inf.json
    └── ...
```

## Key Files

### log.out

The main orchestration log showing node assignments, worker launches, and the frontend URL:

```
Node 0: watchtower-aqua-cn01
Node 1: watchtower-aqua-cn02
...
Master IP address (node 1): 10.30.1.49
Nginx node (node 0): watchtower-aqua-cn01
...
Prefill worker 0 leader: watchtower-aqua-cn01 (10.30.1.163)
Launching prefill worker 0, node 0 (local_rank 0): watchtower-aqua-cn01
...
Decode worker 0 leader: watchtower-aqua-cn05 (10.30.1.153)
...
Frontend available at: http://watchtower-aqua-cn01:8000
```

### benchmark.out

Shows benchmark progress and results:

```
Polling http://localhost:8000/health every 5 seconds...
Model is not ready, waiting for 4 prefills and 1 decodes to spin up.
Model is ready.

Warming up model with concurrency 128
============ Serving Benchmark Result ============
Successful requests:                     640
Benchmark duration (s):                  93.97
Request throughput (req/s):              6.81
Output token throughput (tok/s):         6278.02
---------------Time to First Token----------------
Mean TTFT (ms):                          1924.07
Median TTFT (ms):                        342.39
P99 TTFT (ms):                           13652.77
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.78
Median TPOT (ms):                        15.48
P99 TPOT (ms):                           22.36
==================================================
```

### Worker Logs ({node}\_prefill_w0.err, {node}\_decode_w0.err)

SGLang worker logs showing model loading, memory allocation, and runtime info. Check these for debugging CUDA errors, OOM issues, or NCCL failures.

### hwinfo/before.out, hwinfo/after.out

NVLink, MNNVL and GPU memory state on every worker node, captured before any load
and again when the run ends (on success as well as failure). Each command is
recorded with its output and, when it failed, its exit code:

```
===== theia0245 | before | 2026-08-13T07:53:30Z =====

# ---------- NVLink ----------

$ nvidia-smi nvlink -e
GPU 0: NVIDIA GB300
	 Link 0: Replay Errors: 0
	 Link 0: Recovery Errors: 0
	 Link 0: CRC Errors: 0
```

The commands run per node in the order below, grouped into the sections that
appear in the file.

**GPU inventory** — which card is which.

| Command | What it answers |
| --- | --- |
| `nvidia-smi --query-gpu=index,name,serial,uuid,pci.bus_id --format=csv` | Which physical GPU a rank in a crash log refers to. Serials are what a hardware ticket needs. |
| `nvidia-smi --version` | Whether one node in the domain runs a different driver than the rest. |

**GPU memory and processes** — what was already on the cards before the job.

| Command | What it answers |
| --- | --- |
| `nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv` | Whether the GPUs were clean at startup. A worker that dies with `Free memory on device ... is less than desired GPU memory utilization` names no culprit; this shows how much was missing and on which cards. |
| `nvidia-smi --query-gpu=index,memory.reserved --format=csv` | How much the driver itself holds. Explains memory that is used while no process owns it. |
| `nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv` | Which processes hold device memory. Usually a previous job that has not finished dying. |
| `fuser -v /dev/nvidia*` | Processes holding the device open that NVML does not report, for example one in another container. |
| `ps -eo pid,ppid,user,rss,etime,args --sort=-rss \| head -15` | Identifies a leftover worker by its command line and how long it has been running. |

**NVLink** — the links themselves.

| Command | What it answers |
| --- | --- |
| `nvidia-smi nvlink -s` | Link state and negotiated speed. A link that is down or trained low shows up here. |
| `nvidia-smi nvlink -e` | Replay, recovery and CRC counters. The before/after diff of this command is what identifies a degrading link. |
| `nvidia-smi topo -m` | Which GPU pairs are actually NVLink-connected rather than falling back to PCIe. |

**MNNVL / IMEX** — whether remote peers are reachable at all.

| Command | What it answers |
| --- | --- |
| `ls -al /dev/nvidia-caps-imex-channels/` | Whether the kernel side of MNNVL is wired up on this node. |
| `cat /etc/nvidia-imex/nodes_config.cfg` | Whether the node list matches the allocation. A mismatch is a classic cause of remote NVLink faults. |
| `cat /etc/nvidia-imex/config.cfg` | The daemon's own settings, including the ports it expects. |
| `systemctl is-active nvidia-imex`, `systemctl status nvidia-imex` | Whether the daemon is running, and what it said if it is not. |
| `nvidia-imex-ctl -c <copy> -N -H` | Domain state and the hosts in it, as the daemon sees them. Run against a copy of the config with log and stats paths redirected to `/tmp`, because the shipped paths are root-only and an unprivileged run otherwise reports a misleading parse error. |
| `nvidia-smi -q \| grep -A8 -i "^ *Fabric"` | Per-GPU fabric State, Status and CliqueId. A GPU outside the clique cannot reach remote peers even though its local links look healthy. |

**Driver and GPU faults** — what the machine logged on its own.

| Command | What it answers |
| --- | --- |
| `dmesg -T \| grep -iE "xid\|nvlink\|nvswitch\|imex" \| tail -60` | Xid codes and driver-level link events, with timestamps to correlate against the crash. |
| `nvidia-smi -q -d ROW_REMAPPER` | Whether the GPU was already degrading before the run started. |

The point is the difference between the two files:

```bash
diff hwinfo/before.out hwinfo/after.out
```

A job that died with `CUDA error: uncorrectable NVLink error` reports nothing
about which link failed. Error counters that moved between the snapshots do, and
the GPU serials in the same file are what a hardware ticket needs. Counters that
grow during a run that still passed are the earliest warning that a link is
about to take the next job down.

Everything here is best effort: a missing tool or a read the job user is not
allowed to make is recorded and skipped, and a hung driver call is cut off after
20 seconds (`HWINFO_CMD_TIMEOUT`). Collection never fails a job.

#### Preflight

The `before` snapshot is also read back at startup, and a run whose hardware
already looks fatal is stopped there instead of failing twenty minutes later:

```
RuntimeError: Preflight found 1 problem(s) that would break this run:
  - theia0282.lyris.clusters.nvidia.com: IMEX domain node #11 is UNAVAILABLE
Full snapshot: /.../logs/hwinfo/before.out
Set preflight.enabled: false to run anyway.
```

Four conditions are treated as fatal:

| Finding | Why it stops the run |
| --- | --- |
| A process still holds device memory | A worker will die on the memory check with a message that names no owner. The pid and its size come from `--query-compute-apps`. |
| A card has less free memory than `gpu_memory_utilization x total` | Exactly the arithmetic the engine does in `init_device`, so this card is already known to fail. |
| A node in the IMEX domain is not `READY` | MNNVL spans every node in `nodes_config.cfg`, so a broken peer breaks an MNNVL all2all backend even when that node is outside this allocation. |
| A GPU did not join the fabric clique | `State`, `Status` or health other than Completed/Success/Healthy means remote peers are unreachable while local links still look fine. |

A finding about the shared domain is reported once for the offending peer, not
once per node that observed it. A command that was missing or timed out yields no
finding, so an older snapshot or a node without `nvidia-smi` still passes.

The utilization threshold is taken from the recipe — the largest
`gpu-memory-utilization` (or SGLang's `mem-fraction-static`) across
`prefill`, `decode` and `aggregated` — and can be overridden:

```yaml
preflight:
  enabled: true                  # false: log the findings and run anyway
  gpu_memory_utilization: 0.92   # default: whatever the recipe asks for
```

### config.yaml

The fully resolved configuration showing exactly what ran, with all aliases expanded and defaults applied.

## Common Commands

```bash
# List your running jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# Cancel a job
scancel <job_id>

# Watch logs
tail -f logs/4459_*/*_prefill_*.err logs/4459_*/*_decode_*.err

# Watch benchmark progress
tail -f logs/4459_*/benchmark.out
```

## Connecting to Running Jobs

The `log.out` file includes commands to connect to running nodes
