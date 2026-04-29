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
