# Host attribution metrics: pinning vs frontend placement

srt-slurm's host sampler collects the telemetry needed to answer, **from a
baseline run alone**, two questions that end-to-end serving metrics cannot:

1. Are worker ranks losing CPU time to scheduler contention/migration?
   (remedy: `backend.numa_cpu_bind: true`)
2. Is the frontend/etcd interfering with the workers sharing its node?
   (remedy: `frontend.dedicated_node` / `infra.etcd_nats_dedicated_node`)

Both effects are real and independently worth ~1% output throughput each at
high concurrency on GB300 disaggregated serving — but they are invisible in
throughput/TTFT alone, which is why the collectors below exist.

## Collection

With `observability.enabled: true`, the `/proc` host sampler runs on the
orchestrator node (in-process) **and on every other allocated node**
(`observability.host_sampler_all_nodes`, default true: one persistent
`srun --overlap` per node group running `host_sampler.py` standalone). Each
node writes `host_samples_<node>.jsonl` into the run's log dir; the ingest
merges them into `host_series.json` with a per-node `hosts` map. This closes
the previous gaps: worker nodes had no per-process host telemetry, and a
dedicated frontend node had none at all.

## Metric set

Per sampled process (workers, frontend, benchmark client — matched by cmdline):

| Field (raw JSONL) | Ingest series | Diagnoses | Points at |
|---|---|---|---|
| `run_delay_ns` (`/proc/pid/schedstat`) | `run_delay_ms_per_s` | Task runnable but not running: scheduler contention on its cores | pinning |
| `nr_migrations` (`/proc/pid/sched`) | `migrations_rate` | Cross-core churn; near-zero when pinned | pinning |
| `affinity_ncpus` (`sched_getaffinity`) | `affinity_ncpus` | Direct pinning-state observable (144 = floating, 36 = pinned rank on GB200/GB300) | pinning (config state) |
| `ctx_invol` (`/proc/pid/status`) | `ctx_invol_rate` | Involuntary descheduling (lock convoys, neighbor pressure) | pinning / placement |
| `cpu_jiffies` | `cpu_pct` | Per-process CPU use — splits a shared node's load into frontend vs etcd vs ranks | placement |
| host `procs_running/blocked` (`/proc/stat`) | `procs_runnable` | Whole-node run-queue pressure vs core count | either (localizes with the per-process rows) |
| `t` + `t_mono` | — | Per-node clock-offset estimation; cross-node wall clocks have been observed seconds apart | metric hygiene |

## Attribution logic (draft rubric, thresholds pending validation runs)

Comparing the node hosting frontend+etcd against its clean peers **within one
baseline run** (same model, same traffic mix, cache-aware normalization):

- **Placement signal**: the co-located node's worker ranks show elevated
  `run_delay_ms_per_s` / `ctx_invol_rate` correlated with `cpu_pct` bursts of
  the frontend/etcd processes, and its GPUs run ~2–3 pp lower utilization than
  peer nodes → move the frontend (`frontend.dedicated_node: true`).
- **Pinning signal**: elevated `migrations_rate` and `run_delay_ms_per_s`
  across **all** worker nodes (not just the shared one) with `affinity_ncpus`
  at the full core count → pin the ranks (`backend.numa_cpu_bind: true`).
- **Double dissociation** (how a metric earns its place): a pinning metric
  must go green when `numa_cpu_bind` flips on and stay unchanged when only the
  frontend moves; a placement metric the reverse.

Cross-node timing comparisons must estimate per-node clock offsets first
(pair `t` with `t_mono`, or use a constant frontend→worker dispatch offset);
raw cross-node wall-clock deltas are unreliable at millisecond scale.
