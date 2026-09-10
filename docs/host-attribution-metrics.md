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

## Decision rubric (thresholds from the c1010 validation matrix, GB300/oci-aga)

You are looking at `host_series.json` from ONE run. You do not need to know what
"taskset" or "frontend placement" are — the rubric names the config change.

**Step 1 — is the bottleneck host-CPU-side at all?**
Look at the per-node `procs` map for the busiest process per worker node
(highest `cpu_pct`). If every worker node shows `run_delay_ms_per_s` p50
< 0.1 and `migrations_rate` ≈ 0, host-CPU scheduling is NOT the problem —
stop here. (Validated: clean nodes sit at 0.00–0.01 ms/s.)

**Step 2 — check the pinning state directly.**
`affinity_ncpus` of the worker ranks equals the node's full logical-CPU count
(e.g. 144 or 288) → the ranks are NOT pinned. Remedy:
`backend.numa_cpu_bind: true`. Validated effect at c1010: +4.0% output
throughput on 288-CPU GB300 nodes (+1.1% on 144-CPU nodes in the reference
campaign — the gain grows with core count). If `affinity_ncpus` equals
(CPUs ÷ GPUs per node), the ranks are already pinned.

**Step 3 — look for the single-node asymmetry.**
Compare each worker rank's `run_delay_ms_per_s` p50 against the median of its
peers on other nodes. Threshold: **>10× the peer median AND >0.5 ms/s absolute,
on exactly the node(s) that also host a non-worker process with
`affinity_ncpus` = full width and `cpu_pct` > 1000** (the frontend: measured
~4,100–5,000% of one core at c1010). That is co-location interference.
Remedy: `frontend.dedicated_node: true`. Validated effect: +0.5% throughput
(+0.85% in the reference campaign) — and the asymmetry itself is huge even
when the throughput cost is small: measured 140–350× on the shared node,
collapsing to 1× in all three dedicated-frontend runs.
Note the dissociation, confirmed both ways across 7 runs: this asymmetry is
UNCHANGED by pinning (190× with ranks pinned), and `affinity_ncpus` is
UNCHANGED by moving the frontend. Each signal names exactly one remedy.

**Expected-gain estimate**: single-node asymmetry affecting 1 of N prefill
groups → small-percent gain (≈ its share of prefill capacity); full-width
affinity on all ranks → the pinning gain for your node's core count.

Cross-node timing comparisons must estimate per-node clock offsets first
(pair `t` with `t_mono`, or use a constant frontend→worker dispatch offset);
raw cross-node wall-clock deltas are unreliable at millisecond scale —
observed inter-node skew up to 2.1 s.
