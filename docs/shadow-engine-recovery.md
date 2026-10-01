# Shadow Engine Recovery (vLLM)

[Shadow engine recovery](https://developer.nvidia.com/blog/restore-llm-inference-capacity-in-seconds-with-shadow-engine-recovery-in-nvidia-dynamo/) keeps a fully initialized standby `dynamo.vllm` engine parked on the same GPUs as the serving engine. Dynamo's GPU Memory Service (GMS) owns the model weights in a process of its own, so the standby maps the one copy already in HBM instead of loading another. When the serving engine dies, the standby takes over in seconds instead of the minutes a cold restart costs; the dead engine is relaunched in place and becomes the new standby.

`engine.failover` turns this on for a vLLM recipe behind the Dynamo frontend. It implies a `gms` service, one instance per worker, and launches the extra engines with the environment Dynamo's Kubernetes operator would give them. `roles.<role>.restart` relaunches an engine that exits.

## Table of Contents

- [Why no DRA](#why-no-dra)
- [Quick Start](#quick-start)
- [What Runs](#what-runs)
- [What srtslurm Owns vs What You Set](#what-srtslurm-owns-vs-what-you-set)
- [Configuration Reference](#configuration-reference)
- [Sizing GPU Memory](#sizing-gpu-memory)
- [Testing a Failover](#testing-a-failover)
- [Validation](#validation)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

---

## Why no DRA

The blog lists Dynamic Resource Allocation (Kubernetes 1.34 with the NVIDIA GPU DRA driver) as a requirement. That requirement is about Kubernetes, not about the feature: a pod's containers each get their own GPU allocation, and the default device plugin cannot hand the same GPU to the GMS sidecar and to two engine containers at once. DRA can.

On SLURM the whole node is allocated to the job and every step on it sees the same `/dev/nvidia*`. The three steps of one worker share its GPUs by setting the same `CUDA_VISIBLE_DEVICES`. Everything else the feature needs is a process property: the GMS server's CUDA VMM allocations are reference counted and survive any engine's exit as long as one mapping remains, the election is a POSIX `flock` on a file, and the relaunch is a supervisor watching step exits. None of that knows about Kubernetes.

The two things Kubernetes gives for free that srtslurm provides through existing machinery are a directory all containers of a worker share on the node (an `emptyDir` there; a node-local host path here) and the "restart the failed container" policy (the pod's `restartPolicy: Always`; `roles.<role>.restart` here).

## Quick Start

```yaml
schema: 2
model:
  container: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2    # ships gpu_memory_service
dynamo:
  install: false
frontend:
  type: dynamo
engine:
  type: vllm
  failover: {}                    # defaults: one shadow, /dev/shm
roles:
  agg:
    nodes: 1
    workers: 2
    gpus: 1
    restart:
      policy: always              # relaunch an exited engine in place as the new shadow
    args:
      tensor-parallel-size: 1
      gpu-memory-utilization: 0.4   # leave room for the shadow, see Sizing GPU Memory
```

`examples/features/vllm-failover.yaml` is the runnable version. `srtctl dry-run -f <recipe>` prints a "Shadow Engine Recovery" panel with the layout and lists the `gms` service under Services before you submit.

The container must ship the `gpu_memory_service` package. The `ai-dynamo` PyPI wheel does not include it, so the `dynamo.source: pypi` path does not work here; the `nvcr.io/nvidia/ai-dynamo/vllm-runtime` images from 1.4 on do, with `dynamo.install: false`.

## What Runs

Per worker and node, in this order:

```
node im-b200-c021                                        /dev/shm/srtctl-<job>/agg_0/
                                                          +- gms_<GPU-UUID>_weights.sock
  step service_gms_agg_0_<node>    gms service (per worker) +- gms_<GPU-UUID>_kv_cache.sock
    python3 -m gpu_memory_service --device 0  ------------> +- failover.lock
    (one server per GPU of the worker)                         ^          ^
                                                                |          |
  step agg_0_<node>                engine 0  ENGINE_ID=0 -------+  flock --+   holds the lock: serving
    python3 -m dynamo.vllm ... --load-format gms --gms-shadow-mode                      (registered)
                                                                |          |
  step agg_0_<node>_e1             engine 1  ENGINE_ID=1 -------+  flock --+   waiting: parked shadow
    python3 -m dynamo.vllm ... --load-format gms --gms-shadow-mode                      (not registered)
```

- **The gms service** (`service_gms_<role>_<index>_<node>`). A [service](services.md) implied by `engine.failover`, `type: gms`, `placement.per: worker`: one instance per worker on every worker node, in the `before_workers` phase, in the job container, with the worker's `CUDA_VISIBLE_DEVICES`. It starts one `gpu_memory_service` server per GPU of the worker, each binding a `weights` and a `kv_cache` socket in the worker's directory, and prints `GMS ready:` once every socket exists; the service stage waits for that line before moving on. Critical (a worker without its weight server cannot recover), stopped after the engines at cleanup (shutdown tier 1), when it removes the directory. Declare it by name to change its container or its readiness timeout.
- **Engine 0** (`<role>_<index>_<node>`). `ENGINE_ID=0` loads the weights from disk into GMS, or imports them read-only when they are already there after a relaunch. Every other engine imports read-only, so no two engines ever hold GMS's write lock at once.
- **Shadow engines** (`<role>_<index>_<node>_e<k>`). Same command, same GPUs, own ports (`DYN_SYSTEM_PORT`, KV events, NIXL side channel, `VLLM_PORT`). Each engine initializes fully (weights mapped, CUDA graphs captured, communicators up), sleeps, and blocks on `failover.lock`. The one that acquires it wakes, materializes its KV cache and registers with the frontend. The frontend health gate counts registered instances, so shadows do not count toward the expected worker total.
- **Relaunch** (`roles.<role>.restart`). The worker supervisor's unit is one engine of a worker, so when the serving engine dies only its step is relaunched, with the role's backoff and `max_restarts`, its log rotated to `<log>.out.<n>`, and the event recorded in `worker_restarts.json` and the lockfile; the engine that just took over is left alone. The relaunched engine loads through GMS (fast) and parks as the new shadow; the supervisor's readiness probe passes once it is parked, because Dynamo reports a standby healthy on purpose. Without a restart policy an exited engine's step simply ends and `roles.<role>.critical` decides whether the run continues (the shadow still takes over either way).

Multi-node workers get the same treatment per node; every engine of a multi-node worker gets its own `--master-port` (`29500 + 100 * ENGINE_ID`) so their `torch.distributed` stores do not collide. Only single-node workers have been run so far.

Why the gms service is per worker rather than per node: GMS names its sockets after the NVML UUID of "device k" while the engine picks device k by CUDA index. With a subset `CUDA_VISIBLE_DEVICES` those disagree (NVML ignores it), so a node-level server would hand the worker on GPU 1 the socket of GPU 0. In the worker's own device view both sides name the same socket. This is what `placement.per: worker` exists for; see [services.md](services.md#placement).

## What srtslurm Owns vs What You Set

| Piece | Owner | Value |
| --- | --- | --- |
| the `gms` service | srtslurm | implied by `engine.failover`; declare `services: [{name: gms, type: gms, ...}]` to change its container or readiness timeout |
| `--load-format gms --gms-shadow-mode` | srtslurm | added to every engine; a `load-format` in `roles.<role>.args` must be `gms` or absent |
| `ENGINE_ID`, `GMS_SOCKET_DIR`, `FAILOVER_LOCK_PATH`, `DYN_VLLM_GMS_SHADOW_MODE`, `DYN_SYSTEM_STARTING_HEALTH_STATUS` | srtslurm | per engine, the same names the Dynamo operator injects |
| `CUDA_VISIBLE_DEVICES` | srtslurm | pinned on the engines and the gms instance (no `--device-ids`), so "device k" is the same GPU for all of them |
| ports | srtslurm | every engine is its own `Process` with its own ports, from the usual allocators |
| `--master-port` (multi-node) | srtslurm | `29500 + 100 * ENGINE_ID`; a `master-port` in `args` moves the base |
| `gpu-memory-utilization` | you | see [Sizing GPU Memory](#sizing-gpu-memory) |
| the container | you | must ship `gpu_memory_service`; `dynamo.install: false` |
| `roles.<role>.restart` | you | relaunches an exited engine as the new shadow; `roles.<role>.critical` applies once its `max_restarts` are spent, or from the start without a policy |

## Configuration Reference

```yaml
engine:
  type: vllm
  failover:
    shadow_engines: 1                  # standby engines per worker
    shared_dir: /dev/shm               # node-local host path every container on the node sees
roles:
  agg:
    restart:                           # see docs/config-reference.md, section "restart"
      policy: always
      backoff_seconds: 5
services:                              # optional: only to change the implied gms service
  - name: gms
    type: gms
    readiness:
      log: {pattern: "GMS ready:"}
      timeout_seconds: 300             # also bounds the servers' own startup
```

| Key | Default | Meaning |
| --- | --- | --- |
| `shadow_engines` | `1` | Standby engines per worker. Each one costs its CUDA context, captured graphs and communicator buffers in HBM while parked, and no weights. |
| `shared_dir` | `/dev/shm` | Where `srtctl-<job_id>/<role>_<index>/` (sockets and lock file) is created. It must be the same directory in every container on the node and it must be node-local: enroot bind-mounts the host's `/dev/shm` into every container, while `/tmp` is a fresh tmpfs per container and a Unix socket cannot be shared through the cluster filesystem. |

Validation refuses `engine.failover` without `frontend.type: dynamo` (the election lives in `dynamo.vllm`; a static router would also list the parked shadows), with `dynamo.sidecar: true`, with `data-parallel-size` on any role, or with a `load-format` other than `gms`. It warns when Dynamo is pip-installed at job start, because that path cannot supply `gpu_memory_service`. A declared `gms` service is refused without `engine.failover` or with `placement.per: node`.

## Sizing GPU Memory

Two engines share each GPU. The weights are counted once (GMS owns them; both engines map the same physical pages). Everything else is per engine:

- the serving engine holds its KV cache, sized from `gpu-memory-utilization` as usual;
- a parked shadow holds its CUDA context, captured CUDA graphs and NCCL/NIXL buffers, typically one to a few GB, and no KV cache (it reserves the address range and materializes it on promotion).

So `gpu-memory-utilization` must leave the shadow's standing cost free. The shadow's own profiling run accounts for the active engine being present, so a value that works is one where `weights + active KV cache + one shadow's context and graphs` fits. Start conservatively (the example uses 0.4 for a small model), read both engines' logs for `[GMS] Scratch-KV engaged` and the KV cache sizes, and raise it from there. A shadow that OOMs during its warm-up exits; with a restart policy it is relaunched until `max_restarts` is spent, then the role's `critical` flag decides.

## Testing a Failover

Kill the engine process, not its step. `scancel --signal=KILL <job>.<step>` ends the step, and the supervisor (or, without a policy, the process monitor) treats that as the worker exit it is. What the blog measures is a process crash, and the engines run in the host PID namespace, so from the login node:

```bash
# which engine is serving: the lock file names the holder
srun --jobid <job> --overlap -w <node> -N1 cat /dev/shm/srtctl-<job>/agg_0/failover.lock   # engine-0

# SIGKILL that engine (the dynamo.vllm parent and its EngineCore children)
srun --jobid <job> --overlap -w <node> -N1 bash -c '
  for pid in $(pgrep -f "^python3 -m dynamo.vllm"); do
    env=$(tr "\0" "\n" < /proc/$pid/environ 2>/dev/null) || continue
    if echo "$env" | grep -qx "ENGINE_ID=0" &&
       echo "$env" | grep -qx "FAILOVER_LOCK_PATH=/dev/shm/srtctl-'"$SLURM_JOB_ID"'/agg_0/failover.lock"; then
      echo "killing $pid"; kill -9 $pid $(pgrep -P $pid)
    fi
  done'
```

Then watch:

- the shadow's log (`<node>_agg_w0_e1.out`): `[Shadow] Lock acquired, waking engine`, then `[Shadow] Engine awake, registering with discovery` and `failover_state engine=1 -> active`;
- the frontend's `/health`: the dead instance id disappears and a new one takes its place;
- the sweep log: `Worker agg_0_<node> exited with code 137; relaunching agg_0 in 5s (restart 1/3)`, then `Relaunched agg_0 as agg_0_<node>_r1`;
- engine 0's new log (`<node>_agg_w0.out`; the crash log moved to `<node>_agg_w0.out.1`): `Connected with rw_or_ro lock (granted=ro)` and `Read mode: imported 1.18 GiB` (weights imported from GMS, no load from disk), then `[Shadow] Engine sleeping, startup probe now passing, waiting for lock`. The roles have swapped.

The lock file now reads `engine-1`. Kill again and it swaps back. The step logs are written by `srun`, which buffers, so read timings from the timestamps inside the lines rather than from when they appear in the file.

## Validation

Validated on sa-b200 (one B200 node, two TP1 Qwen3-0.6B workers with one shadow each, `examples/features/vllm-failover.yaml`, `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2`: Dynamo 1.4.2, vLLM 0.26.0, GMS 0.9.0), job 16074 on 2026-09-17. Times are from the timestamps inside the step logs.

| Event | Observed |
| --- | --- |
| GMS start to `GMS ready:` (both sockets bound) | 9 s |
| Engine 0 start to serving: weights loaded from disk into GMS, `init engine` 34 s (13.5 s compile), lock taken, KV cache (69 GiB) materialized, registered | 1 min 40 s |
| Shadow start to parked: weights imported from GMS, `init engine` 34 s, `Scratch-KV engaged` (0.5 GiB physical), sleeping on the lock | 1 min 40 s (started 2 s after engine 0) |
| SIGKILL of the serving engine to the shadow holding the lock | 0.1 to 0.2 s (the lock is polled every 100 ms) |
| Lock held to awake: weights remapped (60 allocations, 1.18 GiB), KV cache reallocated (28 allocations, 69.23 GiB) | 0.27 s (`It took 0.273367 seconds to wake up tags {'weights', 'kv_cache'}`) |
| Awake to registered: the frontend logs `added model` | 0.05 s |
| SIGKILL to the frontend routing to the shadow | about 0.5 s; the test script saw `/health` with the new instance id at 7.5 s and 11.7 s in two runs, at its own poll granularity |
| Frontend drops the dead instance's event publisher (lease expiry) | 10 s after the kill |
| Killed engine relaunched in place: `Connected with rw_or_ro lock (granted=ro)`, `Read mode: imported 1.18 GiB` in 1 s (no load from disk), `init engine` 21.9 s, parked | 49 s after the kill (in-step relaunch, compile cache warm) |
| Completions before, during (against the survivor), and after | all answered |

`nvidia-smi` during the run: 78.5 GiB used of 183 GiB per GPU, of which each engine process holds 2.7 GiB (CUDA context, graphs, communicators) and the rest is the GMS-owned weights plus the serving engine's KV cache. The weights appear under neither process.

A second cutover on the same job, killing engine 1 so the relaunched engine 0 (which had imported the weights read-only) took the lock back, behaved the same way.

Job 16086 (same recipe plus `roles.agg.restart: {policy: always, backoff_seconds: 5}`, on a build that merges this branch with `roles.<role>.restart`) is the layout as merged: the `gms` service launched `service_gms_agg_0_<node>` and `service_gms_agg_1_<node>` (each ready 9 s after its start) before the engines, and the supervisor tracked four units (`agg_0`, `agg_0_e1`, `agg_1`, `agg_1_e1`).

| Event | Observed |
| --- | --- |
| SIGKILL of engine 0 to the shadow active and registered | 0.4 s (lock 08:12:34.96, active 08:12:35.35) |
| Supervisor notices the exit (monitor tick) and schedules the relaunch | 5 s after the exit: `Worker agg_0_<node> exited with code 137; relaunching agg_0 in 5s (restart 1/3)` |
| Relaunched as `agg_0_<node>_r1`; the shadow's step untouched | 11 s later (5 s backoff plus the srun) |
| Relaunched engine imports the weights from GMS (`granted=ro`, `imported 1.18 GiB`), `init engine` 34 s (13.5 s compile: a new step is a new container, so the compile cache is cold), parked | 95 s after the kill; `worker_restarts.json`: `outcome: ready, ready_seconds: 90.7`, crash log rotated to `<node>_agg_w0.out.1` |
| Reverse cutover (kill engine 1; the relaunched engine 0 takes the lock; the supervisor relaunches engine 1 as `agg_0_<node>_e1_r1`) | same shape, both engines back |

A relaunched engine pays the compile again because each srun step is a fresh container. Mounting a persistent `TORCHINDUCTOR_CACHE_DIR` / `VLLM_CACHE_ROOT` through `container_mounts` brings the relaunch close to the in-step number above.

## Limitations

- vLLM only, behind the Dynamo frontend. SGLang and TRT-LLM have GMS weight-loading integrations but no shadow election in their Dynamo workers yet.
- No `data-parallel-size`. Each DP rank would need its own GMS session and lock.
- Multi-node workers are launched with the right ports but have not been run.
- Process failures only. A node or GPU failure still ends the run (SLURM has no spare in the allocation).
- The gms service is critical and not relaunched. If it dies, the engines keep their mappings and serve, but a relaunched engine cannot import the weights, so the run fails on the next engine exit rather than limping on.
- A promoted shadow starts with an empty KV cache (the current Dynamo preview does not carry it), so TTFT bumps briefly after a cutover.
- Tachometer scrapes every engine's `DYN_SYSTEM_PORT`, shadows included; a parked shadow reports healthy and no traffic.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `services[gms] exited with code 1 on <node> before its readiness probe passed` and the log says `No module named gpu_memory_service` | the container has no GMS | use a `vllm-runtime` image with `dynamo.install: false` |
| `GMS startup timed out: 0 of 2 sockets` while the servers are running | the servers bound their sockets somewhere the wrapper cannot see, or `shared_dir` is not writable | check the service log's `Socket path:` lines; keep `shared_dir` on `/dev/shm` |
| engines log `connection refused` on `gms_*.sock` | `shared_dir` is not shared between the containers (a per-container `/tmp`) | use `/dev/shm` or another host bind mount that enroot gives every container |
| both engines serve at once (two instances per worker in `/health`) | the lock directory was removed or recreated while engines ran, so they lock different inodes | do not touch `srtctl-<job>/` during a run; a relaunched engine reopens the same path, which is fine as long as the file stays |
| a shadow is relaunched every backoff with a CUDA OOM | not enough free HBM for its context and graphs next to the active engine's KV cache | lower `gpu-memory-utilization` |
| `--gms-shadow-mode requires --load-format gms` | a `load-format` in `args` other than `gms` | remove it; validation catches this before submit |
| the shadow took over but the dead engine never came back | no `roles.<role>.restart` policy | add one; without it the step ends and `critical` decides |
