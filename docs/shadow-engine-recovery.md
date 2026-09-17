# Shadow Engine Recovery (vLLM)

[Shadow engine recovery](https://developer.nvidia.com/blog/restore-llm-inference-capacity-in-seconds-with-shadow-engine-recovery-in-nvidia-dynamo/) keeps a fully initialized standby `dynamo.vllm` engine parked on the same GPUs as the serving engine. Dynamo's GPU Memory Service (GMS) owns the model weights in a process of its own, so the standby maps the one copy already in HBM instead of loading another. When the serving engine dies, the standby takes over in seconds instead of the minutes a cold restart costs; the dead engine is relaunched in place and becomes the new standby.

`engine.failover` turns this on for a vLLM recipe behind the Dynamo frontend. srtslurm launches the GMS sidecar and the extra engines, hands every engine the environment Dynamo's Kubernetes operator would, and relaunches an engine that exits.

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

On SLURM the whole node is allocated to the job and every step on it sees the same `/dev/nvidia*`. Three steps of one worker share its GPUs by setting the same `CUDA_VISIBLE_DEVICES`. Everything else the feature needs is a process property: the GMS server's CUDA VMM allocations are reference counted and survive any engine's exit as long as one mapping remains, the election is a POSIX `flock` on a file, and the relaunch is a shell loop. None of that knows about Kubernetes.

The two things Kubernetes gives for free that srtslurm has to provide are a directory that all three containers of a worker share on the node (an `emptyDir` there; a node-local host path here) and the "restart the failed container" policy (the `restartPolicy: Always` of the pod; a respawn loop around the engine here).

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
  failover: {}                    # defaults: one shadow, restart always, /dev/shm
roles:
  agg:
    nodes: 1
    workers: 2
    gpus: 1
    args:
      tensor-parallel-size: 1
      gpu-memory-utilization: 0.4   # leave room for the shadow, see Sizing GPU Memory
```

`examples/features/vllm-failover.yaml` is the runnable version. `srtctl dry-run -f <recipe>` prints a "Shadow Engine Recovery" panel with the layout before you submit.

The container must ship the `gpu_memory_service` package. The `ai-dynamo` PyPI wheel does not include it, so the `dynamo.source: pypi` path does not work here; the `nvcr.io/nvidia/ai-dynamo/vllm-runtime` images from 1.4 on do, with `dynamo.install: false`.

## What Runs

Per worker and node, in this order:

```
node im-b200-c021                                   /dev/shm/srtctl-<job>/agg_0/
                                                     +- gms_<GPU-UUID>_weights.sock
  step gms_agg_0_<node>          GMS sidecar         +- gms_<GPU-UUID>_kv_cache.sock
    python3 -m gpu_memory_service --device 0  ---->  +- failover.lock
    (one server per GPU of the worker)                   ^          ^
                                                          |          |
  step agg_0_<node>              engine 0  ENGINE_ID=0 ---+  flock --+   holds the lock: serving
    python3 -m dynamo.vllm ... --load-format gms --gms-shadow-mode                  (registered)
                                                          |          |
  step agg_0_<node>_e1           engine 1  ENGINE_ID=1 ---+  flock --+   waiting: parked shadow
    python3 -m dynamo.vllm ... --load-format gms --gms-shadow-mode                  (not registered)
```

- **GMS sidecar** (`gms_<role>_<index>_<node>`). Runs in the job container with the worker's `CUDA_VISIBLE_DEVICES` and starts one `gpu_memory_service` server per GPU of the worker, each binding a `weights` and a `kv_cache` socket in the worker's directory. The step prints `GMS ready:` once every socket exists and the worker stage waits for that line before starting the engines. It is critical (a worker without its weight server cannot recover) and is stopped after the engines at cleanup, when it removes the directory.
- **Engine 0** (`<role>_<index>_<node>`). `ENGINE_ID=0` loads the weights from disk into GMS, or imports them read-only when they are already there after a relaunch. Every other engine imports read-only, so no two engines ever hold GMS's write lock at once.
- **Shadow engines** (`<role>_<index>_<node>_e<k>`). Same command, same GPUs, own ports (`DYN_SYSTEM_PORT`, KV events, NIXL side channel, `VLLM_PORT`). Each engine initializes fully (weights mapped, CUDA graphs captured, communicators up), sleeps, and blocks on `failover.lock`. The one that acquires it wakes, materializes its KV cache and registers with the frontend. The frontend health gate counts registered instances, so shadows do not count toward the expected worker total.
- **Relaunch** (`restart: always`). Each engine step is a loop: when the engine exits, the step logs `[srtctl] agg_0 engine 0 exited with code 137; relaunching in 5s` and starts it again. The step itself never exits, so the process monitor sees nothing and `roles.<role>.critical` never fires for an engine death. The relaunched engine loads through GMS (fast) and parks as the new shadow.

Multi-node workers get the same treatment per node; every engine of a multi-node worker gets its own `--master-port` (`29500 + 100 * ENGINE_ID`) so their `torch.distributed` stores do not collide. Only single-node workers have been run so far.

## What srtslurm Owns vs What You Set

| Piece | Owner | Value |
| --- | --- | --- |
| `--load-format gms --gms-shadow-mode` | srtslurm | added to every engine; a `load-format` in `roles.<role>.args` must be `gms` or absent |
| `ENGINE_ID`, `GMS_SOCKET_DIR`, `FAILOVER_LOCK_PATH`, `DYN_VLLM_GMS_SHADOW_MODE`, `DYN_SYSTEM_STARTING_HEALTH_STATUS` | srtslurm | per engine, the same names the Dynamo operator injects |
| `CUDA_VISIBLE_DEVICES` | srtslurm | pinned on the engines and the sidecar (no `--device-ids`), so "device k" is the same GPU for all of them |
| ports | srtslurm | every engine is its own `Process` with its own ports, from the usual allocators |
| `--master-port` (multi-node) | srtslurm | `29500 + 100 * ENGINE_ID`; a `master-port` in `args` moves the base |
| `gpu-memory-utilization` | you | see [Sizing GPU Memory](#sizing-gpu-memory) |
| the container | you | must ship `gpu_memory_service`; `dynamo.install: false` |
| `roles.<role>.critical` | you | only matters with `restart: never`, when a dead engine's step exits |

## Configuration Reference

```yaml
engine:
  type: vllm
  failover:
    shadow_engines: 1                  # standby engines per worker
    restart: always                    # always | never
    restart_backoff_seconds: 5         # delay before an engine is relaunched
    shared_dir: /dev/shm               # node-local host path every container on the node sees
    gms_startup_timeout_seconds: 120   # how long the sidecar may take to bind its sockets
```

| Key | Default | Meaning |
| --- | --- | --- |
| `shadow_engines` | `1` | Standby engines per worker. Each one costs its CUDA context, captured graphs and communicator buffers in HBM while parked, and no weights. |
| `restart` | `always` | `always` wraps every engine in a relaunch loop (Kubernetes `restartPolicy: Always`). `never` leaves the step to exit with the engine; the surviving engine still takes over, and `roles.<role>.critical` decides whether the run ends. |
| `restart_backoff_seconds` | `5` | Fixed delay before a relaunch. |
| `shared_dir` | `/dev/shm` | Where `srtctl-<job_id>/<role>_<index>/` (sockets and lock file) is created. It must be the same directory in every container on the node and it must be node-local: enroot bind-mounts the host's `/dev/shm` into every container, while `/tmp` is a fresh tmpfs per container and a Unix socket cannot be shared through the cluster filesystem. |
| `gms_startup_timeout_seconds` | `120` | The sidecar exits 1 if its sockets are not all up by then, and the job fails with a pointer to its log. |

Validation refuses `engine.failover` without `frontend.type: dynamo` (the election lives in `dynamo.vllm`; a static router would also list the parked shadows), with `dynamo.sidecar: true`, with `data-parallel-size` on any role, or with a `load-format` other than `gms`. It warns when Dynamo is pip-installed at job start, because that path cannot supply `gpu_memory_service`.

## Sizing GPU Memory

Two engines share each GPU. The weights are counted once (GMS owns them; both engines map the same physical pages). Everything else is per engine:

- the serving engine holds its KV cache, sized from `gpu-memory-utilization` as usual;
- a parked shadow holds its CUDA context, captured CUDA graphs and NCCL/NIXL buffers, typically one to a few GB, and no KV cache (it reserves the address range and materializes it on promotion).

So `gpu-memory-utilization` must leave the shadow's standing cost free. The shadow's own profiling run accounts for the active engine being present, so a value that works is one where `weights + active KV cache + one shadow's context and graphs` fits. Start conservatively (the example uses 0.4 for a small model), read both engines' logs for `[GMS] Scratch-KV engaged` and the KV cache sizes, and raise it from there. A shadow that OOMs during its warm-up exits, is relaunched, and OOMs again: the relaunch loop makes that visible as a repeating `relaunching in 5s` line in its log.

## Testing a Failover

Kill the engine process, not its step. `scancel --signal=KILL <job>.<step>` ends the whole step, including the relaunch loop, and the process monitor then treats it as a worker exit (`roles.<role>.critical` applies). What the blog measures is a process crash, and the engines run in the host PID namespace, so from the login node:

```bash
# which engine is serving: the lock file names the holder
srun --jobid <job> --overlap -w <node> -N1 cat /dev/shm/srtctl-<job>/agg_0/failover.lock   # engine-0

# find that engine's process and SIGKILL it
srun --jobid <job> --overlap -w <node> -N1 bash -c '
  for pid in $(pgrep -f "dynamo.vllm"); do
    if tr "\0" "\n" < /proc/$pid/environ 2>/dev/null | grep -qx "ENGINE_ID=0" &&
       tr "\0" "\n" < /proc/$pid/environ | grep -q "FAILOVER_LOCK_PATH=/dev/shm/srtctl-'"$SLURM_JOB_ID"'/agg_0/"; then
      echo "killing $pid"; kill -9 $pid; break
    fi
  done'
```

Then watch:

- the shadow's log (`<node>_agg_w0_e1.out`): `[Shadow] Lock acquired, waking engine`, then `[Shadow] Engine awake, registering with discovery`;
- the frontend's `/health`: the instance count drops by one when the dead engine's lease expires and comes back with a new instance id when the shadow registers;
- engine 0's log (`<node>_agg_w0.out`): `[srtctl] agg_0 engine 0 exited with code 137; relaunching in 5s`, then a GMS import (no weight load from disk) and `[Shadow] Engine paused, startup probe now passing, waiting for lock`. The roles have swapped.

The lock file now reads `engine-1`. Kill again and it swaps back.

## Validation

Validated on sa-b200 (B200, one node, two TP1 Qwen3-0.6B workers with one shadow each, `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2`, Dynamo 1.4.2, vLLM 0.26.0, GMS 0.9.0). See the timeline in the PR description; the numbers below are from that run.

| Event | Observed |
| --- | --- |
| GMS sidecar ready | see PR |
| Engine 0 serving (weights loaded through GMS) | see PR |
| Shadow parked | see PR |
| SIGKILL engine 0 to shadow registered with the frontend | see PR |
| Killed engine relaunched and parked as the new shadow | see PR |

## Limitations

- vLLM only, behind the Dynamo frontend. SGLang and TRT-LLM have GMS weight-loading integrations but no shadow election in their Dynamo workers yet.
- No `data-parallel-size`. Each DP rank would need its own GMS session and lock.
- Multi-node workers are launched with the right ports but have not been run.
- Process failures only. A node or GPU failure still ends the run (SLURM has no spare in the allocation).
- The GMS sidecar is critical and not relaunched. If it dies, the engines keep their mappings and serve, but a relaunched engine cannot import the weights, so the run fails on the next engine exit rather than limping on.
- A promoted shadow starts with an empty KV cache (the current Dynamo preview does not carry it), so TTFT bumps briefly after a cutover.
- Tachometer scrapes every engine's `DYN_SYSTEM_PORT`, shadows included; a parked shadow reports healthy and no traffic.
- `restart: always` hides engine exits from the process monitor by design. A shadow that can never come up (OOM at warm-up) shows only as a repeating relaunch line in its log.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `GMS sidecar gms_agg_0_<node> exited with code 1 before its sockets came up` and the log says `No module named gpu_memory_service` | the container has no GMS | use a `vllm-runtime` image with `dynamo.install: false` |
| `GMS startup timed out: 0 of 2 sockets` while the servers are running | the servers bound their sockets somewhere the wrapper cannot see, or `shared_dir` is not writable | check the sidecar log's `Socket path:` lines; keep `shared_dir` on `/dev/shm` |
| engines log `connection refused` on `gms_*.sock` | `shared_dir` is not shared between the containers (a per-container `/tmp`) | use `/dev/shm` or another host bind mount that enroot gives every container |
| both engines serve at once (two instances per worker in `/health`) | the lock directory was removed or recreated while engines ran, so they lock different inodes | do not touch `srtctl-<job>/` during a run; a relaunched engine reopens the same path, which is fine as long as the file stays |
| a shadow relaunches every few seconds with a CUDA OOM | not enough free HBM for its context and graphs next to the active engine's KV cache | lower `gpu-memory-utilization` |
| `--gms-shadow-mode requires --load-format gms` | a `load-format` in `args` other than `gms` | remove it; validation catches this before submit |
