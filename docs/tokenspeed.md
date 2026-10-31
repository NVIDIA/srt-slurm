# TokenSpeed

Use `engine: tokenspeed` with `frontend.type: dynamo` or `frontend.type: smg`.

- **Dynamo.** Every worker runs Dynamo's TokenSpeed backend, `python3 -m dynamo.tokenspeed`,
  which registers with the Dynamo frontend like the other Dynamo engines.
- **SMG.** Every worker runs `python3 -m tokenspeed.cli serve --headless`.
  SMG binds a TCP ZMQ handshake for each logical worker on the frontend node;
  the engine ranks connect to it. The gateway tokenizes and serves the OpenAI API.
  This supports aggregated and prefill/decode serving, including multi-node workers.

Recipes:

- Dynamo: [aggregated](https://github.com/NVIDIA/srt-slurm/blob/main/examples/tokenspeed/dynamo-agg.yaml), [prefill/decode](https://github.com/NVIDIA/srt-slurm/blob/main/examples/tokenspeed/dynamo-disagg.yaml)
- SMG: [aggregated](https://github.com/NVIDIA/srt-slurm/blob/main/examples/tokenspeed/smg-agg.yaml), [prefill/decode](https://github.com/NVIDIA/srt-slurm/blob/main/examples/tokenspeed/smg-disagg.yaml)

Configuration loading rejects the engine-specific routers (`sglang-router`,
`vllm-router`, ...), `dynamo.sidecar`, and `roles.<role>.kv_events`.

The Dynamo integration follows
[TokenSpeed](https://github.com/lightseekorg/tokenspeed/tree/22251686ff26a2b2f495263b348db80980b8ac9e),
Dynamo's
[`dynamo.tokenspeed`](https://github.com/ai-dynamo/dynamo/tree/7778c8d0cddb2a1ab7b2782c92cf97b30b2a5dcd/components/src/dynamo/tokenspeed)
with the matching TokenSpeed version in its image. The SMG integration targets
[TokenSpeed `0.1.1.post20261101`](https://github.com/lightseekorg/whl/releases/tag/tokenspeed-nightly-v0.1.1.post20261101)
(source [5d670a1](https://github.com/lightseekorg/tokenspeed/tree/5d670a1659ed3ac23383dbfab43f6907c4e3011a))
and its pinned `tokenspeed-smg==1.13.0.post20261027`.

## Image

Behind Dynamo, the worker image must contain TokenSpeed and an `ai-dynamo` build with
`dynamo.tokenspeed`. Aggregated serving needs Dynamo 1.2.0 or newer; prefill/decode
needs a build that includes [ai-dynamo/dynamo#9237](https://github.com/ai-dynamo/dynamo/pull/9237),
which Dynamo 1.5.0 does not; the PyPI nightlies from `1.6.0.dev20260922` on do. Either
let srtctl install one at job start (`dynamo.source.pypi: "1.6.0.dev20260922"`) into an
image that ships TokenSpeed, or build an image with Dynamo's
[TokenSpeed Dockerfile](https://github.com/ai-dynamo/dynamo/blob/7778c8d0cddb2a1ab7b2782c92cf97b30b2a5dcd/recipes/kimi-k2.5/tokenspeed/agg/nvidia/Dockerfile)
from a TokenSpeed runner and the Dynamo checkout. The examples set
`dynamo.install: false` because their image already ships Dynamo.

Behind SMG, install the published TokenSpeed daily wheel and its dependencies,
including the matching kernel wheel for the image's CUDA version and architecture.
TokenSpeed pins `tokenspeed-smg`, which provides the `smg` command. The same image
runs the workers and SMG unless `frontend.container_image` selects another image
with that matching gateway build. No gRPC servicer is launched.

## Arguments

`roles.<role>.args` are TokenSpeed
[server flags](https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/docs/configuration/server.md),
such as `tensor-parallel-size`, `data-parallel-size`, `enable-expert-parallel`,
`max-model-len` and `gpu-memory-utilization`. Set the parallelism to match the
role's `gpus`. A role's `served-model-name` is the name every worker serves; without
one, workers serve the model directory name.

srtctl sets these flags on every worker, under either frontend:

| Flag | Value |
| --- | --- |
| `--model` | `/model`, the staged model path, or the Hugging Face ID |
| `--served-model-name` | the role's `served-model-name`, else the model directory name |
| `--host` | the worker's own IP on `network_interface` |
| `--port` | an allocated engine port-scan base, 10000 and up, 1024 apart on a node |
| `--dist-init-addr` | the endpoint leader's IP and an allocated port |
| `--nnodes`, `--node-rank` | the endpoint's node count and this node's rank |
| `--disaggregation-mode` | `prefill` or `decode`, on prefill/decode workers |
| `--disaggregation-bootstrap-port` | an allocated port, on prefill workers |

TokenSpeed derives several listeners from two of these. Its
[`PortArgs.init_new`](https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/python/tokenspeed/runtime/utils/server_args.py#L2513-L2602)
picks the NCCL port 100 to 1000 above `--port`, and binds a control-plane cluster
on the leader that starts at the `--dist-init-addr` port: six ports, plus one per
attention-DP rank
([`data_parallel_controller.py`](https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/python/tokenspeed/runtime/engine/data_parallel_controller.py#L293-L294)).
srtctl reserves a block of six ports plus the endpoint's GPU count for that cluster,
and spaces `--port` 1024 apart, so workers sharing a node do not collide. Under the Dynamo integration, a
follower node serves its health check on `--host` and `--port`
([`engine.py`](https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/python/tokenspeed/runtime/entrypoints/engine.py#L636-L642)).
Each node runs its own single-node srun step, so srtctl passes `--nnodes`,
`--node-rank` and `--dist-init-addr` instead of relying on TokenSpeed's
[Slurm step detection](https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/docs/serving/parallelism.md#under-a-launcher).

## Parallelism

Parallelism is in the role's args and needs nothing else from srtctl. Attention DP with
expert parallelism (DEP) is `data-parallel-size: <gpus>` with `enable-expert-parallel: true`;
tensor parallelism with expert parallelism (TEP) is `tensor-parallel-size: <gpus>` with
`enable-expert-parallel: true`. A worker that spans nodes sizes these over all of its GPUs:
srtctl passes its node count and rank, and TokenSpeed runs `world size / nnodes` ranks
on each node
([parallelism.md](https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/docs/serving/parallelism.md#multi-node)).
Behind Dynamo, prefill and decode workers need attention DP 1 (see below); behind SMG a
decode worker can be DEP.

## KV cache offload

Unless a role sets `disable-kvstore`, every TokenSpeed rank keeps evicted KV blocks in a
pinned host-memory KVStore (L2). Each rank allocates its own: `kvstore-size` sets it in GB,
otherwise it is `kvstore-ratio` (default 2.0) times the rank's GPU KV pool, so size it to
the node's memory.

`kvstore-storage-backend: mooncake` adds Mooncake Store under it as L3
([server.md](https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/docs/configuration/server.md#host-l2-and-mooncake-store-l3)).
Declare a `mooncake-master` service: srtctl starts the master on the infra node and sets
`MOONCAKE_MASTER`, `MOONCAKE_TE_META_DATA_SERVER` and `MOONCAKE_LOCAL_HOSTNAME` on every
worker, which TokenSpeed's Mooncake client reads
([`mooncake.py`](https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/python/tokenspeed/runtime/cache/l3/mooncake.py#L94-L140)).
Set `MOONCAKE_GLOBAL_SEGMENT_SIZE`, `MOONCAKE_PROTOCOL` or `MOONCAKE_DEVICE` in the role's
env. A TokenSpeed rank stops when Mooncake rejects a write for lack of space, so size the
segments for the workload and let the master evict before it fills (the service's `args`,
for example `--eviction_high_watermark_ratio=0.7`). The service runs `mooncake_master` from
TokenSpeed's `tokenspeed-mooncake` dependency in the model image; give it a `preamble` that
installs TokenSpeed when the image gets it from `setup_script`, which services do not run.

```yaml
services:
  - name: mooncake-master
    type: mooncake-master
roles:
  agg:
    args:
      kvstore-size: 64
      kvstore-storage-backend: mooncake
```

## Prefill/decode

TokenSpeed moves the KV cache with Mooncake. A prefill worker advertises its
bootstrap server at `--host` and the bootstrap port
([`runtime_disaggregated_endpoint`](https://github.com/ai-dynamo/dynamo/blob/7778c8d0cddb2a1ab7b2782c92cf97b30b2a5dcd/components/src/dynamo/tokenspeed/disagg.py#L71-L105));
the Dynamo frontend passes it to the decode worker with each request. Prefill
workers register as the `prefill` component and decode workers as `backend`
([`args.py`](https://github.com/ai-dynamo/dynamo/blob/7778c8d0cddb2a1ab7b2782c92cf97b30b2a5dcd/components/src/dynamo/tokenspeed/args.py#L105-L108)).

Dynamo checks the remaining requirements when the worker starts
([`validate_disagg_compatibility`](https://github.com/ai-dynamo/dynamo/blob/7778c8d0cddb2a1ab7b2782c92cf97b30b2a5dcd/components/src/dynamo/tokenspeed/disagg.py#L50-L68)):
the transfer backend is Mooncake, attention DP is 1, and `prefix-granularity` is
positive (TokenSpeed's default is 64). Scale prefill or decode with more workers.
Set `disaggregation-ib-device` in the role's args when Mooncake does not pick the
right RDMA device.

## SMG

Set `frontend.enable_multiple_frontends: false` and leave frontend placement on
`head`. A ZMQ engine connects to one gateway, so multiple frontend replicas and
non-head frontend placement are rejected at configuration load.

For each endpoint, srtctl allocates one gateway handshake port (31000 and up,
unique across the job). SMG receives `tcp://<head-ip>:<handshake-port>` in
`--worker-urls`, or `--prefill` / `--decode` for disaggregation, together with
`--backend tokenspeed`. Its input and output sockets use OS-assigned TCP ports on
the same frontend host, returned to the engines during the handshake.
Every node of the endpoint runs `ts serve --headless` with:

- `--data-parallel-address <head-ip>`
- `--data-parallel-rpc-port <handshake-port>`
- `--zmq-engine-index 0`

Attention-DP ranks register their own identities and declare their group size;
SMG learns that size from the handshake. Different endpoints may have different
DP sizes. Each endpoint retains its own distributed rendezvous and each node its
own engine port-scan base. See TokenSpeed's
[headless entrypoint](https://github.com/lightseekorg/tokenspeed/blob/5d670a1659ed3ac23383dbfab43f6907c4e3011a/python/tokenspeed/runtime/entrypoints/engine.py)
and [gateway socket contract](https://github.com/lightseekorg/tokenspeed/blob/5d670a1659ed3ac23383dbfab43f6907c4e3011a/python/tokenspeed/cli/_proc.py).

SMG must bind the handshake before an engine can become ready. srtctl therefore
does not probe ZMQ URLs as HTTP or gRPC worker listeners before starting SMG.
The gateway's `/workers` readiness gate still runs before the benchmark.

For prefill/decode, the engine advertises its own KV bootstrap host and allocated
bootstrap port in its handshake metadata. No bootstrap port is appended to the
router's `--prefill tcp://...` URL: that URL names the gateway, not the prefill
node. The engine-side `--host` remains the worker's own network-interface IP.

SMG receives the model path/ID and a `--model-alias` for a differing
`served-model-name`. A role's `tokenizer` supplies `--tokenizer-path` unless the
frontend sets one explicitly. Local models use the shared `/model` mount in the
gateway even when weights are staged on the worker nodes. Set model-specific
`tool-call-parser`, `reasoning-parser`, and other gateway options in
`frontend.args`.

With `roles.<role>.args.enable-metrics: true` and the Prometheus reporter,
every worker node gets a separate allocated `--metrics-port` (6800 and up on each
node). `metrics-host` binds all interfaces unless explicitly supplied. Tachometer
and the benchmark's `AIPERF_SERVER_METRICS_URLS` scrape these HTTP `/metrics`
listeners, including multi-node followers. Disabled metrics and non-Prometheus
reporters advertise no engine metrics URL. SMG's own metrics remain on its
separate Prometheus listener. The headless engine exposes no HTTP profiling
control API.

The engine's handshake address, engine index and metrics port are managed by
srtctl; conflicting role arguments are rejected rather than silently overwritten.
