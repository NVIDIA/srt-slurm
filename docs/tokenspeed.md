# TokenSpeed

Use `engine: tokenspeed` with `frontend.type: dynamo` or `frontend.type: smg`.

- **Dynamo.** Every worker runs Dynamo's TokenSpeed backend, `python3 -m dynamo.tokenspeed`,
  which registers with the Dynamo frontend like the other Dynamo engines.
- **SMG.** Every worker runs TokenSpeed's gRPC engine, `python3 -m smg_grpc_servicer.tokenspeed`,
  and srtctl hands SMG its `grpc://` URL. This is the pair `ts serve` runs on one node: the
  same engine module with `--host`/`--port`, and `smg launch --worker-urls grpc://<host:port>`
  ([`_proc.py`](https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/python/tokenspeed/cli/_proc.py#L39-L104)).

Recipes:

- Dynamo: [aggregated](https://github.com/NVIDIA/srt-slurm/blob/main/examples/tokenspeed/dynamo-agg.yaml), [prefill/decode](https://github.com/NVIDIA/srt-slurm/blob/main/examples/tokenspeed/dynamo-disagg.yaml)
- SMG: [aggregated](https://github.com/NVIDIA/srt-slurm/blob/main/examples/tokenspeed/smg-agg.yaml), [prefill/decode](https://github.com/NVIDIA/srt-slurm/blob/main/examples/tokenspeed/smg-disagg.yaml)

Configuration loading rejects the engine-specific routers (`sglang-router`,
`vllm-router`, ...), `dynamo.sidecar`, and `roles.<role>.kv_events`.

The backend follows
[TokenSpeed](https://github.com/lightseekorg/tokenspeed/tree/22251686ff26a2b2f495263b348db80980b8ac9e),
Dynamo's
[`dynamo.tokenspeed`](https://github.com/ai-dynamo/dynamo/tree/7778c8d0cddb2a1ab7b2782c92cf97b30b2a5dcd/components/src/dynamo/tokenspeed)
and SMG [v1.11.0](https://github.com/smg-project/smg/tree/3be823a700fabaff3add8a390cf78f163479d686).

## Image

The worker image must contain TokenSpeed and an `ai-dynamo` build with
`dynamo.tokenspeed`. Aggregated serving needs Dynamo 1.2.0 or newer; prefill/decode
needs a build that includes [ai-dynamo/dynamo#9237](https://github.com/ai-dynamo/dynamo/pull/9237),
which Dynamo 1.5.0 does not; the PyPI nightlies from `1.6.0.dev20260922` on do. Either
let srtctl install one at job start (`dynamo.source.pypi: "1.6.0.dev20260922"`) into an
image that ships TokenSpeed, or build an image with Dynamo's
[TokenSpeed Dockerfile](https://github.com/ai-dynamo/dynamo/blob/7778c8d0cddb2a1ab7b2782c92cf97b30b2a5dcd/recipes/kimi-k2.5/tokenspeed/agg/nvidia/Dockerfile)
from a TokenSpeed runner and the Dynamo checkout. The examples set
`dynamo.install: false` because their image already ships Dynamo.

Behind SMG the image needs TokenSpeed only. Installing it pulls in the gRPC engine
(`tokenspeed-smg-grpc-servicer`) and the SMG build it pins (`tokenspeed-smg`, which
provides the `smg` command)
([`pyproject.toml`](https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/python/pyproject.toml#L75-L77)), so SMG runs in the model
container and needs no `frontend.container_image`.

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
| `--port` | an allocated port, 10000 and up, 1024 apart on a node; behind SMG, the leader's gRPC port |
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
and spaces `--port` 1024 apart, so workers sharing a node do not collide. A
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

The engine's `--host` and `--port` are its gRPC listener
([`server.py`](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/grpc_servicer/smg_grpc_servicer/tokenspeed/server.py)), so srtctl
advertises the leader of each worker as `grpc://<host>:<port>`
(`TokenSpeedBackend.is_grpc_mode` is always true: TokenSpeed has no HTTP-only engine
server). A follower node of a multi-node worker runs the same module, as `ts serve` does
on a non-zero node rank
([`serve_smg.py`](https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/python/tokenspeed/cli/serve_smg.py#L873-L879)), and is not routed.
Before SMG starts, srtctl waits for every gRPC port to accept connections.

For prefill/decode srtctl passes `--pd-disaggregation --prefill grpc://<host:port>
<bootstrap-port> --decode grpc://<host:port>`. SMG's gRPC P/D table gives TokenSpeed
parallel dispatch with a KV bootstrap room
([`pd_protocol.rs`](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/model_gateway/src/routers/grpc/common/stages/pd_protocol.rs#L63-L85)):
it mints one room per request and sends both workers the prefill worker's host and
bootstrap port
([`helpers.rs`](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/model_gateway/src/routers/grpc/common/stages/helpers.rs#L764-L824)),
which the engine hands to Mooncake
([`servicer.py`](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/grpc_servicer/smg_grpc_servicer/tokenspeed/servicer.py#L1031-L1071)).
The prefill engine serves that bootstrap port
([`async_llm.py`](https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/python/tokenspeed/runtime/engine/async_llm.py#L245-L253)).
srtctl sets `TOKENSPEED_SKIP_GRPC_WARMUP=1` on prefill and decode workers: the engine's
startup warmup is a generate without a bootstrap room
([`server.py`](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/grpc_servicer/smg_grpc_servicer/tokenspeed/server.py#L211-L282)),
which a prefill engine cannot finish without a decode peer and an attention-DP decode engine
fails to place on a DP rank
([`data_parallel_controller.py`](https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/python/tokenspeed/runtime/engine/data_parallel_controller.py#L425-L430)).

In gRPC mode SMG tokenizes and applies the chat template itself; pass
`tool-call-parser` / `reasoning-parser` in `frontend.args` when the model needs them.
The gRPC engine starts no HTTP listener, so these workers serve no Prometheus
`/metrics`; SMG's own metrics stay on its Prometheus port. TokenSpeed's ZMQ engine
(`ts serve --headless`) is not used: SMG reaches a ZMQ engine only over local `ipc://`
sockets.
