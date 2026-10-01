# TokenSpeed

Use `engine: tokenspeed` with `frontend.type: dynamo`. Every worker runs Dynamo's
TokenSpeed backend, `python3 -m dynamo.tokenspeed`, which registers with the Dynamo
frontend like the other Dynamo engines. Recipes:

- [Aggregated](../examples/tokenspeed/dynamo-agg.yaml)
- [Prefill/decode](../examples/tokenspeed/dynamo-disagg.yaml)

Configuration loading rejects any other frontend, `dynamo.sidecar`, and
`roles.<role>.kv_events`.

The backend follows
[TokenSpeed](https://github.com/lightseekorg/tokenspeed/tree/22251686ff26a2b2f495263b348db80980b8ac9e)
and Dynamo's
[`dynamo.tokenspeed`](https://github.com/ai-dynamo/dynamo/tree/7778c8d0cddb2a1ab7b2782c92cf97b30b2a5dcd/components/src/dynamo/tokenspeed).

## Image

The worker image must contain TokenSpeed and an `ai-dynamo` build with
`dynamo.tokenspeed`. Aggregated serving needs Dynamo 1.2.0 or newer; prefill/decode
needs a build that includes [ai-dynamo/dynamo#9237](https://github.com/ai-dynamo/dynamo/pull/9237),
which Dynamo 1.5.0 does not. Dynamo's
[TokenSpeed Dockerfile](https://github.com/ai-dynamo/dynamo/blob/7778c8d0cddb2a1ab7b2782c92cf97b30b2a5dcd/recipes/kimi-k2.5/tokenspeed/agg/nvidia/Dockerfile)
builds such an image from a TokenSpeed runner and the Dynamo checkout. The examples
set `dynamo.install: false` because the image already ships Dynamo.

## Arguments

`roles.<role>.args` are TokenSpeed
[server flags](https://github.com/lightseekorg/tokenspeed/blob/22251686ff26a2b2f495263b348db80980b8ac9e/docs/configuration/server.md),
such as `tensor-parallel-size`, `data-parallel-size`, `enable-expert-parallel`,
`max-model-len` and `gpu-memory-utilization`. Set the parallelism to match the
role's `gpus`. A role's `served-model-name` is the name every worker serves; without
one, workers serve the model directory name.

srtctl sets these flags on every worker:

| Flag | Value |
| --- | --- |
| `--model` | `/model`, the staged model path, or the Hugging Face ID |
| `--served-model-name` | the role's `served-model-name`, else the model directory name |
| `--host` | the worker's own IP on `network_interface` |
| `--port` | an allocated port, 10000 and up, 1024 apart on a node |
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
