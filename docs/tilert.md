# TileRT decode

Use `roles.prefill.engine: vllm`, `roles.decode.engine: tilert`, and
`frontend.type: tilert-router`. Do not set a top-level `engine` alongside role engines.
Set `frontend.enable_multiple_frontends: false`.
Recipes:

- [B200 / NIXL](https://github.com/NVIDIA/srt-slurm/blob/main/examples/tilert/glm5-disagg.yaml)
- [MI355X / Mooncake](https://github.com/NVIDIA/srt-slurm/blob/main/examples/tilert/glm5-rocm-mooncake-disagg.yaml)

The supported pairing is **vLLM prefill + TileRT decode + TileRT router**.
Configuration loading rejects other engine/router combinations before Slurm
submission. Other prefill engines need a TileRT protocol integration and an
update to the frontend adapter's validation; using NIXL or Mooncake alone is not
enough. Set the model profile, sequence limit, KV layout, and transport consistently
in vLLM's `TileRTConnector` and the decode arguments. The configuration checks do
not verify the connector installed inside the image or prove KV-layout compatibility.

Single-node decode with one prefill worker follows TileRT 0.1.6's
[decode server](https://github.com/tile-ai/TileRT/blob/0ea19371f120977761bfa0cf6a2d08415d8fce0e/tilert/pd_vllm/decode_server.py)
and [router](https://github.com/tile-ai/TileRT/blob/0ea19371f120977761bfa0cf6a2d08415d8fce0e/tilert/pd_vllm/pd_router.py).
Several prefill workers, [multi-node decode](#multi-node-decode), and the `dsv41`
profile need TileRT 0.1.8 or later. 0.1.8 is published as the
[`tilert==0.1.8` wheel on PyPI](https://pypi.org/project/tilert/0.1.8/)
(`tilert-0.1.8-cp312-cp312-manylinux_2_34_aarch64.whl`, sha256
`17ae557029a14c2676e20ae5d61c895fb5658a06f0bdd232fe7b21d478f1f838`); its source
is not on GitHub, whose newest tag is 0.1.6. In that wheel:

- `tilert/pd_vllm/pd_router.py` splits `--vllm-url` on commas and sends each
  prefill to the URL with the fewest requests in flight. 0.1.6 takes one URL.
- `tilert/pd_vllm/decode_server.py` `main()` re-executes under
  `torch.distributed.run` when the profile needs more than one rank per node and
  `LOCAL_RANK` is unset. With `TILERT_PD_NNODES` > 1 it passes `--nnodes`,
  `--node-rank` from `TILERT_PD_NODE_RANK`, `--rdzv-backend static`, and
  `--master-addr`/`--master-port` from `TILERT_PD_RDZV`.
- The `dsv41` profile (`tilert/pd_vllm/profiles/dsv41.py`, 4 ranks per node) builds
  the engine on every rank through `start_node`. Followers return before the HTTP
  socket is bound, so only global rank 0 listens on `--http-port`.
- `TILERT_MP_GROUP_SIZE` (`tilert/runtime/multiproc.py`) and `TILERT_DSV41_TP8XN`
  (`tilert/models/deepseek_v4_flash/tp8xn.py`) select the dsv41 TP8 / TP8x3 layouts
  listed in `tilert/pd_vllm/profiles/dsv41_decode.py`.

## Images and weights

Use images containing the engine and required connector. Set
`roles.prefill.container` and `roles.decode.container` for the worker images.
`model.container` supplies the shared image for non-worker tasks.
The router image defaults to `model.container`; use `frontend.container_image`
to override it. It must contain `tilert.pd_vllm.pd_router` and its dependencies.
The B200 recipe's image aliases must be defined in `srtslurm.yaml`.

Convert weights with TileRT before serving and mount them at
`roles.decode.args.model-weights-dir`. Set `frontend.args.model-path` to the
tokenizer path or Hugging Face model ID, including when `parser: none`.

## Limits

The router supports one or more decode workers. It supports one prefill worker
with TileRT 0.1.6 and several with TileRT 0.1.8 or later. Each decode worker runs
on a single node unless [multi-node decode](#multi-node-decode) is enabled. TileRT requires transferred
KV state and cannot serve aggregate requests through this adapter.

srtctl sets the decode HTTP/control ports and `--engine tilert`; other options
come from `roles.decode.args`. Workers must pass `/health` before the router
starts. TileRT decode has no `/metrics`, so only prefill metrics are scraped.

The [role-engine restrictions](topology.md#roles) also apply.

## Multi-node decode

Set `roles.decode.engine.multi_node_launch` to let one decode worker span several
nodes, for example TP8 across two GB300 NVL72 trays:

- `node-env` (recommended) runs one `python -m tilert.pd_vllm.decode_server` per
  node with `TILERT_PD_NNODES`, `TILERT_PD_NODE_RANK`, and `TILERT_PD_RDZV`; the
  server starts that node's ranks.
- `torchrun` runs `torchrun --nnodes --nproc-per-node --node-rank --master-addr
  --master-port -m tilert.pd_vllm.decode_server` on each node. Every local rank
  gets the same `--http-port`, so use it only with a profile whose followers do
  not bind it, such as `dsv41` in TileRT 0.1.8.

srtctl allocates one rendezvous port on the worker's first node and an HTTP port
on every node. Only the first node is registered with and health-checked by the
router. Unset, decode workers must fit on one node. Either way, a multi-node
decode worker must use whole nodes. Both modes need TileRT 0.1.8 or later.

## Benchmarks

Use `custom` with a client that sends `/v1/chat/completions` and an explicit
model name, as in the InferenceX validations. The public model name comes from
vLLM prefill's `served-model-name`; it need not be repeated on the decode engine.
An explicit, conflicting decode model name is rejected.

TileRT 0.1.6 has no `/v1/models` endpoint and only supports streaming on
`/v1/chat/completions`. The built-in `sa-bench` runner uses streaming
`/v1/completions` and is therefore incompatible. GPQA's automatic model discovery
also cannot work; `lm-eval` needs an explicit `MODEL_NAME` in `benchmark.env`
to bypass discovery. Neither evaluation path was validated by the TileRT sweeps.
