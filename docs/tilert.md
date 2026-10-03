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

The adapter follows TileRT 0.1.6's
[decode server](https://github.com/tile-ai/TileRT/blob/0ea19371f120977761bfa0cf6a2d08415d8fce0e/tilert/pd_vllm/decode_server.py)
and [router](https://github.com/tile-ai/TileRT/blob/0ea19371f120977761bfa0cf6a2d08415d8fce0e/tilert/pd_vllm/pd_router.py).

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

The router supports one prefill worker and one or more decode workers, each
decode worker on a single node. TileRT requires transferred KV state and cannot
serve aggregate requests through this adapter.

srtctl sets the decode HTTP/control ports and `--engine tilert`; other options
come from `roles.decode.args`. Workers must pass `/health` before the router
starts. TileRT decode has no `/metrics`, so only prefill metrics are scraped.

The [role-engine restrictions](topology.md#roles) also apply.

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
