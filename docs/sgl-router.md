# SGLang Router (sgl-router)

`frontend.type: sgl-router` runs [`sgl-router`](https://github.com/sgl-project/sglang/tree/main/sgl-router),
the Rust router that ships in the SGLang repository, in front of direct
`sglang.launch_server` workers. No Dynamo, NATS, or etcd is involved.

It is a different binary from the Model Gateway that
[`frontend.type: sglang-router`](sglang-router.md) launches
(`python -m sglang_router.launch_router`), and it rejects that gateway's flags.

Upstream references below are pinned to SGLang
[`324668d8d5b5`](https://github.com/sgl-project/sglang/tree/324668d8d5b5/sgl-router).

## What srtctl launches

```
sgl-router --worker-urls <url> <url> ... --host 0.0.0.0 --port <port> \
  --model-id <served model name> --tokenizer-path <tokenizer> <frontend.args>
```

- Aggregated and prefill/decode topologies use the same flat `--worker-urls`
  list. Each engine discloses its role and bootstrap port in `/server_info`, and
  the router classifies it from that
  ([`static_urls.rs`](https://github.com/sgl-project/sglang/blob/324668d8d5b5/sgl-router/src/discovery/static_urls.rs)).
  There is no `--pd-disaggregation`, `--prefill`, or `--decode`.
- The workers are the same commands as for the other static routers: HTTP on the
  allocated port, with `--disaggregation-mode` and the prefill bootstrap port
  added by srtctl. Only `engine: sglang` over HTTP is supported; `grpc-mode` on
  a role is rejected.
- `--model-id` is required by the router and must match the engines' served
  model name; srtctl passes the recipe's served name. Set `model-id` in
  `frontend.args` to override.
- `--tokenizer-path` is the mounted model's `tokenizer.json` for a local model
  and the repo id for an `hf:` model, so the router does not download a
  tokenizer for `--model-id`. Set `tokenizer-path` or `no-tokenizer` in
  `frontend.args` to override.
- The router runs in `frontend.container_image` when set (must include `/bin/sh`
  for Pyxis/Enroot), else in the model container with `/configs` appended to
  `PATH`. Upstream `lmsysorg/sgl-router` is distroless (no `/bin/sh`) and fails
  `enroot-switchroot` if launched directly as a Pyxis container; extract
  `/usr/local/bin/sgl-router` from it into `configs/sgl-router` (or build from
  source / layer a shell-capable image) to run inside the model container without
  a separate image or `setup_script`.

## Readiness and metrics

srtctl waits for every engine's `/health` before starting the router. The
router introspects a static worker three times, 100 ms apart, and otherwise
registers it as a plain worker serving no model
([`introspect.rs`](https://github.com/sgl-project/sglang/blob/324668d8d5b5/sgl-router/src/workers/introspect.rs)).

The job is ready when `GET /workers` lists the expected number of prefill and
decode (or plain) workers that are registered, healthy, and resolved a model.
`/readyz` alone only says that one usable pool exists.

The router serves Prometheus on its routing port at `/metrics`; workers are
scraped on their own HTTP ports.

## Routing options

Routing flags go in `frontend.args` verbatim. `chat-routing: reorg` selects the
current routing path; without it the legacy path runs with `round_robin`.
`policy: cache_aware` reads the engines' KV events, so set `kv_events` on the
roles; load-aware admission reads load reports, enabled on the engine with
`load-publish-endpoint: auto` in the role's `args`.

## Example

[`examples/sglang/sgl-router-disagg.yaml`](https://github.com/NVIDIA/srt-slurm/blob/main/examples/sglang/sgl-router-disagg.yaml):

```yaml
frontend:
  type: sgl-router
  enable_multiple_frontends: false
  args:
    chat-routing: "reorg"

engine: sglang
```
