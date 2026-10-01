# Shepherd Model Gateway (SMG)

`frontend.type: smg` runs the [Shepherd Model Gateway](https://github.com/smg-project/smg)
in front of direct engine workers. SMG is an engine-agnostic Rust gateway with an
OpenAI-compatible API; srtctl hands it the statically allocated worker URLs, the same
way it does for the other static routers. No Dynamo, NATS, or etcd is involved.

Upstream references below are pinned to SMG
[v1.11.0](https://github.com/smg-project/smg/tree/3be823a700fabaff3add8a390cf78f163479d686).

## Supported topologies

| Topology | Backends | What srtctl launches |
| --- | --- | --- |
| Aggregated | any engine whose workers serve HTTP (`sglang`, `vllm`, `trtllm`, ...) | `smg launch --worker-urls <url> ...` |
| Prefill/decode | `sglang` | `smg launch --pd-disaggregation --prefill <url> <bootstrap-port> --decode <url>` |

SMG detects each HTTP worker's engine itself (`/v1/models` `owned_by`, then `/version`
and `/server_info`) and reads the served model name from the worker
([`detect_backend.rs`](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/model_gateway/src/workflow/steps/local/detect_backend.rs#L182-L191),
[`discover_metadata.rs`](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/model_gateway/src/workflow/steps/local/discover_metadata.rs#L230-L239)),
so srtctl passes no `--backend` and no model name. SMG routes a request only to workers
serving the model it names
([`router.rs`](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/model_gateway/src/routers/http/router.rs#L249-L252)),
so clients send the workers' served model name, as srtctl's benchmarks do.

P/D is listed for SGLang only. SMG's HTTP P/D router hands the KV cache over through
SGLang's bootstrap rendezvous (`bootstrap_host` / `bootstrap_port` / `bootstrap_room`,
[`pd_router.rs`](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/model_gateway/src/routers/http/pd_router.rs#L303-L305)).
A vLLM worker registered by URL carries no KV connector, so SMG sends the request through
and the decode worker recomputes the prompt
([upstream test note](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/e2e_test/router/test_pd_topologies.py#L105-L110)).
srtctl does not reject that layout, but it is not a KV-disaggregated run.

## Configuration

```yaml
frontend:
  type: smg
  container_image: smg # an image that ships SMG, e.g. lightseekorg/smg:1.11.0
  enable_multiple_frontends: false
  args:
    policy: cache_aware # any `smg launch` flag
```

- `container_image`: an engine image need not ship SMG. Point this at an image
  with the `smg` command (the official `lightseekorg/smg` image, or any image where
  `pip install smg` was run). Without it the model container is reused.
- `args`: passed to `smg launch` as `--<key> <value>`; `true` adds a bare flag and a
  list repeats the flag. `prometheus-port` is managed by srtctl and rejected here.
- `enable_multiple_frontends` / `num_additional_frontends`: as for every router,
  several SMG replicas behind nginx, or one SMG on the public port.

See [`examples/vllm/smg-agg.yaml`](../examples/vllm/smg-agg.yaml) (vLLM, aggregated)
and [`examples/sglang/smg-disagg.yaml`](../examples/sglang/smg-disagg.yaml) (SGLang, P/D).

## Launch and readiness

- **Command.** `smg launch`
  ([subcommand](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/model_gateway/src/main.rs#L153-L160)) with `--host 0.0.0.0 --port <frontend port>` and the
  workers from the allocated topology: `--worker-urls` for aggregated workers;
  `--pd-disaggregation`, one `--prefill URL BOOTSTRAP_PORT` per prefill worker and one
  `--decode URL` per decode worker for P/D
  ([`main.rs`](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/model_gateway/src/main.rs#L35-L72),
  [CLI arguments](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/model_gateway/src/main.rs#L213-L237),
  [P/D arguments](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/model_gateway/src/main.rs#L494-L509)).
  The Python package (what `pip install smg` and the official image install) provides
  the same `smg launch` subcommand and flags
  ([`cli.py`](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/bindings/python/src/smg/cli.py#L76-L82),
  [`router_args.py`](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/bindings/python/src/smg/router_args.py#L843-L876)).
- **Worker probe before start.** SMG waits at most `--worker-startup-timeout-secs`
  (1800 s by default,
  [`main.rs`](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/model_gateway/src/main.rs#L522-L524))
  for a startup worker to register, and a large model can load for longer, so srtctl
  waits for every advertised worker's `/health` to answer 200 before it starts SMG.
- **Readiness.** srtctl polls SMG's `GET /workers` and waits until
  `stats.prefill_count`, `stats.decode_count` and `stats.regular_count` cover the
  expected workers; aggregated workers count as `regular`
  ([route](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/model_gateway/src/server.rs#L993),
  [response](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/model_gateway/src/worker/service.rs#L204-L241)).
- **Metrics.** SMG serves Prometheus on its own listener. srtctl passes
  `--prometheus-port 29000` (on every interface, SMG's default host) and points
  tachometer's frontend target there
  ([`main.rs`](https://github.com/smg-project/smg/blob/3be823a700fabaff3add8a390cf78f163479d686/model_gateway/src/main.rs#L741-L748)).
  Workers are scraped on their own HTTP ports, as with the other static routers.
- **Logs.** Each replica writes `<node>_smg_<index>.out` in the job log directory and
  runs as the Slurm step `smg_<index>`.

## Not wired

srtctl advertises every worker over HTTP. SMG's gRPC, ZMQ and encode (EPD) worker
modes, its Kubernetes service discovery, mesh, and the cloud-provider, history, MCP
and WASM features are not configured by srtctl; `frontend.args` can still pass any
`smg launch` flag that needs no srtctl-managed process or port.
