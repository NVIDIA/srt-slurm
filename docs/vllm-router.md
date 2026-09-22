# vLLM Router

`frontend.type: vllm-router` runs the official
[vLLM Router](https://github.com/vllm-project/router) in front of direct
`vllm serve` workers. srtctl supplies the statically allocated worker URLs;
vLLM remains responsible for the DP/TP/PP engine topology inside each worker.

## Responsibilities

- The existing vLLM backend allocates endpoints and derives `per_node` DP,
  cross-node TP/PP rendezvous, device IDs, and headless followers.
- The frontend adapter starts Router, supplies aggregate or P/D worker URLs,
  adds NIXL bootstrap ports for P/D, and derives Router's node-local DP
  expansion factor.
- Readiness first validates Router's `/workers` counts, then requires HTTP 200
  from every exact base `/health` URL advertised to Router. This prevents
  benchmark and eval traffic from racing a partially started DP pool.
- Router stdout/stderr is captured in the normal job log directory alongside
  backend and benchmark logs. `frontend.container_image` can select a separate
  Router image; otherwise the model container is reused.

DP recipes must use upstream's default `engine.dp_launch_mode: per_node`.
The deprecated `per_gpu` mode launches Dynamo registrations rather than
independently routable API servers and is rejected for vLLM Router DP.

## Aggregate cases

### One direct engine behind Router

This is the smallest smoke topology. Router is functionally optional, but it
validates the adapter and provides the same public interface as larger cases.

```yaml
frontend:
  type: vllm-router
  enable_multiple_frontends: false
  args:
    policy: consistent_hash

resources:
  gpus_per_node: 8

engine: vllm
roles:
  agg:
    nodes: 1
    workers: 1
    args:
      tensor-parallel-size: 8
```

### Multiple aggregate replicas

Each logical aggregate worker contributes one or more routable base URLs.
Router receives them through `--worker-urls` and distributes sessions between
the expanded ranks.

```yaml
resources:
  gpus_per_node: 8

roles:
  agg:
    nodes: 4
    workers: 4
```

### Node-local data parallelism

For DEP8 on two four-GPU nodes, upstream srt-slurm creates one hybrid-LB
`vllm serve` process on each node. Router receives both base URLs and srtctl
adds `--intra-node-data-parallel-size 4`, exposing all eight DP ranks.

```yaml
resources:
  gpus_per_node: 4

engine: vllm
roles:
  agg:
    nodes: 2
    workers: 1
    args:
      data-parallel-size: 8
      enable-expert-parallel: true
```

### Model parallelism spanning nodes

Upstream's native topology is preserved. If one TP/PP replica spans nodes,
only its global API leader has a positive HTTP port and is advertised to
Router; the remaining processes stay headless.

```yaml
resources:
  gpus_per_node: 4

engine: vllm
roles:
  agg:
    nodes: 2
    workers: 1
    args:
      tensor-parallel-size: 8
```

## Disaggregated P/D

Both pools are direct vLLM servers. Router is launched with
`--vllm-pd-disaggregation`, repeated `--prefill URL NIXL_PORT` entries, and
repeated `--decode URL` entries. The same aggregate topology rules apply
independently to every prefill and decode endpoint.

```yaml
frontend:
  type: vllm-router
  enable_multiple_frontends: false
  args:
    policy: consistent_hash

resources:
  gpus_per_node: 4

engine:
  type: vllm
  connector: nixl
roles:
  prefill:
    nodes: 1
    workers: 1
    args:
      data-parallel-size: 4
  decode:
    nodes: 2
    workers: 2
    args:
      data-parallel-size: 4
```

Router has one global `--intra-node-data-parallel-size`, so every advertised
P/D base must represent the same number of local DP ranks. srtctl derives and
validates that value; do not set it manually unless it exactly matches the
allocated topology.

### MoRI-IO discovery

`engine.connector: moriio` runs the P/D pair over AMD's MoRI-IO connector (ROCm). MoRI-IO workers do not take peer addresses on the command line; they register with the Router over a ZMQ discovery endpoint and learn each other's transfer addresses from it. srtctl therefore launches the Router in discovery mode and builds each worker's connector config from the realized topology instead of listing URLs:

```yaml
frontend:
  type: vllm-router
  enable_multiple_frontends: false

engine:
  type: vllm
  connector: moriio
roles:
  prefill:
    nodes: 1
    workers: 1
    gpus: 1
  decode:
    nodes: colocate
    workers: 1
    gpus: 1
```

| Piece | srtctl sets | Why |
| --- | --- | --- |
| Router command | `--vllm-pd-disaggregation --kv-connector moriio --vllm-discovery-address 0.0.0.0:36367`, no `--prefill`/`--decode` | Discovery mode in vllm-project/router (`RouterArgs.vllm_discovery_address`); the address is the one listener workers register with |
| `kv_connector` / `kv_role` | `MoRIIOConnector`; `kv_producer` on prefill, `kv_consumer` on decode | The connector's role follows the worker's mode |
| `proxy_ip`, `proxy_ping_port` | the head node's IP on the cluster interface, `36367` | Where the worker registers; hence one Router, on the head node |
| `http_port`, `host_ip` | the worker's allocated HTTP port and its own routable IP | What the Router routes to and what peers connect to; upstream falls back to its own interface guess for `host_ip`, which is wrong on multi-homed nodes |
| `handshake_port`, `notify_port` | allocated per process by `NodePortAllocator` (`moriio_handshake`, `moriio_notify` kinds); both reserve one port per local rank | vLLM adds rank offsets to both bases, so colocated workers need disjoint blocks. Defaults start at 26000/27000, below Linux's default ephemeral range, to avoid MoRI's other dynamically allocated listeners |
| `read_mode` | `true` | The Router's discovery flow drives reads |
| `VLLM_PORT` | unset for these workers | MoRI-IO opens its other listeners inside the TP child processes, which inherit `VLLM_PORT`; a fixed scan base there hands several ranks the same unbound port, so vLLM takes ephemeral ports from the kernel |

Readiness is the Router's `/health`: it answers 503 `Waiting for discovered workers` until a prefill and a decode have registered, then 200 (vllm-project/router 43140bc8e2). This is a workaround for a gap in the Router, not a MoRI-IO property: in discovery mode the Router keeps registered workers in a separate registry but still serves `/workers` and `/get_server_info` from the static URL list, which is empty, so the count-based `/workers` gate srtctl uses for static routing has nothing to count. `/health` is the only endpoint that reads the discovery registry, and it only proves one worker of each role, so with several workers per role the benchmark can start before the Router knows all of them; the per-worker `/health` gate that follows proves the workers are up, not that they registered. The upstream fix is for `/workers` to merge the discovery registry (`src/routers/router_manager.rs`); when it does, `VLLMRouterFrontend.probe_ready` loses its discovery branch and the count-based probe covers both modes.

Rules the recipe must satisfy: both `prefill` and `decode` run the connector (an engine-wide `connector: moriio`, or the same value in both roles' `args`), the topology is prefill/decode, `frontend.enable_multiple_frontends` is `false`, and `frontend.orchestrator_placement` is `head`. `srtctl dry-run` rejects anything else, and any other `frontend.type` with this connector. Upstream key names are those `moriio_common.py` reads (vllm-project/vllm 9679173788). `examples/vllm/vllm-router-moriio-disagg.yaml` is the reference recipe.

## Multiple Router processes

With `enable_multiple_frontends: true`, srtctl starts nginx on the public port
and multiple identical Router processes on the internal frontend port. Each
Router receives the same backend topology. Use nginx session affinity when a
client session must remain on one Router process.
