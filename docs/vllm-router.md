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

DP recipes must use upstream's default `backend.dp_launch_mode: per_node`.
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
  agg_nodes: 1
  agg_workers: 1
  gpus_per_node: 8

backend:
  type: vllm
  vllm_config:
    aggregated:
      tensor-parallel-size: 8
```

### Multiple aggregate replicas

Each logical aggregate worker contributes one or more routable base URLs.
Router receives them through `--worker-urls` and distributes sessions between
the expanded ranks.

```yaml
resources:
  agg_nodes: 4
  agg_workers: 4
  gpus_per_node: 8
```

### Node-local data parallelism

For DEP8 on two four-GPU nodes, upstream srt-slurm creates one hybrid-LB
`vllm serve` process on each node. Router receives both base URLs and srtctl
adds `--intra-node-data-parallel-size 4`, exposing all eight DP ranks.

```yaml
resources:
  agg_nodes: 2
  agg_workers: 1
  gpus_per_node: 4

backend:
  type: vllm
  vllm_config:
    aggregated:
      data-parallel-size: 8
      enable-expert-parallel: true
```

### Model parallelism spanning nodes

Upstream's native topology is preserved. If one TP/PP replica spans nodes,
only its global API leader has a positive HTTP port and is advertised to
Router; the remaining processes stay headless.

```yaml
resources:
  agg_nodes: 2
  agg_workers: 1
  gpus_per_node: 4

backend:
  type: vllm
  vllm_config:
    aggregated:
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
  prefill_nodes: 1
  prefill_workers: 1
  decode_nodes: 2
  decode_workers: 2
  gpus_per_node: 4

backend:
  type: vllm
  connector: nixl
  vllm_config:
    prefill:
      data-parallel-size: 4
    decode:
      data-parallel-size: 4
```

Router has one global `--intra-node-data-parallel-size`, so every advertised
P/D base must represent the same number of local DP ranks. srtctl derives and
validates that value; do not set it manually unless it exactly matches the
allocated topology.

## Multiple Router processes

With `enable_multiple_frontends: true`, srtctl starts nginx on the public port
and multiple identical Router processes on the internal frontend port. Each
Router receives the same backend topology. Use nginx session affinity when a
client session must remain on one Router process.
