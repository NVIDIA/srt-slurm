# llm-d router

`frontend.type: llm-d` routes requests with the [llm-d](https://llm-d.ai) router: the
Endpoint Picker (EPP) behind Envoy, in front of direct `vllm serve` workers. The EPP
scores the workers with the plugins in `frontend.epp_config` (prefix-cache affinity,
queue depth, load, ...) and, for prefill/decode, picks a prefill and a decode worker for
every request. No Dynamo, NATS, or etcd is involved.

Upstream references are pinned to llm-d-router
[v0.11.0](https://github.com/llm-d/llm-d-router/tree/a5cbe600ebade00cf3e9885beaf2bfacddeabce1)
and the llm-d [no-Kubernetes guide](https://github.com/llm-d/llm-d/tree/7fb84b0adf8e1d41eb2cef105fc1aafaf5a9b64f/guides/no-kubernetes-deployment).

## What runs where

| Process | Where | Launched as | Port |
| --- | --- | --- | --- |
| Envoy | router node | the frontend (`llm-d-envoy_0`) | the public port (8000); admin `/ready` on 9901 |
| EPP | router node | the frontend (`llm-d-epp_0`) | ext_proc gRPC 9002, gRPC health 9003, Prometheus 9090 |
| P/D sidecar | every routable decode worker (prefill/decode jobs only) | the implied `llm-d-sidecar` service | `Process.proxy_port`, from the allocator |
| vLLM | every worker | the backend, as for the other static routers | `Process.http_port` |

Envoy owns the public port. For every request it asks the EPP over ext_proc
(`FULL_DUPLEX_STREAMED`) which endpoint to use; the EPP sets the
`x-gateway-destination-endpoint` header and Envoy's `ORIGINAL_DST` cluster forwards the
request there. For prefill/decode the EPP picks a decode endpoint and names the chosen
prefill worker in `x-prefiller-host-port`
([`disagg_profile_handler.go`](https://github.com/llm-d/llm-d-router/blob/a5cbe600ebade00cf3e9885beaf2bfacddeabce1/pkg/epp/framework/plugins/scheduling/profilehandler/disagg/disagg_profile_handler.go#L533-L536)).
The decode endpoint is llm-d's P/D sidecar: it sends the request to that prefill worker
first, then hands the returned `kv_transfer_params` to its own vLLM, whose KV connector
pulls the cache from the prefill worker
([`connector_nixlv2.go`](https://github.com/llm-d/llm-d-router/blob/a5cbe600ebade00cf3e9885beaf2bfacddeabce1/pkg/sidecar/proxy/connector_nixlv2.go#L42)).
Everything else, `/metrics` included, the sidecar proxies to its vLLM.

### Why the EPP and Envoy are the frontend and the sidecar is a service

The frontend is the process that owns the public port and decides when the job is
ready, so Envoy and the EPP are the frontend. They also have to start at the frontend's
moment: llm-d discovers workers through Kubernetes or a file, and the workers here are
plain engines that register nowhere, so srtctl is the registrar. It writes the endpoints
file once every worker answers `/health` (the static routers' pre-start gate), then starts
the EPP and Envoy; no service phase sits between the workers and the frontend.

The sidecar is a different process with its own image, placement, and health check, one
per decode worker, on the worker's node; that is what `services` with
`placement.per: worker` is for. Its `GET /health` answers 200 on its own
([`proxy.go`](https://github.com/llm-d/llm-d-router/blob/a5cbe600ebade00cf3e9885beaf2bfacddeabce1/pkg/sidecar/proxy/proxy.go#L586-L588)),
so it starts in the `before_workers` phase and gates on that probe. It is critical: a
dead sidecar takes its decode worker out of the routing pool. The frontend implies it,
as Dynamo implies etcd; declare a service named `llm-d-sidecar` to change it:

```yaml
services:
  - name: llm-d-sidecar
    type: llm-d-sidecar
    container: my-llm-d-image            # default: frontend.container_image, else the job container
    args:
      - --enable-prefiller-sampling      # appended after the flags srtctl sets
```

srtctl sets `--port` (the worker's `Process.proxy_port`), `--model-server-port` (its
`Process.http_port`), `--kv-connector`, and `--secure-proxy=false`; `args` come last.
The allocator hands out the proxy port (`PROXY_PORTS`) for every routable worker of a
mode the frontend proxies (`Frontend.proxied_worker_modes`), so two decode workers on
one node never collide. The `--kv-connector` protocol comes from the decode workers' vLLM
connector class (`VLLMBackend.kv_connector_class`, which reads a role's own
`kv-transfer-config` before `connector`):

| Decode connector | Sidecar protocol |
| --- | --- |
| `NixlConnector` | `nixlv2` |

Any other decode connector is rejected at load time.

## Endpoint discovery

The EPP runs without Kubernetes when its configuration names a discovery plugin
([`runner.go`](https://github.com/llm-d/llm-d-router/blob/a5cbe600ebade00cf3e9885beaf2bfacddeabce1/cmd/epp/runner/runner.go#L264-L273),
[`runWithFileDiscovery`](https://github.com/llm-d/llm-d-router/blob/a5cbe600ebade00cf3e9885beaf2bfacddeabce1/cmd/epp/runner/runner.go#L1044-L1196)).
srtctl writes three files into the log directory (every container sees it at `/logs`):

- `llm-d-endpoints.yaml`: one endpoint per routable worker process (the one with an HTTP
  port: a multi-node worker's leader), at its IPv4 address on the cluster's network
  interface, labelled `llm-d.ai/role: prefill`, `decode`, or `both` (aggregate). A decode
  endpoint is its sidecar's port.
- `llm-d-epp-config.yaml`: `frontend.epp_config` with the
  [`file-discovery`](https://github.com/llm-d/llm-d-router/blob/a5cbe600ebade00cf3e9885beaf2bfacddeabce1/pkg/epp/framework/plugins/datalayer/discovery/file/plugin.go)
  plugin (`watchFile: false`) and `dataLayer.discovery` added. `discovery.pluginRef` is
  the spelling v0.10 and v0.11 both accept. A recipe that configures discovery itself is
  rejected.
- `llm-d-envoy.yaml`: the guide's Envoy configuration with the public port, an admin
  listener srtctl can reach, no request timeout, `/metrics` answered with 404 (routed, it
  would return a different worker's counters on every scrape), and an access log
  (`llm-d-envoy-access.log`) recording, for every request, the endpoint the EPP picked
  (`upstream=`) and the prefill worker it named (`prefiller=`).

## Readiness and metrics

The job is ready when Envoy's admin `/ready` answers 200 and the EPP's
`llm_d_epp_ready_endpoints` gauge, the endpoints whose metrics it scraped within its
staleness window
([`llm_d_router_metrics.go`](https://github.com/llm-d/llm-d-router/blob/a5cbe600ebade00cf3e9885beaf2bfacddeabce1/pkg/epp/metrics/llm_d_router_metrics.go#L234-L241),
[`logger.go`](https://github.com/llm-d/llm-d-router/blob/a5cbe600ebade00cf3e9885beaf2bfacddeabce1/pkg/epp/datalayer/logger/logger.go#L106-L117)),
counts every endpoint in the file. The EPP scrapes a worker on the address it routes
to, so a decode worker is scraped through its sidecar. Tachometer scrapes the EPP's
Prometheus listener as the frontend; worker metrics stay on vLLM's own port.

## Images

The EPP, Envoy, and the sidecar run from `frontend.container_image` (the sidecar unless
its service sets `container`), or the job container when it is unset, as the executables
`epp`, `envoy`, and `pd-sidecar` on `PATH`. llm-d publishes the EPP and the sidecar as
distroless images (`/app/epp`, `/app/pd-sidecar`), which pyxis cannot start: it enters a
container through `sh`
([`pyxis_slurmstepd.c`](https://github.com/NVIDIA/pyxis/blob/107519944221822ea1dace4db8e7234b2eaa4cd5/pyxis_slurmstepd.c#L1128-L1135)). Build an image with a shell that carries the three executables,
for example on top of the vLLM image the workers run:

<!-- docs-yaml: skip -->
```dockerfile
FROM vllm/vllm-openai:<tag>
COPY --from=ghcr.io/llm-d/llm-d-router-endpoint-picker:v0.11.0 /app/epp /usr/local/bin/epp
COPY --from=ghcr.io/llm-d/llm-d-router-disagg-sidecar:v0.11.0 /app/pd-sidecar /usr/local/bin/pd-sidecar
COPY --from=envoyproxy/envoy:distroless-v1.33.2 /usr/local/bin/envoy /usr/local/bin/envoy
```

## Recipe

```yaml
frontend:
  type: llm-d
  enable_multiple_frontends: false
  container_image: llm-d-vllm       # optional: the router image
  epp_config:                       # EndpointPickerConfig without discovery
    plugins: [...]
    schedulingProfiles: [...]
  args:                             # extra EPP flags, e.g. log verbosity
    v: 2
engine:
  type: vllm
  connector: nixl
```

`frontend.args` are passed to the EPP; the pool, configuration, and port flags are
srtctl's. Prefill/decode needs `epp_config` with `prefill` and `decode` scheduling
profiles and the `disagg-profile-handler`; an aggregate job without `epp_config` runs
the EPP's built-in default profile. See
[`examples/vllm/llm-d-disagg.yaml`](https://github.com/NVIDIA/srt-slurm/blob/main/examples/vllm/llm-d-disagg.yaml) and
[`examples/vllm/llm-d-agg.yaml`](https://github.com/NVIDIA/srt-slurm/blob/main/examples/vllm/llm-d-agg.yaml).

## Routing to DP ranks

A vLLM worker with `data-parallel-size` is one `vllm serve` that balances its DP ranks
itself (the leader serves the API, other nodes run headless ranks), so it is one EPP
endpoint, as it is one SMG worker. Set `data-parallel-external-lb: true` in the role's
`args` to have the EPP route to the ranks themselves, as llm-d's wide-EP guide does: srtctl
then launches every DP rank as its own `vllm serve --data-parallel-rank <r>
--data-parallel-address <rank 0's node>` (vLLM's external load balancing) on its own
allocated HTTP port, on one node or across several, and lists each rank in the endpoints
file. Each rank's step and log carry `_dp<r>` (`agg_0_<node>_dp3`, `<node>_agg_w0_dp3.out`),
and in a prefill/decode job every decode rank gets its own P/D sidecar. vLLM accepts
external load balancing for MoE models only. See
[`examples/vllm/llm-d-dp-ranks.yaml`](https://github.com/NVIDIA/srt-slurm/blob/main/examples/vllm/llm-d-dp-ranks.yaml).

```yaml
roles:
  agg:
    gpus: 16
    args:
      data-parallel-size: 16
      data-parallel-external-lb: true
```

## Precise prefix-cache routing

The approximate prefix scorers guess where a prefix is cached from the requests the EPP
routed. With `roles.<role>.kv_events` the workers report it instead: every vLLM worker of
the role publishes its KV-cache events (blocks stored and evicted) to the EPP's
[`precise-prefix-cache-producer`](https://github.com/llm-d/llm-d-router/tree/a5cbe600ebade00cf3e9885beaf2bfacddeabce1/pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache),
which indexes them per endpoint; `prefix-cache-scorer` with
`prefixMatchInfoProducerName: precise-prefix-cache-producer` scores the endpoints from
that index. The producer hashes the prompt's real token IDs, so the configuration also
needs a `token-producer`, which tokenizes through a worker's `/v1/*/render` endpoints:

```yaml
frontend:
  epp_config:
    plugins:
      - type: token-producer
      - type: precise-prefix-cache-producer
        parameters:
          indexerConfig:
            kvBlockIndexConfig:
              enableMetrics: true           # index admissions and lookup hits on /metrics
      - type: prefix-cache-scorer
        parameters:
          prefixMatchInfoProducerName: precise-prefix-cache-producer
      - type: max-score-picker
    schedulingProfiles: [...]
roles:
  agg:
    kv_events: true
```

srtctl wires the rest:

- The producer's `kvEventsConfig.zmqEndpoint` is `tcp://*:5557` (`discoverPods: false`):
  the EPP binds one ZMQ socket on the router node and every worker connects to it. The
  producer's per-pod mode dials the same port on every endpoint address, which workers
  sharing a node cannot all bind (llm-d-router v0.11's file discovery gives every
  endpoint rank 0).
- Every publishing worker gets `--kv-events-config` with `endpoint:
  tcp://<router>:5557` and `topic: kv@<address>:<port>@<served model>`, naming the
  endpoint the EPP routes to (a decode worker's sidecar port) and the model requests
  name; the producer files the events under that endpoint
  ([`vllm_adapter.go`](https://github.com/llm-d/llm-d-router/blob/a5cbe600ebade00cf3e9885beaf2bfacddeabce1/pkg/kvevents/engineadapter/vllm_adapter.go#L52-L58)).
  vLLM adds a publisher's DP rank to its port, so an external-LB rank `r` is given
  `5557 - r`.
- The token producer's `vllm.url` is the first prefill worker's (else the first worker's)
  own vLLM API, and `modelName` defaults to the served model name.

The roles that publish and the producer go together: either without the other is
rejected, as are a producer `kvEventsConfig` that sets the socket or discovery, a
`token-producer` `vllm.url`, and a router not on the head node. A role with
`data-parallel-size` publishes only with `data-parallel-external-lb: true`: each DP rank
publishes its own events, and the EPP can place a prefix only on an endpoint that is
that rank.

## Limitations

- vLLM workers over HTTP only; gRPC roles are rejected.
- One router replica: the EPP keeps its scheduling state (the approximate prefix index,
  in-flight load) in memory, so `enable_multiple_frontends` with more than one router is
  rejected.
- The endpoints file is written once (`watchFile: false`) and the EPP does not eject a
  worker that fails; a crashed worker or sidecar fails the job instead.
- vLLM's `data-parallel-multi-port-external-lb` (one supervisor per node serving a port
  per rank) is not used: its ranks publish KV events on per-rank ports that file
  discovery can only reach with llm-d-router's per-endpoint `rankIndex` (after v0.11.0).
- The precise prefix index lives in the one EPP; KV events go to that EPP alone.
