# Frontends and Dynamo

The `frontend:` block (which router fronts the workers) and the `dynamo:` block (how Dynamo is installed and wired). Router-specific pages: [SGLang Router](sglang-router.md), [vLLM Router](vllm-router.md), [Shepherd Model Gateway](smg.md), [llm-d Router](llm-d.md).

## frontend

Frontend/router configuration.

```yaml
frontend:
  # Frontend type: "dynamo" (default), "sglang-router", "vllm-router", "smg", "llm-d", direct "sglang", "vllm", "trtllm_serve",
  # or "none" for a services-only job (no router, no OpenAI endpoint, no worker-count health gate; only
  # valid without engine roles, see services[].nodes)
  type: dynamo

  # Where it runs; see placement
  placement:
    node: head

  # Scaling
  enable_multiple_frontends: true     # Enable nginx + multiple routers
  num_additional_frontends: 9         # Additional routers (total = 1 + this)

  # Optional: raise nofile for nginx (shell ulimit + worker_rlimit_nofile in nginx.conf).
  # Default false. Set true on clusters that allow it; can also set nginx_raise_ulimit in srtslurm.yaml.
  # nginx_raise_ulimit: true

  # CLI args passed to the frontend/router
  args:
    router-mode: "kv"                 # dynamo: router-mode
    policy: "cache_aware"             # sglang-router: policy
    no-kv-events: true                # boolean flags

  # Dynamo only: inline worker-selection policy config. srtctl writes the
  # router policy YAML and passes --router-policy-config automatically.
  worker_selection:
    prefill: max-kv-overlap
    decode: default
    instances:
      - name: max-kv-overlap
        type: dynamo-two-tier-cost-fn
        parameters:
          cache_threshold: 0.0
          balance_abs_threshold: 1000000000
          balance_rel_threshold: 1000000000.0

  # Environment variables for frontend processes
  env:
    MY_VAR: "value"

  # Optional static-router image; defaults to model.container
  # container_image: vllm-router
```

Fields: [FrontendConfig](schema-reference.md#frontendconfig), [PlacementConfig](schema-reference.md#placementconfig). `worker_selection` is written to `/logs/router_policy_config.yaml` and passed with `--router-policy-config`, so it cannot be combined with a policy-config argument or environment variable of your own.

See [SGLang Router](sglang-router.md) for detailed architecture.

### trtllm_serve frontend

`type: trtllm_serve` runs the `trtllm-serve disaggregated` orchestrator as the router (for `engine: trtllm`). Instead of the dynamo request plane, srtctl collects the prefill/decode worker addresses and writes a static `ser.yaml` (`context_servers` = prefill, `generation_servers` = decode), then launches the orchestrator on the head node. The trtllm workers are started as `trtllm-serve` OpenAI servers rather than `dynamo.trtllm`.

Because the orchestrator is a single process, set `enable_multiple_frontends: false` (the nginx + multi-router path is not supported). A configuration can be switched between the two TRT-LLM serving stacks by changing only `frontend.type` between `dynamo` and `trtllm_serve`; start from the `examples/trtllm/dynamo-disagg.yaml` and `examples/trtllm/trtllm-serve-disagg.yaml` examples.

**Worker metrics default.** srtctl sets `return_perf_metrics: true` in the `args` of every role a `trtllm_serve` recipe uses (prefill and decode, or agg), creating the mapping when the recipe has none. This is a setdefault: an explicit `return_perf_metrics: false` in the recipe wins and is warned about. trtllm-serve mounts a worker's `/prometheus/metrics` route only when the engine runs with that flag, and TensorRT-LLM's own default is `false`, so without it Tachometer's `backend_*` endpoints answer HTTP 404 and the capture has no worker-level data. The route carries the per-request series (request latency, TTFT, TPOT, queue/prefill/decode time, token counters); it applies independently of `observability.enabled`, which keeps its own expansion. The same load step also sets `enable_iter_perf_stats: false` unless the recipe or observability says otherwise (see [Iteration statistics default](engines.md#trt-llm-metrics-publication)), so the route carries the per-request series only and the workers skip TensorRT-LLM's per-iteration statistics.

### vllm frontend

`type: vllm` runs aggregate vLLM jobs **without Dynamo**. The OpenAI-compatible HTTP server is the aggregate `vllm serve` worker itself: there is no separate router/frontend process, and srtctl skips NATS/etcd startup.

Use this for aggregate throughput benchmarks where Dynamo orchestration is not needed. Disaggregated prefill/decode layouts still require a real router such as Dynamo (`frontend.type: dynamo`).

**Requirements**

| Constraint | Value |
| ---------- | ----- |
| `engine` | `vllm` |
| Job layout | Aggregate only (`roles.agg`); no prefill/decode roles |
| `roles.agg.workers` | Exactly `1`; scale across nodes with `roles.agg.nodes`, not with replicas |
| `enable_multiple_frontends` | `false` (nginx + multi-router path is unsupported) |

Nothing load-balances between aggregate endpoints here, so `roles.agg.workers: 2` is rejected at load time: the extra replica would either idle behind the single public address or collide on the port. Use `frontend.type: dynamo` when you want several aggregate replicas behind one endpoint.

**Single-node example**

```yaml
frontend:
  type: vllm
  enable_multiple_frontends: false

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

**Multi-node example (TP/PP across nodes)**

```yaml
frontend:
  type: vllm
  enable_multiple_frontends: false

resources:
  gpus_per_node: 8

engine: vllm
roles:
  agg:
    nodes: 2
    workers: 1
    args:
      tensor-parallel-size: 8
      pipeline-parallel-size: 2
```

srtctl launches one `vllm serve` process per node. The endpoint leader (`node_rank=0`) binds the public OpenAI port; follower ranks run headless engine workers. Multi-node coordination flags (`--master-addr`, `--nnodes`, `--node-rank`, `--headless`) are derived from the allocated topology: **do not set them in the recipe**.

`master-port` / `master_port` remains an optional recipe override and is passed to every node rank. Set it when jobs may share a leader node and need distinct vLLM rendezvous ports; otherwise vLLM's default is used.

**Topology-managed `args` keys**

The following keys are owned by srtctl and are stripped at runtime if present in `roles.<role>.args` for a vLLM role:

- `headless`
- `host`, `port`
- `master-addr` / `master_addr`
- `nnodes`
- `node-rank` / `node_rank`

Existing recipes that still contain these keys generally continue to work because the values are ignored. One exception is `headless` combined with the default `dp_launch_mode: per_node` and `data-parallel-size`: engine validation rejects that combination before direct-vLLM command construction, so remove `headless` from such recipes. `srtctl dry-run` emits a **WARNING** for each accepted key so operators can clean up recipes over time.

Health checks, benchmark clients, and `SRT_FRONTEND_HOST` target the **aggregate endpoint leader** (the node running the public `vllm serve`), not necessarily the Slurm head node.

To use vLLM's Rust OpenAI frontend in managed-engine mode, set `engine.vllm_serve_binary` to `vllm-rs`. An absolute path is also accepted when the executable is installed in the container but is not on `PATH`:

```yaml
frontend:
  type: vllm
  enable_multiple_frontends: false

engine:
  type: vllm
  vllm_serve_binary: /usr/local/lib/python3.12/dist-packages/vllm/vllm-rs
roles:
  agg:
    nodes: 1
    workers: 1
    args:
      tensor-parallel-size: 4
      tokenizer-mode: hf
      reasoning-parser: auto
      tool-call-parser: auto
```

The default remains `vllm`, so existing recipes continue to use the Python frontend. This setting only changes direct `frontend.type: vllm` jobs; Dynamo, sidecar, and `vllm-router` launch paths are unchanged.

Compare with `frontend.type: dynamo` + `engine: vllm`, which keeps Dynamo as the request router and uses `python3 -m dynamo.vllm` workers discovered through etcd.

### vllm-router frontend

`type: vllm-router` launches the official vLLM Router in front of direct `vllm serve` workers. It supports aggregate replicas and disaggregated P/D topologies without Dynamo or NATS/etcd. See [vLLM Router](vllm-router.md) for complete topology examples and the division of responsibility between the upstream vLLM backend topology and Router adapter.

`engine.connector: moriio` (AMD MoRI-IO on ROCm) switches the same frontend to the Router's discovery mode. srtctl launches one Router on the head node with `--kv-connector moriio --vllm-discovery-address 0.0.0.0:36367` and no worker URLs, gives every prefill and decode worker a role-aware `MoRIIOConnector` `--kv-transfer-config` that carries the Router's address, the worker's own routable IP and HTTP port, and the handshake and notify listeners the port allocator reserved for it, and waits on the Router's `/health`, which answers 503 until a prefill and a decode have registered. Both roles must run the connector, the topology must be prefill/decode, and `frontend.enable_multiple_frontends` must be `false`. See [MoRI-IO discovery](vllm-router.md#mori-io-discovery).

## dynamo

Dynamo installation configuration. `source` says where Dynamo comes from; exactly one of `pypi`, `wheel`, or `git` + `rev`.

```yaml
dynamo:
  source:
    git: https://github.com/ai-dynamo/dynamo
    rev: refs/pull/14000/head # a commit, a tag, or a PR head; never a branch name
    # sha: <filled in by srtctl apply>
```

```yaml
dynamo:
  source:
    pypi: "1.4.2"             # a release from PyPI
```

```yaml
dynamo:
  source:
    wheel: "1.5.0.dev20260901" # a staged nightly wheel
```

Fields: [DynamoConfig](schema-reference.md#dynamoconfig), [DynamoSourceConfig](schema-reference.md#dynamosourceconfig).

| `source` key | Meaning |
| --- | --- |
| `pypi` | An `ai-dynamo` release from PyPI, e.g. `"1.4.2"` |
| `wheel` | An exact `ai-dynamo` nightly version installed from staged wheels; the matching `ai-dynamo-runtime` wheel is installed automatically |
| `git` + `rev` | Clone and build with maturin. `git` defaults to the upstream repository when only `rev` is given, so a fork is `git: https://github.com/<you>/dynamo` |
| `patches` | With `git`: Cargo dependency replacements applied tree-wide before the build. Each entry is a full `<crate> = <spec>` TOML line |
| `sha` | Written by `srtctl apply`; the commit `rev` resolved to |

**Notes**:

- Set `install: false` if your container already has dynamo pre-installed.
- `source.wheel` stages the exact `ai-dynamo` and `ai-dynamo-runtime` wheels, then installs those files with dependency resolution inside the container. Missing Python dependencies are fetched from PyPI and the NVIDIA package index, so compute nodes need access to those indexes. Set `DYNAMO_INDEX_URL` and `DYNAMO_EXTRA_INDEX_URL` in the recipe's top-level `environment:` to override the indexes for both prefetch on the batch host and installation inside worker containers. Overrides under `roles.<role>.env` apply only to installation in that role's worker containers.
- `source` is the same shape `services[].source` uses.
- `rev` must be immutable: a commit SHA, a tag such as `v1.4.2`, or `refs/pull/<n>/head` for an unmerged PR. `main`, `master`, and `HEAD` are rejected; pin the commit you mean.
- `srtctl apply` resolves a non-commit `rev` with `git ls-remote`, writes the commit as `source.sha` into the submitted `config.yaml` (comments preserved, the recipe on disk is untouched), and echoes it as `pinned_sources` in `--json` output. The job builds that commit and the `/configs/dynamo-wheels` cache is keyed by it, so two runs of one recipe cannot silently build different code because the PR moved. If the login node cannot reach the remote, the submit continues with a warning and the compute node fetches the ref by name.
- `srtctl dry-run` prints the resolved Dynamo source.

The v1 spelling of this (`dynamo.version`, `dynamo.hash`, `dynamo.top_of_tree`, `dynamo.wheel`, `dynamo.cargo_patches`) is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it.

### Native sidecar mode

Native sidecar mode runs each framework's native engine beside a Dynamo sidecar. See [Native Sidecars](sidecars.md) for configuration, worker launch behavior, and [vLLM versions and timeline](sidecars.md#vllm-sidecar-versions-and-timeline).
