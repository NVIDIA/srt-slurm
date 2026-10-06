# Frontends and Dynamo

The `frontend:` block (which router fronts the workers) and the `dynamo:` block (how Dynamo is installed and wired). Router-specific pages: [SGLang Router](sglang-router.md), [vLLM Router](vllm-router.md).

## frontend

Frontend/router configuration.

```yaml
frontend:
  # Frontend type: "dynamo" (default), "sglang-router", "vllm-router", direct "sglang", "vllm", "trtllm_serve",
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
- `source` is the same shape `services[].source` uses.
- `rev` must be immutable: a commit SHA, a tag such as `v1.4.2`, or `refs/pull/<n>/head` for an unmerged PR. `main`, `master`, and `HEAD` are rejected; pin the commit you mean.
- `srtctl apply` resolves a non-commit `rev` with `git ls-remote`, writes the commit as `source.sha` into the submitted `config.yaml` (comments preserved, the recipe on disk is untouched), and echoes it as `pinned_sources` in `--json` output. The job builds that commit and the `/configs/dynamo-wheels` cache is keyed by it, so two runs of one recipe cannot silently build different code because the PR moved. If the login node cannot reach the remote, the submit continues with a warning and the compute node fetches the ref by name.
- `srtctl dry-run` prints the resolved Dynamo source.

The v1 spelling of this (`dynamo.version`, `dynamo.hash`, `dynamo.top_of_tree`, `dynamo.wheel`, `dynamo.cargo_patches`) is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it.

### Native sidecar mode

Set `sidecar: true` on every role to run the framework's native engine process beside a CPU-only Dynamo sidecar instead of launching `python3 -m dynamo.<framework>`. The mode is job-wide, so every role must agree; the sidecar knobs (`sidecar_port`, `sidecar_args`, ...) stay under `dynamo`. `dynamo.sidecar: true` is the equivalent job-wide spelling. The engine and sidecar share one Slurm step and have a coupled lifecycle: if either exits, srtctl terminates the other and marks the worker failed.

For SGLang, srtctl also adds `--incremental-streaming-output` to the engine (the sidecar consumes deltas) and sets `SGLANG_RUST_BUILD_MODE=never` in the worker environment so the native gRPC extension is loaded from the image instead of being rebuilt with cargo, which images that run SGLang from a source checkout cannot do. Set either one in the role's `args` or `env` to override.

By default, srtctl launches `python3 -m dynamo.<framework>.sidecar`. The `ai-dynamo` package supplies this module and pins the matching `ai-dynamo-runtime` wheel, which embeds the native Rust sidecar. The configured Dynamo source or preinstalled container runtime must include the selected framework's launcher. No separate Cargo build is performed at job startup.

Nightly deployments should select an exact `dynamo.source.wheel` version so srtctl stages and installs the matching `ai-dynamo` and `ai-dynamo-runtime` artifacts on every worker. Set `dynamo.sidecar_binary` only to launch a compatible standalone executable already present in the container or a bind mount.

```yaml
frontend:
  type: dynamo

engine: vllm  # sglang, vllm, or trtllm
roles:
  agg:
    nodes: 1
    workers: 1
    sidecar: true
    args:
      tensor-parallel-size: 8

dynamo:
  source:
    wheel: "<nightly-with-sidecars>"
  sidecar_port: 50051
  sidecar_args:
    - --grpc-connections
    - "4"
```

The default sidecar commands are `python3 -m dynamo.sglang.sidecar`, `python3 -m dynamo.vllm.sidecar`, and `python3 -m dynamo.trtllm.sidecar`. All three use the shared `--grpc-endpoint` flag.

SGLang exposes gRPC and starts the sidecar only on an endpoint leader; distributed followers are engine-only. srtctl also adds `--incremental-streaming-output` to every SGLang sidecar engine (and logs that it did): the sidecar treats each gRPC chunk as a delta, and without the flag current SGLang builds stream the cumulative text per chunk, which shows up as repeated prefixes in responses and inflated token counts. Set `incremental-streaming-output` in the role's `args` yourself to override. TensorRT-LLM supports sidecars for aggregated workers only and runs the sidecar on MPI rank zero. `dynamo.sidecar_context_length` can override the TRT-LLM context length inferred from `roles.agg.args.max_seq_len`.

For multi-node vLLM data parallelism, srtctl launches one Rust gRPC frontend and one Dynamo sidecar on **every node** using hybrid load balancing. All nodes receive the global `--data-parallel-size`, while `--data-parallel-size-local` and `--data-parallel-start-rank` restrict each frontend to its colocated engines. For example, DP12 on three four-GPU nodes launches local DP4 with starting ranks 0, 4, and 8. Dynamo registers three endpoints, each covering its four local ranks. No node runs `--headless`, and readiness requires all three sidecars. vLLM sidecars require `engine.dp_launch_mode: per_node` (the default; `backend.dp_launch_mode` in schema 1); `per_gpu` is rejected during configuration validation, including for single-rank jobs.

The multi-node DP command uses Python-supervised Rust frontends (`python3 -m vllm.entrypoints.cli.main serve` with `VLLM_USE_RUST_FRONTEND=1`). Python coordinates the shared DP rendezvous and starts `vllm-rs frontend` with each node's local engine count and starting rank; requests are handled by Rust. This is a current launcher limitation: `vllm-rs serve` expects to own the complete DP group and does not implement hybrid startup. Python supervision is not a fundamental requirement of the sidecar architecture. Single-node sidecars continue to use `vllm-rs serve`.

Each Rust frontend has one API server; do not set `api-server-count` to another value or enable `grpc` (which selects the separate Python gRPC server) or `data-parallel-external-lb`.

For a TP/PP replica spanning nodes (including TP+EP with `data-parallel-size: 1`),
srtctl starts the Rust gRPC frontend and Dynamo sidecar **only on the leader**.
Followers run native `vllm serve --headless`, without an API server or sidecar.
All nodes share the leader address on the configured network interface and a
per-endpoint rendezvous port, with derived `--nnodes`, `--node-rank`, and
`--distributed-executor-backend mp`. TP × PP × PCP must equal the total allocated
GPUs, distributed evenly across nodes. For example, TP8 on two four-GPU nodes is
one Dynamo registration, not two. Recipe rendezvous overrides are replaced by
these allocation-derived values; `api-server-count` is omitted on followers.
With the default role setting `critical: true`, an unexpected exit on any node
fails the job and stops all workers through the job-wide process monitor. Multiple cross-node DP replicas within one endpoint
remain unsupported; use separate endpoints with DP=1 instead.

This path requires the same Python-supervised Rust frontend `--grpc-port`
integration described below. Native vLLM headless support alone is insufficient
to provide the leader's sidecar transport.

Hybrid sidecars require compatible changes in **both** projects: vLLM's Python `serve` command must support `--grpc-port` for its Rust frontend ([vLLM #59659](https://github.com/vllm-project/vllm/pull/59659)), the Rust Control service must report `ParallelismInfo.data_parallel_size_local` ([vLLM #57116](https://github.com/vllm-project/vllm/pull/57116)), and Dynamo's sidecar must register that local range using the reported global start rank. Equivalent backports are sufficient. The `--grpc-port` requirement also applies to the leader of a cross-node TP/PP replica. Without it, Python exits with an unrecognized-argument error before the sidecar starts; switching to `vllm-rs serve` cannot recover hybrid DP because its launcher does not implement that topology.

Pin a compatible image or source build; merely upgrading the Dynamo wheel is insufficient. srtctl does not automatically verify these capabilities in the worker image. To use a specific Rust binary in the multi-node path, set `VLLM_RUST_FRONTEND_PATH` in the recipe's worker environment. srtctl preserves that path and supplies the Rust frontend selection automatically.

vLLM sidecar mode sets `VLLM_PLUGINS` to an empty value by default. This prevents image-installed plugins from replacing native engine output types that must match the fixed `vllm-rs` MessagePack contract. A recipe can explicitly set `VLLM_PLUGINS` in a role's `env` when every selected plugin is compatible with the sidecar protocol.

#### vLLM sidecar versions and timeline

The following checks describe the **native vLLM sidecar path**, as of **2026-10-06**. They are interface requirements, not an end-to-end qualification of every model, connector, or topology. Direct vLLM jobs and other backends have different requirements.

For multi-node hybrid DP, use srt-slurm with [#420](https://github.com/NVIDIA/srt-slurm/pull/420) (`4bb5f649`), a vLLM main/nightly build containing [#59659](https://github.com/vllm-project/vllm/pull/59659) (`9d2a6f52b2`) and [#57116](https://github.com/vllm-project/vllm/pull/57116) (`9639cbde04`), and a Dynamo main/nightly build containing [#14697](https://github.com/ai-dynamo/dynamo/pull/14697) (`805a77f053`). Equivalent backports also satisfy these interfaces. Cross-node TP/PP additionally needs srt-slurm [#457](https://github.com/NVIDIA/srt-slurm/pull/457) (`71113bbd`); its leader needs the Python gRPC-port support, while the hybrid local-DP ownership requirement applies to hybrid DP.

**Minimum vLLM main/nightly for both multi-node paths:** use **`0.30.1rc1.dev562+g9d2a6f52b`**, published for x86_64 and aarch64 in the [official commit wheel index](https://wheels.vllm.ai/9d2a6f52b264c20b5816c5a094b191bfbcadd74b/vllm/), or a later main build containing commit `9d2a6f52b264c20b5816c5a094b191bfbcadd74b` (merged October 2, 2026). This commit includes the earlier local-DP metadata and protocol changes. Pin the worker image or source/wheel revision to that commit or a descendant; a build dated October 2 can still predate the merge. The compatible Dynamo build above remains required for hybrid DP.

Treat this as a **commit cutoff, not a package-version `>=` constraint**: stock `0.31.0` is numerically newer but lacks the required Python CLI change. vLLM's [commit-specific nightly indices](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#install-specific-revisions) allow an exact revision to be selected. The wheel labels below are copied from those indices, including the historical `0.2.1.dev*` labels; they are not a monotonic compatibility scale. For container packaging changes, verify the image's source revision as well as its installed wheel.

Release dates alone do not establish that a change is included: release branches can omit changes already merged to main. The table below was checked against the tagged source, including the Python CLI field, the protobuf local-DP field, and Dynamo's local-range discovery code.

| Component / stock release | Relevant interfaces included | Remaining requirement for multi-node hybrid DP |
| --- | --- | --- |
| [vLLM 0.29.0](https://github.com/vllm-project/vllm/tree/v0.29.0) | Neither Python `--grpc-port` nor local-DP Control metadata | Both #59659 and #57116, plus a compatible Dynamo build |
| [vLLM 0.30.0](https://github.com/vllm-project/vllm/tree/v0.30.0) | Engine-error propagation and selected-token logprob fixes; neither hybrid prerequisite | Both #59659 and #57116, plus a compatible Dynamo build |
| [vLLM 0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0) | Local-DP metadata, `vllm-proto` 0.4.0 (including the local-DP field introduced in 0.3.0), appended-output-field compatibility, and `vllm-rs` on the CUDA image's `PATH` | Still lacks Python `--grpc-port` (#59659); backport it or use a main/nightly build containing it |
| [Dynamo 1.5.0](https://github.com/ai-dynamo/dynamo/tree/v1.5.0) | Wheel-installed vLLM sidecar launcher and complete-group DP routing | Lacks hybrid local-range ownership (#14697); use a main/nightly build containing it or an equivalent backport |

For example, **vLLM 0.31.0 plus #59659**, together with **Dynamo containing #14697**, satisfies these hybrid launch/discovery prerequisites. Stock vLLM 0.31.0 plus stock Dynamo 1.5.0 does not. Pin matching `ai-dynamo` and `ai-dynamo-runtime` artifacts, and keep Python vLLM and its Rust frontend from the same build. Single-node sidecars already use `vllm-rs serve`, so they do not require Python's `--grpc-port` addition; the image must still provide the compatible Rust executable on `PATH`.

Merge dates below use America/Los_Angeles time. The `vllm-proto` crate version is separate from the vLLM Python package version.

| Merge date (2026) | Change | Effect on srt-slurm sidecars |
| --- | --- | --- |
| Aug 27 | [Dynamo #13923](https://github.com/ai-dynamo/dynamo/pull/13923) | Adds the wheel-installed `python3 -m dynamo.vllm.sidecar` launcher |
| Sep 11–14 | [vLLM #56405](https://github.com/vllm-project/vllm/pull/56405), [#56406](https://github.com/vllm-project/vllm/pull/56406) | Propagates engine generation errors and preserves selected-token logprob mode; these are serving fixes, not the hybrid launch prerequisites |
| Sep 16 | [vLLM #57116](https://github.com/vllm-project/vllm/pull/57116) | Reports the frontend's local DP size. Published main wheel: [`0.2.1.dev29+g9639cbde0`](https://wheels.vllm.ai/9639cbde04f58c185ea6e5122dbd2039908abb7a/vllm/). Still needs the Python `--grpc-port` change for hybrid DP |
| Sep 16 | [vLLM #57233](https://github.com/vllm-project/vllm/pull/57233) | Bumps `vllm-proto` to 0.3.0 for downstream consumers of local-DP metadata. Published main wheel: [`0.2.1.dev59+gd2a2a8295`](https://wheels.vllm.ai/d2a2a82954e4151cbf398c6c9c46ea7b037b0915/vllm/) |
| Sep 16 | [vLLM #56533](https://github.com/vllm-project/vllm/pull/56533) | Accepts appended `EngineCoreOutput` fields, avoiding failures from plugins such as vLLM-Omni; does not add full Omni serving support |
| Sep 18 | [Dynamo #14697](https://github.com/ai-dynamo/dynamo/pull/14697) | Consumes local-DP metadata and registers each frontend's actual rank range and per-rank KV capacity |
| Sep 18 | [srt-slurm #420](https://github.com/NVIDIA/srt-slurm/pull/420), [#457](https://github.com/NVIDIA/srt-slurm/pull/457) | Launches Python-supervised Rust frontends per hybrid-DP node, or one frontend with headless followers for cross-node TP/PP |
| Sep 21 | [vLLM #57606](https://github.com/vllm-project/vllm/pull/57606) | Exposes the bundled `vllm-rs` on the CUDA image's `PATH` for the single-node launcher. The same commit's wheel is [`0.29.1rc1.dev507+gf07e227ee`](https://wheels.vllm.ai/f07e227ee9acdb8d7e48d69f178c897fb54ecd88/vllm/); installing that wheel alone does not apply the Dockerfile's `PATH` fix. Multi-node paths still need #59659 |
| Oct 2 | [vLLM #59659](https://github.com/vllm-project/vllm/pull/59659) | Exposes Python `vllm serve --grpc-port` in [`0.30.1rc1.dev562+g9d2a6f52b`](https://wheels.vllm.ai/9d2a6f52b264c20b5816c5a094b191bfbcadd74b/vllm/). Completes the vLLM launch/discovery prerequisites for hybrid DP and cross-node TP/PP with the srt-slurm and Dynamo revisions above |
