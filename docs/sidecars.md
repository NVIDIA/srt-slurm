# Native Sidecars

Native sidecar mode runs a framework's native engine beside a Dynamo sidecar that connects it to the Dynamo frontend. This guide covers configuration, worker placement, runtime compatibility, and the vLLM support timeline.

## Configuration

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

SGLang's endpoint leader runs the serving sidecar and registers the complete global DP range. With attention DP and `roles.<role>.kv_events: true`, srtctl also exposes native gRPC and runs a sidecar on each follower that owns a KV publisher. Dynamo automatically selects serving or telemetry-only mode from the local engine's `GetServerInfo` metadata; no mode flag is passed. These followers relay their node-local events with the leader's serving-worker identity and the source's global DP rank; they do not add inference workers. Followers without publishers, including TP-only ranks and later pipeline stages, remain engine-only. For example, TP8/DP2 across four nodes publishes on nodes 0 and 2, so only node 2 needs a telemetry sidecar. Each sidecar discovers its local sources through SGLang's `GetServerInfo`; no context file or manually assigned DP rank is needed.

The [multinode attention-DP example](https://github.com/NVIDIA/srt-slurm/blob/main/examples/features/sglang-sidecar-multinode-dp.yaml) uses one logical aggregate worker across two one-GPU nodes, with a custom prebuilt image and KV routing. Set the image path before submitting. Restart the entire distributed engine group and its sidecars together after a failure; the follower relay does not provide independent sidecar restart or event replay recovery.

The [disaggregated multinode attention-DP example](https://github.com/NVIDIA/srt-slurm/blob/main/examples/features/sglang-sidecar-multinode-disagg-dp.yaml) uses four one-GPU nodes: one TP2/DP2 prefill worker on the first pair and one TP2/DP2 decode worker on the second pair. Both leaders run serving sidecars. Prefill enables KV events and a follower telemetry relay; decode keeps prefix caching disabled by default and sets `kv_events: false`, so its follower remains engine-only. KV routing can reuse prefixes on either prefill DP rank. Decode prefix-cache affinity is not enabled or implied. srtctl supplies the role, distributed rendezvous, prefill bootstrap port, and the prefill leader's routable `--bootstrap-host`; the recipe selects NIXL transfer on both roles. The image must include compatible Dynamo and SGLang multinode sidecar support, SGLang's prebuilt native gRPC extension, and NIXL.

TensorRT-LLM supports sidecars for aggregated workers only and runs the sidecar on MPI rank zero. `dynamo.sidecar_context_length` can override the TRT-LLM context length inferred from `roles.agg.args.max_seq_len`.

## vLLM topology and launch requirements

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

## vLLM sidecar versions and timeline

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
