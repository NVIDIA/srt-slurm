# Engines

The top-level `engine:` block: which inference engine runs, and the engine-wide knobs whose behavior needs more than a table row. Per-engine field tables: [Engine types](schema-reference.md#engine-types) (generated).

## engine

GPU scheduling uses upstream's existing cluster settings. For eight-GPU
allocations on GRES-only clusters, set `use_gpus_per_node_directive: false`
and `default_sbatch_directives: {gres: "gpu:8"}`.

### GPU visibility on AMD

Set `visible_devices_env: ROCR_VISIBLE_DEVICES` in the cluster profile for ROCm
workers. GPU subsets then use only that mask, without applying a second mask to
already-renumbered devices. Set `default_gpu_exporter: null` to disable the
NVIDIA GPU exporter, or configure an exporter image, port, command and
its `gpu_labels` and `gpu_metrics` once for the cluster. Other telemetry is unchanged; an explicit
recipe exporter wins.

```yaml
visible_devices_env: ROCR_VISIBLE_DEVICES
default_gpu_exporter:
  container_image: "docker://rocm/device-metrics-exporter:v1.5.2"
  command: "/home/amd/tools/entrypoint.sh"
  port: 5000
  kind: custom
  gpu_labels:
    index: gpu_id
    identity: serial_number
  gpu_metrics:
    power:
      metric: gpu_power_usage
      scope: amd_device_metrics_exporter_gpu_power_usage
    gpu_util:
      metric: gpu_gfx_activity
    temperature:
      metric: gpu_junction_temperature
```

With this block a recipe that sets `telemetry: {enabled: true}` and nothing else
under `telemetry` collects GPU power from the AMD exporter; see
[GPU power telemetry](power-telemetry.md#gpu-exporter-labels-and-metrics).

For vLLM builds without `--device-ids`, set `engine.set_visible_devices: true`.
This is one explicit boolean, not automatic vLLM version detection. The default
is false: vLLM binds devices with `--device-ids`. There is no CUDA-named alias.

`engine:` supplies the default inference engine for worker roles. It is optional when every role sets its own `engine`. A bare string is the common form; a mapping carries engine options:

```yaml
engine: sglang
```

```yaml
engine:
  type: vllm
  connector: nixl               # vLLM KV connector for disaggregation
```

```yaml
engine:
  type: trtllm
  served_model_name: "Qwen/Qwen3-0.6B"
```

For `trtllm_serve`, an explicit `served_model_name` is passed to the worker's
`--served_model_name` option so the server and benchmark/eval clients use the same
API model name. Leave it unset to retain the server's default. This is a server
option, not a key in `roles.*.args` (the engine YAML).

```yaml
engine:
  type: mocker
  engine_type: vllm
  speedup_ratio: 100
```

Valid types are `atom`, `sglang`, `tilert`, `trtllm`, `vllm`, and `mocker`. Everything that is per role (the role's environment, its engine CLI flags, `extra_args`, `kv_events`) lives under [roles](topology.md#roles); everything else about the engine lives here. The generated tables under [Engine types](schema-reference.md#engine-types) list every engine-wide knob per engine; the ones worth knowing are:

| Engine | Engine-wide knobs |
| --- | --- |
| `sglang-router` | none beyond `type` |
| `vllm` | `connector` (default `nixl`), `dp_launch_mode`, `vllm_serve_binary`, `set_visible_devices`, `allow_prefill_decode_colocation`, `allow_prefill_decode_colocation_across_nodes` |
| `trtllm` | `served_model_name`, `publish_metrics`, `publish_events_and_metrics`, `sequential_node_start`, `numa_memory_bind`, `numa_cpu_bind` |
| `mocker` | the simulation parameters: `engine_type`, `speedup_ratio`, `decode_speedup_ratio`, `num_gpu_blocks_override`, `max_num_seqs`, `max_num_batched_tokens`, `block_size`, `data_parallel_size`, ... |

The v1 spelling of this (`backend.type` plus the engine-wide keys under `backend:`) is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it.

### TRT-LLM CPU and memory placement

To place worker CPUs and memory on the NUMA node associated with each task's GPU:

```yaml
engine:
  type: trtllm
  numa_cpu_bind: true
  numa_memory_bind: local
```

The launcher resolves the GPU through `CUDA_VISIBLE_DEVICES` and
`SLURM_LOCALID`, applies its CPU mask, and sets `numactl --membind=<node>`
before starting the worker. Allocations governed by this policy cannot fall
back to another node. Insufficient local memory can cause allocation failure
or OOM, even when another node has free memory. Existing or shared pages are
not migrated. The container must provide `numactl`. Local mode requires
`numa_cpu_bind: true`. The wrapper uses `CUDA_VISIBLE_DEVICES`; alternate
cluster GPU visibility variables are not supported by this wrapper.

`numa_memory_bind: false` keeps CPU binding without a memory policy change.
`numa_memory_bind: true` uses `numactl -m 0,1` for any GPU type or worker mode.
When omitted or null, this two-node policy applies only to `gb200`, `gb300`,
and `vrnvl72` prefill and decode workers. Enabling CPU binding does not change
these memory policies.

In local mode, the launcher fails if the GPU's NUMA affinity cannot be resolved,
its CPU list is missing or empty, or the memory policy cannot be applied.
Without local mode, unknown GPU NUMA affinity skips CPU binding and retains
the selected memory policy. When profiling in local mode, the outer `nsys`
process also inherits the strict memory policy.

See [the local-binding example](https://github.com/NVIDIA/srt-slurm/blob/main/examples/trtllm/trtllm-serve-agg-numa-local.yaml).

### vLLM DP launch mode

vLLM data-parallel endpoints use one process per node by default. srtslurm derives whether each TP/PP replica is node-local or spans multiple nodes:

```yaml
engine: vllm
roles:
  prefill:
    args:
      data-parallel-size: 8
  decode:
    args:
      data-parallel-size: 16
```

| Value      | Process layout                                                               |
| ---------- | ---------------------------------------------------------------------------- |
| `per_node` | One process per node (default); supports node-local or distributed TP/PP      |
| `per_gpu`  | One process per DP rank (TP x PP GPUs each; deprecated compatibility mode)    |

Set `engine.dp_launch_mode: per_gpu` only when temporarily preserving the legacy process layout. srtslurm emits a configuration-time deprecation warning for Dynamo-backed DP configurations that select it. `per_gpu` will be removed in a future release.

When `TP x PP` fits on one node, srtslurm derives `--data-parallel-size-local` and `--data-parallel-start-rank`, then enables `--data-parallel-hybrid-lb` so every node-local process registers with the Dynamo frontend. When `TP x PP` is larger than the node-local GPU allocation, srtslurm instead derives the multi-node rendezvous arguments and makes every process except the global leader headless. For example, both DP4 x TP4 and DP2 x TP8 are selected automatically on four-GPU nodes.

Do not set `data-parallel-size-local`, `data-parallel-start-rank`, `data-parallel-hybrid-lb`, or `headless` manually; srtslurm owns those values. The allocation must be regular: `DP x TP x PP` must match the endpoint GPU count, and a TP/PP replica must divide evenly within or across nodes.

### TRT-LLM metrics publication

With `frontend.type: dynamo`, prefill, decode, and aggregated TRT-LLM workers publish engine metrics by default using `--publish-metrics`, regardless of whether observability is enabled. `publish_events_and_metrics` is retained only for backward compatibility with older Dynamo builds. Observability does not enable this legacy flag automatically.

```yaml
engine:
  type: trtllm
  publish_metrics: true               # default: metrics only
  publish_events_and_metrics: false   # use publish_metrics
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `publish_metrics` | bool | true | Pass `--publish-metrics` unless the legacy combined flag is selected |
| `publish_events_and_metrics` | bool or null | unset | `true`: pass only the legacy combined flag; `false` or unset/null: use `publish_metrics` |

The flags are mutually exclusive. Explicit `engine.publish_events_and_metrics: true` passes only `--publish-events-and-metrics`, even when `publish_metrics` is true. False, omitted, and null all select the metrics-only setting.

| `publish_events_and_metrics` | `publish_metrics` | Publication flag (with or without observability) |
| --- | --- | --- |
| omitted / null / `false` | `true` | `--publish-metrics` |
| omitted / null / `false` | `false` | none |
| `true` | either | `--publish-events-and-metrics` |

**Compatibility:** the metrics-only flag requires a Dynamo build containing [ai-dynamo/dynamo#12162](https://github.com/ai-dynamo/dynamo/pull/12162) or equivalent support. For older builds, set `engine.publish_events_and_metrics: true` to select the legacy flag, which also enables KV events. To omit both flags, set `engine.publish_metrics: false` and leave the legacy flag false or unset. Omitting the flag does not override metrics-related environment variables supplied by the user. Metrics collection adds engine telemetry work; metrics-only does not mean zero overhead, but with the iteration-statistics default below the remaining cost is the per-request perf metrics.

**Migration from the previous publication behavior:** `engine.publish_events_and_metrics: false` previously omitted both publication flags. It now uses `publish_metrics`, which defaults to true. Recipes that used false to disable publication, especially on older Dynamo builds that reject `--publish-metrics`, must also set `engine.publish_metrics: false`. To publish metrics on those older builds, select `engine.publish_events_and_metrics: true` instead. Null remains accepted for existing serialized recipes and behaves the same as false.

**KV events and observability:** `observability.enabled: true` no longer automatically enables TRT-LLM KV events. On Dynamo builds supporting the independent controls, keep the default metrics-only flag and set `DYN_TRTLLM_PUBLISH_KV_EVENTS: "true"` in every worker role that should publish events:

```yaml
engine:
  type: trtllm
roles:
  agg:
    env:
      DYN_TRTLLM_PUBLISH_KV_EVENTS: "true"
```

For disaggregated recipes, set the same environment variable under both `roles.prefill.env` and `roles.decode.env`. This uses Dynamo's [independent KV-event control](https://github.com/ai-dynamo/dynamo/blob/aacb1abae25204fa16a5c3cfeb1b748fc7252df0/components/src/dynamo/trtllm/backend_args.py#L179-L190); it does not require the legacy combined flag. Older builds without that control must use `engine.publish_events_and_metrics: true` to enable events and metrics together.

**Iteration statistics default.** srtctl bakes `enable_iter_perf_stats: false` into every TRT-LLM engine section a recipe uses (prefill and decode, or aggregated), under both `frontend.type: dynamo` and `trtllm_serve`, creating the section when the recipe has none. This is a setdefault: an explicit `enable_iter_perf_stats: true` in the recipe wins, and `observability.enabled: true` keeps its own `true` because its expansion runs first. The default exists because `dynamo.trtllm` turns `--publish-metrics` into `enable_iter_perf_stats: true` in the engine arguments, and the engine YAML is merged over those arguments and wins on conflicts; without the explicit key every Dynamo worker collects TensorRT-LLM's per-iteration statistics (KV-cache stats and CUDA-event step timing on every executor loop). The request-level `trtllm_*` series (request latency, TTFT, TPOT, queue/prefill/decode time, token counters) do not need the key: they come from the per-request perf metrics, which `--publish-metrics` sets on the Dynamo path and `return_perf_metrics: true` sets for trtllm-serve. What the default drops is the iteration-level `trtllm_*` gauges (`trtllm_kv_cache_*`, running/waiting requests, iteration latency) and, on Dynamo, the `dynamo_component_kvstats_*` gauges, the router worker-load sample and the Planner's forward-pass metrics; set the key to `true` or enable `observability` to get them back. Engine sections whose `backend` is the legacy `tensorrt` engine are left alone: its `LlmArgs` rejects the key on containers older than TensorRT-LLM v1.3.0rc21, and that backend always collected the statistics anyway.

```yaml
roles:
  decode:
    args:
      enable_iter_perf_stats: true   # opt back in for one role
```

Saved and locked recipes carry the resolved key, so a later `observability.enabled: true` on such a file meets an explicit `false` rather than an omission; srtctl warns at load time and `srtctl dry-run` shows the value, and removing the line restores the default.

These options do not change native `trtllm_serve` or sidecar worker commands. `srtctl dry-run` shows the publication flag selected for Dynamo TRT-LLM workers and, for every TRT-LLM backend, the per-role `enable_iter_perf_stats` / `return_perf_metrics` values the engine YAML will carry.

TRT-LLM workers can span partially occupied nodes. For example, on four-GPU nodes,
two DEP6 prefill workers need three nodes:

```yaml
roles:
  prefill:
    nodes: 3
    workers: 2
    gpus: 6
    args:
      tensor_parallel_size: 6
      moe_expert_parallel_size: 6
      pipeline_parallel_size: 1
      enable_attention_dp: true
```

The workers use `A[0,1,2,3] + B[0,1]` and `C[0,1,2,3] + B[2,3]`.
Each endpoint launches exactly six MPI ranks and gets its own per-node
`CUDA_VISIBLE_DEVICES`. Full nodes lead the rank order to keep TRT-LLM's local
device mapping consistent. Layouts incompatible with that mapping (for example,
seven ranks split 4+3) are rejected before launch. Backend-specific communication
requirements still apply; this does not enable arbitrary uneven layouts in every
TRT-LLM communication backend.

TRT-LLM endpoints also set `MASTER_ADDR` to the rank-zero node and use a distinct
`MASTER_PORT` per endpoint. This overrides container hooks that infer rank zero
from Slurm's sorted node list. Explicit recipe environment values take precedence.

Every multi-node endpoint, uniform (two full nodes) or uneven, is launched with
`--distribution=arbitrary` and one host per task in srt-slurm's node order, for
example `--ntasks 8 --ntasks-per-node=4 --nodelist A,A,A,A,B,B,B,B` (the
`--nodes` flag is omitted, as Slurm requires for this distribution). A plain
`--nodelist` only names the nodes; Slurm may still order tasks by its own
topology, which can put rank 0 on a node other than the `MASTER_ADDR` leader and
stall the rendezvous. Single-node endpoints are unchanged. A `distribution` set in
the cluster's or the role's `srun_options` is replaced for such an endpoint, with a
warning in the job log.

**Other TRT-LLM launch facts**: TRT-LLM supports prefill, decode, and aggregated roles, uses MPI-style launching (one srun per endpoint with all of its nodes) through `trtllm-llmapi-launch`, and sets `TRTLLM_EPLB_SHM_NAME` to a unique UUID per endpoint.

## TokenSpeed with Dynamo

Use `engine: tokenspeed` with `frontend.type: dynamo` to launch
`python3 -m dynamo.tokenspeed` workers, aggregated or prefill/decode. TokenSpeed flags
go under `roles.<role>.args`; srtctl sets the model, host, ports and multi-node
rendezvous. See [TokenSpeed](tokenspeed.md) and the
[aggregated](https://github.com/NVIDIA/srt-slurm/blob/main/examples/tokenspeed/dynamo-agg.yaml) and
[prefill/decode](https://github.com/NVIDIA/srt-slurm/blob/main/examples/tokenspeed/dynamo-disagg.yaml) recipes.

## ATOM with AToMesh

Use `engine: atom` with `frontend.type: atomesh` to launch native
`atom.entrypoints.openai_server` workers and the official AToMesh router. Both
aggregate workers and prefill/decode topologies use static HTTP endpoints;
disaggregated workers receive topology-owned Mooncake handshake ports.

Engine flags belong under `roles.prefill.args`, `roles.decode.args`, or
`roles.agg.args` (schema v2).
srt-slurm owns the model path, HTTP port, tensor parallel size, and KV-transfer
contract, so recipes cannot override those arguments. See the complete
[ATOM/AToMesh recipe](https://github.com/NVIDIA/srt-slurm/blob/main/examples/atom/atomesh-disagg.yaml).

To run another KV connector next to the Mooncake one (for example ATOM's in-process
LMCache CPU offload on prefill), list it under `extra-kv-connectors` in that role's
args. srtctl keeps generating the Mooncake entry, with its handshake port, and wraps
both in ATOM's `multi` connector. An aggregate role with one extra connector runs it
on its own. An `lmcache_mp` connector (ATOM's client for the `lmcache-server`
service) dials the server on its own node, port 8750, unless its
`kv_connector_extra_config` sets `lmcache.mp.port` or `lmcache.mp.server_urls`.

```yaml
roles:
  prefill:
    env:
      PYTHONHASHSEED: "0"      # LMCache prefix hashes must agree across offload workers
    args:
      extra-kv-connectors:
        - kv_connector: lmcache_offload
          kv_role: offload
          lmcache.local_cpu: true
          lmcache.max_local_cpu_size: 180   # GB per worker
          lmcache.chunk_size: 256
```

```yaml
schema: 2                      # Required: recipe layout version
name: "my-benchmark"           # Required: job name

model:                         # Required: model settings
  path: "deepseek-r1"
  container: "sglang"
  precision: "fp8"

resources:                     # Cluster facts: GPU type and GPUs per node
  gpu_type: "gb200"
  gpus_per_node: 4

slurm:                         # Optional: SLURM overrides
  time_limit: "02:00:00"

frontend:                      # Optional: router/frontend config
  type: dynamo

engine: sglang                 # Default engine; omit when every role sets engine
roles:                         # Required: one block per worker role
  prefill:
    nodes: 1
    workers: 2
    gpus: 2
    args:
      tensor-parallel-size: 2
  decode:
    nodes: 2
    workers: 2
    gpus: 4
    args:
      tensor-parallel-size: 4

benchmark:                     # Optional: benchmark config
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [256, 512]

dynamo:                        # Optional: where Dynamo comes from
  source:
    pypi: "1.4.2"

profiling:                     # Optional: profiling config
  type: "none"

output:                        # Optional: output paths
  log_dir: "./outputs/{job_id}/logs"

health_check:                  # Optional: health check settings
  max_attempts: 180
  interval_seconds: 10

setup_script: "my-setup.sh"    # Optional: custom setup script
```

The v1 spelling of this layout (`backend:`, `resources.prefill_nodes` and friends, `infra:`, `dynamo.version`) is documented in [legacy-v1.md](legacy-v1.md); `srtctl migrate` rewrites it.
