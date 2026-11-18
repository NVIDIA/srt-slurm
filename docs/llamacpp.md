# llama.cpp

`engine.type: llamacpp` and `frontend.type: llamacpp` run one aggregate
`llama-server` process. The worker owns the public OpenAI-compatible endpoint;
SRT does not start a router, nginx, Dynamo, etcd, or NATS for this frontend.
The initial topology is one worker on one node, with one or more GPUs on that
node. GPU visibility uses the existing cluster `visible_devices_env` setting.

## Linux with Docker

Install Docker with the NVIDIA Container Toolkit and the SRT dependencies and
helper binaries described in [Installation](installation.md). Select the
[Docker launcher](cluster-config.md#launcher) in `srtslurm.yaml`:

```yaml
launcher: docker
docker_args: ["--no-healthcheck"]
```

The official image's built-in Docker healthcheck targets port 8080, while SRT
uses its public port 8000. `--no-healthcheck` disables that image-level check;
SRT still waits for `/health` and `/v1/models` on the actual serving port.

Copy [the one-GPU example](https://github.com/NVIDIA/srt-slurm/blob/main/examples/docker/llamacpp-agg-1gpu.yaml).
Set `model.path` to an existing directory containing your GGUF files and
`engine.model_file` to the filename relative to that directory. For split
models, keep all shards together and select the first shard. Adjust
`resources.gpu_type`, `resources.gpus_per_node`, and the worker's GPU settings
for the machine. The example pins the official CUDA server image by digest.

```yaml
model:
  path: /models/Qwen3-0.6B-GGUF
  container: ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:ae96e52f8b2106ea82bad786ee2d08bb9b0a8c93403df9c27ae1bf065a118089
  precision: q8_0
engine:
  type: llamacpp
  model_file: Qwen3-0.6B-Q8_0.gguf
frontend:
  type: llamacpp
  enable_multiple_frontends: false
roles:
  agg:
    nodes: 1
    workers: 1
    gpus: 1
    args:
      alias: Qwen3-0.6B
      ctx-size: 4096
      n-gpu-layers: 99
      parallel: 1
benchmark:
  type: manual
```

The directory is mounted using SRT's existing model mount resolution, normally
at `/model`. The generated command selects the GGUF within that mount and
explicitly sets `--host 0.0.0.0` and the SRT public port (8000). The default
executable is `/app/llama-server`, the location in the official server image;
`engine.server_binary` selects an executable in a custom image.

```bash
uv run srtctl dry-run -f examples/docker/llamacpp-agg-1gpu.yaml
uv run srtctl apply -f examples/docker/llamacpp-agg-1gpu.yaml --serve-only
```

`manual` runs also keep the server up without a benchmark. Wait for the
readiness message, then send requests to `http://127.0.0.1:8000/v1/chat/completions`
or `/v1/completions`. Ctrl+C uses the existing launcher/process-registry cleanup
to stop the worker container. Logs are written under `outputs/<job_id>/logs/`.

## Arguments, model names, and metrics

Pass server flags in `roles.agg.args`, using names such as `ctx-size`,
`batch-size`, `threads`, `n-gpu-layers`, `tensor-split`, and `parallel`.
Underscores are normalized to hyphens. Boolean `true` emits a bare flag;
`false` omits it, so use the actual `no-*` flag with `true` when upstream
requires an explicit disabling flag. Values that llama.cpp expects as a
comma-separated string, such as `tensor-split`, should be quoted strings.

SRT manages `model`, `host`, `port`, and `metrics`; setting those flags under
`roles.agg.args` is rejected at config load. Set `engine.model_file` or
`engine.metrics` instead. `alias` controls the API model name; without it SRT
uses the GGUF filename without `.gguf`. With comma-separated aliases, clients
use the first alias.

`engine.metrics` defaults to `true`, enabling `/metrics` on the same HTTP
port. The existing Tachometer and custom-benchmark metrics discovery use that
endpoint. Setting it to `false` omits `--metrics` and removes the worker from
the metrics targets; health checks and inference still work. This is engine
telemetry, independent of the Slurm-only job monitor.

Field reference: [LlamaCppBackend](schema-reference.md#llamacppbackend).

## Benchmark clients

The official `server-cuda` image is a serving image without a Python benchmark
environment. Use `benchmark.type: custom` with a separate client image and
the existing `SRT_FRONTEND_HOST` / `SRT_FRONTEND_PORT` environment variables.
For example, replace the example's `benchmark` block with this inference smoke
test (it verifies a completion; it does not measure benchmark latency):

```yaml
benchmark:
  type: custom
  container_image: python:3.12-slim
  command: |
    python3 - <<'PY'
    import json
    import os
    from urllib.request import Request, urlopen

    base = "http://" + os.environ["SRT_FRONTEND_HOST"] + ":" + os.environ["SRT_FRONTEND_PORT"]
    with urlopen(base + "/v1/models", timeout=30) as response:
        model = json.load(response)["data"][0]["id"]
    body = {"model": model, "prompt": "Hello", "max_tokens": 16, "stream": False}
    request = Request(base + "/v1/completions", json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    assert result["choices"], result
    print(json.dumps(result, indent=2))
    PY
```

Use a client image containing your actual benchmark and set `benchmark.command`
to that command for performance runs. The GGUF directory is not a Hugging Face
tokenizer directory: built-in clients that expect HF tokenizer files need
appropriate tokenizer inputs and a Python-capable image. SRT does not install
benchmark dependencies into the minimal server image.

## Supported shape and limitations

- Exactly one aggregate worker on one node; no P/D, multiple server replicas,
  Dynamo sidecars, or multi-node/RPC serving.
- `model.path` is a local directory. HF IDs, automatic model downloading, and
  conversion to GGUF are not part of this backend.
- API authentication is not configured by SRT's readiness probe; use the
  default unauthenticated endpoint for this integration.
- Iteration-triggered profiling APIs are not available. Time-based nsys
  profiling can wrap a custom image that provides nsys.
- The Docker launcher's existing foreground lifecycle and Slurm-only monitor
  limitations apply. See [Launcher](cluster-config.md#launcher).

## Upstream contract

The example's multi-architecture image digest was checked against its OCI
labels: build `b12464`, source commit
[`d3c3e242c16b8cbf51400fad410cd69b22d64f35`](https://github.com/ggml-org/llama.cpp/commit/d3c3e242c16b8cbf51400fad410cd69b22d64f35).
The matching [Dockerfile](https://github.com/ggml-org/llama.cpp/blob/d3c3e242c16b8cbf51400fad410cd69b22d64f35/.devops/cuda.Dockerfile)
places the server at `/app/llama-server`. The matching
[server API and CLI reference](https://github.com/ggml-org/llama.cpp/blob/d3c3e242c16b8cbf51400fad410cd69b22d64f35/tools/server/README.md)
documents `--model`, `--alias`, `--metrics`, `/health` (503 while loading, 200
when ready), and `/v1/models`. SRT reuses the direct-server readiness probe:
both `/health` and a nonempty `/v1/models` listing must succeed.
