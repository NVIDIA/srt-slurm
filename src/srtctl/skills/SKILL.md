---
name: srtctl
description: Author, validate, submit, and read back srt-slurm (srtctl) benchmark jobs on a Slurm cluster
---

# srtctl

`srtctl` runs LLM inference benchmarks on Slurm: it reads a recipe (YAML), asks Slurm for the nodes, launches the engine workers (SGLang, vLLM, TRT-LLM, or the Dynamo mocker), the frontend (Dynamo, or the engine's own router), the services they need (etcd, NATS, the Mooncake master, the metrics exporters, anything declared under `services:`), the tachometer metrics scraper, and the benchmark client, then collects logs, a benchmark rollup, the tachometer parquet, and a per-run HTML dashboard under `outputs/<job_id>/`.

## Fresh checkout on a new cluster

Do this once per cluster, from a login node. Nothing is installed system-wide.

```bash
git clone https://github.com/NVIDIA/srt-slurm.git && cd srt-slurm
uv sync --no-dev                      # Python 3.10+; the system interpreter is fine
make setup ARCH=aarch64               # the COMPUTE nodes' arch (aarch64 for Grace/Vera, x86_64 otherwise)
```

`make setup` downloads etcd, nats-server, uv and the tachometer binaries into the checkout (keep the checkout on a filesystem the compute nodes mount) and writes a first `srtslurm.yaml` after prompting for account, partition and GPUs per node. Then edit `srtslurm.yaml` so every recipe can stay portable:

```yaml
default_partition: "batch"            # default_account too, if the cluster enforces one
default_time_limit: "04:00:00"
gpus_per_node: 4
default_gpu_type: "gb200"
network_interface: "eth0"             # the interface workers reach each other on
srtctl_root: "/path/to/srt-slurm"
model_paths:                          # alias -> path; recipes say `path: <alias>`
  qwen3-0.6b: "/shared/models/Qwen3-0.6B"
default_mounts:                       # bind mounts every container needs
  /shared: /shared
preflight: false                      # only if model paths exist solely on compute nodes (node-local NVMe)
```

Rules for that file: unknown keys reject the whole file with a WARNING and srtctl falls back to built-in defaults (a dry-run rendering `--partition=default` means this happened); `default_container` is not a key. Check the result with `srtctl dry-run -f examples/sglang/dynamo-agg.yaml`.

Containers: `model.container` is a registry reference pulled by pyxis at job start (`lmsysorg/sglang:nightly-cu134`; NGC images are `nvcr.io#nvidia/<image>:<tag>`) or a path to an enroot `.sqsh` file. Prefer writing it in the recipe over a `containers:` alias so a shared recipe says which image it ran on. If the image already ships Dynamo, set `dynamo.install: false`; otherwise give `dynamo.source`.

## Before anything else

- Work from the srt-slurm checkout that has the cluster's `srtslurm.yaml` (account, partition, `srtctl_root`, model aliases). Never invent aliases; read that file.
- Run `srtctl dry-run -f <recipe>` before `srtctl apply`. It renders the sbatch script, every mount and environment variable, and every service (implied ones are marked `implied by:`). A recipe that fails dry-run will fail on the cluster.
- Recipes are schema 2 (`schema: 2`). Write block-style YAML, never `{}` or `[]` flow style. Keep every recipe loadable by `srtctl dry-run`.

## The 2.0 recipe shape

```yaml
schema: 2
name: "qwen3-0.6b-sglang-dynamo-agg"

model:
  path: "qwen3-0.6b"          # alias from srtslurm.yaml, an hf:<repo> spec, or a path
  container: "sglang"         # alias or image
  precision: "bf16"

resources:
  gpu_type: "h100"
  gpus_per_node: 8

frontend:
  type: dynamo                # dynamo | sglang-router | vllm-router | sglang | vllm | trtllm_serve (last three: direct, one worker)
  args:
    router-mode: "kv"

engine: sglang                # sglang | vllm | trtllm | mocker, or a mapping with engine-wide knobs
roles:                        # one block per worker role: prefill, decode, agg
  agg:
    nodes: 1
    workers: 2
    gpus: 1
    env:
      PYTHONUNBUFFERED: "1"
    args:                     # the engine's own CLI flags, as a mapping
      tensor-parallel-size: 1
      mem-fraction-static: 0.5

benchmark:
  type: "sa-bench"            # sa-bench | sglang-bench | gsm8k | custom | manual | ...
  isl: 128
  osl: 128
  concurrencies: "4x8"
```

- `dynamo.source` chooses how Dynamo is installed: `pypi: "1.4.2"`, `wheel: <path>`, or `git: <url>` with `rev: <sha, tag, or refs/pull/N/head>`; `srtctl apply` pins the rev to a commit.
- `placement.node: dedicated` on `frontend` or `benchmark` reserves a node for it.
- Disaggregated (`roles.prefill` + `roles.decode`): `roles.decode.nodes: colocate` shares the prefill nodes (give both roles an explicit `gpus`; the loader rejects a split that does not fit). `roles.<role>.sidecar: true` runs the engine's own server with a Dynamo sidecar instead of the Python `dynamo.<engine>` worker (`frontend.type: dynamo` only).
- `services:` declares sidecars. etcd (Dynamo; NATS only when `dynamo.request_plane` or `event_plane` is `nats`), the Mooncake master (when a `mooncake-master` service is declared), and the DCGM and node exporters (tachometer) are implied; declare one by name only to change it (`placement.node: dedicated`, `container`, `options`, `external: <address>`, `enabled: false`).
- `--set KEY=VALUE` and `--unset KEY` on `apply` and `dry-run` override any recipe key without editing the file: `--set resources.gpu_type=b200 --set roles.agg.gpus=2`.
- `srtctl migrate -f <recipe> --in-place` rewrites a v1 recipe (`backend:`, `*_environment`, `infra:`) to this shape; `--verify` proves the two resolve identically.

## Commands

```bash
srtctl dry-run -f recipe.yaml [--set K=V ...]
srtctl apply -f recipe.yaml -y --json          # one JSON line per submission: slurm_job_id, output_dir
srtctl apply -f recipe.yaml --serve-only        # keep the endpoint up, no benchmark
srtctl monitor                                  # live view of your jobs
srtctl migrate -f recipes/ --verify
squeue --me ; sacct -j <id> -X ; scancel <id>
```

## Reading a run

Everything is under `outputs/<job_id>/`:

- `logs/sweep_<job_id>.log`: the orchestrator. Stages in order: services (infra), workers, frontend, health, benchmark, cleanup. `[ERROR]` lines and `Critical process ... exited` tell you what died.
- `logs/<node>_<mode>_w<i>.out`: one per worker. `logs/<node>_frontend_0.out`, `logs/<node>_router_0.out`: the frontend.
- `logs/service_<name>.out`: etcd, nats, dcgm-exporter, node-exporter, mooncake-master, and declared services.
- `logs/benchmark.out` and `logs/benchmark-rollup.json`: the client and its normalized result. A run is only good if `Total generated tokens` in `benchmark.out` is plausible: sa-bench reports success even when every response is empty. In disaggregated runs, `Decode transfer failed` lines in the decode worker log mean the KV transfer is broken.
- `logs/tachometer/raw/scrape/final.parquet`: every scraped metric sample; `logs/perf_dashboard.html`: the rendered dashboard.
- `recipe.lock.yaml`: the exact resolved recipe, pinned sources, and container identity.

Cleanup is graceful: workers, frontends, and services are SIGTERMed through their Slurm steps, then etcd (and NATS when it ran). A `scancel` of the job triggers the same path.

## MCP

`srtctl-mcp` exposes the schema tools (`schema_summary`, `explain_field`, `validate_config`, `resolve_config`, `get_config_reference`) anywhere, and the job tools (`submit_job`, `dry_run`, `job_status`, `job_logs`, `list_jobs`, `cancel_job`) when it runs on a Slurm login node of the cluster.
