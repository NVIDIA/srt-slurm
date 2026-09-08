---
name: srtctl
description: Author, validate, submit, and read back srt-slurm (srtctl) benchmark jobs on a Slurm cluster
---

# srtctl

`srtctl` runs LLM inference benchmarks on Slurm: it reads a recipe (YAML), asks Slurm for the nodes, launches the engine workers (SGLang, vLLM, TRT-LLM, or the Dynamo mocker), the frontend (Dynamo, or the engine's own router), the services they need (etcd, NATS, the Mooncake master, the metrics exporters, anything declared under `services:`), the tachometer metrics scraper, and the benchmark client, then collects logs, a benchmark rollup, the tachometer parquet, and a per-run HTML dashboard under `outputs/<job_id>/`.

## Before anything else

- Work from the srt-slurm checkout that has the cluster's `srtslurm.yaml` (account, partition, `srtctl_root`, model and container aliases). Never invent aliases; read that file.
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
  type: dynamo                # dynamo | sglang | vllm-router | vllm | trtllm_serve
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
- `services:` declares sidecars. etcd and NATS (Dynamo), the Mooncake master (`backend.mooncake_kv_store`), and the DCGM and node exporters (tachometer) are implied; declare one by name only to change it (`placement.node: dedicated`, `container`, `options`, `external: <address>`, `enabled: false`).
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
- `logs/benchmark.out` and `logs/benchmark-rollup.json`: the client and its normalized result.
- `logs/tachometer/raw/scrape/final.parquet`: every scraped metric sample; `logs/perf_dashboard.html`: the rendered dashboard.
- `recipe.lock.yaml`: the exact resolved recipe, pinned sources, and container identity.

Cleanup is graceful: workers, frontends, and services are SIGTERMed through their Slurm steps, then etcd and NATS. A `scancel` of the job triggers the same path.

## MCP

`srtctl-mcp` exposes the schema tools (`schema_summary`, `explain_field`, `validate_config`, `resolve_config`, `get_config_reference`) anywhere, and the job tools (`submit_job`, `dry_run`, `job_status`, `job_logs`, `list_jobs`, `cancel_job`) when it runs on a Slurm login node of the cluster.
