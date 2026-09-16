# benchmarks/rl/openenv

Clients for [OpenEnv](https://github.com/huggingface/OpenEnv) environment servers that a recipe runs as pool services. An OpenEnv environment is an HTTP and WebSocket server with a Gym-shaped API (reset, step, state); Docker is only how the upstream project packages one. Under srt-slurm the server runs as a service on its own pool or as a rider on the trainer's pool, no Docker and no nested container involved, as long as the environment itself does not spawn containers per episode.

| File | Purpose |
| --- | --- |
| `tbench2_smoke.py` | Benchmark-step smoke against a `tbench2_env` server in `TB2_MODE=local`: health, `reset(task_id)`, one `exec`, `evaluate`. Recipe: `examples/miles/openenv-tbench2-smoke.yaml`. |

The Miles adapter for Terminal-Bench-2 (`examples/experimental/openenv/` in the Miles repo) talks to the same server through `--openenv-env-url`; `docs/miles.md` covers wiring it to a pool service.
