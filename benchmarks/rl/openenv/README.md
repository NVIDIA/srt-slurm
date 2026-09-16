# benchmarks/rl/openenv

Clients for [OpenEnv](https://github.com/huggingface/OpenEnv) environment servers that a recipe runs as pool services. An OpenEnv environment is an HTTP and WebSocket server with a Gym-shaped API (reset, step, state); Docker is only how the upstream project packages one. Under srt-slurm the server runs as a service on its own pool or as a rider on the trainer's pool, no Docker and no nested container involved, as long as the environment itself does not spawn containers per episode.

| File | Purpose |
| --- | --- |
| `tbench2_smoke.py` | Benchmark-step smoke against a `tbench2_env` server in `TB2_MODE=local`: health, `reset(task_id)`, one `exec`, `evaluate`. Recipe: `examples/miles/openenv-tbench2-smoke.yaml`. |

The Miles adapter for Terminal-Bench-2 (`examples/experimental/openenv/` in the Miles repo) talks to the same server through `--openenv-env-url`; `benchmarks/rl/miles/recipes/openenv_tbench2_qwen3.py` and `examples/miles/qwen3-4b-openenv-tbench2.yaml` wire it to a rider on the Ray pool.

## Two installs, not one

The env server and the trainer's rollout workers need different Python environments, and the Miles image's `python3` is itself a venv (`/opt/sglang`), which trips the obvious approach.

- **Env server:** a full venv on shared disk. `python3 -m venv --system-site-packages venv && venv/bin/pip install OpenEnv/envs/tbench2_env`. Because the image's interpreter is a venv, `--system-site-packages` reaches the base interpreter, not `/opt/sglang`, so pip installs its own numpy, openai, pydantic, camel-ai and friends. Fine for the server process, which is only the env. Point `PYTHONPATH` at `venv/lib/python3.12/site-packages` on the service.
- **Rollout workers:** a client-only site directory that adds exactly what the image lacks and nothing it already has, so sglang and Miles keep their own openai, pydantic, psutil and websockets. Resolve against the image's interpreter, keep every distribution that is already installed, install the rest with `--no-deps`, then pin the one real conflict: openenv's FastMCP client needs `mcp` 1.x (`McpError`), the image ships `mcp` 2.x, and neither sglang nor Miles import `mcp`.

  ```bash
  python3 -m pip install --dry-run --report /tmp/r.json ./OpenEnv          # resolver sees the image's packages
  python3 - <<'PY'
  import json, importlib.metadata as m
  need = []
  for it in json.load(open("/tmp/r.json"))["install"]:
      name, ver = it["metadata"]["name"], it["metadata"]["version"]
      if name.lower() == "openenv":
          continue
      try:
          m.version(name)                                                  # already in the image: keep it
      except m.PackageNotFoundError:
          need.append(f"{name}=={ver}")
  open("/tmp/need.txt", "w").write("\n".join(need) + "\n")
  PY
  python3 -m pip install --target client-site --no-deps -r /tmp/need.txt
  python3 -m pip install --target client-site --no-deps ./OpenEnv ./OpenEnv/envs/tbench2_env mcp==1.30.0
  PYTHONPATH=client-site python3 -c "from tbench2_env import Tbench2Env; import sglang, miles"
  ```

  Hand `client-site` to the recipe as `OPENENV_SITE`; the recipe appends it to the ray job's `PYTHONPATH`.
