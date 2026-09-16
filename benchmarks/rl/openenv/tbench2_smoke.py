#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for an OpenEnv terminal-bench env server that runs without Docker.

The benchmark step of examples/miles/openenv-tbench2-smoke.yaml runs this against
the `env` service (tbench2_env in TB2_MODE=local): GET /health, reset(task_id), one
exec step, evaluate. Exit 0 when reset and exec succeeded; the evaluate reward is
printed, not judged, since an unsolved task legitimately scores 0.

Environment:
  OPENENV_ENV_URL        http://host:port of the env server. Default: the first IP of
                         the `env` service (SRT_SERVICE_ENV_IPS) on port 8003.
  OPENENV_TASK_ID        Terminal-Bench-2 task to reset into (default headless-terminal).
  OPENENV_SMOKE_COMMAND  Shell command to run inside the task's shell.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import urllib.request


def env_url() -> str:
    url = os.environ.get("OPENENV_ENV_URL")
    if url:
        return url
    ips = os.environ.get("SRT_SERVICE_ENV_IPS", "")
    if not ips:
        sys.exit("OPENENV_ENV_URL is unset and SRT_SERVICE_ENV_IPS is not in the environment")
    return f"http://{ips.split(',')[0]}:8003"


def _observation(result):
    return getattr(result, "observation", result)


def check_health(url: str) -> None:
    with urllib.request.urlopen(f"{url}/health", timeout=30) as response:
        print(f"health {response.status}: {response.read(200).decode(errors='replace').strip()}")


async def main(url: str) -> int:
    from tbench2_env import Tbench2Action, Tbench2Env

    task_id = os.environ.get("OPENENV_TASK_ID", "headless-terminal")
    command = os.environ.get("OPENENV_SMOKE_COMMAND", "pwd; whoami; hostname; ls -la")

    started = time.monotonic()
    async with Tbench2Env(base_url=url, message_timeout_s=600) as env:
        obs = _observation(await env.reset(task_id=task_id))
        print(
            f"reset task_id={obs.task_id} task_path={obs.task_path} success={obs.success} "
            f"error={obs.error!r} ({time.monotonic() - started:.1f}s)"
        )
        print("instruction:", (obs.instruction or "")[:400].replace("\n", " "))
        if not obs.success:
            return 2

        step = await env.step(Tbench2Action(action_type="exec", command=command))
        obs = _observation(step)
        print(f"exec success={obs.success} error={obs.error!r}")
        print((obs.output or "")[:1500])
        if not obs.success:
            return 3

        evaluated = await env.step(Tbench2Action(action_type="evaluate"))
        obs = _observation(evaluated)
        print(
            f"evaluate reward={getattr(evaluated, 'reward', None)} done={getattr(evaluated, 'done', None)} success={obs.success}"
        )
        print((obs.output or "")[:800])

    print("openenv smoke ok: the env server answered reset, exec and evaluate without Docker")
    return 0


if __name__ == "__main__":
    target = env_url()
    check_health(target)
    sys.exit(asyncio.run(main(target)))
