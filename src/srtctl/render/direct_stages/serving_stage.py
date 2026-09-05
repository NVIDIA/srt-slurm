# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker, router, readiness, and smoke-request stage for direct execution."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class ServingStageMixin:
    """Start Dynamo workers/router and wait for real request readiness."""

    plan: dict[str, Any]
    log_dir: Path

    def log(self, message: str) -> None:
        raise NotImplementedError

    def _die(self, message: str) -> None:
        raise NotImplementedError

    def _launch_shell(self, label: str, log_name: str, command: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _assert_services_alive(self) -> None:
        raise NotImplementedError

    def _start_workers_and_router(self) -> None:
        for worker in self.plan["worker_processes"]:
            self._launch_shell(
                str(worker["label"]), str(worker["log_name"]), str(worker["command"]), env=dict(os.environ)
            )
        self._launch_shell("router", "router.log", str(self.plan["router_command"]), env=dict(os.environ))
        self._wait_router_ready()

    def _wait_router_ready(self) -> None:
        url = f"http://127.0.0.1:{self.plan['frontend_port']}/health"
        deadline = time.monotonic() + int(self.plan["health_timeout_seconds"])
        interval = int(self.plan["health_interval_seconds"])
        readiness_log = self.log_dir / "readiness.log"
        with readiness_log.open("a", encoding="utf-8") as handle:
            while time.monotonic() < deadline:
                self._assert_services_alive()
                try:
                    with urllib.request.urlopen(url, timeout=5) as response:
                        payload = json.loads(response.read().decode())
                    prefill, decode = router_counts(payload)
                    if prefill >= int(self.plan["expected_prefill"]) and decode >= int(self.plan["expected_decode"]):
                        handle.write(
                            f"Router ready: prefill={prefill}/{self.plan['expected_prefill']} decode={decode}/{self.plan['expected_decode']}\n"
                        )
                        self.log(
                            f"Router ready: prefill={prefill}/{self.plan['expected_prefill']} decode={decode}/{self.plan['expected_decode']}"
                        )
                        return
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                    pass
                time.sleep(interval)
        self._die(f"Router did not report expected workers before timeout: {url}")

    def _smoke_chat(self) -> None:
        payload = json.dumps(
            {
                "model": self.plan["model_name"],
                "messages": [{"role": "user", "content": "Reply with one word: ready"}],
                "max_tokens": 16,
                "temperature": 0,
            }
        ).encode()
        request = urllib.request.Request(
            f"{os.environ['SRT_FRONTEND_URL']}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read())
        choices = body.get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else None
        if not isinstance(content, str) or not content.strip():
            self._die(f"Smoke chat returned no content: {body}")
        (self.log_dir / "smoke.json").write_text(json.dumps(body), encoding="utf-8")


def router_counts(payload: dict[str, Any]) -> tuple[int, int]:
    """Count Dynamo prefill and decode workers in a router health response."""
    prefill = decode = 0
    for instance in payload.get("instances", []):
        if instance.get("endpoint") != "generate":
            continue
        if instance.get("component") == "prefill":
            prefill += 1
        elif instance.get("component") in ("decode", "tensorrt_llm", "backend"):
            decode += 1
    return prefill, decode
