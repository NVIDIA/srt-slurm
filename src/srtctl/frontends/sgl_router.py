# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The standalone upstream experimental/sgl-router binary, not Model Gateway."""

from typing import Any, ClassVar

import requests
from prometheus_client.parser import text_string_to_metric_families

from srtctl.core.health import WorkerHealthResult
from srtctl.core.topology import Process
from srtctl.frontends.base import register_frontend
from srtctl.frontends.static_router import RouterWorker, StaticRouterFrontend


@register_frontend("sgl-router")
class SGLRouterFrontend(StaticRouterFrontend):
    """Launch the stock standalone router with static aggregate or P/D workers."""

    type: ClassVar[str] = "sgl-router"
    required_backend: ClassVar[str | None] = "sglang"
    executable: ClassVar[tuple[str, ...]] = ("/usr/local/bin/sgl-router",)
    # This binary discovers P/D roles from each worker's /server_info.
    pd_flag: ClassVar[str] = ""
    process_name: ClassVar[str] = "sgl_router"
    wait_for_workers_before_start: ClassVar[bool] = True
    use_bash_wrapper: ClassVar[bool] = False

    def validate(self, config: Any) -> None:
        if config.frontend.numa_bind:
            raise ValueError("sgl-router's standalone image does not contain numactl; unset frontend.numa_bind")
        for mode in config.roles:
            if config.backend.is_grpc_mode(mode):
                raise ValueError("sgl-router requires HTTP SGLang workers, not grpc-mode")
        args = {key.replace("_", "-"): value for key, value in (config.frontend.args or {}).items()}
        if "model-id" in args and args["model-id"] != config.served_model_name:
            raise ValueError("frontend.args.model-id must match the workers' served model name")

    def build_router_command(self, workers: list[RouterWorker], host: str, port: int, backend: Any) -> list[str]:
        if not workers:
            raise ValueError("sgl-router needs at least one worker")
        modes = {worker.mode for worker in workers}
        if modes != {"agg"} and modes != {"prefill", "decode"}:
            raise ValueError("sgl-router requires aggregate workers or a complete prefill/decode pair")
        return [*self.executable, "--host", host, "--port", str(port), "--worker-urls", *(w.url for w in workers)]

    def get_managed_frontend_args(self, config: Any, backend: Any, backend_processes: list[Process]) -> list[str]:
        args = {key.replace("_", "-"): value for key, value in (config.frontend.args or {}).items()}
        if "model-id" in args:
            return []
        return ["--model-id", config.served_model_name]

    def get_backend_health_urls(
        self, backend: Any, backend_processes: list[Process], network_interface: str | None = None
    ) -> list[str]:
        return [f"{w.url}/health" for w in self.collect_workers(backend, backend_processes, network_interface)]

    def probe_ready(
        self, host: str, port: int, expected_prefill: int, expected_decode: int, config: Any
    ) -> WorkerHealthResult:
        # /readyz only promises a usable pool, not that every configured worker
        # joined. The same registry snapshot supplies pool sizes and breaker
        # health in /metrics. Direct worker health is checked separately.
        root = f"http://{host}:{port}"
        ready = requests.get(f"{root}/readyz", timeout=5)
        ready.raise_for_status()
        metrics = requests.get(f"{root}/metrics", timeout=5)
        metrics.raise_for_status()
        counts: dict[str, int] = {}
        healthy: dict[str, bool] = {}
        for family in text_string_to_metric_families(metrics.text):
            for sample in family.samples:
                if sample.name == "sgl_router_workers":
                    counts[sample.labels["mode"]] = int(sample.value)
                elif sample.name == "sgl_router_worker_health":
                    healthy[sample.labels["worker_url"]] = sample.value == 1
        prefills = counts.get("prefill", 0)
        decodes = counts.get("decode", 0) + counts.get("plain", 0)
        all_healthy = len(healthy) == prefills + decodes and all(healthy.values())
        complete = prefills >= expected_prefill and decodes >= expected_decode
        return WorkerHealthResult(
            ready=complete and all_healthy,
            message=f"sgl-router: {prefills} prefill, {decodes} decode/aggregate, {sum(healthy.values())} healthy",
            prefill_ready=prefills if all_healthy else 0,
            prefill_expected=expected_prefill,
            decode_ready=decodes if all_healthy else 0,
            decode_expected=expected_decode,
        )
