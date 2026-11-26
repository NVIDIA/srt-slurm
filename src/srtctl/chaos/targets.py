# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target worker resolution for chaos fault injection events."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import requests

from srtctl.core.ip_utils import url_host
from srtctl.core.slurm import get_hostname_ip
from srtctl.frontends import FRONTEND_NONE, get_frontend

if TYPE_CHECKING:
    from srtctl.backends.base import WorkerMode
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import ChaosEventConfig, SrtConfig
    from srtctl.core.supervisor import EndpointKey
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)

# Fallback Prometheus gauges used when the primary hot_metric is absent on every worker.
FALLBACK_HOT_METRICS: tuple[str, ...] = (
    "vllm:num_requests_running",
    "sglang:num_running_reqs",
    "trtllm_num_requests_running",
)


@dataclass(frozen=True)
class WorkerTarget:
    """One logical worker endpoint that a chaos event can target."""

    key: EndpointKey
    mode: WorkerMode
    index: int
    processes: tuple[Process, ...]
    step_names: tuple[str, ...]

    @property
    def label(self) -> str:
        mode, index, engine = self.key
        return f"{mode}_{index}" + (f"_e{engine}" if engine else "")

    @property
    def leader(self) -> Process:
        return self.processes[0]


def parse_prometheus_metric(text: str, metric_name: str) -> float | None:
    """Sum all non-comment Prometheus samples for ``metric_name``; None when absent."""
    total = 0.0
    found = False
    prefix_brace = f"{metric_name}{{"
    prefix_space = f"{metric_name} "
    prefix_tab = f"{metric_name}\t"
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith((prefix_brace, prefix_space, prefix_tab)):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            total += float(parts[-1])
            found = True
        except ValueError:
            continue
    return total if found else None


def worker_metrics_url(target: WorkerTarget, config: SrtConfig, runtime: RuntimeContext) -> str | None:
    """HTTP Prometheus ``/metrics`` URL for ``target``'s leader process, or None when unavailable."""
    if config.frontend.type == FRONTEND_NONE:
        return None
    frontend = get_frontend(config.frontend.type)
    leader = target.leader
    port = frontend.worker_metrics_port(leader, runtime)
    if port is None or port <= 0:
        return None
    backend = config.backend_for_role(target.mode)
    path = frontend.worker_metrics_path(backend, target.mode)
    if not path:
        return None
    host = get_hostname_ip(leader.node, runtime.network_interface)
    return f"http://{url_host(host)}:{port}{path}"


def scrape_prometheus_text(url: str, *, timeout: float = 3.0) -> str | None:
    """Fetch raw Prometheus exposition text from ``url``; returns None on error."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return None
        return response.text
    except requests.RequestException:
        return None


def resolve_worker_targets(
    event: ChaosEventConfig,
    candidates: list[WorkerTarget],
    *,
    config: SrtConfig,
    runtime: RuntimeContext,
    hot_metric: str,
    rng: random.Random,
    scraper: Callable[[str], str | None] | None = None,
) -> tuple[list[WorkerTarget], dict[str, Any]]:
    """Select target worker endpoint(s) from ``candidates`` according to ``event.workers``.

    ``candidates`` must be sorted deterministically by ``(index, engine)``.
    Returns ``(chosen_targets, selection_details)``.
    """
    scraper = scraper or scrape_prometheus_text
    ordered = sorted(candidates, key=lambda t: (t.index, t.key[2]))
    mode, indices, count = event.parse_workers()
    if not ordered:
        return [], {"workers": event.workers, "candidates": 0}

    if mode == "all":
        return ordered, {"workers": "all", "chosen": [t.label for t in ordered]}

    if mode == "index":
        by_index: dict[int, list[WorkerTarget]] = {}
        for target in ordered:
            by_index.setdefault(target.index, []).append(target)
        chosen: list[WorkerTarget] = []
        seen: set[EndpointKey] = set()
        for idx in indices:
            for target in by_index.get(idx, ()):
                if target.key not in seen:
                    seen.add(target.key)
                    chosen.append(target)
        return chosen, {"workers": event.workers, "indices": list(indices), "chosen": [t.label for t in chosen]}

    if mode == "random":
        k = min(count, len(ordered))
        chosen = sorted(rng.sample(ordered, k), key=lambda t: (t.index, t.key[2]))
        return chosen, {"workers": event.workers, "count": k, "chosen": [t.label for t in chosen]}

    # mode == "hottest"
    k = min(count, len(ordered))
    metric_chain = tuple(dict.fromkeys((hot_metric, *FALLBACK_HOT_METRICS)))
    raw_texts: dict[str, str | None] = {}
    for target in ordered:
        url = worker_metrics_url(target, config, runtime)
        raw_texts[target.label] = scraper(url) if url is not None else None

    used_metric: str | None = None
    scores: list[tuple[float, int, WorkerTarget]] = []
    for metric in metric_chain:
        candidate_scores: list[tuple[float, int, WorkerTarget]] = []
        any_positive = False
        for pos, target in enumerate(ordered):
            text = raw_texts.get(target.label)
            val = parse_prometheus_metric(text, metric) if text else None
            num = val if val is not None else 0.0
            if num > 0.0:
                any_positive = True
            candidate_scores.append((num, pos, target))
        if any_positive:
            used_metric = metric
            scores = candidate_scores
            break

    if used_metric is None:
        chosen = ordered[:k]
        logger.warning(
            "Chaos workers=%r: %s (and fallbacks) returned 0/absent on every %s worker; falling back to %s",
            event.workers,
            hot_metric,
            event.role,
            [t.label for t in chosen],
        )
        return chosen, {
            "workers": event.workers,
            "metric": hot_metric,
            "fallback": True,
            "chosen": [t.label for t in chosen],
        }

    ranked = sorted(scores, key=lambda item: (-item[0], item[1]))[:k]
    chosen = sorted((t for _, _, t in ranked), key=lambda t: (t.index, t.key[2]))
    return chosen, {
        "workers": event.workers,
        "metric": used_metric,
        "fallback": False,
        "scores": {t.label: val for val, _, t in scores},
        "chosen": [t.label for t in chosen],
    }
