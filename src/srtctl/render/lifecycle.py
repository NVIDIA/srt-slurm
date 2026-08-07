# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle render helpers for standalone bash output."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

from srtctl.core.schema import SrtConfig
from srtctl.ports import FRONTEND_PUBLIC_PORT


@dataclass(frozen=True)
class LifecycleRenderContext:
    """Values needed to render a standalone server/benchmark lifecycle script."""

    lifecycle_runtime_text: str
    server_config_filename: str
    server_config_text: str
    benchmark_config_filename: str
    benchmark_config_text: str
    expected_prefill: int
    expected_decode: int
    frontend_type: str
    frontend_port: int
    health_timeout_seconds: int
    health_interval_seconds: int


def heredoc_marker(payload: str, *, prefix: str = "SRTCTL_RUNTIME_CONFIG") -> str:
    """Return a here-doc marker that cannot collide with the payload."""
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    marker = f"{prefix}_{digest}"
    while marker in payload:
        marker = f"{marker}_END"
    return marker


def render_lifecycle_runtime() -> str:
    """Render the reusable bash function library embedded in standalone scripts."""
    template_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), keep_trailing_newline=True)
    return env.get_template("lifecycle_runtime.sh.j2").render()


def make_manual_server_config_text(benchmark_config_text: str) -> str:
    """Return a copy of a benchmark config that starts the server and waits manually."""
    raw_config = yaml.safe_load(benchmark_config_text)
    if not isinstance(raw_config, dict):
        raise ValueError("Expected benchmark config YAML to load as a mapping")

    server_config = copy.deepcopy(raw_config)
    benchmark = server_config.setdefault("benchmark", {})
    if not isinstance(benchmark, dict):
        raise ValueError("Expected benchmark config 'benchmark' field to be a mapping")
    benchmark["type"] = "manual"
    return yaml.safe_dump(server_config, sort_keys=False)


def expected_worker_counts(config: SrtConfig) -> tuple[int, int]:
    """Return expected prefill/decode counts for frontend readiness checks."""
    resources = config.resources
    if resources.num_agg > 0:
        return 0, resources.num_agg
    return resources.num_prefill, resources.num_decode


def build_lifecycle_render_context(
    config: SrtConfig,
    benchmark_config_text: str,
    *,
    server_config_filename: str = "config_server.yaml",
    benchmark_config_filename: str = "config.yaml",
) -> LifecycleRenderContext:
    """Build render context for a standalone lifecycle script."""
    expected_prefill, expected_decode = expected_worker_counts(config)
    health_timeout = int(float(config.health_check.max_attempts) * float(config.health_check.interval_seconds))
    health_interval = max(1, int(float(config.health_check.interval_seconds)))

    return LifecycleRenderContext(
        lifecycle_runtime_text=render_lifecycle_runtime().rstrip("\n"),
        server_config_filename=server_config_filename,
        server_config_text=make_manual_server_config_text(benchmark_config_text),
        benchmark_config_filename=benchmark_config_filename,
        benchmark_config_text=benchmark_config_text,
        expected_prefill=expected_prefill,
        expected_decode=expected_decode,
        frontend_type=config.frontend.type,
        frontend_port=FRONTEND_PUBLIC_PORT,
        health_timeout_seconds=health_timeout,
        health_interval_seconds=health_interval,
    )
