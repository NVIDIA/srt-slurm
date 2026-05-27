# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml

from srtctl.core.schema import SrtConfig
from srtctl.render.lifecycle import build_lifecycle_render_context


def test_lifecycle_render_context_splits_server_and_benchmark_configs() -> None:
    benchmark_config_text = yaml.safe_dump(
        {
            "name": "standalone-render",
            "model": {
                "path": "hf:fake/mock-model",
                "container": "nvcr.io/fake:latest",
                "precision": "fp8",
            },
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "agg_nodes": 1,
                "agg_workers": 1,
            },
            "benchmark": {
                "type": "custom",
                "command": "echo benchmark",
            },
        },
        sort_keys=False,
    )
    config = SrtConfig.Schema().load(yaml.safe_load(benchmark_config_text))

    context = build_lifecycle_render_context(config, benchmark_config_text)

    server_config = yaml.safe_load(context.server_config_text)
    benchmark_config = yaml.safe_load(context.benchmark_config_text)
    assert server_config["benchmark"]["type"] == "manual"
    assert benchmark_config["benchmark"]["type"] == "custom"
    assert context.expected_prefill == 0
    assert context.expected_decode == 1
    assert "srt_start_tachometer" in context.lifecycle_runtime_text
    assert "srt_wait_workers_ready" in context.lifecycle_runtime_text
