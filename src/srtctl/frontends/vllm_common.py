# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker contracts shared by routers of native vLLM servers."""

from typing import Any


def validate_mooncake_workers(config: Any) -> None:
    """A router caching bootstrap engine IDs needs matching, stable P/D workers."""
    backend = config.backend
    if config.topology.num_agg:
        raise ValueError("the Mooncake connector requires a prefill/decode topology; aggregate workers transfer no KV")
    connectors = {mode: backend.bootstrap_connector(mode) for mode in ("prefill", "decode")}
    if connectors["prefill"] != connectors["decode"]:
        names = ", ".join(f"{mode}={backend.kv_connector_classes(mode)}" for mode in connectors)
        raise ValueError(f"the Mooncake connector must be set on both prefill and decode; got {names}")
    if config.health_check is None:
        raise ValueError("the Mooncake connector requires health_check so prefill registers before the Router starts")
    if config.topology.worker_restart("prefill").enabled:
        raise ValueError(
            "the Mooncake connector requires prefill restart: never; the Router caches engine IDs at startup"
        )
    if (backend._get_dp_size("prefill") or 1) > 1:
        raise ValueError("the Mooncake connector currently requires prefill data-parallel-size: 1")
