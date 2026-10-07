# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-schema normalizer: a declared ``mooncake-master`` service sets ``engine.mooncake_kv_store``.

This runs on the raw recipe dict before ``SrtConfig`` loads it: the recipe
declares the master under ``services:``, and the consumers of
``engine.mooncake_kv_store`` read the field this fills. The declared entry stays
in the list: ``effective_services`` then treats it as an override of the
implicit one.
"""

from __future__ import annotations

from typing import Any


def expand_services(config: dict[str, Any]) -> dict[str, Any]:
    """Map a declared ``mooncake-master`` service onto ``engine.mooncake_kv_store``, in place."""
    services = config.get("services")
    if not isinstance(services, list):
        return config
    declared = [s for s in services if isinstance(s, dict)]

    masters = [s for s in declared if s.get("type") == "mooncake-master" and s.get("enabled", True)]
    if len(masters) > 1:
        raise ValueError("only one mooncake-master service is supported")
    if masters:
        master = masters[0]
        engine_raw = config.get("engine")
        engine: dict[str, Any]
        if engine_raw is None:
            roles = config.get("roles")
            if isinstance(roles, dict) and any(
                isinstance(spec, dict) and spec.get("engine") is not None for spec in roles.values()
            ):
                raise ValueError(
                    "a mooncake-master service needs a top-level engine; role-specific engines do not support it"
                )
            engine = {"type": "sglang"}
        elif isinstance(engine_raw, str):
            engine = {"type": engine_raw}
        elif isinstance(engine_raw, dict):
            engine = engine_raw
        else:
            raise TypeError("engine must be a type string or a mapping")
        config["engine"] = engine
        mapped: dict[str, Any] = {}
        if master.get("container"):
            mapped["container"] = master["container"]
        if master.get("args"):
            mapped["master_extra_args"] = list(master["args"])
        store_config = (master.get("options") or {}).get("store_config")
        if store_config:
            mapped["store_config"] = store_config
        options = master.get("options") or {}
        if "device_names_by_gpu" in options:
            mapped["device_names_by_gpu"] = options["device_names_by_gpu"]
        for key in ("eviction_ratio", "master_timeout_s", "store_role"):
            if key in options:
                mapped[key] = options[key]
        engine["mooncake_kv_store"] = mapped
    return config
