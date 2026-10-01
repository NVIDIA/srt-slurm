# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-schema normalizer: a declared ``mooncake-master`` service sets the internal field the runtime reads.

Like ``expand_roles``, this runs on the raw recipe dict before ``SrtConfig``
loads it: the recipe declares the master under ``services:``, and the consumers
of ``backend.mooncake_kv_store`` read the field this fills. The declared entry
stays in the list: ``effective_services`` then treats it as an override of the
implicit one.
"""

from __future__ import annotations

from typing import Any


def expand_services(config: dict[str, Any]) -> dict[str, Any]:
    """Map a declared ``mooncake-master`` service onto ``backend.mooncake_kv_store``, in place."""
    services = config.get("services")
    if not isinstance(services, list):
        return config
    declared = [s for s in services if isinstance(s, dict)]

    masters = [s for s in declared if s.get("type") == "mooncake-master" and s.get("enabled", True)]
    if len(masters) > 1:
        raise ValueError("only one mooncake-master service is supported")
    if masters:
        master = masters[0]
        backend = config.setdefault("backend", {})
        if not isinstance(backend, dict):
            raise TypeError("backend must be a mapping")
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
        backend["mooncake_kv_store"] = mapped
    return config
