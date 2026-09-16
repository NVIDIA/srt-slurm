# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-schema normalizer: declared infra-class services set the internal fields the runtime reads.

Like ``expand_roles`` and ``expand_placement``, this runs on the raw recipe dict
before ``SrtConfig`` loads it: the recipe declares ``etcd`` / ``nats`` /
``mooncake-master`` entries under ``services:``, and the consumers of
``infra.etcd_nats_dedicated_node``, ``infra.nats_max_payload_mb``, and
``backend.mooncake_kv_store`` read the fields this fills. The recipe cannot set
those fields itself (``srtctl.core.config.require_current_schema`` rejects the
pre-2.0 ``infra:`` and ``backend:`` blocks), so there is nothing to reconcile
against. The declared entries stay in the list: ``effective_services`` then
treats them as overrides of the implicit ones.
"""

from __future__ import annotations

from typing import Any

_INFRA_KINDS = ("etcd", "nats")


def expand_services(config: dict[str, Any]) -> dict[str, Any]:
    """Map declared ``etcd`` / ``nats`` / ``mooncake-master`` services onto ``infra`` and ``backend`` fields, in place."""
    services = config.get("services")
    if not isinstance(services, list):
        return config
    declared = [s for s in services if isinstance(s, dict)]

    infra_entries = [s for s in declared if s.get("type") in _INFRA_KINDS and s.get("enabled", True)]
    if infra_entries:
        infra = config.setdefault("infra", {})
        if not isinstance(infra, dict):
            raise TypeError("infra must be a mapping")
        dedicated = {bool((s.get("placement") or {}).get("node") == "dedicated") for s in infra_entries}
        if len(dedicated) > 1:
            raise ValueError(
                "services etcd and nats must agree on placement.node: dedicated (they share the infra node)"
            )
        infra["etcd_nats_dedicated_node"] = dedicated.pop()
        for entry in infra_entries:
            if entry.get("type") == "nats":
                payload = (entry.get("options") or {}).get("max_payload_mb")
                if payload is not None:
                    infra["nats_max_payload_mb"] = payload

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
        backend["mooncake_kv_store"] = mapped
    return config
