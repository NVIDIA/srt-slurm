# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The 2.0 ``placement:`` authoring surface for where the frontend, benchmark client, and infra run.

One vocabulary replaces the per-block placement knobs::

    frontend:  {placement: {node: head | first_decode | dedicated}}
    benchmark: {placement: {node: head | last_decode | dedicated}}
    infra:     {placement: {node: head | dedicated}}

``node: dedicated`` reserves a node for that component (and implies the head
location, which the legacy validation already required). Any other value is a
location string passed through. :func:`expand_placement` normalizes these blocks
into the existing internal fields before schema load, so no consumer changes and
the legacy fields still load.

| Block       | node: dedicated sets                          | node: <other> sets                    |
|-------------|-----------------------------------------------|---------------------------------------|
| frontend    | dedicated_node=True, orchestrator_placement=head | orchestrator_placement=<other>     |
| benchmark   | client_dedicated_node=True, client_placement=head | client_placement=<other>          |
| infra       | etcd_nats_dedicated_node=True                 | (only ``head`` is valid; = False)     |
"""

from __future__ import annotations

from typing import Any

DEDICATED = "dedicated"

# block name -> (placement-str field, dedicated-bool field or None, legacy fields it fills)
_FRONTEND = ("orchestrator_placement", "dedicated_node")
_BENCHMARK = ("client_placement", "client_dedicated_node")


def _expand_block(section: dict[str, Any], *, place_field: str, dedicated_field: str, block: str) -> None:
    placement = section.get("placement")
    if not isinstance(placement, dict):
        return
    collisions = [f for f in (place_field, dedicated_field) if f in section]
    if collisions:
        raise ValueError(f"{block}.placement cannot be combined with {block}." + f", {block}.".join(collisions))
    node = placement.get("node")
    if node is None:
        raise ValueError(f"{block}.placement requires a 'node' value")
    unknown = set(placement) - {"node"}
    if unknown:
        raise ValueError(f"{block}.placement has unknown keys: {', '.join(sorted(unknown))}")
    if node == DEDICATED:
        section[dedicated_field] = True
        section[place_field] = "head"
    else:
        section[place_field] = node
    section.pop("placement", None)


def expand_placement(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize ``placement:`` blocks on frontend / benchmark / infra into the internal fields, in place."""
    frontend = config.get("frontend")
    if isinstance(frontend, dict):
        _expand_block(frontend, place_field=_FRONTEND[0], dedicated_field=_FRONTEND[1], block="frontend")

    benchmark = config.get("benchmark")
    if isinstance(benchmark, dict):
        _expand_block(benchmark, place_field=_BENCHMARK[0], dedicated_field=_BENCHMARK[1], block="benchmark")

    infra = config.get("infra")
    if isinstance(infra, dict) and isinstance(infra.get("placement"), dict):
        if "etcd_nats_dedicated_node" in infra:
            raise ValueError("infra.placement cannot be combined with infra.etcd_nats_dedicated_node")
        placement = infra["placement"]
        node = placement.get("node")
        unknown = set(placement) - {"node"}
        if unknown:
            raise ValueError(f"infra.placement has unknown keys: {', '.join(sorted(unknown))}")
        if node == DEDICATED:
            infra["etcd_nats_dedicated_node"] = True
        elif node == "head":
            infra["etcd_nats_dedicated_node"] = False
        else:
            raise ValueError(f"infra.placement.node must be 'head' or 'dedicated', got {node!r}")
        infra.pop("placement", None)

    return config
