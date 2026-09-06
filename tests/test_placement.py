# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from srtctl.core.placement import expand_placement
from srtctl.core.schema import SrtConfig


def test_frontend_placement_location_and_dedicated() -> None:
    cfg = {"frontend": {"type": "dynamo", "placement": {"node": "first_decode"}}}
    expand_placement(cfg)
    assert cfg["frontend"]["orchestrator_placement"] == "first_decode"
    assert "placement" not in cfg["frontend"]

    cfg = {"frontend": {"type": "dynamo", "placement": {"node": "dedicated"}}}
    expand_placement(cfg)
    assert cfg["frontend"]["dedicated_node"] is True
    assert cfg["frontend"]["orchestrator_placement"] == "head"


def test_benchmark_placement_location_and_dedicated() -> None:
    cfg = {"benchmark": {"type": "sa-bench", "placement": {"node": "last_decode"}}}
    expand_placement(cfg)
    assert cfg["benchmark"]["client_placement"] == "last_decode"

    cfg = {"benchmark": {"type": "sa-bench", "placement": {"node": "dedicated"}}}
    expand_placement(cfg)
    assert cfg["benchmark"]["client_dedicated_node"] is True
    assert cfg["benchmark"]["client_placement"] == "head"


def test_infra_placement() -> None:
    cfg = {"infra": {"placement": {"node": "dedicated"}}}
    expand_placement(cfg)
    assert cfg["infra"]["etcd_nats_dedicated_node"] is True
    assert "placement" not in cfg["infra"]

    cfg = {"infra": {"placement": {"node": "head"}}}
    expand_placement(cfg)
    assert cfg["infra"]["etcd_nats_dedicated_node"] is False


def test_placement_and_legacy_load_identically() -> None:
    base = {
        "name": "placement",
        "model": {"path": "/m", "container": "/c.sqsh", "precision": "fp8"},
        "resources": {
            "gpu_type": "h100",
            "gpus_per_node": 8,
            "prefill_nodes": 1,
            "prefill_workers": 1,
            "decode_nodes": 1,
            "decode_workers": 1,
        },
        "backend": {"type": "sglang"},
        "benchmark": {"type": "sa-bench", "isl": 128, "osl": 128, "concurrencies": "4"},
    }
    legacy = {
        **base,
        "frontend": {"type": "dynamo", "orchestrator_placement": "first_decode"},
        "infra": {"etcd_nats_dedicated_node": True},
    }
    placed = {
        **base,
        "frontend": {"type": "dynamo", "placement": {"node": "first_decode"}},
        "infra": {"placement": {"node": "dedicated"}},
    }

    schema = SrtConfig.Schema()
    a = schema.dump(schema.load(expand_placement(dict(placed))))
    b = schema.dump(schema.load(legacy))
    assert a == b


def test_mixing_placement_with_legacy_fields_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        expand_placement({"frontend": {"placement": {"node": "head"}, "orchestrator_placement": "head"}})
    with pytest.raises(ValueError, match="cannot be combined"):
        expand_placement({"infra": {"placement": {"node": "head"}, "etcd_nats_dedicated_node": False}})


def test_invalid_placement_values_rejected() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        expand_placement({"frontend": {"placement": {"node": "head", "extra": 1}}})
    with pytest.raises(ValueError, match="requires a 'node'"):
        expand_placement({"frontend": {"placement": {}}})
    with pytest.raises(ValueError, match="must be 'head' or 'dedicated'"):
        expand_placement({"infra": {"placement": {"node": "first_decode"}}})


def test_no_placement_is_a_no_op() -> None:
    cfg = {"frontend": {"type": "dynamo", "orchestrator_placement": "head"}}
    assert expand_placement(dict(cfg)) == cfg
