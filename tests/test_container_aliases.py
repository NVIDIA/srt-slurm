# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single container-alias resolver behind resolve_config_with_defaults."""

from __future__ import annotations

from srtctl.core.config import CONTAINER_ALIAS_KEYS, resolve_config_with_defaults, resolve_container_aliases

CONTAINERS = {
    "sglang": "/sqsh/sglang.sqsh",
    "nginx": "/sqsh/nginx.sqsh",
    "router": "/sqsh/router.sqsh",
    "evals": "/sqsh/evals.sqsh",
    "dcgm-exporter": "/sqsh/dcgm.sqsh",
    "node-exporter": "/sqsh/node.sqsh",
    "mooncake": "/sqsh/mooncake.sqsh",
    "sidecar": "/sqsh/sidecar.sqsh",
}


def _recipe() -> dict:
    return {
        "name": "aliases",
        "model": {"path": "/models/m", "container": "sglang", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1},
        "frontend": {"nginx_container": "nginx", "container_image": "router"},
        "benchmark": {"type": "custom", "command": "echo", "container_image": "evals"},
        "observability": {
            "enabled": True,
            "tachometer": {
                "dcgm_exporter": {"container_image": "dcgm-exporter", "port": 9401},
                "node_exporter": {"container_image": "node-exporter", "port": 9101},
            },
        },
        "telemetry": {"enabled": True, "dcgm_exporter": {"container_image": "dcgm-exporter", "port": 9401}},
        "backend": {"type": "sglang", "mooncake_kv_store": {"container": "mooncake", "env": {"image": "sglang"}}},
    }


def test_every_known_container_key_resolves_in_one_pass() -> None:
    resolved = resolve_config_with_defaults(_recipe(), {"containers": CONTAINERS})

    assert resolved["model"]["container"] == "/sqsh/sglang.sqsh"
    assert resolved["frontend"]["nginx_container"] == "/sqsh/nginx.sqsh"
    assert resolved["frontend"]["container_image"] == "/sqsh/router.sqsh"
    assert resolved["benchmark"]["container_image"] == "/sqsh/evals.sqsh"
    assert resolved["observability"]["tachometer"]["dcgm_exporter"]["container_image"] == "/sqsh/dcgm.sqsh"
    assert resolved["observability"]["tachometer"]["node_exporter"]["container_image"] == "/sqsh/node.sqsh"
    assert resolved["telemetry"]["dcgm_exporter"]["container_image"] == "/sqsh/dcgm.sqsh"
    # Newly covered: the Mooncake master container used to be the one image key no block resolved.
    assert resolved["backend"]["mooncake_kv_store"]["container"] == "/sqsh/mooncake.sqsh"


def test_free_form_maps_and_identity_are_never_touched() -> None:
    recipe = _recipe()
    recipe["identity"] = {"container": {"image": "sglang"}}
    recipe["environment"] = {"image": "sglang", "container": "nginx"}
    recipe["backend"]["aggregated_environment"] = {"container_image": "sglang"}
    recipe["backend"]["sglang_config"] = {"aggregated": {"image": "sglang"}}

    resolved = resolve_config_with_defaults(recipe, {"containers": CONTAINERS})

    assert resolved["identity"] == {"container": {"image": "sglang"}}
    assert resolved["environment"] == {"image": "sglang", "container": "nginx"}
    assert resolved["backend"]["aggregated_environment"] == {"container_image": "sglang"}
    assert resolved["backend"]["sglang_config"] == {"aggregated": {"image": "sglang"}}
    assert resolved["backend"]["mooncake_kv_store"]["env"] == {"image": "sglang"}


def test_literal_paths_registry_uris_and_unknown_aliases_pass_through() -> None:
    recipe = _recipe()
    recipe["model"]["container"] = "/direct/container.sqsh"
    recipe["frontend"]["nginx_container"] = "nginx:1.27.4"
    recipe["benchmark"]["container_image"] = "not-an-alias"

    resolved = resolve_config_with_defaults(recipe, {"containers": CONTAINERS})

    assert resolved["model"]["container"] == "/direct/container.sqsh"
    assert resolved["frontend"]["nginx_container"] == "nginx:1.27.4"
    assert resolved["benchmark"]["container_image"] == "not-an-alias"


def test_new_blocks_with_image_or_container_image_keys_resolve_without_resolver_code() -> None:
    """A future `services:` list (or anything else) that names an image just works."""
    recipe = _recipe()
    recipe["services"] = [
        {"name": "a", "image": "sidecar", "command": ["x"]},
        {"name": "b", "container_image": "router", "command": ["y"], "env": {"image": "sidecar"}},
    ]

    resolved = resolve_config_with_defaults(recipe, {"containers": CONTAINERS})

    assert resolved["services"][0]["image"] == "/sqsh/sidecar.sqsh"
    assert resolved["services"][1]["container_image"] == "/sqsh/router.sqsh"
    assert resolved["services"][1]["env"] == {"image": "sidecar"}


def test_walker_reports_each_resolution_and_mutates_in_place() -> None:
    recipe = _recipe()
    notes = resolve_container_aliases(recipe, CONTAINERS)

    assert recipe["model"]["container"] == "/sqsh/sglang.sqsh"
    assert "Resolved container alias model.container: 'sglang' -> '/sqsh/sglang.sqsh'" in notes
    assert any(note.startswith("Resolved container alias frontend.nginx_container:") for note in notes)
    assert len(notes) == 8
    assert resolve_container_aliases(recipe, CONTAINERS) == [], "second pass finds nothing left to resolve"


def test_no_cluster_containers_map_leaves_aliases_as_written() -> None:
    resolved = resolve_config_with_defaults(_recipe(), {"default_account": "acct"})
    assert resolved["model"]["container"] == "sglang"
    assert resolve_config_with_defaults(_recipe(), None)["frontend"]["nginx_container"] == "nginx"


def test_alias_key_set_is_the_documented_one() -> None:
    assert set(CONTAINER_ALIAS_KEYS) == {"container", "container_image", "image", "nginx_container"}
