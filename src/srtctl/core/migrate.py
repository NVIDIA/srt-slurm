# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Upgrade recipe YAML to the current schema version, preserving comments.

``srtctl migrate -f recipe.yaml`` rewrites a plain recipe, an override file
(``base`` plus ``override_*`` / ``zip_override_*`` variants), a sweep file, or a
lockfile so that it declares ``schema: <current>`` and uses the current layout.
The transformation is done on a ruamel round-trip document, so comments, key
order, and quoting survive; keys that move between blocks carry their comments.

This module is the only part of srtctl that still reads the pre-2.0 (v1)
layout: the loader rejects it (``srtctl.core.config.require_current_schema``),
so a v1 recipe has to pass through here first. The key-by-key mapping is
documented in ``docs/legacy-v1.md``.

v1 -> v2 rewrites (each is a pure re-spelling of the same resolved config):

- ``resources.<role>_nodes/_workers``, ``gpus_per_<role>``, ``<role>_critical``,
  ``backend.<mode>_environment``, ``backend.<engine>_config.<mode>``, and
  ``backend.<mode>_extra_args`` fold into ``roles.<role>``.
- ``frontend.orchestrator_placement`` / ``dedicated_node``,
  ``benchmark.client_placement`` / ``client_dedicated_node``, and
  ``infra.etcd_nats_dedicated_node`` fold into ``placement.node``.
- ``dynamo.hash`` / ``cargo_patches`` / ``wheel`` / ``version`` fold into
  ``dynamo.source``. ``top_of_tree`` has no immutable equivalent and is left.
- ``benchmark`` fields the recipe's benchmark type never reads are removed
  (schema 2 rejects them; they were silent no-ops).
- schema 1 ``frontend.type: sglang`` (the router) becomes ``sglang-router``;
  in schema 2 ``sglang`` is the router-free single worker.
- ``services[mooncake-master].options.store_config`` (where ``backend.mooncake_kv_store.store_config``
  lands) moves into ``roles.<role>.mooncake_store_config`` of each role whose KV connector
  uses Mooncake, under that role's own keys. An override or sweep file keeps it in place.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marshmallow import ValidationError
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import CommentMark
from ruamel.yaml.tokens import CommentToken

from srtctl.core.roles import COLOCATE, ENGINE_CONFIG_KEY, ROLE_NAMES, ROLE_TO_MODE
from srtctl.core.schema import CURRENT_SCHEMA_VERSION, BackendConfigField, RoleConfig
from srtctl.core.yaml_utils import dump_yaml_with_comments, load_yaml_text_with_comments

_VARIANT_PREFIXES = ("override_", "zip_override_")
_ENGINE_KEYS = frozenset(ENGINE_CONFIG_KEY.values())

# Every schema version the migrator reads. The loader accepts only the current one.
MIGRATABLE_SCHEMA_VERSIONS: tuple[int, ...] = (1, CURRENT_SCHEMA_VERSION)

# Renamed frontend types: {schema-1 value: schema-2 value}. In schema 1 recipes
# ``frontend.type: sglang`` was the SGLang Model Gateway; in 2.0 that router is
# ``sglang-router`` and ``sglang`` is the router-free single worker.
SCHEMA1_FRONTEND_RENAMES: dict[str, str] = {"sglang": "sglang-router"}


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of migrating one recipe document."""

    text: str
    changed: bool
    from_version: int
    to_version: int
    notes: tuple[str, ...]


# --- ruamel helpers -------------------------------------------------------------------


def _move(src: CommentedMap, key: str, dst: CommentedMap, new_key: str) -> None:
    """Move ``src[key]`` to ``dst[new_key]``, carrying the key's comment tokens along."""
    value = src.pop(key)
    dst[new_key] = value
    if key in src.ca.items:
        dst.ca.items[new_key] = src.ca.items.pop(key)


def _insert_after(mapping: CommentedMap, anchor: str, key: str, value: Any) -> None:
    """Insert ``key`` right after ``anchor`` (or append when the anchor is absent)."""
    keys = list(mapping.keys())
    if anchor in keys:
        mapping.insert(keys.index(anchor) + 1, key, value)
    else:
        mapping[key] = value


def _child_map(parent: CommentedMap, key: str, *, after: str | None = None) -> CommentedMap:
    """``parent[key]`` as a mapping, created (after ``after`` when given) if missing."""
    existing = parent.get(key)
    if isinstance(existing, CommentedMap):
        return existing
    created = CommentedMap()
    keys = list(parent.keys())
    if after is not None and after in keys:
        parent.insert(keys.index(after) + 1, key, created)
    else:
        parent[key] = created
    return created


def _services_list(variant: CommentedMap, *, near: str | None = None) -> CommentedSeq:
    """``variant["services"]`` as a list, created (right after ``near`` when given) if missing."""
    existing = variant.get("services")
    if isinstance(existing, CommentedSeq):
        return existing
    created = CommentedSeq()
    keys = list(variant.keys())
    if near is not None and near in keys:
        variant.insert(keys.index(near) + 1, "services", created)
    else:
        variant["services"] = created
    return created


def _service_entry(services: CommentedSeq, name: str) -> CommentedMap:
    """The service named ``name`` (its ``type`` is the name for the built-in kinds), appended if missing."""
    for entry in services:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    entry = CommentedMap()
    entry["name"] = name
    entry["type"] = name
    services.append(entry)
    return entry


def _drop_if_empty(parent: CommentedMap, key: str) -> None:
    value = parent.get(key)
    if isinstance(value, dict) and not value:
        parent.pop(key)
        parent.ca.items.pop(key, None)


def _neutralize_if_empty(parent: CommentedMap, key: str) -> bool:
    """Keep an emptied block as ``key: {}`` (comments cleared so ruamel emits valid flow style).

    The key stays because override variants may refer to it: a zip list with a
    ``null`` deletes a base key only when the block exists in the base.
    """
    value = parent.get(key)
    if isinstance(value, CommentedMap) and not value:
        value.ca.comment = None
        value.ca.items.clear()
        value.ca.end = None
        return True
    return False


_ENGINE_FOR_KEY = {config_key: engine for engine, config_key in ENGINE_CONFIG_KEY.items()}


def _declared_version(doc: CommentedMap) -> int:
    raw = doc.get("schema", 1)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"schema must be an integer version, got {raw!r}")
    if raw not in MIGRATABLE_SCHEMA_VERSIONS:
        raise ValueError(f"schema {raw} is not supported; known versions: {list(MIGRATABLE_SCHEMA_VERSIONS)}")
    return raw


def _variants(doc: CommentedMap) -> Iterator[tuple[str, CommentedMap]]:
    """The recipe mappings a document holds: the document itself, or base plus every override variant."""
    if "base" in doc:
        for key, value in doc.items():
            if (key == "base" or str(key).startswith(_VARIANT_PREFIXES)) and isinstance(value, CommentedMap):
                yield str(key), value
    else:
        yield "", doc


# --- v1 -> v2 transforms --------------------------------------------------------------


def _engine_key_for(variant: CommentedMap, base: CommentedMap) -> str:
    """The ``backend.<engine>_config`` key from the variant's or base's ``backend.type``, else from what is present."""
    for source in (variant, base):
        backend = source.get("backend")
        if isinstance(backend, dict) and backend.get("type"):
            return ENGINE_CONFIG_KEY.get(str(backend["type"]), "sglang_config")
    backend = variant.get("backend")
    if isinstance(backend, dict):
        for key in _ENGINE_KEYS:
            if key in backend:
                return key
    return "sglang_config"


def _fold_roles(variant: CommentedMap, engine_key: str, label: str) -> list[str]:
    notes: list[str] = []
    resources = variant.get("resources")
    backend = variant.get("backend")
    resources = resources if isinstance(resources, CommentedMap) else None
    backend = backend if isinstance(backend, CommentedMap) else None
    engine_cfg = backend.get(engine_key) if backend is not None else None
    engine_cfg = engine_cfg if isinstance(engine_cfg, CommentedMap) else None

    roles: CommentedMap | None = None
    for role in ROLE_NAMES:
        mode = ROLE_TO_MODE[role]
        moves: list[tuple[CommentedMap, str, str]] = []
        if resources is not None:
            for legacy, new in (
                (f"{role}_nodes", "nodes"),
                (f"{role}_workers", "workers"),
                (f"gpus_per_{role}", "gpus"),
                (f"{role}_critical", "critical"),
            ):
                if legacy in resources:
                    moves.append((resources, legacy, new))
        if backend is not None:
            if f"{mode}_environment" in backend:
                moves.append((backend, f"{mode}_environment", "env"))
            if f"{mode}_extra_args" in backend:
                moves.append((backend, f"{mode}_extra_args", "extra_args"))
        if engine_cfg is not None and mode in engine_cfg:
            moves.append((engine_cfg, mode, "args"))
        if not moves:
            continue
        if roles is None:
            anchor = "backend" if "backend" in variant else ("resources" if "resources" in variant else None)
            roles = _child_map(variant, "roles", after=anchor)
        spec = _child_map(roles, role)
        for src, legacy, new in moves:
            if new in spec:
                notes.append(f"{label}roles.{role}.{new} already set; kept it and dropped legacy {legacy}")
                src.pop(legacy)
                src.ca.items.pop(legacy, None)
                continue
            _move(src, legacy, spec, new)
        if role == "decode" and spec.get("nodes") == 0:
            # roles: spells shared-node decode as `nodes: colocate`; the bare v1 sentinel is rejected there,
            # and a colocated split must state gpus on both roles. Materialize what v1 derived implicitly:
            # prefill = prefill_nodes * gpus_per_node // prefill_workers, decode inherits prefill.
            spec["nodes"] = COLOCATE
            notes.append(f"{label}rewrote roles.decode.nodes: 0 as nodes: colocate")
            prefill_spec = roles.get("prefill") if roles is not None else None
            gpus_per_node = resources.get("gpus_per_node") if resources is not None else None
            if isinstance(prefill_spec, CommentedMap) and isinstance(gpus_per_node, int):
                if "gpus" not in prefill_spec:
                    prefill_nodes, prefill_workers = prefill_spec.get("nodes"), prefill_spec.get("workers")
                    if isinstance(prefill_nodes, int) and isinstance(prefill_workers, int) and prefill_workers > 0:
                        _insert_after(
                            prefill_spec, "workers", "gpus", (prefill_nodes * gpus_per_node) // prefill_workers
                        )
                        notes.append(f"{label}materialized roles.prefill.gpus: {prefill_spec['gpus']} (v1 derived it)")
                if "gpus" not in spec and "gpus" in prefill_spec:
                    _insert_after(spec, "workers", "gpus", prefill_spec["gpus"])
                    notes.append(f"{label}materialized roles.decode.gpus: {spec['gpus']} (v1 inherited prefill)")
            elif "gpus" not in spec or not (isinstance(prefill_spec, CommentedMap) and "gpus" in prefill_spec):
                notes.append(
                    f"{label}roles.decode.nodes: colocate needs gpus: on prefill and decode; "
                    "set them by hand (gpus_per_node is not in this recipe)"
                )
        notes.append(f"{label}folded {role} fields into roles.{role}")

    if backend is not None:
        _drop_if_empty(backend, engine_key)
    return notes


def _fold_placement_block(section: CommentedMap, place_key: str, dedicated_key: str, label: str) -> list[str]:
    if place_key not in section and dedicated_key not in section:
        return []
    dedicated = bool(section.pop(dedicated_key, False))
    section.ca.items.pop(dedicated_key, None)
    location = section.pop(place_key, "head")
    section.ca.items.pop(place_key, None)
    placement = _child_map(section, "placement")
    placement["node"] = "dedicated" if dedicated else location
    return [f"{label}placement.node: {placement['node']}"]


def _fold_placement(variant: CommentedMap, label: str) -> list[str]:
    notes: list[str] = []
    frontend = variant.get("frontend")
    if isinstance(frontend, CommentedMap):
        notes += _fold_placement_block(frontend, "orchestrator_placement", "dedicated_node", f"{label}frontend.")
    benchmark = variant.get("benchmark")
    if isinstance(benchmark, CommentedMap):
        notes += _fold_placement_block(benchmark, "client_placement", "client_dedicated_node", f"{label}benchmark.")
    return notes


def _uses_discovery_plane(variant: CommentedMap, base: CommentedMap) -> bool:
    """Whether the frontend (variant's, else base's, default dynamo) runs on etcd and NATS."""
    for source in (variant, base):
        frontend = source.get("frontend")
        if isinstance(frontend, dict) and frontend.get("type") is not None:
            return str(frontend["type"]) == "dynamo"
    return True


def _wants_nats(variant: CommentedMap, base: CommentedMap, infra: Any) -> bool:
    """Whether the migrated recipe should declare a ``nats`` service at all.

    Mirrors :func:`srtctl.services.implicit.nats_implied_reasons`: only a NATS request or
    event plane, or a ``nats_max_payload_mb`` knob, means NATS was in use. Everything else
    ran NATS for nothing under v1 and gets etcd alone in 2.0.
    """
    for source in (variant, base):
        dynamo = source.get("dynamo") if isinstance(source, dict) else None
        if isinstance(dynamo, dict) and (dynamo.get("request_plane") == "nats" or dynamo.get("event_plane") == "nats"):
            return True
    return isinstance(infra, dict) and infra.get("nats_max_payload_mb") is not None


def _fold_infra_services(variant: CommentedMap, base: CommentedMap, label: str, *, is_override: bool) -> list[str]:
    """``infra:`` -> ``etcd`` and ``nats`` services.

    ``etcd_nats_dedicated_node: true`` becomes ``placement.node: dedicated`` on both;
    ``nats_max_payload_mb`` becomes ``options.max_payload_mb`` on nats. A ``false``
    is the default and is dropped, except in an override variant, where it has to
    undo a base ``true`` and so is spelled out as ``placement.node: infra``. An
    ``infra: null`` override (delete the base block) is spelled out the same way.

    Under a static frontend (no discovery plane) declaring the services would
    launch etcd and NATS that nothing uses, so the block is not folded: the
    payload knob is dropped (it had no effect) and ``etcd_nats_dedicated_node``
    stays in its v1 spelling, since it still reserves a node.
    """
    if "infra" not in variant:
        return []
    notes: list[str] = []
    infra = variant.get("infra")
    if not _uses_discovery_plane(variant, base):
        if isinstance(infra, CommentedMap):
            if "nats_max_payload_mb" in infra:
                infra.pop("nats_max_payload_mb")
                infra.ca.items.pop("nats_max_payload_mb", None)
                notes.append(f"{label}dropped infra.nats_max_payload_mb (no discovery plane under this frontend)")
            if infra.get("etcd_nats_dedicated_node"):
                notes.append(
                    f"{label}infra.etcd_nats_dedicated_node left as is: it reserves a node although this "
                    "frontend runs no etcd/NATS; drop it to give the node back to the workers"
                )
            elif "etcd_nats_dedicated_node" in infra:
                infra.pop("etcd_nats_dedicated_node")
                infra.ca.items.pop("etcd_nats_dedicated_node", None)
            if not infra:
                variant.pop("infra")
                variant.ca.items.pop("infra", None)
        return notes
    if infra is None:
        if is_override:
            services = _services_list(variant, near="infra")
            names = ("etcd", "nats") if _wants_nats(variant, base, base.get("infra")) else ("etcd",)
            for name in names:
                _child_map(_service_entry(services, name), "placement")["node"] = "infra"
            notes.append(f"{label}infra: null -> services {'/'.join(names)} placement.node: infra")
        variant.pop("infra")
        variant.ca.items.pop("infra", None)
        return notes
    if not isinstance(infra, CommentedMap):
        return notes

    dedicated: bool | None = None
    if "etcd_nats_dedicated_node" in infra:
        dedicated = bool(infra.pop("etcd_nats_dedicated_node"))
        infra.ca.items.pop("etcd_nats_dedicated_node", None)
    placement = infra.pop("placement", None)  # an earlier 2.0 draft spelled it infra.placement.node
    if isinstance(placement, dict):
        infra.ca.items.pop("placement", None)
        dedicated = placement.get("node") == "dedicated"
    payload = None
    if "nats_max_payload_mb" in infra:
        payload = infra.pop("nats_max_payload_mb")
        infra.ca.items.pop("nats_max_payload_mb", None)

    if dedicated or payload is not None or (dedicated is False and is_override):
        services = _services_list(variant, near="infra")
        wants_nats = payload is not None or _wants_nats(variant, base, {"nats_max_payload_mb": payload})
        names = ("etcd", "nats") if wants_nats else ("etcd",)
        if dedicated is not None:
            node = "dedicated" if dedicated else "infra"
            for name in names:
                _child_map(_service_entry(services, name), "placement")["node"] = node
            notes.append(f"{label}infra.etcd_nats_dedicated_node -> services {'/'.join(names)} placement.node: {node}")
        if payload is not None:
            _child_map(_service_entry(services, "nats"), "options")["max_payload_mb"] = payload
            notes.append(f"{label}infra.nats_max_payload_mb -> services nats options.max_payload_mb")
    elif dedicated is False:
        notes.append(f"{label}dropped infra.etcd_nats_dedicated_node: false (the default)")
    if not infra:
        variant.pop("infra")
        variant.ca.items.pop("infra", None)
    return notes


def _fold_mooncake(variant: CommentedMap, label: str) -> list[str]:
    """``backend.mooncake_kv_store`` -> a ``mooncake-master`` service plus ``roles.*.env``.

    ``container`` and ``master_extra_args`` (as ``args``) and ``store_config`` (as
    ``options.store_config``) describe the master; ``env`` was injected into every
    worker and lands in each role's ``env`` (Mooncake values win, as they did at
    launch). Runs after ``_fold_roles`` so the roles exist, before ``_fold_engine``.
    """
    backend = variant.get("backend")
    if not isinstance(backend, CommentedMap):
        return []
    store = backend.get("mooncake_kv_store")
    if not isinstance(store, CommentedMap):
        return []
    roles = variant.get("roles")
    role_specs = [v for v in roles.values() if isinstance(v, CommentedMap)] if isinstance(roles, CommentedMap) else []
    env = store.get("env")
    if env and not role_specs:
        return [f"{label}backend.mooncake_kv_store left as is (no roles in this variant to carry its env)"]
    notes: list[str] = []
    services = _services_list(variant, near="roles" if "roles" in variant else None)
    entry = _service_entry(services, "mooncake-master")
    if "container" in store:
        _move(store, "container", entry, "container")
    if "master_extra_args" in store:
        _move(store, "master_extra_args", entry, "args")
    if "store_config" in store:
        _move(store, "store_config", _child_map(entry, "options"), "store_config")
    if "device_names_by_gpu" in store:
        _move(store, "device_names_by_gpu", _child_map(entry, "options"), "device_names_by_gpu")
    if "env" in store:
        moved = store.pop("env")
        store.ca.items.pop("env", None)
        if moved:
            for spec in role_specs:
                role_env = _child_map(spec, "env")
                for key, value in moved.items():
                    role_env[key] = value
            notes.append(f"{label}backend.mooncake_kv_store.env -> roles.*.env")
    if not store:
        backend.pop("mooncake_kv_store")
        backend.ca.items.pop("mooncake_kv_store", None)
    notes.append(f"{label}backend.mooncake_kv_store -> services mooncake-master")
    return notes


def _mooncake_store_modes(engine: Any, role_specs: dict[str, CommentedMap]) -> tuple[str, ...] | None:
    """The roles whose workers read a Mooncake client config, as the engine resolves their connectors.

    None when the engine cannot be loaded from this document (a sweep placeholder, say).
    """
    if engine is None:
        return ()  # the default engine, SGLang, reads no client config file
    try:
        backend = BackendConfigField().deserialize(engine)
        roles = {name: RoleConfig(args=dict(spec.get("args") or {})) for name, spec in role_specs.items()}
        return dataclasses.replace(backend, roles=roles).mooncake_store_modes()
    except (ValidationError, ValueError, TypeError):
        return None


def _split_comment(token: CommentToken | None) -> tuple[CommentToken | None, str]:
    """A key's end-of-line comment alone, and the comment lines after it, which belong to what follows."""
    if token is None:
        return None, ""
    line, _, trailing = token.value.partition("\n")
    return (CommentToken(line + "\n", token.start_mark, None) if line.strip() else None), trailing


def _keep_trailing_comment(mapping: CommentedMap, key: Any, trailing: str) -> None:
    """Emit ``trailing`` comment lines right after ``mapping[key]``, where a removed block left them."""
    if not trailing:
        return
    slot = mapping.ca.items.setdefault(key, [None, None, None, None])
    if slot[2] is None:
        slot[2] = CommentToken("\n" + trailing, CommentMark(0), None)
    else:
        slot[2].value += trailing


def _fold_mooncake_store_config(variant: CommentedMap, label: str, *, varies: str | None) -> list[str]:
    """``services[mooncake-master].options.store_config`` -> ``roles.<role>.mooncake_store_config``.

    Those keys were the client config every role's workers read. They move to each role
    whose KV connector uses Mooncake, under the role's own keys, which won at launch
    too; other roles never read them. The engine names those roles, so connectors resolve
    as they do at launch. Runs after ``_fold_engine``. ``varies`` names a document whose
    variants or sweep points can change a role's connector (an override or sweep file);
    there the option is left in place, as is ``engine.mooncake_kv_store.store_config``.
    """
    services = variant.get("services")
    masters = (
        [
            s
            for s in services
            if isinstance(s, CommentedMap) and s.get("type") == "mooncake-master" and s.get("enabled", True)
        ]
        if isinstance(services, CommentedSeq)
        else []
    )
    engine = variant.get("engine")
    engine_store = engine.get("mooncake_kv_store") if isinstance(engine, CommentedMap) else None
    if not masters and isinstance(engine_store, CommentedMap) and engine_store.get("store_config"):
        return [
            f"{label}engine.mooncake_kv_store.store_config left as is; move it into each Mooncake role's mooncake_store_config"
        ]
    if len(masters) != 1:
        return []
    entry = masters[0]
    options = entry.get("options")
    store_config = options.get("store_config") if isinstance(options, CommentedMap) else None
    if not isinstance(options, CommentedMap) or not isinstance(store_config, CommentedMap) or not store_config:
        return []
    where = f"{label}services mooncake-master options.store_config"
    if varies:
        return [f"{where} left as is ({varies}); move it into each Mooncake role's mooncake_store_config"]
    roles = variant.get("roles")
    role_specs = (
        {k: v for k, v in roles.items() if isinstance(v, CommentedMap)} if isinstance(roles, CommentedMap) else {}
    )
    modes = _mooncake_store_modes(engine, role_specs)
    if modes is None:
        return [f"{where} left as is (the engine could not be loaded to resolve the roles' connectors)"]
    if not modes or any(mode not in role_specs for mode in modes):
        return [f"{where} left as is (no role's KV connector resolves to Mooncake)"]
    keys = list(store_config)
    comments = {key: _split_comment((store_config.ca.items.get(key) or [None] * 4)[2])[0] for key in keys}
    for mode in modes:
        spec = role_specs[mode]
        own = spec.get("mooncake_store_config")
        if not isinstance(own, CommentedMap):
            own = CommentedMap()
            _insert_after(spec, "args" if "args" in spec else "env", "mooncake_store_config", own)
        # The role's keys won over the shared ones at launch; keep them and put the rest first.
        for position, key in enumerate(k for k in keys if k not in own):
            own.insert(position, key, copy.deepcopy(store_config[key]))
            comment = comments[key]
            if comment is not None:
                own.ca.items[key] = [None, None, CommentToken(comment.value, comment.start_mark, None), None]
    # Comment lines after the block (a section banner, say) belong to what follows it: keep them there.
    _, trailing = _split_comment((store_config.ca.items.get(keys[-1]) or [None] * 4)[2])
    option_keys = list(options)
    position = option_keys.index("store_config")
    options.pop("store_config")
    options.ca.items.pop("store_config", None)
    if position:
        _keep_trailing_comment(options, option_keys[position - 1], trailing)
    elif options:
        if options.ca.comment:
            options.ca.comment[1] = None  # the lines before store_config described it
        _keep_trailing_comment(entry, "options", trailing)
    else:
        entry_keys = list(entry)
        before = entry_keys[entry_keys.index("options") - 1] if entry_keys.index("options") else None
        _drop_if_empty(entry, "options")
        if before is not None:
            _keep_trailing_comment(entry, before, trailing)
    return [f"{where} -> roles.{'/'.join(modes)}.mooncake_store_config"]


def _fold_dynamo_source(variant: CommentedMap, label: str) -> list[str]:
    dynamo = variant.get("dynamo")
    if not isinstance(dynamo, CommentedMap) or "source" in dynamo:
        return []
    notes: list[str] = []
    if dynamo.get("top_of_tree"):
        notes.append(f"{label}dynamo.top_of_tree left as is (no immutable rev to pin; choose a commit for source.rev)")
        return notes
    has_git = dynamo.get("hash") is not None
    has_wheel = dynamo.get("wheel") is not None
    has_version = dynamo.get("version") is not None
    if not (has_git or has_wheel or has_version):
        return notes
    source = _child_map(dynamo, "source", after="install" if "install" in dynamo else None)
    if has_git:
        _move(dynamo, "hash", source, "rev")
        if "cargo_patches" in dynamo:
            _move(dynamo, "cargo_patches", source, "patches")
        if has_version:  # version is auto-cleared when hash is set; the legacy loader ignored it
            dynamo.pop("version")
            dynamo.ca.items.pop("version", None)
            notes.append(f"{label}dropped dynamo.version (ignored alongside hash)")
        notes.append(f"{label}dynamo.hash -> dynamo.source.rev")
    elif has_wheel:
        _move(dynamo, "wheel", source, "wheel")
        if has_version:
            dynamo.pop("version")
            dynamo.ca.items.pop("version", None)
        notes.append(f"{label}dynamo.wheel -> dynamo.source.wheel")
    else:
        _move(dynamo, "version", source, "pypi")
        notes.append(f"{label}dynamo.version -> dynamo.source.pypi")
    return notes


def _strip_unused_benchmark_fields(variant: CommentedMap, base: CommentedMap, label: str) -> list[str]:
    """Remove benchmark fields the recipe's type never reads (schema 2 rejects them)."""
    benchmark = variant.get("benchmark")
    if not isinstance(benchmark, CommentedMap):
        return []
    btype = benchmark.get("type")
    if btype is None:
        base_benchmark = base.get("benchmark")
        btype = base_benchmark.get("type", "manual") if isinstance(base_benchmark, dict) else "manual"
    try:
        import srtctl.benchmarks  # noqa: F401 - registers runners
        from srtctl.benchmarks.base import benchmark_config_fields, list_benchmarks
    except Exception:  # noqa: BLE001
        return []
    if btype not in {*list_benchmarks(), "manual"}:
        return []  # unknown type: the loader reports it; nothing to strip safely
    accepted = benchmark_config_fields(str(btype)) | {"placement"}
    notes: list[str] = []
    for key in [k for k in benchmark if k not in accepted]:
        benchmark.pop(key)
        benchmark.ca.items.pop(key, None)
        notes.append(f"{label}removed benchmark.{key} (unused by type {btype})")
    return notes


def _migrate_1_to_2(doc: CommentedMap) -> list[str]:
    """Structural v1 -> v2 rewrites, applied to every variant a document holds."""
    notes: list[str] = []
    base = doc.get("base") if isinstance(doc.get("base"), CommentedMap) else doc
    for name, variant in _variants(doc):
        label = f"{name}: " if name else ""
        notes += _fold_roles(variant, _engine_key_for(variant, base), label)
        notes += _fold_placement(variant, label)
        notes += _fold_infra_services(variant, base, label, is_override=name not in ("", "base"))
        notes += _fold_dynamo_source(variant, label)
        notes += _strip_unused_benchmark_fields(variant, base, label)
        notes += _fold_mooncake(variant, label)
        notes += _fold_engine(variant, base, label)
        varies = "an override file" if name else "a sweep file" if "sweep" in doc else None
        notes += _fold_mooncake_store_config(variant, label, varies=varies)
        for key in ("resources", "dynamo", "frontend"):
            _neutralize_if_empty(variant, key)
    return notes


def _fold_engine(variant: CommentedMap, base: CommentedMap, label: str) -> list[str]:
    """``backend:`` -> top-level ``engine:``, ``kv_events_config`` -> ``roles.*.kv_events``, ``dynamo.sidecar`` -> ``roles.*.sidecar``.

    Runs after the per-mode folds, so what is left in ``backend`` is the engine
    type plus engine-wide knobs. ``engine`` is a bare string when only the type
    remains, else a mapping. The key is placed where ``backend`` was.
    """
    notes: list[str] = []
    roles = variant.get("roles")
    role_specs = (
        {k: v for k, v in roles.items() if isinstance(v, CommentedMap)} if isinstance(roles, CommentedMap) else {}
    )
    backend = variant.get("backend")
    if isinstance(backend, CommentedMap):
        engine_type = backend.get("type")
        if engine_type is None and (not backend or "type" not in backend) and not base.get("engine"):
            # Only when this variant had a backend block that is now empty of
            # per-mode fields: make the implicit default explicit so an override
            # variant's `null` can still delete it (deletion needs the base key).
            base_backend = base.get("backend") if isinstance(base.get("backend"), dict) else None
            if not backend and not (base_backend and base_backend.get("type")):
                engine_type = _ENGINE_FOR_KEY.get(_engine_key_for(variant, base), "sglang")
                notes.append(f"{label}engine: {engine_type} (was the implicit default)")

        kv_events = backend.get("kv_events_config")
        if kv_events is not None and role_specs:
            if isinstance(kv_events, dict):
                for role_name in ROLE_NAMES:
                    mode = ROLE_TO_MODE[role_name]
                    if mode in kv_events and role_name in role_specs:
                        role_specs[role_name]["kv_events"] = kv_events[mode]
                        notes.append(f"{label}backend.kv_events_config.{mode} -> roles.{role_name}.kv_events")
                if all(ROLE_TO_MODE[r] not in kv_events or r in role_specs for r in ROLE_NAMES):
                    backend.pop("kv_events_config")
                    backend.ca.items.pop("kv_events_config", None)
            elif kv_events is True:
                base_backend = base.get("backend") if isinstance(base.get("backend"), dict) else {}
                resolved_type = engine_type or base_backend.get("type") or "sglang"
                covered = ("prefill", "decode", "agg") if resolved_type == "sglang" else ("prefill", "decode")
                for role_name in covered:
                    if role_name in role_specs:
                        role_specs[role_name]["kv_events"] = True
                backend.pop("kv_events_config")
                backend.ca.items.pop("kv_events_config", None)
                notes.append(f"{label}backend.kv_events_config: true -> roles.*.kv_events")

        remaining = CommentedMap()
        for key in [k for k in backend if k != "type"]:
            _move(backend, key, remaining, key)
        if engine_type is not None or remaining:
            keys = list(variant.keys())
            position = keys.index("backend")
            if remaining:
                engine_value: Any = remaining
                if engine_type is not None:
                    remaining.insert(0, "type", engine_type)
            else:
                engine_value = engine_type
            variant.insert(position, "engine", engine_value)
            if "backend" in variant.ca.items:
                variant.ca.items["engine"] = variant.ca.items.pop("backend")
            notes.append(f"{label}backend -> engine")
        variant.pop("backend")
        variant.ca.items.pop("backend", None)

    dynamo = variant.get("dynamo")
    if isinstance(dynamo, CommentedMap) and dynamo.get("sidecar") is True and role_specs:
        for spec in role_specs.values():
            spec["sidecar"] = True
        dynamo.pop("sidecar")
        dynamo.ca.items.pop("sidecar", None)
        notes.append(f"{label}dynamo.sidecar -> roles.*.sidecar")
    return notes


# --- entry points -----------------------------------------------------------------------


def _rename_schema1_frontends(doc: CommentedMap) -> list[str]:
    """``frontend.type: sglang`` (schema 1, the router) -> ``sglang-router`` in every variant."""
    notes: list[str] = []
    for label, variant in _variants(doc):
        frontend = variant.get("frontend")
        if not isinstance(frontend, CommentedMap):
            continue
        old_value = frontend.get("type")
        renamed = SCHEMA1_FRONTEND_RENAMES.get(old_value)
        if renamed:
            frontend["type"] = renamed
            prefix = f"{label}: " if label else ""
            notes.append(f"{prefix}frontend.type: {old_value} -> {renamed} (the router)")
    return notes


def migrate_recipe_text(text: str) -> MigrationResult:
    """Migrate one YAML document (plain, override, sweep, or lock format) to the current schema."""
    doc = load_yaml_text_with_comments(text)
    from_version = _declared_version(doc)
    notes: list[str] = []

    # Renamed values are NOT re-spellings: in schema 1 `frontend.type: sglang`
    # was the router, in schema 2 it is the router-free worker. Only a schema 1
    # document gets the rename.
    if from_version < 2:
        notes.extend(_rename_schema1_frontends(doc))

    # The v2 layout folds are pure re-spellings, so they apply to a schema: 2
    # document that still uses the legacy layout as well (a no-op once folded).
    if from_version <= 2:
        notes.extend(_migrate_1_to_2(doc))

    if doc.get("schema") != CURRENT_SCHEMA_VERSION:
        if "schema" in doc:
            doc["schema"] = CURRENT_SCHEMA_VERSION
        else:
            doc.insert(0, "schema", CURRENT_SCHEMA_VERSION)
        notes.append(f"set schema: {CURRENT_SCHEMA_VERSION}")

    migrated = dump_yaml_with_comments(doc) or ""
    return MigrationResult(
        text=migrated,
        changed=migrated != text,
        from_version=from_version,
        to_version=CURRENT_SCHEMA_VERSION,
        notes=tuple(notes),
    )


def migrate_recipe_file(path: Path, *, in_place: bool = False, output: Path | None = None) -> MigrationResult:
    """Migrate a recipe file. Writes back when ``in_place`` or to ``output`` when given."""
    if in_place and output is not None:
        raise ValueError("choose either --in-place or --output, not both")
    result = migrate_recipe_text(path.read_text(encoding="utf-8"))
    if in_place:
        if result.changed:
            path.write_text(result.text, encoding="utf-8")
    elif output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.text, encoding="utf-8")
    return result


def recipe_files(paths: Iterable[Path]) -> list[Path]:
    """Expand files and directories (recursively, ``*.yaml`` / ``*.yml``) into a sorted list of recipe files."""
    found: set[Path] = set()
    for path in paths:
        if path.is_dir():
            found.update(p for p in path.rglob("*") if p.suffix in {".yaml", ".yml"} and p.is_file())
        else:
            found.add(path)
    return sorted(found)
