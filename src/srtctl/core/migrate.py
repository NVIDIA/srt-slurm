# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Upgrade recipe YAML to the current schema version, preserving comments.

``srtctl migrate -f recipe.yaml`` rewrites a plain recipe, an override file
(``base`` plus ``override_*`` / ``zip_override_*`` variants), a sweep file, or a
lockfile so that it declares ``schema: <current>`` and uses the current layout.
The transformation is done on a ruamel round-trip document, so comments, key
order, and quoting survive; keys that move between blocks carry their comments.

v1 -> v2 rewrites (each is a pure re-spelling; the resolved config is identical):

- ``resources.<role>_nodes/_workers``, ``gpus_per_<role>``,
  ``backend.<mode>_environment``, ``backend.<engine>_config.<mode>``, and
  ``backend.<mode>_extra_args`` fold into ``roles.<role>``.
- ``frontend.orchestrator_placement`` / ``dedicated_node``,
  ``benchmark.client_placement`` / ``client_dedicated_node``, and
  ``infra.etcd_nats_dedicated_node`` fold into ``placement.node``.
- ``dynamo.hash`` / ``cargo_patches`` / ``wheel`` / ``version`` fold into
  ``dynamo.source``. ``top_of_tree`` has no immutable equivalent and is left.
- ``benchmark`` fields the recipe's benchmark type never reads are removed
  (schema 2 rejects them; they were silent no-ops).

``srtctl migrate --verify`` proves the equivalence: it migrates in memory,
resolves both documents through the same loader, and compares the results.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ruamel.yaml.comments import CommentedMap, CommentedSeq

from srtctl.core.roles import ENGINE_CONFIG_KEY, ROLE_NAMES, ROLE_TO_MODE
from srtctl.core.schema import CURRENT_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS
from srtctl.core.yaml_utils import dump_yaml_with_comments, load_yaml_text_with_comments

_VARIANT_PREFIXES = ("override_", "zip_override_")
_ENGINE_KEYS = frozenset(ENGINE_CONFIG_KEY.values())


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
    if raw not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"schema {raw} is not supported; known versions: {list(SUPPORTED_SCHEMA_VERSIONS)}")
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
            for name in ("etcd", "nats"):
                _child_map(_service_entry(services, name), "placement")["node"] = "infra"
            notes.append(f"{label}infra: null -> services etcd/nats placement.node: infra")
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
        if dedicated is not None:
            node = "dedicated" if dedicated else "infra"
            for name in ("etcd", "nats"):
                _child_map(_service_entry(services, name), "placement")["node"] = node
            notes.append(f"{label}infra.etcd_nats_dedicated_node -> services etcd/nats placement.node: {node}")
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


def migrate_recipe_text(text: str) -> MigrationResult:
    """Migrate one YAML document (plain, override, sweep, or lock format) to the current schema."""
    doc = load_yaml_text_with_comments(text)
    from_version = _declared_version(doc)
    notes: list[str] = []

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


# --- golden equality ----------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyResult:
    """Golden-equality outcome for one recipe file."""

    path: Path
    status: str  # ok | mismatch | skipped | error
    detail: str = ""
    variants: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "skipped"}


def _expand(raw: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every concrete recipe a raw document produces: plain, override variants, or sweep points."""
    from srtctl.core.config import generate_override_configs

    if "base" in raw:
        return generate_override_configs(raw)
    if "sweep" in raw:
        from srtctl.core.sweep import generate_sweep_configs

        return [(str(params), cfg) for cfg, params in generate_sweep_configs(copy.deepcopy(raw))]
    return [("", raw)]


def _resolved_dump(raw: dict[str, Any]) -> dict[str, Any]:
    """Resolve and load a raw recipe exactly as the loader does, then dump it for comparison."""
    from srtctl.core.config import resolve_config_with_defaults
    from srtctl.core.schema import SrtConfig

    schema = SrtConfig.Schema()
    loaded = schema.load(resolve_config_with_defaults(raw, None))
    dumped = schema.dump(loaded)
    dumped.pop("schema", None)
    # kv_events_config is compared by effect, not spelling: `true` and a per-mode
    # map that enables the same modes resolve to the same worker flags. Only modes
    # with workers matter.
    backend = dumped.get("backend")
    if (
        isinstance(backend, dict)
        and "kv_events_config" in backend
        and hasattr(loaded.backend, "get_kv_events_config_for_mode")
    ):
        active = {
            "prefill": loaded.resources.num_prefill,
            "decode": loaded.resources.num_decode,
            "agg": loaded.resources.num_agg,
        }
        backend["kv_events_config"] = {
            mode: loaded.backend.get_kv_events_config_for_mode(mode) for mode, count in active.items() if count > 0
        }
    # services are compared by effect: the list the job would run, implied ones
    # included, so `infra:` and declared etcd/nats entries (or mooncake_kv_store
    # and a declared master) resolve to the same thing.
    from srtctl.services.implicit import effective_services

    effective = [entry.service for entry in effective_services(loaded)]
    dumped["services"] = schema.fields["services"]._serialize(effective, "services", loaded)
    if not any(service.type in ("etcd", "nats") for service in effective) and isinstance(dumped.get("infra"), dict):
        # No discovery plane: the NATS payload knob never had an effect, and the migrator drops it.
        dumped["infra"]["nats_max_payload_mb"] = None
    for service, item in zip(effective, dumped["services"], strict=True):
        # Kind defaults left implicit on one side and spelled out on the other are the same service.
        item["placement"] = {"node": service.effective_placement}
        item["start"] = service.effective_start
        item["critical"] = service.effective_critical
    # mooncake_kv_store.env is compared by effect too: it was injected into every
    # worker on top of the per-mode env, which is where the migrator puts it.
    mooncake = backend.get("mooncake_kv_store") if isinstance(backend, dict) else None
    if isinstance(mooncake, dict) and mooncake.get("env"):
        for mode, key in (
            ("prefill", "prefill_environment"),
            ("decode", "decode_environment"),
            ("agg", "aggregated_environment"),
        ):
            if getattr(loaded.resources, f"num_{mode}") > 0:
                backend[key] = {**(backend.get(key) or {}), **mooncake["env"]}
        mooncake["env"] = {}
    return dumped


def _mask_spelling_only_fields(dump: dict[str, Any]) -> None:
    """Blank fields that only record how the recipe was spelled, not what it resolves to.

    ``dynamo.source`` maps onto ``hash`` / ``version`` / ``wheel`` / ``cargo_patches``
    in ``DynamoConfig.__post_init__``; those mapped fields are what gets compared.
    """
    dynamo = dump.get("dynamo")
    if isinstance(dynamo, dict):
        dynamo["source"] = None


def _mask_unused_benchmark_fields(dump: dict[str, Any]) -> None:
    """Blank benchmark fields the type never reads: the migrator removes them, and they never had an effect."""
    try:
        import srtctl.benchmarks  # noqa: F401
        from srtctl.benchmarks.base import benchmark_config_fields
    except Exception:  # noqa: BLE001
        return
    benchmark = dump.get("benchmark")
    if not isinstance(benchmark, dict):
        return
    accepted = benchmark_config_fields(str(benchmark.get("type", "manual")))
    for key in list(benchmark):
        if key not in accepted:
            benchmark[key] = None


def _diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            out += _diff(a.get(key), b.get(key), f"{path}.{key}" if path else str(key))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += _diff(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [f"{path}: v1={a!r} v2={b!r}"]


def verify_migration_text(text: str, path: Path = Path("<text>")) -> VerifyResult:
    """Migrate in memory and prove the v1 and v2 documents resolve to the same config."""
    import yaml

    try:
        original = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return VerifyResult(path, "error", f"YAML parse error: {exc}")
    if not isinstance(original, dict):
        return VerifyResult(path, "error", "not a YAML mapping")

    try:
        result = migrate_recipe_text(text)
        migrated = yaml.safe_load(result.text)
    except Exception as exc:  # noqa: BLE001 - a migrator crash is a finding, not a skip
        detail = next((line for line in str(exc).splitlines() if "duplicate key" in line), str(exc).splitlines()[0])
        return VerifyResult(path, "error", f"migration failed: {detail} (fix the recipe, then re-run)")
    if not isinstance(migrated, dict):
        return VerifyResult(path, "error", "migrated document is not a YAML mapping", notes=result.notes)

    try:
        before = _expand(original)
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(path, "skipped", f"v1 document does not expand: {exc}", notes=result.notes)
    try:
        after = _expand(migrated)
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(path, "mismatch", f"migrated document does not expand: {exc}", notes=result.notes)
    if len(before) != len(after):
        return VerifyResult(path, "mismatch", f"{len(before)} variants before, {len(after)} after", notes=result.notes)

    for (name_a, raw_a), (_name_b, raw_b) in zip(before, after, strict=True):
        where = f" [{name_a}]" if name_a else ""
        try:
            dump_a = _resolved_dump(raw_a)
        except Exception as exc:  # noqa: BLE001
            return VerifyResult(path, "skipped", f"v1 does not load{where}: {exc}", notes=result.notes)
        try:
            dump_b = _resolved_dump(raw_b)
        except Exception as exc:  # noqa: BLE001
            return VerifyResult(path, "mismatch", f"migrated recipe does not load{where}: {exc}", notes=result.notes)
        for dump in (dump_a, dump_b):
            _mask_spelling_only_fields(dump)
            _mask_unused_benchmark_fields(dump)
        differences = _diff(dump_a, dump_b)
        if differences:
            return VerifyResult(path, "mismatch", f"resolved configs differ{where}: " + "; ".join(differences[:5]))
    return VerifyResult(path, "ok", variants=len(before), notes=result.notes)


def verify_migration_file(path: Path) -> VerifyResult:
    return verify_migration_text(path.read_text(encoding="utf-8"), path)
