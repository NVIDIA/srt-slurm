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

from ruamel.yaml.comments import CommentedMap

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
    infra = variant.get("infra")
    if isinstance(infra, CommentedMap) and "etcd_nats_dedicated_node" in infra:
        dedicated = bool(infra.pop("etcd_nats_dedicated_node"))
        infra.ca.items.pop("etcd_nats_dedicated_node", None)
        _child_map(infra, "placement")["node"] = "dedicated" if dedicated else "head"
        notes.append(f"{label}infra.placement.node: {'dedicated' if dedicated else 'head'}")
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
        notes += _fold_dynamo_source(variant, label)
        notes += _strip_unused_benchmark_fields(variant, base, label)
        # A `backend:` that only held per-mode env and engine config is empty now.
        # Make the implicit engine explicit rather than dropping the block: an
        # override variant may `null` its way back to the default type, which
        # only works when the base still has the key.
        backend = variant.get("backend")
        if isinstance(backend, CommentedMap) and not backend:
            engine = _ENGINE_FOR_KEY.get(_engine_key_for(variant, base), "sglang")
            backend.ca.comment = None
            backend.ca.items.clear()
            backend.ca.end = None
            backend["type"] = engine
            notes.append(f"{label}backend.type: {engine} (was the implicit default)")
        for key in ("resources", "dynamo", "infra", "frontend"):
            _neutralize_if_empty(variant, key)
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
    dumped = schema.dump(schema.load(resolve_config_with_defaults(raw, None)))
    dumped.pop("schema", None)
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
