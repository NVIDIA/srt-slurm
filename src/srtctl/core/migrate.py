# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Upgrade recipe YAML to the current schema version, preserving comments.

``srtctl migrate -f recipe.yaml`` rewrites a plain recipe, an override file
(``base`` plus ``override_*`` variants), or a lockfile so that it declares
``schema: <current>`` and uses the current layout. The transformation is done on
a ruamel round-trip document, so comments, key order, and quoting survive.

Each schema step registers its structural rewrite in :func:`_migrate_1_to_2`;
this module starts with the version key alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ruamel.yaml.comments import CommentedMap

from srtctl.core.schema import CURRENT_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS
from srtctl.core.yaml_utils import dump_yaml_with_comments, load_yaml_text_with_comments


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of migrating one recipe document."""

    text: str
    changed: bool
    from_version: int
    to_version: int
    notes: tuple[str, ...]


def _declared_version(doc: CommentedMap) -> int:
    raw = doc.get("schema", 1)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"schema must be an integer version, got {raw!r}")
    if raw not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"schema {raw} is not supported; known versions: {list(SUPPORTED_SCHEMA_VERSIONS)}")
    return raw


def _migrate_1_to_2(doc: CommentedMap) -> list[str]:
    """Structural v1 -> v2 rewrites. Later schema steps add their transforms here."""
    del doc
    return []


def migrate_recipe_text(text: str) -> MigrationResult:
    """Migrate one YAML document (plain, override, or lock format) to the current schema."""
    doc = load_yaml_text_with_comments(text)
    from_version = _declared_version(doc)
    notes: list[str] = []

    if from_version < 2:
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
