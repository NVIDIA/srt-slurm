# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every YAML block in docs/ is checked against the published JSON Schema.

A block is classified by its top-level keys: an override file (``base:``) and a ``srtslurm.yaml``
are validated as written; a recipe fragment is laid onto a minimal valid recipe first, with the blocks
it names (``engine``, ``roles``, ...) replacing the base's, so a snippet that shows only ``frontend:``
is still checked for unknown keys, bad enums and wrong types. Blocks whose top level is a nested
section (``readiness:``, ``args`` flags) are not recipe-shaped and are not checked. Cross-field rules
are ``srtctl dry-run``'s job, not this test's. Mark an intentionally partial or proposed block with
``<!-- docs-yaml: skip -->`` on the line above its fence.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from srtctl.core.schema import ClusterConfig, SrtConfig
from srtctl.core.schema_docs import json_schema

jsonschema = pytest.importorskip("jsonschema")

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
GENERATED = {"schema-reference.md", "cli-reference.md"}
SKIP_MARKER = "<!-- docs-yaml: skip -->"
FENCE = re.compile(r"^```ya?ml[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
PLACEHOLDER = re.compile(r"\{\w+\}")
REPLACED_BLOCKS = {"engine", "roles", "benchmark", "frontend"}
BASE_RECIPE = REPO_ROOT / "examples" / "mocker" / "dynamo-agg.yaml"

RECIPE_SCHEMA = json_schema(SrtConfig)
CLUSTER_SCHEMA = json_schema(ClusterConfig)
RECIPE_KEYS = set(RECIPE_SCHEMA["$defs"]["Recipe"]["properties"])
CLUSTER_KEYS = set(CLUSTER_SCHEMA["properties"])


def _blocks() -> list[tuple[str, str]]:
    out = []
    for md in sorted(DOCS.rglob("*.md")):
        rel = md.relative_to(DOCS)
        if md.name in GENERATED or rel.parts[0] == "design" or md.name == "AGENTS.md":
            continue
        text = md.read_text()
        for match in FENCE.finditer(text):
            before = text[: match.start()].rstrip("\n").rsplit("\n", 1)[-1]
            if before.strip() == SKIP_MARKER:
                continue
            line = text[: match.start()].count("\n") + 1
            out.append((f"{rel}:{line}", match.group(1)))
    return out


def _kind(doc: Any) -> str | None:
    if not isinstance(doc, dict) or not doc:
        return None
    keys = set(doc)
    if "base" in keys:
        return "override"
    if keys <= RECIPE_KEYS:
        return "recipe"
    if keys <= CLUSTER_KEYS:
        return "cluster"
    return None


def _on_base(fragment: dict[str, Any]) -> dict[str, Any]:
    def merge(base: dict[str, Any], top: dict[str, Any], depth: int) -> dict[str, Any]:
        out = copy.deepcopy(base)
        for key, value in top.items():
            if depth == 0 and key in REPLACED_BLOCKS:
                out[key] = value
            elif isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = merge(out[key], value, depth + 1)
            else:
                out[key] = value
        return out

    return merge(yaml.safe_load(BASE_RECIPE.read_text()), fragment, 0)


BLOCKS = _blocks()
PARSED = [(block_id, yaml.safe_load(src)) for block_id, src in BLOCKS if src.strip()]
CHECKED = [(block_id, doc) for block_id, doc in PARSED if _kind(doc)]


@pytest.mark.parametrize(("block_id", "src"), BLOCKS, ids=[b for b, _ in BLOCKS])
def test_docs_yaml_block_parses(block_id: str, src: str) -> None:
    yaml.safe_load(src)


@pytest.mark.parametrize(("block_id", "doc"), CHECKED, ids=[b for b, _ in CHECKED])
def test_docs_yaml_block_matches_the_schema(block_id: str, doc: dict[str, Any]) -> None:
    kind = _kind(doc)
    schema = CLUSTER_SCHEMA if kind == "cluster" else RECIPE_SCHEMA
    instance = _on_base(doc) if kind == "recipe" else doc
    errors = [
        error
        for error in jsonschema.Draft202012Validator(schema).iter_errors(instance)
        # `{param}` is a sweep placeholder; it is substituted with a typed value before validation.
        if not (isinstance(error.instance, str) and PLACEHOLDER.fullmatch(error.instance))
    ]
    assert not errors, "\n".join(f"{'.'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errors)


def test_most_docs_yaml_blocks_are_checked() -> None:
    """Guard against a classification change that silently stops checking the docs."""
    assert len(CHECKED) >= 120, len(CHECKED)
