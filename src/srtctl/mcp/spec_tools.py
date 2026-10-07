# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from srtctl.core.config import (
    generate_override_configs,
    resolve_config_with_defaults,
)
from srtctl.core.schema import SrtConfig
from srtctl.core.schema_docs import (
    BACKEND_TYPES,
    authoring_rows,
    benchmark_types,
    describe_field,
    resolve_field_path,
)
from srtctl.core.validation import preflight_config_variants, validate_topology

DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
# Hand-written recipe pages searched for prose context; field facts come from the schema, not from here.
DOC_PATHS = tuple(
    DOCS_DIR / name
    for name in (
        "config-reference.md",
        "cluster-config.md",
        "engines.md",
        "topology.md",
        "frontends.md",
        "sidecars.md",
        "benchmarks.md",
        "observability.md",
        "runtime-env.md",
        "profiling.md",
        "services.md",
        "overrides.md",
        "sweeps.md",
    )
)
COMPUTE_SIDE_HINT = (
    "Host-side srtslurm.yaml is not used by srtctl MCP. For cluster defaults, "
    "aliases, containers, model paths, filesystem checks, or dry-run behavior, "
    "run srtctl on the compute side or use IBAR remote_preflight/remote_dry_run/"
    "cluster_aliases with the target compute profile."
)


def schema_summary() -> dict[str, Any]:
    """Return the top-level recipe keys with descriptions, plus the engine and benchmark types."""
    return {
        "config_type": "SrtConfig",
        "reference": "docs/schema-reference.md",
        "json_schema": "srtctl schema",
        "top_level_fields": [describe_field(SrtConfig, row) for row in authoring_rows(SrtConfig, markdown=False)],
        "engine_types": [name for name, _ in BACKEND_TYPES],
        "benchmark_types": [
            {"type": name, "description": summary, "keys": list(own)} for name, summary, own in benchmark_types()
        ],
    }


def explain_field(path: str) -> dict[str, Any]:
    """Describe a config field path from the schema, with recipe-guide prose as supplemental context.

    ``schema.leaf`` carries the field's type, default, description, and allowed values straight from the
    dataclasses (the data behind docs/schema-reference.md). When the path does not resolve,
    ``schema.unresolved`` names the first unknown segment and ``schema.available`` lists the valid keys there.
    """
    resolved = resolve_field_path(path)
    docs = get_config_reference(query=path, max_matches=3)
    if not docs["matches"]:
        tail = path.split(".")[-1]
        docs = get_config_reference(query=tail, max_matches=3)
    return {
        "path": path,
        "resolved": resolved["unresolved"] is None and resolved["leaf"] is not None,
        "schema": resolved,
        "docs": docs,
    }


def get_config_reference(query: str | None = None, max_matches: int = 5) -> dict[str, Any]:
    """Search the hand-written recipe pages (DOC_PATHS) and return matching snippets with their page."""
    sections = _parse_doc_sections()
    if not query:
        return {
            "docs_dir": str(DOCS_DIR),
            "matches": [
                {"doc": section["doc"], "heading": section["heading"]} for section in sections[: max_matches or 5]
            ],
        }

    lowered = query.lower()
    matches: list[dict[str, Any]] = []
    for section in sections:
        lines = section["body"].splitlines()
        hit_indexes = [idx for idx, line in enumerate(lines) if lowered in line.lower()]
        if not hit_indexes and lowered not in section["heading"].lower():
            continue
        if hit_indexes:
            hit = hit_indexes[0]
            start = max(0, hit - 3)
            end = min(len(lines), hit + 4)
            snippet = "\n".join(lines[start:end]).strip()
        else:
            snippet = "\n".join(lines[:7]).strip()
        matches.append(
            {
                "doc": section["doc"],
                "heading": section["heading"],
                "snippet": snippet,
                "score": len(hit_indexes) + int(lowered in section["heading"].lower()),
            }
        )
    matches.sort(key=lambda item: item["score"], reverse=True)
    if not matches:
        leaf = resolve_field_path(query)["leaf"]
        if leaf is not None:
            matches.append(
                {
                    "doc": "schema-reference.md",
                    "heading": f"Schema: {query}",
                    "snippet": (
                        f"{leaf['name']}: type={leaf['type']}, default={leaf['default']}. {leaf['description']}"
                    ),
                    "score": 1,
                }
            )
    return {"docs_dir": str(DOCS_DIR), "query": query, "matches": matches[:max_matches]}


def validate_config(
    *,
    config: dict[str, Any] | None = None,
    config_yaml: str | None = None,
    apply_cluster_defaults: bool = False,
) -> dict[str, Any]:
    """Validate one plain config or an override config against the real schema."""
    _reject_cluster_defaults(apply_cluster_defaults)
    raw = _load_raw_config(config=config, config_yaml=config_yaml)
    cluster_config = None
    context = _cluster_context()
    schema = SrtConfig.Schema()

    variants: list[tuple[str, dict[str, Any]]] = generate_override_configs(raw) if "base" in raw else [("base", raw)]

    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    for suffix, variant in variants:
        try:
            resolved = resolve_config_with_defaults(variant, cluster_config)
            loaded = schema.load(resolved)
        except Exception as exc:  # noqa: BLE001 - a pre-2.0 layout is reported like any other invalid recipe
            errors.append(f"{suffix}: {exc}")
            continue
        normalized.append({"variant": suffix, "config": schema.dump(loaded)})
        for issue in validate_topology(resolved.get("roles")):
            errors.append(f"{suffix}: {issue.field}: {issue.message}")

    return {
        "valid": not errors,
        "variant_count": len(variants),
        "errors": errors,
        "normalized": normalized,
        **context,
    }


def preflight_config(
    *,
    config: dict[str, Any] | None = None,
    config_yaml: str | None = None,
    apply_cluster_defaults: bool = False,
) -> dict[str, Any]:
    """Check explicit local paths only; cluster state must be checked compute-side."""
    _reject_cluster_defaults(apply_cluster_defaults)
    raw = _load_raw_config(config=config, config_yaml=config_yaml)
    cluster_config = None
    context = _cluster_context()
    results = preflight_config_variants(raw, cluster_config=cluster_config)
    return {
        "scope": "explicit-local-paths",
        "ok": all(result.ok for result in results),
        "variant_count": len(results),
        "variants": [result.as_dict() for result in results],
        "operator_hint": COMPUTE_SIDE_HINT,
        **context,
    }


def resolve_config(
    *,
    config: dict[str, Any] | None = None,
    config_yaml: str | None = None,
    apply_cluster_defaults: bool = False,
) -> dict[str, Any]:
    """Resolve schema-only defaults without reading host-side srtslurm.yaml."""
    _reject_cluster_defaults(apply_cluster_defaults)
    raw = _load_raw_config(config=config, config_yaml=config_yaml)
    cluster_config = None
    context = _cluster_context()
    if "base" in raw:
        resolved_variants = [
            {
                "variant": suffix,
                "config": resolve_config_with_defaults(variant, cluster_config),
            }
            for suffix, variant in generate_override_configs(raw)
        ]
        return {
            "scope": "schema-only",
            "variant_count": len(resolved_variants),
            "variants": resolved_variants,
            **context,
        }
    return {
        "scope": "schema-only",
        "variant_count": 1,
        "variants": [{"variant": "base", "config": resolve_config_with_defaults(raw, cluster_config)}],
        **context,
    }


def _load_raw_config(*, config: dict[str, Any] | None = None, config_yaml: str | None = None) -> dict[str, Any]:
    if config is not None:
        return config
    if config_yaml is None:
        raise ValueError("Provide either config or config_yaml")
    loaded = yaml.safe_load(config_yaml)
    if not isinstance(loaded, dict):
        raise TypeError("Config must be a YAML mapping")
    return loaded


def _reject_cluster_defaults(apply_cluster_defaults: bool) -> None:
    if apply_cluster_defaults:
        raise ValueError(COMPUTE_SIDE_HINT)


def _cluster_context() -> dict[str, Any]:
    return {
        "cluster_defaults_applied": False,
        "cluster_defaults_source": "not-used-by-mcp",
        "cluster_config_path": None,
        "operator_boundary": COMPUTE_SIDE_HINT,
    }


def _parse_doc_sections() -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    for path in DOC_PATHS:
        current_heading = "Introduction"
        body: list[str] = []
        in_code = False
        for line in path.read_text().splitlines():
            if line.startswith("```"):
                in_code = not in_code
            if line.startswith("#") and not in_code:
                if body:
                    sections.append({"doc": path.name, "heading": current_heading, "body": "\n".join(body).strip()})
                current_heading = line.lstrip("#").strip()
                body = []
            else:
                body.append(line)
        if body:
            sections.append({"doc": path.name, "heading": current_heading, "body": "\n".join(body).strip()})
    return sections
