# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from srtctl.core.config import (
    generate_override_configs,
    load_cluster_config,
    resolve_config_with_defaults,
)
from srtctl.core.schema import SrtConfig
from srtctl.core.validation import preflight_config_variants

DOC_PATH = Path(__file__).resolve().parents[3] / "docs" / "config-reference.md"


def schema_summary() -> dict[str, Any]:
    """Return a compact summary of top-level SrtConfig fields."""
    return {
        "config_type": "SrtConfig",
        "top_level_fields": [_field_summary(SrtConfig, f.name) for f in fields(SrtConfig)],
    }


def explain_field(path: str) -> dict[str, Any]:
    """Return best-effort schema and docs context for a config field path."""
    resolved = _resolve_field_path(path)
    docs = get_config_reference(query=path, max_matches=3)
    if not docs["matches"]:
        tail = path.split(".")[-1]
        docs = get_config_reference(query=tail, max_matches=3)
    return {
        "path": path,
        "resolved": resolved is not None,
        "schema": resolved,
        "docs": docs,
    }


def get_config_reference(query: str | None = None, max_matches: int = 5) -> dict[str, Any]:
    """Search config-reference.md and return matching snippets."""
    sections = _parse_doc_sections()
    if not query:
        return {
            "doc_path": str(DOC_PATH),
            "matches": [{"heading": section["heading"]} for section in sections[: max_matches or 5]],
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
                "heading": section["heading"],
                "snippet": snippet,
                "score": len(hit_indexes) + int(lowered in section["heading"].lower()),
            }
        )
    matches.sort(key=lambda item: item["score"], reverse=True)
    if not matches:
        resolved = _resolve_field_path(query)
        if resolved is not None and resolved["leaf"] is not None:
            leaf = resolved["leaf"]
            matches.append(
                {
                    "heading": f"Schema: {query}",
                    "snippet": (f"{leaf['name']}: type={leaf['type']}, default={leaf['default']}"),
                    "score": 1,
                }
            )
    return {"doc_path": str(DOC_PATH), "query": query, "matches": matches[:max_matches]}


def validate_config(
    *,
    config: dict[str, Any] | None = None,
    config_yaml: str | None = None,
    apply_cluster_defaults: bool = True,
) -> dict[str, Any]:
    """Validate one plain config or an override config against the real schema."""
    raw = _load_raw_config(config=config, config_yaml=config_yaml)
    cluster_config = load_cluster_config() if apply_cluster_defaults else None
    schema = SrtConfig.Schema()

    variants: list[tuple[str, dict[str, Any]]] = generate_override_configs(raw) if "base" in raw else [("base", raw)]

    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    for suffix, variant in variants:
        resolved = resolve_config_with_defaults(variant, cluster_config)
        try:
            loaded = schema.load(resolved)
            normalized.append({"variant": suffix, "config": schema.dump(loaded)})
        except Exception as exc:
            errors.append(f"{suffix}: {exc}")

    return {
        "valid": not errors,
        "variant_count": len(variants),
        "errors": errors,
        "normalized": normalized,
        "cluster_defaults_applied": apply_cluster_defaults and cluster_config is not None,
    }


def preflight_config(
    *,
    config: dict[str, Any] | None = None,
    config_yaml: str | None = None,
    apply_cluster_defaults: bool = True,
) -> dict[str, Any]:
    """Check that model and container references are locally resolvable."""
    raw = _load_raw_config(config=config, config_yaml=config_yaml)
    cluster_config = load_cluster_config() if apply_cluster_defaults else None
    results = preflight_config_variants(raw, cluster_config=cluster_config)
    return {
        "ok": all(result.ok for result in results),
        "variant_count": len(results),
        "variants": [result.as_dict() for result in results],
        "cluster_defaults_applied": apply_cluster_defaults and cluster_config is not None,
    }


def resolve_config(
    *,
    config: dict[str, Any] | None = None,
    config_yaml: str | None = None,
    apply_cluster_defaults: bool = True,
) -> dict[str, Any]:
    """Resolve config defaults without requiring the caller to understand srtslurm.yaml."""
    raw = _load_raw_config(config=config, config_yaml=config_yaml)
    cluster_config = load_cluster_config() if apply_cluster_defaults else None
    if "base" in raw:
        resolved_variants = [
            {
                "variant": suffix,
                "config": resolve_config_with_defaults(variant, cluster_config),
            }
            for suffix, variant in generate_override_configs(raw)
        ]
        return {"variant_count": len(resolved_variants), "variants": resolved_variants}
    return {
        "variant_count": 1,
        "variants": [{"variant": "base", "config": resolve_config_with_defaults(raw, cluster_config)}],
    }


def _load_raw_config(*, config: dict[str, Any] | None = None, config_yaml: str | None = None) -> dict[str, Any]:
    if config is not None:
        return config
    if config_yaml is None:
        raise ValueError("Provide either config or config_yaml")
    loaded = yaml.safe_load(config_yaml)
    if not isinstance(loaded, dict):
        raise ValueError("Config must be a YAML mapping")
    return loaded


def _parse_doc_sections() -> list[dict[str, str]]:
    text = DOC_PATH.read_text()
    sections: list[dict[str, str]] = []
    current_heading = "Introduction"
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if body:
                sections.append({"heading": current_heading, "body": "\n".join(body).strip()})
            current_heading = line.lstrip("#").strip()
            body = []
        else:
            body.append(line)
    if body:
        sections.append({"heading": current_heading, "body": "\n".join(body).strip()})
    return sections


def _resolve_field_path(path: str) -> dict[str, Any] | None:
    current_cls: type[Any] | None = SrtConfig
    trail: list[dict[str, Any]] = []
    for segment in path.split("."):
        if current_cls is None or not is_dataclass(current_cls):
            return None
        try:
            field_info = _field_summary(current_cls, segment)
        except KeyError:
            return None
        trail.append(field_info)
        next_type = _unwrap_type(get_type_hints(current_cls, include_extras=True).get(segment))
        current_cls = next_type if isinstance(next_type, type) and is_dataclass(next_type) else None
    return {"segments": trail, "leaf": trail[-1] if trail else None}


def _field_summary(cls: type[Any], field_name: str) -> dict[str, Any]:
    field_map = {item.name: item for item in fields(cls)}
    if field_name not in field_map:
        raise KeyError(field_name)
    item = field_map[field_name]
    hints = get_type_hints(cls, include_extras=True)
    annotation = hints.get(field_name, item.type)
    default: Any
    if item.default is not MISSING:
        default = item.default
    elif item.default_factory is not MISSING:  # type: ignore[attr-defined]
        factory_name = getattr(item.default_factory, "__name__", type(item.default_factory).__name__)
        default = f"<factory:{factory_name}>"
    else:
        default = "<required>"
    return {
        "name": item.name,
        "type": _type_label(annotation),
        "default": default,
    }


def _unwrap_type(annotation: Any) -> Any:
    if annotation is None:
        return None
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    if str(origin) == "typing.Annotated":
        return _unwrap_type(get_args(annotation)[0])
    if origin is types.UnionType or str(origin) == "typing.Union":
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        dataclass_args = [arg for arg in args if isinstance(arg, type) and is_dataclass(arg)]
        return dataclass_args[0] if dataclass_args else (args[0] if args else annotation)
    return annotation


def _type_label(annotation: Any) -> str:
    if annotation is None:
        return "unknown"
    origin = get_origin(annotation)
    if origin is None:
        return getattr(annotation, "__name__", str(annotation))
    if str(origin) == "typing.Annotated":
        return _type_label(get_args(annotation)[0])
    args = ", ".join(_type_label(arg) for arg in get_args(annotation))
    origin_name = getattr(origin, "__name__", str(origin).replace("typing.", ""))
    return f"{origin_name}[{args}]"
