# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line recipe overrides: ``--set KEY=VALUE`` and ``--unset KEY``.

Paths are dotted, with ``[N]`` for list indexes and quotes for segments that
contain dots (``container_mounts."/a/b.c"``). Values are parsed as YAML scalars
or sequences: ``720`` is an int, ``"720"`` a string, ``true`` a bool,
``[4, 8]`` a list. A value that parses as a mapping (``{"a": 1}``) is kept as
the literal string, because that is how engine flags take JSON.

Overrides are applied to the raw recipe document before cluster defaults,
observability expansion, sweep expansion, and schema validation, so an explicit
``--set`` always wins and ``{placeholder}`` values still expand. On override
files (``base`` plus ``override_*`` / ``zip_override_*``) a ``--set`` is written
into ``base`` and into every variant so no variant can shadow it; a ``--unset``
removes the key from all of them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, MutableMapping, MutableSequence, Sequence
from dataclasses import dataclass
from typing import Any

import yaml
from ruamel.yaml.comments import CommentedMap

Segment = str | int


@dataclass(frozen=True)
class Override:
    """One ``--set`` or ``--unset``."""

    path: tuple[Segment, ...]
    value: Any = None
    unset: bool = False

    def render(self) -> str:
        key = format_path(self.path)
        if self.unset:
            return f"--unset {key}"
        return f"--set {key}={_render_value(self.value)}"


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(value)


def format_path(path: Sequence[Segment]) -> str:
    out = ""
    for segment in path:
        if isinstance(segment, int):
            out += f"[{segment}]"
        else:
            text = segment if "." not in segment and "[" not in segment and "]" not in segment else f'"{segment}"'
            out += text if not out else f".{text}"
    return out


def parse_path(text: str) -> tuple[Segment, ...]:
    """Parse ``a.b[2]."c.d".e`` into ``("a", "b", 2, "c.d", "e")``."""
    text = text.strip()
    if not text:
        raise ValueError("override path must not be empty")
    segments: list[Segment] = []
    i = 0
    n = len(text)
    expect_segment = True
    while i < n:
        ch = text[i]
        if ch == ".":
            if expect_segment:
                raise ValueError(f"empty segment in override path {text!r}")
            expect_segment = True
            i += 1
            continue
        if ch == "[":
            if expect_segment and not segments:
                raise ValueError(f"override path {text!r} must start with a key, not an index")
            end = text.find("]", i)
            if end == -1 or not text[i + 1 : end].isdigit():
                raise ValueError(f"bad list index in override path {text!r}")
            segments.append(int(text[i + 1 : end]))
            i = end + 1
            expect_segment = False
            continue
        if not expect_segment:
            raise ValueError(f"expected '.' or '[' at position {i} in override path {text!r}")
        if ch in ('"', "'"):
            end = text.find(ch, i + 1)
            if end == -1:
                raise ValueError(f"unterminated quote in override path {text!r}")
            segments.append(text[i + 1 : end])
            i = end + 1
        else:
            j = i
            while j < n and text[j] not in ".[":
                j += 1
            segments.append(text[i:j])
            i = j
        expect_segment = False
    if expect_segment:
        raise ValueError(f"override path {text!r} ends with a separator")
    return tuple(segments)


def parse_value(raw: str) -> Any:
    """YAML-scalar parsing with mappings kept as literal strings."""
    if raw == "":
        return ""
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    if isinstance(parsed, dict):
        return raw
    return parsed


def parse_set(arg: str) -> Override:
    key, sep, raw = arg.partition("=")
    if not sep:
        raise ValueError(f"--set expects KEY=VALUE, got {arg!r}")
    return Override(path=parse_path(key), value=parse_value(raw))


def parse_unset(arg: str) -> Override:
    return Override(path=parse_path(arg), unset=True)


def parse_overrides(sets: Iterable[str] | None, unsets: Iterable[str] | None) -> list[Override]:
    overrides = [parse_set(item) for item in (sets or [])]
    overrides.extend(parse_unset(item) for item in (unsets or []))
    return overrides


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def _new_mapping(like: Any) -> MutableMapping[str, Any]:
    return CommentedMap() if isinstance(like, CommentedMap) else {}


def _step(container: Any, segment: Segment, *, create: bool, path: Sequence[Segment]) -> Any:
    if isinstance(segment, int):
        if not isinstance(container, MutableSequence):
            raise TypeError(f"{format_path(path)}: cannot index a non-list with [{segment}]")
        if segment >= len(container):
            raise ValueError(f"{format_path(path)}: index [{segment}] is out of range (length {len(container)})")
        return container[segment]
    if not isinstance(container, MutableMapping):
        raise TypeError(f"{format_path(path)}: cannot descend into a non-mapping at {segment!r}")
    if segment not in container:
        if not create:
            return None
        container[segment] = _new_mapping(container)
    return container[segment]


def apply_override(doc: MutableMapping[str, Any], override: Override) -> bool:
    """Apply one override to a mapping. Returns True when the document changed."""
    *parents, last = override.path
    container: Any = doc
    for depth, segment in enumerate(parents):
        container = _step(container, segment, create=not override.unset, path=override.path[: depth + 1])
        if container is None:
            return False  # --unset on a missing path is a no-op
    if override.unset:
        if isinstance(last, int):
            if isinstance(container, MutableSequence) and last < len(container):
                del container[last]
                return True
            return False
        if isinstance(container, MutableMapping) and last in container:
            del container[last]
            return True
        return False
    if isinstance(last, int):
        if not isinstance(container, MutableSequence) or last >= len(container):
            raise ValueError(f"{format_path(override.path)}: index out of range")
        container[last] = override.value
        return True
    if not isinstance(container, MutableMapping):
        raise TypeError(f"{format_path(override.path)}: cannot set a key on a non-mapping")
    container[last] = override.value
    return True


def apply_overrides_to_recipe(doc: MutableMapping[str, Any], overrides: Sequence[Override]) -> list[str]:
    """Apply overrides to a plain, sweep, or override-format recipe document.

    Returns the rendered overrides that changed something (for logging).
    """
    if not overrides:
        return []
    applied: list[str] = []
    is_override_file = "base" in doc
    for override in overrides:
        changed = False
        if not is_override_file:
            changed = apply_override(doc, override)
        else:
            changed = apply_override(doc["base"], override)
            for key, section in doc.items():
                if not isinstance(key, str) or not isinstance(section, MutableMapping):
                    continue
                if key.startswith("zip_override_"):
                    # A one-element list broadcasts to every zipped variant, whatever the value.
                    zipped = override if override.unset else Override(path=override.path, value=[override.value])
                    changed = apply_override(section, zipped) or changed
                elif key.startswith("override_"):
                    changed = apply_override(section, override) or changed
        if changed:
            applied.append(override.render())
    return applied
