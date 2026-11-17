# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe composition: merge the shared files a recipe lists in ``extends`` under it.

The recipe passed to ``-f`` (the *final file*) may list shared partial recipes in a top-level
``extends``. They are merged left to right, then the final file on top, with the rules override
variants use: mappings merge key by key, lists and scalars replace, ``null`` deletes a key an
earlier file set (a ``null`` that deletes nothing stays, as it would in the file alone). In an
override file the result is the ``base`` every variant is merged over; an override file with
variants and no ``base`` takes the merged files as its base. A file in ``extends`` is a partial
recipe: it cannot use ``extends`` itself or hold ``base``, ``override_*``, ``zip_override_*`` or
``sweep``.

Documents of both representations go through the same rules: ruamel ``CommentedMap`` documents
(the CLI writes the composed recipe the job runs, comments kept) and plain dicts (``load_config``
and validation). See docs/extends.md.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from ruamel.yaml.comments import CommentedMap

from srtctl.core.yaml_utils import comment_aware_merge, dump_yaml_with_comments, load_yaml_text_with_comments

EXTENDS_KEY = "extends"
STDIN_SOURCE = "<stdin>"
# Keys only the final file may hold: they define what jobs the file expands into.
_SWEEP_KEY = "sweep"
_VARIANT_PREFIXES = ("override_", "zip_override_")
# The first line of the provenance block written at the top of a composed recipe.
_HEADER = "# Composed by srtctl from these files (later files win):"
_HEADER_ENTRY = "#   "


class ExtendsError(ValueError):
    """A recipe's ``extends`` cannot be composed."""


@dataclass(frozen=True)
class ComposedRecipe:
    """A recipe with the files in its ``extends`` merged under it.

    Attributes:
        document: The merged recipe, without ``extends``. A ``CommentedMap`` when the recipe was
            one (comments kept), else a dict.
        sources: The files merged, lowest precedence first; the last entry is the recipe itself.
    """

    document: Any
    sources: tuple[str, ...]

    @property
    def header(self) -> str:
        """The comment block that records ``sources`` at the top of the written recipe."""
        return "\n".join([_HEADER, *(f"{_HEADER_ENTRY}{source}" for source in self.sources)]) + "\n"

    def text(self) -> str:
        """The merged recipe as YAML, headed by the files it came from."""
        return self.header + (dump_yaml_with_comments(self.document) or "")


def has_extends(document: object) -> bool:
    """Whether ``document`` is a mapping with a top-level ``extends``."""
    return isinstance(document, Mapping) and EXTENDS_KEY in document


def _is_variant_key(key: object) -> bool:
    return isinstance(key, str) and key.startswith(_VARIANT_PREFIXES)


def _is_override_file(document: Mapping[str, Any]) -> bool:
    """``base`` or any ``override_*`` / ``zip_override_*`` section makes ``document`` an override file."""
    return "base" in document or any(_is_variant_key(key) for key in document)


def _extends_paths(document: Mapping[str, Any], *, base_dir: Path, source: str) -> list[Path]:
    """The files ``document`` lists in ``extends``, absolute against ``base_dir``, in order.

    ``extends`` is one path or a non-empty list of paths, each listed once. Paths are literal:
    environment variables, ``~`` and globs are not expanded. Symlinks are kept in the returned
    paths (they name the files in errors and in the provenance block) and resolved to tell
    whether two entries are the same file.
    """
    value = document[EXTENDS_KEY]
    entries = [value] if isinstance(value, str) else value
    if not isinstance(entries, list) or not entries or not all(isinstance(e, str) and e.strip() for e in entries):
        raise ExtendsError(f"{source}: extends must be a file path or a non-empty list of file paths, got {value!r}")
    paths: list[Path] = []
    seen: set[Path] = set()
    for entry in entries:
        if entry.startswith("~") or any(char in entry for char in "$*?"):
            raise ExtendsError(
                f"{source}: extends {entry!r}: paths are literal; environment variables, ~ and globs are not expanded"
            )
        path = _absolute(base_dir / entry)
        if path.resolve() in seen:
            raise ExtendsError(f"{source}: extends lists {entry!r} more than once")
        seen.add(path.resolve())
        paths.append(path)
    return paths


def _absolute(path: Path) -> Path:
    """``path`` made absolute and normalized (``..`` folded), without resolving symlinks."""
    return Path(os.path.abspath(path))


def _commented(value: Any) -> Any:
    """``value`` with every mapping in it a ``CommentedMap``, so the comment-keeping merge recurses into it."""
    if isinstance(value, CommentedMap):
        return value
    if isinstance(value, Mapping):
        converted = CommentedMap()
        for key, item in value.items():
            converted[key] = _commented(item)
        return converted
    if isinstance(value, list):
        return [_commented(item) for item in value]
    return value


def _load_layer(path: Path, *, commented: bool, source: str) -> Any:
    if not path.exists():
        raise ExtendsError(f"{source}: extends {path}: file not found")
    if not path.is_file():
        raise ExtendsError(f"{source}: extends {path}: not a file")
    text = path.read_text()
    if not text.strip():
        raise ExtendsError(f"{source}: extends {path}: file is empty")
    try:
        return load_yaml_text_with_comments(text) if commented else yaml.safe_load(text)
    except TypeError as exc:  # load_yaml_text_with_comments: the top level is not a mapping
        raise ExtendsError(f"{source}: extends {path}: not a YAML mapping") from exc
    except Exception as exc:
        raise ExtendsError(f"{source}: extends {path}: {exc}") from exc


def _check_layer(layer: Any, *, label: str, source: str, schema: Any) -> None:
    if not isinstance(layer, Mapping):
        raise ExtendsError(f"{source}: extends {label}: not a YAML mapping")
    if EXTENDS_KEY in layer:
        raise ExtendsError(
            f"{source}: extends {label}: a file in extends cannot use extends itself; "
            f"list every file in the extends of {source}"
        )
    sections = sorted(str(key) for key in layer if key in ("base", _SWEEP_KEY) or _is_variant_key(key))
    if sections:
        raise ExtendsError(
            f"{source}: extends {label}: holds {', '.join(sections)}; "
            "only the file passed to -f can define base, override_*, zip_override_* or sweep"
        )
    if "schema" in layer and layer["schema"] != schema:
        raise ExtendsError(
            f"{source}: extends {label}: declares schema {layer['schema']!r} but the recipe declares {schema!r}"
        )


def _without(mapping: Mapping[str, Any], *keys: str) -> Any:
    """A deep copy of ``mapping`` (its comments too, for a ``CommentedMap``) without ``keys``."""
    result = copy.deepcopy(mapping) if isinstance(mapping, CommentedMap) else copy.deepcopy(dict(mapping))
    for key in keys:
        result.pop(key, None)
    return result


def _reordered(mapping: Mapping[str, Any], first: Sequence[str]) -> Any:
    """``mapping`` with the keys in ``first`` (when present) moved to the front, comments kept."""
    keys = [key for key in first if key in mapping] + [key for key in mapping if key not in first]
    if not isinstance(mapping, CommentedMap):
        return {key: mapping[key] for key in keys}
    result = CommentedMap()
    for key in keys:
        result[key] = mapping[key]
        if key in mapping.ca.items:
            result.ca.items[key] = mapping.ca.items[key]
    return result


def _new_nulls(below: Any, layer: Mapping[str, Any], path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Where ``layer`` sets ``null`` on a key ``below`` does not have.

    The merge deletes a key on ``null``, which is the rule where an earlier file set it. A ``null`` with
    nothing under it is the file's own value (``telemetry.dcgm_exporter: null`` keeps a cluster default
    exporter out), so it is kept, as it would be in the file alone.
    """
    found: list[tuple[Any, ...]] = []
    for key, value in layer.items():
        has_key = isinstance(below, Mapping) and key in below
        if value is None and not has_key:
            found.append((*path, key))
        elif isinstance(value, Mapping) and has_key and isinstance(below[key], Mapping):
            found.extend(_new_nulls(below[key], value, (*path, key)))
    return found


def _merge(merge: Any, below: Any, layer: Mapping[str, Any]) -> Any:
    """``merge(below, layer)``, keeping the ``null`` values that delete nothing (:func:`_new_nulls`)."""
    merged = merge(below, layer)
    for path in _new_nulls(below, layer):
        node = merged
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = None
    return merged


def compose_recipe(
    document: Mapping[str, Any],
    *,
    base_dir: Path,
    source: str = "<recipe>",
    layers: Sequence[Mapping[str, Any] | Path] = (),
) -> ComposedRecipe:
    """Merge the files ``document`` lists in ``extends`` under it.

    ``base_dir`` resolves relative ``extends`` paths (the recipe's directory); ``source`` names the
    recipe in errors and in the provenance block. ``layers`` are merged below the recipe's own
    ``extends`` (lowest precedence first) and follow the same rules; a tool that picks shared files
    at launch passes them here instead of writing them into the recipe. A ``CommentedMap``
    ``document`` composes with comments kept; any other mapping composes as plain dicts.

    Raises:
        ExtendsError: ``extends`` is malformed, a file is missing or not a mapping, a file in
            ``extends`` uses ``extends`` or holds variants, the recipe declares no ``schema``, or
            a file declares a different one.
    """
    if not has_extends(document) and not layers:
        return ComposedRecipe(document, (source,))
    commented = isinstance(document, CommentedMap)
    override_file = _is_override_file(document)
    for name, section in document.items():
        if (name == "base" or _is_variant_key(name)) and isinstance(section, Mapping) and EXTENDS_KEY in section:
            raise ExtendsError(f"{source}: extends must be a top-level key of the file, not inside {name}")
    base = document.get("base") if override_file else None
    schema = document.get("schema", base.get("schema") if isinstance(base, Mapping) else None)
    if schema is None:
        raise ExtendsError(f"{source}: a recipe that uses extends must declare its schema (`schema: 2`)")

    loaded: list[tuple[str, Any]] = []
    for index, layer in enumerate(layers):
        if isinstance(layer, Path):
            path = _absolute(layer)
            loaded.append((str(path), _load_layer(path, commented=commented, source=source)))
        else:
            loaded.append((f"<layer {index}>", _commented(layer) if commented else layer))
    if has_extends(document):
        for path in _extends_paths(document, base_dir=base_dir, source=source):
            loaded.append((str(path), _load_layer(path, commented=commented, source=source)))

    if commented:
        merge: Any = comment_aware_merge
        merged: Any = CommentedMap()
    else:
        from srtctl.core.config import deep_merge

        merge, merged = deep_merge, {}
    for label, layer in loaded:
        _check_layer(layer, label=label, source=source, schema=schema)
        merged = _merge(merge, merged, _without(layer, "schema"))

    body = _without(document, EXTENDS_KEY)
    if override_file:
        body["base"] = _merge(merge, merged, base if isinstance(base, Mapping) else {})
        composed = _reordered(body, ("schema", "base"))
    else:
        composed = _reordered(_merge(merge, merged, body), ("schema", "name"))
    if commented and document.ca.comment is not None:
        composed.ca.comment = document.ca.comment
    return ComposedRecipe(composed, (*(label for label, _ in loaded), source))


def compose_recipe_text(text: str, *, path: Path | None) -> ComposedRecipe | None:
    """Compose the recipe ``text`` read from ``path`` (``None``: stdin) if it has a top-level ``extends``.

    Returns None when it has none, or when the text is not YAML (the recipe's own loader reports
    that). Relative ``extends`` paths resolve against the file's directory, or the working
    directory for stdin.
    """
    try:
        if not has_extends(yaml.safe_load(text)):
            return None
    except yaml.YAMLError:
        return None
    document = load_yaml_text_with_comments(text)
    if path is None:
        return compose_recipe(document, base_dir=Path.cwd(), source=STDIN_SOURCE)
    return compose_recipe(document, base_dir=path.parent, source=str(_absolute(path)))


def load_recipe(path: Path) -> Any:
    """Read the recipe at ``path`` the way srtctl's loaders do, with its ``extends`` merged in.

    Returns whatever the YAML holds (callers report an empty or non-mapping file); a mapping with
    ``extends`` comes back composed, as plain dicts.
    """
    with open(path) as f:
        document = yaml.safe_load(f)
    if has_extends(document):
        return compose_recipe(document, base_dir=path.parent, source=str(_absolute(path))).document
    return document


def layer_files(paths: Iterable[Path]) -> frozenset[Path]:
    """The files another of ``paths`` lists in its ``extends``: shared layers, not jobs.

    Resolved paths; files that do not parse are skipped (they are reported when submitted).
    """
    found: set[Path] = set()
    for path in paths:
        try:
            document = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError):
            continue
        if not has_extends(document):
            continue
        value = document[EXTENDS_KEY]
        entries = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        found.update((path.parent / entry).resolve() for entry in entries if isinstance(entry, str) and entry.strip())
    return frozenset(found)


def composition_header(text: str) -> str:
    """The provenance block :meth:`ComposedRecipe.text` wrote at the top of ``text``, or ``""``."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\n") != _HEADER:
        return ""
    block = [lines[0]]
    for line in lines[1:]:
        if not line.startswith(_HEADER_ENTRY):
            break
        block.append(line)
    return "".join(block)
