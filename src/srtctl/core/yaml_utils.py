# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comment-aware YAML utilities using ruamel.yaml.

Provides load/dump that preserve comments and a merge that keeps base field
order with override fields appended at the end.
"""

from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import IO, Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


def _make_yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 120
    object.__setattr__(y, "best_sequence_indent", 2)
    object.__setattr__(y, "best_map_flow_style", False)
    return y


def load_yaml_with_comments(path: Path) -> CommentedMap:
    """Load a YAML file preserving comments and key insertion order."""
    y = _make_yaml()
    with open(path) as f:
        result = y.load(f)
    if not isinstance(result, CommentedMap):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(result).__name__}")
    return result


def dump_yaml_with_comments(data: Any, stream: IO[str] | None = None) -> str | None:
    """Dump YAML preserving comments. Returns str when stream is None."""
    y = _make_yaml()
    if stream is None:
        buf = io.StringIO()
        y.dump(data, buf)
        return buf.getvalue()
    y.dump(data, stream)
    return None


def comment_aware_merge(base: CommentedMap, override: CommentedMap | dict[str, Any]) -> CommentedMap:
    """Merge *override* into *base*, preserving base field order and comments.

    Rules:
    - Base keys: kept in their original order, comments preserved, values updated from override.
    - ``None`` value in override → key removed from result.
    - Nested dicts: recursively merged with the same rules.
    - New keys in override (absent from base): appended at end in override order,
      with their override comments when the override is a CommentedMap.
    """
    result = CommentedMap()
    last_base_key = None

    # Pass 1 — base keys in base order
    for key in list(base.keys()):
        if key in override:
            val = override[key]
            if val is None:
                continue  # None → delete key
            if isinstance(base[key], CommentedMap) and isinstance(val, dict | CommentedMap):
                result[key] = comment_aware_merge(base[key], val)
            else:
                result[key] = copy.deepcopy(val)
        else:
            result[key] = copy.deepcopy(base[key])

        # Carry over comment tokens attached to this key in the base
        if key in base.ca.items:
            result.ca.items[key] = base.ca.items[key]

        last_base_key = key

    # Collect new keys that pass 2 will append
    new_keys = [k for k in override if k not in base and override[k] is not None]

    if new_keys and last_base_key is not None and last_base_key in result.ca.items:
        # Strip the trailing block comment from the last base key before appending
        # override-only keys.  In the source file this token is a section separator
        # (e.g. "# zip_override_* — ...") that ruamel.yaml attaches as an
        # after-value token on the last key of the preceding block.  Keeping it
        # would place it between the base-inherited fields and the newly appended
        # fields, making the output look broken.
        #
        # Two sub-cases handled:
        # 1. Pure block comment (starts with \n, col 0)  → strip the whole token.
        # 2. Inline comment with a block comment appended → keep the inline line,
        #    truncate everything after the first \n.
        tokens = result.ca.items[last_base_key]
        if tokens is not None and len(tokens) > 2 and tokens[2] is not None:
            tok = tokens[2]
            val = tok.value
            if val.startswith("\n"):
                new_tok: object = None  # pure block separator — drop entirely
            else:
                first_nl = val.find("\n")
                stripped = val[: first_nl + 1] if first_nl != -1 else val
                if stripped != val:
                    new_tok = copy.copy(tok)
                    new_tok.value = stripped  # type: ignore[union-attr]
                else:
                    new_tok = tok  # unchanged
            if new_tok is not tok:
                # Copy the list so we don't mutate the shared source ca.items
                result.ca.items[last_base_key] = [tokens[0], tokens[1], new_tok, tokens[3] if len(tokens) > 3 else None]

    # Pass 2 — new keys from override not present in base
    for key in new_keys:
        result[key] = copy.deepcopy(override[key])
        # Carry over comment from override (only available for CommentedMap)
        if isinstance(override, CommentedMap) and key in override.ca.items:
            result.ca.items[key] = override.ca.items[key]

    # Block comment before the first key (e.g. "# section header")
    if base.ca.comment is not None:
        result.ca.comment = base.ca.comment

    return result
