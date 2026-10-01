# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary of the ``roles:`` block: role names, their engine-side mode names, per-role keys.

A recipe groups everything about a worker role under ``roles.<role>``
(:class:`srtctl.core.schema.RoleConfig`); the roles are ``prefill``, ``decode``
and ``agg``. The engine dataclasses still carry per-mode fields named after the
modes (``prefill`` / ``decode`` / ``aggregated``): :data:`ROLE_TO_MODE` maps a
role onto its mode, and :data:`PER_ROLE_ENGINE_KEYS` lists the engine fields a
recipe may not set directly because ``roles.<role>`` is their one spelling.
``srtctl migrate`` uses the same tables to rewrite a pre-2.0 recipe.
"""

from __future__ import annotations

# engine type -> the engine's per-mode CLI config field.
ENGINE_CONFIG_KEY: dict[str, str] = {
    "atom": "atom_config",
    "sglang": "sglang_config",
    "tilert": "tilert_config",
    "tokenspeed": "tokenspeed_config",
    "vllm": "vllm_config",
    "trtllm": "trtllm_config",
    "mocker": "mocker_config",
}

# role name -> the mode name used in the engine's per-mode fields.
ROLE_TO_MODE: dict[str, str] = {"prefill": "prefill", "decode": "decode", "agg": "aggregated"}
ROLE_NAMES: tuple[str, ...] = ("prefill", "decode", "agg")

# ``roles.decode.nodes`` value meaning "share the prefill nodes".
COLOCATE = "colocate"

# Engine fields that hold a per-role setting. An engine mapping carries engine-wide knobs
# only (mooncake_kv_store is one: the master is shared, so it may ride on the engine or be
# a service); these are refused on ``engine:`` and ``roles.<role>.engine``.
PER_ROLE_ENGINE_KEYS: frozenset[str] = frozenset(
    {f"{mode}_{suffix}" for mode in ROLE_TO_MODE.values() for suffix in ("environment", "extra_args")}
    | set(ENGINE_CONFIG_KEY.values())
    | {"kv_events_config"}
)
