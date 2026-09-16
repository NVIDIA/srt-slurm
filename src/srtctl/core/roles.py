# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The 2.0 ``roles:`` authoring surface for recipe topology.

A recipe can group everything about a worker role under one block::

    roles:
      prefill:
        nodes: 2
        workers: 6
        gpus: 2
        env:
          PYTHONUNBUFFERED: "1"
        args:
          tensor-parallel-size: 2
      decode:
        nodes: colocate
        workers: 2
        gpus: 2
        args:
          tensor-parallel-size: 2

:func:`expand_roles` normalizes a ``roles:`` block into the internal fields the
runtime reads (``resources.prefill_workers`` / ``resources.gpus_per_prefill``,
``backend.prefill_environment``, ``backend.<engine>_config.prefill``, ...) before
schema load, so no downstream consumer changes. Those internal fields keep the
names of the pre-2.0 recipe layout; a recipe that spells them out itself is
rejected by ``srtctl.core.config.require_current_schema``, and ``srtctl migrate``
(``srtctl.core.migrate``) rewrites it into ``roles:``.

Role names are ``prefill``, ``decode``, and ``agg``. The aggregated role is
``agg`` here (matching ``resources.agg_*``); it maps to the ``aggregated`` key in
``backend.aggregated_environment`` and ``backend.<engine>_config.aggregated``.

``decode.nodes: colocate`` places the decode workers on the prefill nodes' spare
GPUs instead of reserving nodes for them. It normalizes to the internal sentinel
``resources.decode_nodes: 0``; under ``roles:`` the sentinel itself is rejected
so the intent is always spelled out. A colocated recipe must give ``gpus`` on
both prefill and decode, and :class:`~srtctl.core.schema.SrtConfig` rejects a
colocated layout whose workers do not fit on the prefill nodes.

The engine itself is a top-level ``engine:`` key (a string, or a mapping with
``type`` plus engine-wide knobs such as vLLM's ``connector``); it maps onto
``backend``. Per-role settings never belong in that mapping: ``roles.<role>.env``,
``.args``, ``.extra_args``, and ``.kv_events`` are the only spellings. Per-role
``kv_events`` maps onto ``backend.kv_events_config.<mode>`` and per-role
``sidecar`` onto ``dynamo.sidecar`` (every role must agree); per-role ``critical``
maps onto ``resources.<role>_critical``.
"""

from __future__ import annotations

from typing import Any

# backend.type -> the engine's per-mode CLI config key.
ENGINE_CONFIG_KEY: dict[str, str] = {
    "sglang": "sglang_config",
    "vllm": "vllm_config",
    "trtllm": "trtllm_config",
    "mocker": "mocker_config",
}
_ALL_ENGINE_CONFIG_KEYS = frozenset(ENGINE_CONFIG_KEY.values())

# roles: role name -> the mode name used in backend env / engine-config keys.
ROLE_TO_MODE: dict[str, str] = {"prefill": "prefill", "decode": "decode", "agg": "aggregated"}
ROLE_NAMES: tuple[str, ...] = ("prefill", "decode", "agg")

# ``roles.decode.nodes`` value meaning "share the prefill nodes"; expands to ``decode_nodes: 0``.
COLOCATE = "colocate"

# Per-role spec keys.
_ROLE_SPEC_KEYS = frozenset(
    {"nodes", "workers", "gpus", "env", "args", "extra_args", "engine", "kv_events", "sidecar", "critical"}
)

# Backend fields that hold a per-role setting. `engine:` only carries engine-wide knobs.
_PER_ROLE_ENGINE_KEYS = frozenset(
    {f"{mode}_{suffix}" for mode in ROLE_TO_MODE.values() for suffix in ("environment", "extra_args")}
    | _ALL_ENGINE_CONFIG_KEYS
    | {"kv_events_config", "mooncake_kv_store"}
)


def _engine_key(config: dict[str, Any]) -> str:
    backend = config.get("backend")
    btype = backend.get("type", "sglang") if isinstance(backend, dict) else "sglang"
    return ENGINE_CONFIG_KEY.get(btype, "sglang_config")


def expand_engine(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize the top-level ``engine:`` key (and per-role ``engine``) into ``backend``, in place.

    ``engine: sglang`` is shorthand for ``engine: {type: sglang}``; a mapping
    carries engine-wide knobs (``connector``, ``served_model_name``, ...) that
    are merged into ``backend``. ``backend.type`` may coexist only when it agrees.
    """
    engine = config.pop("engine", None)
    roles = config.get("roles")
    role_engines = {
        str(spec["engine"])
        for spec in (roles.values() if isinstance(roles, dict) else ())
        if isinstance(spec, dict) and spec.get("engine") is not None
    }
    if engine is None and not role_engines:
        return config

    if isinstance(engine, str):
        engine_map: dict[str, Any] = {"type": engine}
    elif isinstance(engine, dict):
        engine_map = dict(engine)
    elif engine is None:
        engine_map = {}
    else:
        raise TypeError("engine must be a string (the engine type) or a mapping with a 'type' key")

    if len(role_engines) > 1:
        raise ValueError(f"every role must use the same engine; got {', '.join(sorted(role_engines))}")
    if role_engines:
        (role_engine,) = role_engines
        if engine_map.get("type") not in (None, role_engine):
            raise ValueError(f"roles.*.engine {role_engine!r} conflicts with engine.type {engine_map['type']!r}")
        engine_map.setdefault("type", role_engine)
    for spec in roles.values() if isinstance(roles, dict) else ():
        if isinstance(spec, dict):
            spec.pop("engine", None)

    per_role = sorted(set(engine_map) & _PER_ROLE_ENGINE_KEYS)
    if per_role:
        raise ValueError(
            "engine: carries per-role settings (" + ", ".join(per_role) + "); those live under roles.<role> "
            "(env, args, extra_args, kv_events) or as a mooncake-master service, never on the engine"
        )

    backend = config.get("backend")
    if backend is None:
        backend = config["backend"] = {}
    if not isinstance(backend, dict):
        raise TypeError("backend must be a mapping")
    for key, value in engine_map.items():
        if key in backend and backend[key] != value:
            raise ValueError(f"engine.{key} conflicts with backend.{key}; set it in one place")
        backend[key] = value
    return config


def _internal_targets_present(config: dict[str, Any]) -> list[str]:
    """Internal fields, already set on the dict, that a ``roles:`` block would overwrite."""
    present: list[str] = []
    resources = config.get("resources")
    if isinstance(resources, dict):
        for role in ROLE_NAMES:
            for key in (f"{role}_nodes", f"{role}_workers", f"gpus_per_{role}", f"{role}_critical"):
                if key in resources:
                    present.append(f"resources.{key}")
    backend = config.get("backend")
    if isinstance(backend, dict):
        for mode in ("prefill", "decode", "aggregated"):
            for key in (f"{mode}_environment", f"{mode}_extra_args"):
                if key in backend:
                    present.append(f"backend.{key}")
        for engine_key in _ALL_ENGINE_CONFIG_KEYS:
            if backend.get(engine_key):
                present.append(f"backend.{engine_key}")
    roles = config.get("roles")
    specs = [spec for spec in roles.values() if isinstance(spec, dict)] if isinstance(roles, dict) else []
    if isinstance(backend, dict) and "kv_events_config" in backend and any("kv_events" in spec for spec in specs):
        present.append("backend.kv_events_config")
    dynamo = config.get("dynamo")
    if isinstance(dynamo, dict) and "sidecar" in dynamo and any("sidecar" in spec for spec in specs):
        present.append("dynamo.sidecar")
    return present


def _expand_nodes(role_name: str, value: Any) -> int:
    """Map ``roles.<role>.nodes`` onto ``resources.<role>_nodes``.

    ``colocate`` is only meaningful for ``decode`` (share the prefill nodes) and
    becomes the v1 sentinel ``0``. The bare ``0`` is rejected under ``roles:``.
    """
    if isinstance(value, str):
        if value.strip().lower() == COLOCATE:
            if role_name != "decode":
                raise ValueError(f"roles.{role_name}.nodes: only the decode role can colocate (on the prefill nodes)")
            return 0
        raise ValueError(f"roles.{role_name}.nodes must be a positive integer or 'colocate'; got {value!r}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"roles.{role_name}.nodes must be a positive integer or 'colocate'; got {value!r}")
    if value == 0:
        if role_name == "decode":
            raise ValueError("roles.decode.nodes: 0 is not accepted; write nodes: colocate to share the prefill nodes")
        raise ValueError(f"roles.{role_name}.nodes must be at least 1; got 0")
    if value < 0:
        raise ValueError(f"roles.{role_name}.nodes must be at least 1; got {value}")
    return value


def expand_roles(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize a ``roles:`` block into the internal fields, in place.

    A no-op when there is no ``roles:`` key. Rejects a dict that already carries
    the internal ``prefill_*`` / ``<engine>_config`` fields ``roles:`` expands
    into, so a role setting is never silently overwritten.
    """
    expand_engine(config)
    roles = config.get("roles")
    if not isinstance(roles, dict):
        return config

    collisions = _internal_targets_present(config)
    if collisions:
        raise ValueError(
            "roles: cannot be combined with the fields it expands into: "
            + ", ".join(sorted(collisions))
            + ". Put the setting under roles.<role> only."
        )

    engine_key = _engine_key(config)
    resources = config.setdefault("resources", {})
    backend = config.setdefault("backend", {})

    for role_name, spec in roles.items():
        if role_name not in ROLE_TO_MODE:
            raise ValueError(f"unknown role {role_name!r}; valid roles are {', '.join(ROLE_NAMES)}")
        if not isinstance(spec, dict):
            raise TypeError(f"roles.{role_name} must be a mapping")
        unknown = set(spec) - _ROLE_SPEC_KEYS
        if unknown:
            raise ValueError(f"roles.{role_name} has unknown keys: {', '.join(sorted(unknown))}")

        mode = ROLE_TO_MODE[role_name]
        if "nodes" in spec:
            resources[f"{role_name}_nodes"] = _expand_nodes(role_name, spec["nodes"])
        if "workers" in spec:
            resources[f"{role_name}_workers"] = spec["workers"]
        if "gpus" in spec:
            resources[f"gpus_per_{role_name}"] = spec["gpus"]
        if "critical" in spec:
            if not isinstance(spec["critical"], bool):
                raise TypeError(f"roles.{role_name}.critical must be a boolean")
            resources[f"{role_name}_critical"] = spec["critical"]
        if "env" in spec:
            backend[f"{mode}_environment"] = spec["env"]
        if "args" in spec:
            engine_cfg = backend.setdefault(engine_key, {})
            engine_cfg[mode] = {**(engine_cfg.get(mode) or {}), **spec["args"]}
        if "extra_args" in spec:
            backend[f"{mode}_extra_args"] = spec["extra_args"]
        if "kv_events" in spec:
            kv_events = backend.setdefault("kv_events_config", {})
            if not isinstance(kv_events, dict):
                raise ValueError("roles.*.kv_events cannot be combined with a boolean backend.kv_events_config")
            kv_events[mode] = spec["kv_events"]

    decode_spec = roles.get("decode")
    if isinstance(decode_spec, dict) and resources.get("decode_nodes") == 0:
        # A colocated split cannot be derived: the per-node formula would hand prefill every GPU
        # and the decode size would silently inherit it. Both roles must state their worker size.
        missing = [role for role in ("prefill", "decode") if "gpus" not in (roles.get(role) or {})]
        if missing:
            raise ValueError(
                "roles.decode.nodes: colocate requires an explicit gpus: on both prefill and decode "
                f"(missing on {', '.join(missing)}); the GPU split is validated against the prefill nodes at load"
            )

    sidecars = {bool(spec["sidecar"]) for spec in roles.values() if isinstance(spec, dict) and "sidecar" in spec}
    if len(sidecars) > 1:
        raise ValueError("roles.*.sidecar must agree across roles (the Dynamo sidecar mode is job-wide)")
    if sidecars:
        config.setdefault("dynamo", {})["sidecar"] = sidecars.pop()

    config.pop("roles", None)
    return config
