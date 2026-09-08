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
        nodes: 0
        workers: 2
        gpus: 2
        args:
          tensor-parallel-size: 2

instead of spreading it across ``resources.prefill_workers`` /
``resources.gpus_per_prefill``, ``backend.prefill_environment``, and
``backend.<engine>_config.prefill``. :func:`expand_roles` normalizes a
``roles:`` block back into those existing internal fields before schema load, so
no downstream consumer changes; :func:`roles_from_legacy` is the inverse, used to
migrate a v1 recipe and to prove the two forms are equivalent.

Role names are ``prefill``, ``decode``, and ``agg``. The aggregated role is
``agg`` here (matching ``resources.agg_*``); it maps to the ``aggregated`` key in
``backend.aggregated_environment`` and ``backend.<engine>_config.aggregated``.

The engine itself is a top-level ``engine:`` key in 2.0 (a string, or a mapping
with ``type`` plus engine-wide knobs such as vLLM's ``connector``); it maps onto
``backend``. Per-role ``kv_events`` maps onto ``backend.kv_events_config.<mode>``
and per-role ``sidecar`` onto ``dynamo.sidecar`` (every role must agree). A v2
recipe therefore needs no ``backend:`` block at all; a v1 recipe still loads.
"""

from __future__ import annotations

import copy
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

# Per-role spec keys.
_ROLE_SPEC_KEYS = frozenset({"nodes", "workers", "gpus", "env", "args", "extra_args", "engine", "kv_events", "sidecar"})


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


def _legacy_targets_present(config: dict[str, Any]) -> list[str]:
    """Internal v1 fields that would collide with a ``roles:`` block."""
    present: list[str] = []
    resources = config.get("resources")
    if isinstance(resources, dict):
        for role in ROLE_NAMES:
            for key in (f"{role}_nodes", f"{role}_workers", f"gpus_per_{role}"):
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


def expand_roles(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize a ``roles:`` block into the existing internal fields, in place.

    A no-op when there is no ``roles:`` key. Rejects mixing ``roles:`` with the
    v1 ``prefill_*`` / ``<engine>_config`` fields it expands into.
    """
    expand_engine(config)
    roles = config.get("roles")
    if not isinstance(roles, dict):
        return config

    collisions = _legacy_targets_present(config)
    if collisions:
        raise ValueError(
            "roles: cannot be combined with the fields it expands into: "
            + ", ".join(sorted(collisions))
            + ". Use roles: or the legacy fields, not both."
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
            resources[f"{role_name}_nodes"] = spec["nodes"]
        if "workers" in spec:
            resources[f"{role_name}_workers"] = spec["workers"]
        if "gpus" in spec:
            resources[f"gpus_per_{role_name}"] = spec["gpus"]
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

    sidecars = {bool(spec["sidecar"]) for spec in roles.values() if isinstance(spec, dict) and "sidecar" in spec}
    if len(sidecars) > 1:
        raise ValueError("roles.*.sidecar must agree across roles (the Dynamo sidecar mode is job-wide)")
    if sidecars:
        config.setdefault("dynamo", {})["sidecar"] = sidecars.pop()

    config.pop("roles", None)
    return config


def roles_from_legacy(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``config`` with legacy role fields folded into a ``roles:`` block.

    The inverse of :func:`expand_roles`, used by ``srtctl migrate`` and by tests
    that assert the two forms are equivalent. Only non-empty roles appear.
    """
    result = copy.deepcopy(config)
    resources = result.get("resources") if isinstance(result.get("resources"), dict) else {}
    backend = result.get("backend") if isinstance(result.get("backend"), dict) else {}
    engine_key = _engine_key(result)
    engine_cfg = backend.get(engine_key) if isinstance(backend.get(engine_key), dict) else {}

    roles: dict[str, dict[str, Any]] = {}
    for role_name in ROLE_NAMES:
        mode = ROLE_TO_MODE[role_name]
        spec: dict[str, Any] = {}
        if f"{role_name}_nodes" in resources:
            spec["nodes"] = resources[f"{role_name}_nodes"]
        if f"{role_name}_workers" in resources:
            spec["workers"] = resources[f"{role_name}_workers"]
        if f"gpus_per_{role_name}" in resources:
            spec["gpus"] = resources[f"gpus_per_{role_name}"]
        if backend.get(f"{mode}_environment"):
            spec["env"] = backend[f"{mode}_environment"]
        if engine_cfg.get(mode):
            spec["args"] = engine_cfg[mode]
        if backend.get(f"{mode}_extra_args"):
            spec["extra_args"] = backend[f"{mode}_extra_args"]
        if spec:
            roles[role_name] = spec

    kv_events_config = backend.get("kv_events_config")
    if isinstance(kv_events_config, dict):
        for role_name in ROLE_NAMES:
            mode = ROLE_TO_MODE[role_name]
            if mode in kv_events_config:
                roles.setdefault(role_name, {})["kv_events"] = kv_events_config[mode]
        backend.pop("kv_events_config", None)
    elif kv_events_config is True:
        # A bare `true` enables prefill and decode everywhere, and aggregated on SGLang.
        covered = ("prefill", "decode", "agg") if backend.get("type", "sglang") == "sglang" else ("prefill", "decode")
        for role_name in covered:
            if role_name in roles:
                roles[role_name]["kv_events"] = True
        backend.pop("kv_events_config", None)

    dynamo = result.get("dynamo") if isinstance(result.get("dynamo"), dict) else None
    if dynamo is not None and dynamo.get("sidecar") is True and roles:
        for spec in roles.values():
            spec["sidecar"] = True
        dynamo.pop("sidecar")

    if not roles:
        return result

    # Strip the folded fields from resources/backend.
    for role_name in ROLE_NAMES:
        mode = ROLE_TO_MODE[role_name]
        for key in (f"{role_name}_nodes", f"{role_name}_workers", f"gpus_per_{role_name}"):
            resources.pop(key, None)
        backend.pop(f"{mode}_environment", None)
        backend.pop(f"{mode}_extra_args", None)
        if isinstance(engine_cfg, dict):
            engine_cfg.pop(mode, None)
    if isinstance(engine_cfg, dict) and not engine_cfg:
        backend.pop(engine_key, None)
    if isinstance(resources, dict) and not resources:
        result.pop("resources", None)

    result["roles"] = roles
    # The engine moves to the top level: a bare string when nothing else is left in backend.
    if isinstance(result.get("backend"), dict):
        remaining = dict(result["backend"])
        engine_type = remaining.pop("type", None)
        if engine_type is not None or remaining:
            result["engine"] = engine_type if not remaining else {"type": engine_type, **remaining}
        result.pop("backend", None)
    return result
