# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cluster-side model and container materialization helpers."""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml.comments import CommentedMap

from srtctl.core.config import find_cluster_config_path, generate_override_configs
from srtctl.core.yaml_utils import dump_yaml_with_comments, load_yaml_with_comments

_DEFAULT_SRUN_TEMPLATE = "srun --nodes=1 --ntasks=1 bash -lc {q_command}"


@dataclass(frozen=True)
class AssetAction:
    kind: str
    alias: str | None
    source: str | None
    target: str | None
    mode: str
    status: str
    message: str
    command: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "alias": self.alias,
            "source": self.source,
            "target": self.target,
            "mode": self.mode,
            "status": self.status,
            "message": self.message,
            "command": self.command,
        }


@dataclass(frozen=True)
class EnsureAssetsResult:
    cluster_config_path: str
    dry_run: bool
    changed: bool
    actions: list[AssetAction]
    model_paths: dict[str, str]
    containers: dict[str, str]

    @property
    def ok(self) -> bool:
        failures = {"error", "missing-source", "missing-target-root", "missing-template"}
        return all(action.status not in failures for action in self.actions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "cluster_config_path": self.cluster_config_path,
            "dry_run": self.dry_run,
            "changed": self.changed,
            "actions": [action.as_dict() for action in self.actions],
            "model_paths": dict(self.model_paths),
            "containers": dict(self.containers),
        }


def ensure_assets_for_config_variants(
    raw_config: dict[str, Any],
    *,
    selector: str | None = None,
    cluster_config_path: Path | None = None,
    dry_run: bool = False,
    mode: str | None = None,
) -> EnsureAssetsResult:
    """Ensure model/container aliases referenced by one recipe or override file."""
    variants = (
        generate_override_configs(raw_config, selector=selector) if "base" in raw_config else [("base", raw_config)]
    )
    return ensure_assets(
        [variant for _suffix, variant in variants],
        cluster_config_path=cluster_config_path,
        dry_run=dry_run,
        mode=mode,
    )


def ensure_assets(
    configs: list[dict[str, Any]],
    *,
    cluster_config_path: Path | None = None,
    dry_run: bool = False,
    mode: str | None = None,
) -> EnsureAssetsResult:
    """Materialize missing aliases and update the cluster-owned srtslurm.yaml.

    The recipe remains unchanged. If `model.path` or `model.container` names an
    alias that is not present, this function infers the pull source from the
    recipe `identity` block, chooses a target under the configured root, runs
    the configured command template, and records the alias.
    """
    path = cluster_config_path or find_cluster_config_path()
    if path is None:
        raise FileNotFoundError("No srtslurm.yaml found. Set SRTSLURM_CONFIG or run from the repo/cluster config tree.")
    path = Path(path)

    with _asset_lock(path):
        cluster_cm = load_yaml_with_comments(path)
        cluster = _plain_mapping(cluster_cm)
        policy = _plain_mapping(cluster.get("asset_materialization") or {})
        selected_mode = _select_mode(str(mode or policy.get("mode") or "login"), policy)

        model_paths = _string_mapping(cluster.get("model_paths"))
        containers = _string_mapping(cluster.get("containers"))

        actions: list[AssetAction] = []
        changed = False
        for config in configs:
            model_action, model_changed = _ensure_model_alias(
                config,
                model_paths=model_paths,
                policy=policy,
                selected_mode=selected_mode,
                dry_run=dry_run,
            )
            if model_action:
                actions.append(model_action)
                changed = changed or model_changed

            container_action, container_changed = _ensure_container_alias(
                config,
                containers=containers,
                policy=policy,
                selected_mode=selected_mode,
                dry_run=dry_run,
            )
            if container_action:
                actions.append(container_action)
                changed = changed or container_changed

        if changed and not dry_run:
            _set_mapping(cluster_cm, "model_paths", model_paths)
            _set_mapping(cluster_cm, "containers", containers)
            _atomic_write_yaml(path, cluster_cm)

    return EnsureAssetsResult(
        cluster_config_path=str(path),
        dry_run=dry_run,
        changed=changed,
        actions=actions,
        model_paths=model_paths,
        containers=containers,
    )


def _ensure_model_alias(
    config: dict[str, Any],
    *,
    model_paths: dict[str, str],
    policy: dict[str, Any],
    selected_mode: str,
    dry_run: bool,
) -> tuple[AssetAction | None, bool]:
    model = _plain_mapping(config.get("model") or {})
    raw = _string_or_none(model.get("path"))
    if not raw or not _is_model_alias(raw):
        return None, False

    identity = _plain_mapping(config.get("identity") or {})
    identity_model = _plain_mapping(identity.get("model") or {})
    source = _string_or_none(identity_model.get("repo"))
    revision = _string_or_none(identity_model.get("revision"))
    if source is None and raw.startswith("hf:"):
        source = raw[3:]

    existing_target = model_paths.get(raw)
    target = existing_target or _default_target(policy.get("models_root"), raw, suffix=None)
    if target and Path(os.path.expandvars(target)).expanduser().exists():
        if existing_target:
            return (
                AssetAction("model", raw, source, target, selected_mode, "exists", f"model alias {raw!r} exists"),
                False,
            )
        model_paths[raw] = target
        return (
            AssetAction(
                "model", raw, source, target, selected_mode, "registered", f"registered existing model {raw!r}"
            ),
            True,
        )

    if source is None:
        return (
            AssetAction(
                "model",
                raw,
                None,
                target,
                selected_mode,
                "missing-source",
                f"model alias {raw!r} is missing and no identity.model.repo was provided",
            ),
            False,
        )
    if target is None:
        return (
            AssetAction(
                "model",
                raw,
                source,
                None,
                selected_mode,
                "missing-target-root",
                "asset_materialization.models_root is required to create a new model alias",
            ),
            False,
        )

    command = _materialization_command(
        template=_string_or_none(policy.get("model_pull_template")),
        srun_template=_string_or_none(policy.get("srun_template")),
        mode=selected_mode,
        alias=raw,
        source=source,
        target=target,
        revision=revision,
    )
    if command is None:
        return (
            AssetAction(
                "model",
                raw,
                source,
                target,
                selected_mode,
                "missing-template",
                "asset_materialization.model_pull_template is required to pull missing models",
            ),
            False,
        )
    if dry_run:
        return (
            AssetAction(
                "model", raw, source, target, selected_mode, "would-pull", "would pull and register model", command
            ),
            False,
        )

    _run_command(command)
    if not Path(os.path.expandvars(target)).expanduser().is_dir():
        return (
            AssetAction(
                "model",
                raw,
                source,
                target,
                selected_mode,
                "error",
                "model pull command completed but target directory does not exist",
                command,
            ),
            False,
        )
    model_paths[raw] = target
    return (
        AssetAction("model", raw, source, target, selected_mode, "pulled", "pulled and registered model", command),
        True,
    )


def _ensure_container_alias(
    config: dict[str, Any],
    *,
    containers: dict[str, str],
    policy: dict[str, Any],
    selected_mode: str,
    dry_run: bool,
) -> tuple[AssetAction | None, bool]:
    model = _plain_mapping(config.get("model") or {})
    raw = _string_or_none(model.get("container"))
    if not raw or not _is_container_alias(raw):
        return None, False

    identity = _plain_mapping(config.get("identity") or {})
    identity_container = _plain_mapping(identity.get("container") or {})
    source = _string_or_none(identity_container.get("image"))

    existing_target = containers.get(raw)
    target = existing_target or _default_target(policy.get("containers_root"), raw, suffix=".sqsh")
    if target and Path(os.path.expandvars(target)).expanduser().exists():
        if existing_target:
            return (
                AssetAction(
                    "container", raw, source, target, selected_mode, "exists", f"container alias {raw!r} exists"
                ),
                False,
            )
        containers[raw] = target
        return (
            AssetAction(
                "container",
                raw,
                source,
                target,
                selected_mode,
                "registered",
                f"registered existing container {raw!r}",
            ),
            True,
        )

    if source is None:
        return (
            AssetAction(
                "container",
                raw,
                None,
                target,
                selected_mode,
                "missing-source",
                f"container alias {raw!r} is missing and no identity.container.image was provided",
            ),
            False,
        )
    if target is None:
        return (
            AssetAction(
                "container",
                raw,
                source,
                None,
                selected_mode,
                "missing-target-root",
                "asset_materialization.containers_root is required to create a new container alias",
            ),
            False,
        )

    command = _materialization_command(
        template=_string_or_none(policy.get("container_pull_template")),
        srun_template=_string_or_none(policy.get("srun_template")),
        mode=selected_mode,
        alias=raw,
        source=source,
        target=target,
        revision=None,
    )
    if command is None:
        return (
            AssetAction(
                "container",
                raw,
                source,
                target,
                selected_mode,
                "missing-template",
                "asset_materialization.container_pull_template is required to pull missing containers",
            ),
            False,
        )
    if dry_run:
        return (
            AssetAction(
                "container",
                raw,
                source,
                target,
                selected_mode,
                "would-pull",
                "would pull and register container",
                command,
            ),
            False,
        )

    _run_command(command)
    if not Path(os.path.expandvars(target)).expanduser().is_file():
        return (
            AssetAction(
                "container",
                raw,
                source,
                target,
                selected_mode,
                "error",
                "container pull command completed but target file does not exist",
                command,
            ),
            False,
        )
    containers[raw] = target
    return (
        AssetAction(
            "container",
            raw,
            source,
            target,
            selected_mode,
            "pulled",
            "pulled and registered container",
            command,
        ),
        True,
    )


def _materialization_command(
    *,
    template: str | None,
    srun_template: str | None,
    mode: str,
    alias: str,
    source: str,
    target: str,
    revision: str | None,
) -> str | None:
    if not template:
        return None
    values = _template_values(alias=alias, source=source, target=target, revision=revision)
    command = template.format_map(values)
    if mode != "srun":
        return command
    wrapper = srun_template or _DEFAULT_SRUN_TEMPLATE
    wrapped_values = {**values, "command": command, "q_command": shlex.quote(command), "py_command": repr(command)}
    return wrapper.format_map(wrapped_values)


def _template_values(*, alias: str, source: str, target: str, revision: str | None) -> dict[str, str]:
    values = {
        "alias": alias,
        "source": source,
        "target": target,
        "revision": revision or "",
    }
    quoted = {f"q_{key}": shlex.quote(value) for key, value in values.items()}
    py_quoted = {f"py_{key}": repr(value) for key, value in values.items()}
    return {**values, **quoted, **py_quoted}


def _select_mode(mode: str, policy: dict[str, Any]) -> str:
    normalized = mode.lower()
    if normalized not in {"auto", "login", "srun"}:
        raise ValueError("asset materialization mode must be one of: auto, login, srun")
    if normalized != "auto":
        return normalized

    probe = _string_or_none(policy.get("login_probe_command"))
    if probe is None:
        return "login"
    result = subprocess.run(probe, shell=True, capture_output=True, text=True, check=False)
    return "login" if result.returncode == 0 else "srun"


def _run_command(command: str) -> None:
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"asset materialization command failed: {details}")


@contextlib.contextmanager
def _asset_lock(cluster_config_path: Path) -> Iterator[None]:
    raw = _load_raw_cluster_config(cluster_config_path)
    policy = _plain_mapping(raw.get("asset_materialization") or {})
    lock_path = _string_or_none(policy.get("lock_path")) or f"{cluster_config_path}.asset.lock"
    lock_file = Path(os.path.expandvars(lock_path)).expanduser()
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _load_raw_cluster_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"srtslurm.yaml not found: {path}")
    loaded = load_yaml_with_comments(path)
    return _plain_mapping(loaded)


def _atomic_write_yaml(path: Path, data: Any) -> None:
    fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w") as handle:
            dump_yaml_with_comments(data, handle)
        os.replace(temp_path, path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(temp_path)


def _set_mapping(data: CommentedMap, key: str, values: dict[str, str]) -> None:
    existing = data.get(key)
    if isinstance(existing, CommentedMap):
        existing.clear()
        for item_key in sorted(values):
            existing[item_key] = values[item_key]
        return
    replacement = CommentedMap()
    for item_key in sorted(values):
        replacement[item_key] = values[item_key]
    data[key] = replacement


def _plain_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if v is not None}


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _default_target(root: Any, alias: str, *, suffix: str | None) -> str | None:
    root_str = _string_or_none(root)
    if root_str is None:
        return None
    name = _safe_asset_name(alias)
    if suffix and not name.endswith(suffix):
        name = f"{name}{suffix}"
    return str(Path(os.path.expandvars(root_str)).expanduser() / name)


def _safe_asset_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return cleaned or "asset"


def _is_model_alias(value: str) -> bool:
    if value.startswith(("hf:", "/", "./", "../", "~")):
        return False
    return "/" not in value


def _is_container_alias(value: str) -> bool:
    if value.startswith(("/", "./", "../", "~")):
        return False
    if "://" in value:
        return False
    if "/" in value:
        return False
    return ":" not in value
