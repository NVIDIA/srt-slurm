# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only primitives shared by direct execution stages."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ManagedProcess:
    """A direct-run subprocess and its process-group leader."""

    label: str
    process: subprocess.Popen[Any]
    log_path: Path


class DirectRunInterrupted(Exception):
    """Signal delivered to the supervisor while it owns child process groups."""

    def __init__(self, signal_number: int) -> None:
        self.signal_number = signal_number
        super().__init__(f"received signal {signal_number}")


def rust_toolchain(path: Path) -> str | None:
    """Return the source-pinned Rust toolchain, when SGLang specifies one."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("channel") and "=" in stripped:
            return stripped.split("=", 1)[1].strip().strip('"')
    return None


def run_capture(args: list[str]) -> str:
    """Run a command and return its stripped stdout."""
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def shell_quote(value: str) -> str:
    """Return a minimal shell-safe representation for sidecar command files."""
    if value and all(character.isalnum() or character in "@%_+=:,./-" for character in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


# ---------------------------------------------------------------------------
# Dynamo worker-selection policy catalog linking
#
# Canonical home for these values: this module is stdlib-only, so both the
# control-plane schema and the in-container direct runner can share them.
# ---------------------------------------------------------------------------

# Dynamo links exactly one worker-selection policy catalog through this optional
# dependency alias in ``lib/bindings/python/Cargo.toml``, behind the
# ``custom-policy`` cargo feature.
POLICY_CATALOG_DEPENDENCY = "dynamo-worker-selection-policy-catalog"
KV_ROUTER_DEPENDENCY = "dynamo-kv-router"
# Filename the resolved policy configuration is published under, in the same
# build-cache directory as the wheel, so Slurm and direct runs agree on it.
POLICY_CATALOG_CONFIG_NAME = "worker-selection.yaml"
# Directory the catalog crate is materialized into inside a build sandbox.
POLICY_CATALOG_DIRNAME = "policy-catalog"


def policy_catalog_dependency_line(package: str, crate_dir: str) -> str:
    """Return the Cargo declaration that links *package* as Dynamo's catalog."""
    return f'{POLICY_CATALOG_DEPENDENCY} = {{ package = "{package}", path = "{crate_dir}", optional = true }}'


def kv_router_dependency_line(kv_router_dir: str) -> str:
    """Return the Cargo declaration pinning a catalog to this Dynamo checkout.

    The published catalog crates depend on ``dynamo-kv-router`` from git. Left
    alone, cargo resolves that as a second source of the same crate and the
    plugin's ``WorkerSelectionPolicy`` types no longer unify with the ones the
    bindings were compiled against. Repointing it at the checkout being built
    keeps exactly one ``dynamo-kv-router`` in the graph.
    """
    return f'{KV_ROUTER_DEPENDENCY} = {{ path = "{kv_router_dir}", features = ["standalone-selection"] }}'


def apply_dependency_override(text: str, crate: str, replacement: str) -> str:
    """Replace *crate*'s declaration line in one Cargo.toml body."""
    return re.sub(rf"(?m)^{re.escape(crate)}[ \t]*=.*$", lambda _match: replacement, text)
