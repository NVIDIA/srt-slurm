# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only primitives shared by direct execution stages."""

from __future__ import annotations

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
