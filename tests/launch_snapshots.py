# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden srun launch plans for every runnable recipe under ``examples/``.

Each example runs through the real ``SweepOrchestrator`` under
``srtctl.mock.run_mock_sweep``; every ``start_srun_process`` call is recorded and
rendered with run-specific paths replaced by placeholders. The rendered plans
live in ``tests/snapshots/launch/`` and ``tests/test_launch_snapshots.py``
fails when one drifts. Regenerate after an intended change with::

    make snapshots
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import shlex
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots" / "launch"
JOB_ID = "4242"

# Recipes that are not a single runnable job: overrides and sweeps expand into several.
NOT_A_JOB = {"examples/features/override.yaml", "examples/features/sweep.yaml"}

# Values generated fresh on every run, replaced by a stable placeholder.
_RANDOM_VALUES = (re.compile(r"moe_shared_[0-9a-f]{32}"),)
# The worker fingerprint script (core/fingerprint.py) is inlined into every worker preamble; keep one line.
_FINGERPRINT_SCRIPT = re.compile(r"(<<'__FINGERPRINT_EOF__'\n).*?(__FINGERPRINT_EOF__\n)", re.DOTALL)

_RECORDED_KEYS = (
    "nodes",
    "ntasks",
    "cpus_per_task",
    "nodelist",
    "output",
    "container_image",
    "container_mounts",
    "env_to_pass_through",
    "env_to_set",
    "env_to_unset",
    "srun_options",
    "srun_export_env",
    "overlap",
    "use_bash_wrapper",
    "mpi",
    "oversubscribe",
    "cpu_bind",
    "het_group",
)


def example_recipes() -> list[Path]:
    """Every example recipe that describes one job."""
    return sorted(p for p in EXAMPLES_DIR.rglob("*.yaml") if p.relative_to(REPO_ROOT).as_posix() not in NOT_A_JOB)


def snapshot_path(recipe: Path) -> Path:
    rel = recipe.relative_to(EXAMPLES_DIR).with_suffix("")
    return SNAPSHOT_DIR / (rel.as_posix().replace("/", "__") + ".txt")


@contextmanager
def _isolated_cluster(model_dir: Path, container_file: Path) -> Iterator[None]:
    """No srtslurm.yaml; the recipe's model and container-file paths resolve to existing placeholders."""
    from srtctl.core import config as config_module

    original_load = config_module.load_config

    def _load_with_fake_model(path: Path | str):
        cfg = original_load(path)
        container = cfg.model.container
        if container.startswith(("/", "./")):
            container = str(container_file)
        return dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, path=str(model_dir), container=container))

    with (
        patch.object(config_module, "load_cluster_config", lambda: None),
        patch.object(config_module, "load_config", _load_with_fake_model),
    ):
        yield


def _render_value(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return " {}"
        return "".join(f"\n    {k}: {v}".rstrip() for k, v in sorted((str(k), str(v)) for k, v in value.items()))
    if isinstance(value, list | tuple):
        return " " + " ".join(str(v) for v in value)
    return f" {value}"


def _render_call(call: dict[str, Any]) -> str:
    lines = [f"## {call.get('step_name') or '<unnamed>'}"]
    for key in _RECORDED_KEYS:
        value = call.get(key)
        if value is None or value == [] or value == {} or value == ():
            continue
        lines.append(f"{key}:{_render_value(value)}")
    preamble = call.get("bash_preamble")
    if preamble:
        preamble = _FINGERPRINT_SCRIPT.sub(r"\1<fingerprint script>\n\2", str(preamble))
        lines.append("bash_preamble: |")
        lines.extend(f"    {line}" for line in str(preamble).splitlines())
    lines.append("command: |")
    lines.extend(f"    {line}" for line in shlex.join(str(c) for c in call["command"]).splitlines())
    return "\n".join(lines)


def _normalize(text: str, replacements: dict[str, str]) -> str:
    for needle, placeholder in sorted(replacements.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(needle, placeholder)
    for pattern in _RANDOM_VALUES:
        text = pattern.sub("<random>", text)
    return text


def render_launch_plan(recipe: Path) -> str:
    """Run ``recipe`` through the mock orchestrator and render its srun calls."""
    from srtctl.core.config import load_config
    from srtctl.mock import MockOptions, run_mock_sweep

    calls: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="srt-launch-") as tmp:
        tmp_path = Path(tmp).resolve()
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        container_file = tmp_path / "container.sqsh"
        container_file.touch()
        output_dir = tmp_path / "output"
        prior_cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            with _isolated_cluster(model_dir, container_file):
                nodes = load_config(recipe).total_nodes
                options = MockOptions(
                    child_duration_s=0.0,
                    phase_pause_s=0.0,
                    nodelist=tuple(f"node-{i:02d}" for i in range(1, nodes + 3)),
                    on_srun=calls.append,
                    end_manual_hold=True,
                )
                exit_code = run_mock_sweep(config_path=recipe, output_dir=output_dir, job_id=JOB_ID, options=options)
        finally:
            os.chdir(prior_cwd)
        body = "\n\n".join(_render_call(c) for c in calls)
        body = _normalize(
            body,
            {
                str(model_dir): "<model>",
                str(container_file): "<container>",
                str(output_dir): "<output>",
                str(tmp_path): "<tmp>",
                str(REPO_ROOT): "<repo>",
                str(Path.home()): "<home>",
            },
        )
    header = f"# {recipe.relative_to(REPO_ROOT).as_posix()}\n# exit_code: {exit_code}\n# srun calls: {len(calls)}\n"
    return header + "\n" + body + "\n"


def main(argv: list[str]) -> int:
    """Write (default) or ``--check`` every snapshot."""
    logging.disable(logging.CRITICAL)
    check = "--check" in argv
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    expected = {snapshot_path(r): render_launch_plan(r) for r in example_recipes()}
    stale = [p for p, text in expected.items() if not p.exists() or p.read_text() != text]
    orphans = sorted(set(SNAPSHOT_DIR.glob("*.txt")) - set(expected))
    if check:
        for p in [*stale, *orphans]:
            print(f"stale launch snapshot: {p.relative_to(REPO_ROOT)}")
        return 1 if stale or orphans else 0
    for p in stale:
        p.write_text(expected[p])
    for p in orphans:
        p.unlink()
    print(f"{len(stale)} snapshot(s) written, {len(orphans)} removed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
