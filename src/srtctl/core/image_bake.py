# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bake srtctl's runtime dependencies into a container image.

Without this, every job reinstalls the same things before it can serve a single
request: dynamo from pip and the sa-bench Python deps. Doing it once into a new
squashfs takes that off the critical path of every run.

The image surgery is pyxis': ``--container-save`` exports the container
filesystem to a new .sqsh when the step ends, so nothing here has to understand
the squashfs format.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from srtctl.benchmarks.base import SCRIPTS_DIR
from srtctl.core.config import get_srtslurm_setting
from srtctl.core.schema import DynamoConfig
from srtctl.core.slurm import CONTAINER_REMAP_ROOT_EXPORT, get_slurm_job_id, start_srun_process

MANIFEST_VERSION = 1

# Installing a couple of pip packages and writing a ~7 GB squashfs; the cluster
# default (hours) would hold a node far longer than this needs.
DEFAULT_TIME_LIMIT = "0:30:00"

SA_BENCH_DEPS_SCRIPT = SCRIPTS_DIR / "sa-bench" / "deps.sh"

# --container-save captures everything written, so package caches and build
# leftovers would be baked in and inflate the image by gigabytes.
_CLEANUP = (
    "pip cache purge >/dev/null 2>&1 || true\n"
    "rm -rf /root/.cache/pip /tmp/* >/dev/null 2>&1 || true\n"
    'echo "bake-image: install complete"'
)


def _parse_bash_assignment(text: str, name: str, script: Path) -> str:
    match = re.search(rf"^{name}=(.+)$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"{name} not found in {script}")
    return match.group(1).strip()


def read_sa_bench_deps(script: Path = SA_BENCH_DEPS_SCRIPT) -> tuple[tuple[str, ...], str]:
    """Return sa-bench's ``(pip packages, import list)`` as declared in deps.sh.

    bench.sh owns the list; reading it here is what keeps a baked image from
    drifting away from what the benchmark checks for at runtime.
    """
    text = script.read_text()
    packages = _parse_bash_assignment(text, "SA_BENCH_DEPS", script)
    if not (packages.startswith("(") and packages.endswith(")")):
        raise ValueError(f"SA_BENCH_DEPS in {script} is not a bash array")
    imports = _parse_bash_assignment(text, "SA_BENCH_IMPORTS", script)
    return tuple(shlex.split(packages[1:-1])), " ".join(shlex.split(imports))


def default_output_image(source: Path, *, dynamo_version: str | None, sa_bench: bool) -> Path:
    """Name the output after what went into it, next to the source image.

    A directory of ``*.sqsh`` files is unreadable without this; the tags say
    what each one contains without opening the manifest.
    """
    tags = []
    if dynamo_version:
        tags.append(f"dynamo{dynamo_version}")
    if sa_bench:
        tags.append("sa-bench")
    return source.with_name(f"{source.stem}+{'+'.join(tags)}.sqsh")


@dataclass(frozen=True)
class BakePlan:
    """What to install into which image, and where the result goes."""

    source_image: Path
    output_image: Path
    dynamo_version: str | None = None
    sa_bench: bool = False
    sa_bench_deps: tuple[str, ...] = ()
    sa_bench_imports: str = ""
    # Only used when srun has to allocate for itself.
    time_limit: str = DEFAULT_TIME_LIMIT
    slurm_overrides: Mapping[str, str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dynamo_version and not self.sa_bench:
            raise ValueError("nothing to install; pass --dynamo and/or --sa-bench")
        if self.sa_bench and not self.sa_bench_deps:
            raise ValueError("sa-bench requested without a package list")
        if self.source_image == self.output_image:
            raise ValueError("output image must differ from the source image")

    @property
    def manifest_path(self) -> Path:
        return self.output_image.with_suffix(".manifest.json")

    def install_script(self) -> str:
        """Bash run inside the container, before pyxis saves it."""
        steps = ["set -euo pipefail"]

        if self.dynamo_version:
            # serialize=False: a single task, and the flock sentinel would be
            # captured in the image and make later jobs skip their own install.
            steps.append(DynamoConfig(version=self.dynamo_version).get_install_commands(serialize=False))
            steps.append(f"pip show ai-dynamo >/dev/null && echo 'bake-image: ai-dynamo {self.dynamo_version} present'")

        if self.sa_bench:
            packages = " ".join(self.sa_bench_deps)
            steps.append("echo 'bake-image: installing sa-bench deps...'")
            steps.append(f"pip install --break-system-packages --quiet {packages}")
            steps.append(f'python3 -c "import {self.sa_bench_imports}"')
            steps.append("echo 'bake-image: sa-bench deps importable'")

        steps.append(_CLEANUP)
        return "\n".join(steps)

    def manifest(self) -> dict:
        return {
            "version": MANIFEST_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_image": str(self.source_image),
            "output_image": str(self.output_image),
            "installed": {
                "ai-dynamo": self.dynamo_version,
                "sa_bench_deps": list(self.sa_bench_deps) if self.sa_bench else None,
            },
        }


def build_srun_options(plan: BakePlan) -> dict[str, str]:
    """srun flags for the bake step.

    Inside an existing allocation the step just joins it. Standalone, srun
    allocates for itself, which needs the same account/partition/time that a
    submitted job would get from srtslurm.yaml.
    """
    options = {"container-save": str(plan.output_image), "container-writable": ""}
    if get_slurm_job_id():
        return options

    options["time"] = plan.time_limit
    for flag, key in (("account", "default_account"), ("partition", "default_partition")):
        value = plan.slurm_overrides.get(flag) or get_srtslurm_setting(key)
        if value:
            options[flag] = str(value)
    return options


def srun_preview(plan: BakePlan) -> str:
    """The srun invocation, for dry-run output."""
    options = build_srun_options(plan)
    rendered = " ".join(f"--{k}={v}" if v else f"--{k}" for k, v in options.items())
    overlap = "--overlap " if get_slurm_job_id() else ""
    return (
        f"srun {overlap}--nodes 1 --ntasks 1 "
        f"--container-image {plan.source_image} "
        "--no-container-entrypoint --no-container-mount-home "
        f"{rendered} "
        f"--export=ALL,{','.join(f'{k}={v}' for k, v in CONTAINER_REMAP_ROOT_EXPORT.items())} "
        "bash -c '<install script>'"
    )


def bake_image(plan: BakePlan, *, dry_run: bool = False) -> int:
    """Run the install inside the source image and save the result.

    Returns the srun exit code (0 on success). Works both inside an allocation
    (joins it, no queue wait) and standalone, where srun allocates a node using
    the cluster defaults.
    """
    script = plan.install_script()

    if dry_run:
        print(srun_preview(plan))
        print()
        print(script)
        return 0

    if not plan.source_image.exists():
        raise FileNotFoundError(f"source image not found: {plan.source_image}")

    in_allocation = bool(get_slurm_job_id())
    print(f"Baking {plan.source_image} -> {plan.output_image}")
    if not in_allocation:
        print(f"No allocation found; srun will request one (time={plan.time_limit}). This may queue.")

    proc = start_srun_process(
        command=["bash", "-c", script],
        container_image=str(plan.source_image),
        srun_options=build_srun_options(plan),
        srun_export_env=CONTAINER_REMAP_ROOT_EXPORT,
        use_bash_wrapper=False,
        overlap=in_allocation,
    )
    exit_code = proc.wait()

    if exit_code != 0:
        print(f"Install failed (exit {exit_code}); no manifest written")
        return exit_code

    plan.manifest_path.write_text(json.dumps(plan.manifest(), indent=2) + "\n")
    print(f"Wrote {plan.output_image}")
    print(f"Wrote {plan.manifest_path}")
    return 0
