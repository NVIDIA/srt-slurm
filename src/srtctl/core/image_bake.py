# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bake srtctl's runtime dependencies into a container image.

Without this, every job reinstalls the same things before it can serve a single
request: dynamo from pip, the sa-bench Python deps, and recipe ``setup_script``
overlays (vLLM patches, etc.). Doing it once into a new squashfs takes that
off the critical path of every run.

The image surgery is pyxis': ``--container-save`` exports the container
filesystem to a new .sqsh when the step ends, so nothing here has to understand
the squashfs format.

``--script`` mounts the repo ``configs/`` tree at ``/configs`` (same path jobs
use) and runs the named setup script inside the writable container.
``--patch`` mounts a unified diff and applies it to the image's vLLM tree
with ``patch(1)``; a conflict fails the bake and does not keep the output
image. Bind mounts are not part of the exported squashfs; only overlay
writes (patched site-packages, pip installs) persist.
"""

from __future__ import annotations

import json
import os
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

# Jobs mount the repo configs/ tree here; setup scripts hardcode this prefix
# (see configs/vllm-pr51924-one-sided.sh looking for /configs/patches/...).
CONFIGS_IN_CONTAINER = Path("/configs")
# Fallback when --script points at a file that is not inside configs/.
BAKE_SCRIPT_IN_CONTAINER = Path("/bake-script.sh")
# Each --patch file is bind-mounted here so the install script has a stable path.
BAKE_PATCH_DIR = Path("/bake-patches")


def repo_root() -> Path:
    """Checkout that owns ``configs/``, matching how jobs find SRTCTL_SOURCE_DIR."""
    for key in ("SRTCTL_SOURCE_DIR", "SRTCTL_ROOT"):
        value = os.environ.get(key)
        if value:
            return Path(value)
    return Path(__file__).resolve().parent.parent.parent.parent


def default_configs_dir() -> Path:
    return repo_root() / "configs"


def resolve_setup_script(script: str, *, configs_dir: Path) -> Path:
    """Resolve ``--script`` the same way jobs resolve ``setup_script``.

    Accepts a host path, or a name under ``configs/`` / ``configs/patches/``.
    """
    given = Path(script).expanduser()
    if given.is_file():
        return given.resolve()
    for candidate in (configs_dir / script, configs_dir / "patches" / script):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"setup script {script!r} not found as a file, nor in {configs_dir} or {configs_dir / 'patches'}"
    )


def resolve_vllm_patch(patch: str, *, configs_dir: Path | None = None) -> Path:
    """Resolve ``--patch`` to a host file.

    Accepts a path, or a name under ``configs/`` / ``configs/patches/``.
    """
    given = Path(patch).expanduser()
    if given.is_file():
        return given.resolve()
    searched: list[Path] = []
    if configs_dir is not None:
        for candidate in (configs_dir / patch, configs_dir / "patches" / patch):
            searched.append(candidate)
            if candidate.is_file():
                return candidate.resolve()
    extra = f", nor in {' or '.join(str(p.parent) for p in searched)}" if searched else ""
    raise FileNotFoundError(f"vLLM patch {patch!r} not found as a file{extra}")


def infer_vllm_patch_args(diff_text: str) -> tuple[int, str]:
    """Return ``(strip, root)`` for ``patch -pSTRIP -d $ROOT``.

    ``root`` is ``site-packages`` (directory that contains ``vllm/``) or
    ``vllm`` (the package directory itself). Git diffs against the vLLM repo
    look like ``--- a/vllm/foo.py`` and apply with ``-p1`` in site-packages.
    """
    old_path: str | None = None
    for line in diff_text.splitlines():
        if not line.startswith("--- "):
            continue
        raw = line[4:].split("\t", 1)[0].strip()
        if raw != "/dev/null":
            old_path = raw
            break
    if old_path is None:
        return 1, "site-packages"
    if old_path.startswith(("a/", "b/")):
        strip = 1
        rest = old_path[2:]
    else:
        strip = 0
        rest = old_path
    rest = rest.lstrip("./")
    if rest == "vllm" or rest.startswith("vllm/"):
        return strip, "site-packages"
    return (1 if old_path.startswith(("a/", "b/")) else 0), "vllm"


@dataclass(frozen=True)
class VllmPatch:
    """One unified diff applied to the image's vLLM install."""

    host_path: Path
    container_name: str
    strip: int
    root: str  # "site-packages" | "vllm"

    def __post_init__(self) -> None:
        if self.root not in ("site-packages", "vllm"):
            raise ValueError(f"unknown patch root {self.root!r}")
        if self.strip < 0:
            raise ValueError("patch strip must be >= 0")

    @property
    def container_path(self) -> Path:
        return BAKE_PATCH_DIR / self.container_name


def build_vllm_patches(paths: list[Path]) -> tuple[VllmPatch, ...]:
    """Read each diff and pick strip/root so the container ``patch`` is deterministic."""
    used_names: dict[str, int] = {}
    patches: list[VllmPatch] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"vLLM patch not found: {path}")
        text = path.read_text(errors="replace")
        if not text.strip():
            raise ValueError(f"vLLM patch is empty: {path}")
        strip, root = infer_vllm_patch_args(text)
        n = used_names.get(path.name, 0)
        used_names[path.name] = n + 1
        container_name = path.name if n == 0 else f"{n}-{path.name}"
        patches.append(VllmPatch(host_path=path.resolve(), container_name=container_name, strip=strip, root=root))
    return tuple(patches)


# --container-save captures everything written, so package caches and build
# leftovers would be baked in and inflate the image by gigabytes.
_CLEANUP = (
    "pip cache purge >/dev/null 2>&1 || true\n"
    "rm -rf /root/.cache/pip /tmp/* >/dev/null 2>&1 || true\n"
    'echo "bake-image: install complete"'
)

# Locate vLLM, then dry-run each --patch before writing. A failed hunk must
# abort the bake (set -e) so pyxis does not keep a conflicted tree.
_VLLM_PATCH_HELPERS = r"""
SITE_PACKAGES="$(python3 -c 'import pathlib, sys
try:
    import vllm
except ImportError:
    sys.stderr.write("bake-image: vLLM is not installed in this image\n")
    sys.exit(1)
print(pathlib.Path(vllm.__file__).resolve().parent.parent)
')"
VLLM_ROOT="${SITE_PACKAGES}/vllm"
if [[ ! -d "${VLLM_ROOT}" ]]; then
    echo "bake-image: vLLM package directory missing: ${VLLM_ROOT}" >&2
    exit 1
fi
if ! command -v patch >/dev/null 2>&1; then
    echo "bake-image: patch(1) is not in this image; cannot apply --patch" >&2
    exit 1
fi
echo "bake-image: vLLM at ${VLLM_ROOT}"
_bake_apply_patch() {
    local file="$1" strip="$2" root="$3" label="$4"
    echo "bake-image: applying ${label} (patch -p${strip} -d ${root})"
    if [[ ! -f "${file}" ]]; then
        echo "ERROR: patch file not mounted: ${file}" >&2
        exit 1
    fi
    if ! patch --batch --forward --dry-run -p"${strip}" -d "${root}" < "${file}"; then
        echo "ERROR: ${label} does not apply to this vLLM tree" >&2
        echo "ERROR: refusing to bake; fix the patch or use a matching vLLM version" >&2
        patch --batch --forward --dry-run -p"${strip}" -d "${root}" < "${file}" >&2 || true
        exit 1
    fi
    patch --batch --forward -p"${strip}" -d "${root}" < "${file}"
    echo "bake-image: applied ${label}"
}
"""


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


def default_output_image(
    source: Path,
    *,
    dynamo_version: str | None,
    sa_bench: bool,
    setup_script: str | Path | None = None,
    vllm_patches: list[Path] | tuple[VllmPatch, ...] | None = None,
) -> Path:
    """Name the output after what went into it, next to the source image.

    A directory of ``*.sqsh`` files is unreadable without this; the tags say
    what each one contains without opening the manifest.
    """
    tags = []
    if dynamo_version:
        tags.append(f"dynamo{dynamo_version}")
    if sa_bench:
        tags.append("sa-bench")
    if setup_script:
        tags.append(Path(setup_script).stem)
    for item in vllm_patches or ():
        host = item.host_path if isinstance(item, VllmPatch) else Path(item)
        tags.append(host.stem)
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
    # Host path of the recipe setup script; run inside the image at bake time.
    setup_script: Path | None = None
    configs_dir: Path | None = None
    # Unified diffs applied to the image's vLLM tree (--patch). Dry-run first.
    vllm_patches: tuple[VllmPatch, ...] = ()
    # Only used when srun has to allocate for itself.
    time_limit: str = DEFAULT_TIME_LIMIT
    slurm_overrides: Mapping[str, str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dynamo_version and not self.sa_bench and self.setup_script is None and not self.vllm_patches:
            raise ValueError("nothing to install; pass --dynamo, --sa-bench, --script, and/or --patch")
        if self.sa_bench and not self.sa_bench_deps:
            raise ValueError("sa-bench requested without a package list")
        if self.setup_script is not None:
            if self.configs_dir is None:
                raise ValueError("setup script requires a configs directory to mount at /configs")
            if not self.setup_script.is_file():
                raise ValueError(f"setup script not found: {self.setup_script}")
        for patch in self.vllm_patches:
            if not patch.host_path.is_file():
                raise ValueError(f"vLLM patch not found: {patch.host_path}")
        if self.source_image == self.output_image:
            raise ValueError("output image must differ from the source image")

    def container_script_path(self) -> Path:
        """Where the setup script is visible inside the container."""
        assert self.setup_script is not None
        if self.configs_dir is not None:
            try:
                rel = self.setup_script.resolve().relative_to(self.configs_dir.resolve())
                return CONFIGS_IN_CONTAINER / rel
            except ValueError:
                pass
        return BAKE_SCRIPT_IN_CONTAINER

    def container_mounts(self) -> dict[Path, Path]:
        """Host binds the script and --patch files need."""
        mounts: dict[Path, Path] = {}
        if self.setup_script is not None:
            assert self.configs_dir is not None
            mounts[self.configs_dir.resolve()] = CONFIGS_IN_CONTAINER
            if self.container_script_path() == BAKE_SCRIPT_IN_CONTAINER:
                mounts[self.setup_script.resolve()] = BAKE_SCRIPT_IN_CONTAINER
        for patch in self.vllm_patches:
            mounts[patch.host_path.resolve()] = patch.container_path
        return mounts

    @property
    def manifest_path(self) -> Path:
        return self.output_image.with_suffix(".manifest.json")

    def install_script(self) -> str:
        """Bash run inside the container, before pyxis saves it."""
        steps = ["set -euo pipefail"]

        if self.setup_script is not None:
            container_path = shlex.quote(str(self.container_script_path()))
            steps.append(f"echo 'bake-image: running setup script {self.setup_script.name}'")
            steps.append(f"test -f {container_path}")
            steps.append(f"bash {container_path}")
            steps.append(f"echo 'bake-image: {self.setup_script.name} complete'")

        if self.vllm_patches:
            steps.append(_VLLM_PATCH_HELPERS.strip())
            for patch in self.vllm_patches:
                root_expr = '"${SITE_PACKAGES}"' if patch.root == "site-packages" else '"${VLLM_ROOT}"'
                steps.append(
                    "_bake_apply_patch "
                    f"{shlex.quote(str(patch.container_path))} {patch.strip} {root_expr} "
                    f"{shlex.quote(patch.host_path.name)}"
                )

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
                "setup_script": self.setup_script.name if self.setup_script else None,
                "vllm_patches": [patch.host_path.name for patch in self.vllm_patches] or None,
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
    mounts = plan.container_mounts()
    mount_flag = ""
    if mounts:
        mount_str = ",".join(f"{host}:{container}" for host, container in mounts.items())
        mount_flag = f"--container-mounts {mount_str} "
    return (
        f"srun {overlap}--nodes 1 --ntasks 1 "
        f"--container-image {plan.source_image} "
        "--no-container-entrypoint --no-container-mount-home "
        f"{mount_flag}{rendered} "
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
    if plan.setup_script is not None:
        print(f"Setup script: {plan.setup_script} (mounted from {plan.configs_dir} -> {CONFIGS_IN_CONTAINER})")
    if plan.vllm_patches:
        names = ", ".join(patch.host_path.name for patch in plan.vllm_patches)
        print(f"vLLM patches: {names}")
    if not in_allocation:
        print(f"No allocation found; srun will request one (time={plan.time_limit}). This may queue.")

    output_existed = plan.output_image.exists()
    mounts = plan.container_mounts()
    proc = start_srun_process(
        command=["bash", "-c", script],
        container_image=str(plan.source_image),
        container_mounts=mounts or None,
        srun_options=build_srun_options(plan),
        srun_export_env=CONTAINER_REMAP_ROOT_EXPORT,
        use_bash_wrapper=False,
        overlap=in_allocation,
    )
    exit_code = proc.wait()

    if exit_code != 0:
        print(f"Install failed (exit {exit_code}); no manifest written")
        # pyxis --container-save may still write an image; drop it if this run
        # created it so a conflicted --patch does not leave a baked .sqsh.
        if plan.output_image.exists() and not output_existed:
            plan.output_image.unlink()
            print(f"Removed incomplete {plan.output_image}")
        return exit_code

    plan.manifest_path.write_text(json.dumps(plan.manifest(), indent=2) + "\n")
    print(f"Wrote {plan.output_image}")
    print(f"Wrote {plan.manifest_path}")
    return 0
