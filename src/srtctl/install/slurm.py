# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SLURM-mode submission for ``srtctl install``.

When ``--slurm`` is passed, we generate a self-contained sbatch script that
inlines:

1. ``python3 -c "snapshot_download(...)"`` for the HF model download
2. ``enroot import`` for the container ``.sqsh``
3. A small inline Python snippet that registers the aliases in
   ``srtslurm.yaml`` via ruamel.yaml

The compute node only needs a venv with ``huggingface_hub`` and
``ruamel.yaml`` available — srtctl itself does not need to be importable
inside the sbatch. This matches the user's existing per-model download
scripts (one self-contained bash file per artifact, readable top-to-bottom).

Cluster-specific fields (``default_account`` / ``default_partition`` /
``default_time_limit``) are read from ``srtslurm.yaml`` at submit time —
never hardcoded — so the same command works on every cluster.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from srtctl.install.container import container_filename
from srtctl.install.model import model_storage_dirname
from srtctl.install.registry import ModelInstallSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SbatchSubmission:
    """Result of submitting an install job."""

    job_id: str
    script_path: Path
    log_path: Path


def _require(cluster_config: dict[str, Any] | None, key: str) -> str:
    """Pull a required string field from srtslurm.yaml or raise a helpful error."""
    if cluster_config is None:
        raise RuntimeError(
            "srtslurm.yaml not found. Run `make setup ARCH=<arch>` first to "
            "populate cluster defaults (account, partition, time_limit)."
        )
    value = cluster_config.get(key)
    if not value:
        raise RuntimeError(
            f"srtslurm.yaml is missing '{key}'. Add it manually or re-run "
            f"`make setup ARCH=<arch>` to populate cluster defaults."
        )
    return str(value)


def build_sbatch_script(
    *,
    spec: ModelInstallSpec,
    srtctl_root: Path,
    install_base: Path,
    log_path: Path,
    venv_path: Path,
    cluster_config: dict[str, Any] | None,
    strict_auth_preflight: bool = False,
) -> str:
    """Render the self-contained sbatch script that does the full install on a compute node.

    The generated script reads top-to-bottom: activate venv → check token →
    snapshot_download → enroot import → register aliases. No call back into
    srtctl from inside the script.
    """
    account = _require(cluster_config, "default_account")
    partition = _require(cluster_config, "default_partition")
    time_limit = _require(cluster_config, "default_time_limit")
    use_exclusive = bool((cluster_config or {}).get("use_exclusive_sbatch_directive", False))

    model_dir = install_base / "models" / model_storage_dirname(spec.hf_repo_id)
    container_path = install_base / "containers" / container_filename(spec.container_image)
    srtslurm_yaml = srtctl_root / "srtslurm.yaml"

    directives = [
        f'#SBATCH --job-name="srtctl-install-{spec.name}"',
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=128G",
        f"#SBATCH --output={log_path}",
        f"#SBATCH --error={log_path}",
    ]
    if use_exclusive:
        directives.append("#SBATCH --exclusive")

    lines = [
        "#!/bin/bash",
        *directives,
        "",
        "set -eu -o pipefail",
        "",
        "echo '============================================='",
        f"echo 'srtctl install {spec.name}  ({spec.description})'",
        'echo "Start  : $(date)"',
        'echo "Node   : $(hostname)"',
        'echo "Job ID : ${SLURM_JOB_ID:-?}"',
        'echo "Arch   : $(uname -m)"',
        "echo '============================================='",
        "",
        "# Hermetic Python env: drop any inherited PYTHONPATH so the venv we just",
        "# activated is the ONLY place we look for packages. This avoids cases where",
        "# a parent shell has PYTHONPATH set to some other venv's site-packages and",
        "# pip says \"already satisfied\" for things that are not actually here.",
        "unset PYTHONPATH",
        f'source "{venv_path}/bin/activate"',
        "",
        "# Surface exactly where Python resolves huggingface_hub from, so a future",
        "# 'works on my machine' bug is caught immediately in the job log.",
        'python3 -c "import huggingface_hub; print(f\'huggingface_hub @ {huggingface_hub.__file__}\')" || {',
        '    echo "ERROR: huggingface_hub is not importable from this venv ('
        + str(venv_path)
        + ')."',
        '    echo "Inside an aarch64 srun shell:"',
        f'    echo "  source {venv_path}/bin/activate && pip install --ignore-installed huggingface_hub ruamel.yaml"',
        "    exit 1",
        "}",
        "",
        'if [ -z "${HF_TOKEN:-}" ]; then',
        '    echo "ERROR: HF_TOKEN not set. Re-submit with `sbatch --export=ALL ...`"',
        "    exit 1",
        "fi",
        "",
        "export HF_HUB_DISABLE_XET=1",
        "",
        "# ---------------------------------------------------------------",
        "# 1. Download HF model weights",
        "# ---------------------------------------------------------------",
        f'MODEL_DIR="{model_dir}"',
        'mkdir -p "${MODEL_DIR}"',
        "",
        f'echo "Downloading {spec.hf_repo_id} -> ${{MODEL_DIR}}"',
        "python3 - <<'PYEOF'",
        "import os",
        "from huggingface_hub import snapshot_download",
        "snapshot_download(",
        f"    repo_id={spec.hf_repo_id!r},",
        f"    local_dir={str(model_dir)!r},",
        '    token=os.environ["HF_TOKEN"],',
        "    max_workers=8,",
        ")",
        "print('Model download complete.')",
        "PYEOF",
        "",
        "# ---------------------------------------------------------------",
        "# 2. Import the container .sqsh via enroot",
        "# ---------------------------------------------------------------",
        f'CONTAINER_PATH="{container_path}"',
        'mkdir -p "$(dirname "${CONTAINER_PATH}")"',
        "",
        f'CONTAINER_IMAGE="{spec.container_image}"',
        f'STRICT_AUTH_PREFLIGHT={"1" if strict_auth_preflight else "0"}',
        'if [[ "${CONTAINER_IMAGE}" == nvcr.io/* ]]; then',
        '    if [ ! -f "${HOME}/.config/enroot/.credentials" ] && [ ! -f "${HOME}/.docker/config.json" ]; then',
        '        if [ "${STRICT_AUTH_PREFLIGHT}" = "1" ]; then',
        '            echo "ERROR: Missing nvcr.io credentials. Configure enroot/docker auth before running install."',
        '            echo "Expected one of: ${HOME}/.config/enroot/.credentials or ${HOME}/.docker/config.json"',
        "            exit 1",
        "        else",
        '            echo "WARNING: Missing common nvcr.io credential files; continuing with enroot import attempt."',
        "        fi",
        "    fi",
        "fi",
        "",
        'if [ -f "${CONTAINER_PATH}" ]; then',
        '    echo "Container already present at ${CONTAINER_PATH} - skipping import."',
        "else",
        '    LOCAL_TMP="/tmp/${USER}/srtctl_enroot_$$"',
        '    mkdir -p "${LOCAL_TMP}/cache" "${LOCAL_TMP}/data" "${LOCAL_TMP}/tmp"',
        '    export ENROOT_CACHE_PATH="${LOCAL_TMP}/cache"',
        '    export ENROOT_DATA_PATH="${LOCAL_TMP}/data"',
        '    export ENROOT_TEMP_PATH="${LOCAL_TMP}/tmp"',
        '    export TMPDIR="${LOCAL_TMP}/tmp"',
        "",
        '    echo "Importing ${CONTAINER_IMAGE} -> ${CONTAINER_PATH}"',
        '    enroot import --output "${CONTAINER_PATH}" "docker://${CONTAINER_IMAGE}"',
        "",
        '    rm -rf "${LOCAL_TMP}"',
        "fi",
        "",
        "# ---------------------------------------------------------------",
        "# 3. Register aliases in srtslurm.yaml (comment-preserving)",
        "# ---------------------------------------------------------------",
        'echo "Registering aliases in srtslurm.yaml..."',
        "python3 - <<'PYEOF'",
        "import fcntl",
        "import os",
        "import tempfile",
        "from pathlib import Path",
        "from ruamel.yaml import YAML",
        "from ruamel.yaml.comments import CommentedMap",
        "",
        f"yml_path = Path({str(srtslurm_yaml)!r})",
        "y = YAML()",
        "y.preserve_quotes = True",
        "lock_path = yml_path.with_name(yml_path.name + '.lock')",
        "with open(lock_path, 'a+') as lockf:",
        "    fcntl.flock(lockf, fcntl.LOCK_EX)",
        "    with open(yml_path) as f:",
        "        doc = y.load(f) or CommentedMap()",
        "",
        "def ensure(d, k):",
        "    if k not in d or d[k] is None:",
        "        d[k] = CommentedMap()",
        "    return d[k]",
        "",
        "model_paths = ensure(doc, 'model_paths')",
        f"model_paths[{spec.model_alias!r}] = {str(model_dir)!r}",
        "containers = ensure(doc, 'containers')",
        f"containers[{spec.container_image!r}] = {str(container_path)!r}",
        "",
        "fd, tmp_path = tempfile.mkstemp(prefix=yml_path.name + '.', suffix='.tmp', dir=str(yml_path.parent))",
        "os.close(fd)",
        "try:",
        "    with open(tmp_path, 'w') as f:",
        "        y.dump(doc, f)",
        "    os.replace(tmp_path, yml_path)",
        "finally:",
        "    if os.path.exists(tmp_path):",
        "        os.remove(tmp_path)",
        "print('Aliases registered in srtslurm.yaml.')",
        "PYEOF",
        "",
        "echo '============================================='",
        'echo "Done : $(date)"',
        'if [ -n "' + str(spec.default_recipe) + '" ]; then',
        f'    echo "Next step: srtctl apply -f {spec.default_recipe}"',
        "else",
        '    echo "Next step: create/update a recipe with:"',
        f'    echo "  model.path: {spec.model_alias}"',
        f'    echo "  model.container: {spec.container_image}"',
        "fi",
        "echo '============================================='",
        "",
    ]
    return "\n".join(lines)


def submit_install_job(
    *,
    spec: ModelInstallSpec,
    srtctl_root: Path,
    install_base: Path,
    venv_path: Path,
    cluster_config: dict[str, Any] | None,
    strict_auth_preflight: bool = False,
) -> SbatchSubmission:
    """Write the sbatch script and submit it via ``sbatch --export=ALL``.

    Returns the parsed job id, the path of the generated script, and the
    SLURM log path so the caller can print them.

    ``venv_path`` is forwarded to the sbatch script; the user is responsible
    for ensuring the venv exists and contains huggingface_hub + ruamel.yaml
    built for the compute-node architecture.
    """
    if shutil.which("sbatch") is None:
        raise RuntimeError("`sbatch` not found on PATH. Are you on a SLURM cluster?")

    install_base.mkdir(parents=True, exist_ok=True)
    log_path = install_base / f"install_{spec.name}_%j.log"
    script_path = install_base / f"install_{spec.name}.sbatch"

    script = build_sbatch_script(
        spec=spec,
        srtctl_root=srtctl_root,
        install_base=install_base,
        log_path=log_path,
        venv_path=venv_path,
        cluster_config=cluster_config,
        strict_auth_preflight=strict_auth_preflight,
    )
    script_path.write_text(script)
    script_path.chmod(0o755)
    logger.info("Wrote sbatch script: %s", script_path)

    result = subprocess.run(
        ["sbatch", "--export=ALL", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=srtctl_root,
    )
    stdout = result.stdout.strip()
    match = re.search(r"(\d+)\s*$", stdout)
    if not match:
        raise RuntimeError(f"Could not parse job id from sbatch output: {stdout!r}")
    job_id = match.group(1)

    # Resolve the actual log file name (sbatch substitutes %j)
    actual_log = Path(str(log_path).replace("%j", job_id))
    return SbatchSubmission(job_id=job_id, script_path=script_path, log_path=actual_log)
