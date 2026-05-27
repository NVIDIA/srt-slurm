# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face model download for ``srtctl install``.

Mirrors the behaviour of the prototype ``setup_glm5.sh`` (snapshot_download
into a user-controlled directory) but exposes it as a typed Python function.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def model_storage_dirname(hf_repo_id: str) -> str:
    """Return a stable directory name for storing downloaded model artifacts.

    We include namespace + repo (``org__repo``) to avoid collisions between
    repos with the same trailing segment (for example ``orgA/modelX`` and
    ``orgB/modelX``).
    """
    return hf_repo_id.replace("/", "__")


def download_model(
    hf_repo_id: str,
    dest_dir: Path,
    *,
    hf_token: str | None = None,
    cache_dir: Path | None = None,
    max_workers: int = 8,
) -> Path:
    """Snapshot-download ``hf_repo_id`` into ``dest_dir``.

    Raises if HF_TOKEN is missing (gated repos fail silently otherwise) or
    if ``huggingface_hub`` isn't installed.
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Create a token at https://huggingface.co/settings/tokens "
            "and `export HF_TOKEN=hf_xxx` before running `srtctl install`."
        )

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import logging as hf_logging
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for `srtctl install`. "
            "Reinstall srtctl (`pip install -e .`) to pull in the dependency."
        ) from exc

    hf_logging.set_verbosity_error()
    dest_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Silence the Xet warning + symlink noise that adds nothing for our flow.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    logger.info("Downloading %s → %s", hf_repo_id, dest_dir)
    snapshot_download(
        repo_id=hf_repo_id,
        local_dir=str(dest_dir),
        cache_dir=str(cache_dir) if cache_dir else None,
        token=token,
        max_workers=max_workers,
    )
    return dest_dir
