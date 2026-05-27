# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardcoded registry of installable models for the beta integration.

Each entry describes everything ``srtctl install <name>`` needs to fully
provision a model: the Hugging Face repository to pull weights from, the
container image to import via enroot, and the alias keys those resources
should be registered under in ``srtslurm.yaml`` so existing recipes resolve.

The alias keys deliberately match the **full registry strings** the recipes
already reference in ``model.path`` and ``model.container`` (not short
slugs) — adding a new model here must not break recipe resolution.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInstallSpec:
    """Everything ``srtctl install`` needs for one model."""

    name: str
    """Short slug used on the CLI: ``srtctl install <name>``."""

    hf_repo_id: str
    """Hugging Face repository to download weights from (e.g. ``nvidia/GLM-5-NVFP4``)."""

    model_alias: str
    """Alias key written to ``srtslurm.yaml`` ``model_paths``.

    Must match exactly what existing recipes use for ``model.path``.
    """

    container_image: str
    """Docker image reference passed to ``enroot import docker://...``.

    Also used verbatim as the alias key under ``srtslurm.yaml`` ``containers``.
    """

    default_recipe: str
    """Repo-relative recipe path printed as the next-step suggestion."""

    description: str = ""


REGISTRY: dict[str, ModelInstallSpec] = {
    "glm5": ModelInstallSpec(
        name="glm5",
        hf_repo_id="nvidia/GLM-5-NVFP4",
        model_alias="nvidia/GLM5-NVFP4",
        container_image="nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.0-dev.3",
        default_recipe=(
            "recipes/GLM5/disagg/trtllm_dynamo/gb300_nvfp4/ISL8K_OSL1K/STP/"
            "ctx1dep2_gen5tep4_batch4_allconc_eplb0_mtp0.yaml"
        ),
        description="GLM-5 NVFP4 (TRT-LLM + Dynamo disagg, GB300)",
    ),
}


def get_spec(name: str) -> ModelInstallSpec:
    """Look up an install spec by CLI slug. Raises KeyError with available names listed."""
    try:
        return REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(REGISTRY)) or "<none>"
        raise KeyError(f"Unknown model '{name}'. Available: {available}") from exc


def available_models() -> list[str]:
    return sorted(REGISTRY)
