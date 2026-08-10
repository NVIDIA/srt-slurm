# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM Router frontend."""

from typing import ClassVar

from srtctl.frontends.base import register_frontend
from srtctl.frontends.static_router import StaticRouterFrontend


@register_frontend("vllm-router")
class VLLMRouterFrontend(StaticRouterFrontend):
    """Route requests to direct vLLM OpenAI-compatible worker endpoints."""

    type: ClassVar[str] = "vllm-router"
    backend_type: ClassVar[str] = "vllm"
    executable: ClassVar[tuple[str, ...]] = ("vllm-router",)
    pd_flag: ClassVar[str] = "--vllm-pd-disaggregation"
    process_name: ClassVar[str] = "vllm_router"
