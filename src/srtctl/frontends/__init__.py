# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Frontend implementations for routing requests to backend workers.

Each module registers its implementation with ``@register_frontend("<type>")``;
importing it from here is what makes ``frontend.type: <type>`` resolvable.
``list_frontend_types()`` is the registry, ``none`` included.

Supported frontend types:
- dynamo: Dynamo frontend with NATS/etcd communication
- sglang: Direct sglang.launch_server for a single aggregate worker (no router)
- sglang-router: SGLang Model Gateway router in front of static workers
- trtllm_serve: Direct trtllm-serve worker or the disaggregated orchestrator
- vllm: Direct vLLM OpenAI server for aggregate jobs
- vllm-router: Official vLLM Router with static aggregate or P/D workers
"""

from srtctl.frontends.base import (
    FRONTEND_NONE,
    FrontendProtocol,
    get_frontend,
    list_frontend_types,
    register_frontend,
)
from srtctl.frontends.dynamic_frontend import DynamicFrontend
from srtctl.frontends.dynamo import DynamoFrontend
from srtctl.frontends.sglang import SGLangRouterFrontend
from srtctl.frontends.sglang_direct import SGLangFrontend
from srtctl.frontends.trtllm_serve import TRTLLMServeFrontend
from srtctl.frontends.vllm import VLLMFrontend
from srtctl.frontends.vllm_router import VLLMRouterFrontend

__all__ = [
    "FRONTEND_NONE",
    "DynamicFrontend",
    "DynamoFrontend",
    "FrontendProtocol",
    "SGLangFrontend",
    "SGLangRouterFrontend",
    "TRTLLMServeFrontend",
    "VLLMFrontend",
    "VLLMRouterFrontend",
    "get_frontend",
    "list_frontend_types",
    "register_frontend",
]
