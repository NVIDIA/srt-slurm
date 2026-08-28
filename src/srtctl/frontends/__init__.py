# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Frontend implementations for routing requests to backend workers.

Supported frontend types:
- dynamo: Dynamo frontend with NATS/etcd communication
- sglang: SGLang Model Gateway static router (Python launch_router)
- sgl-router: experimental Rust SGL router over aggregate SGLang workers
- vllm: Direct vLLM OpenAI server for aggregate jobs
- vllm-router: Official vLLM Router with static aggregate or P/D workers
"""

from srtctl.frontends.base import (
    FrontendProtocol,
    FrontendType,
    get_frontend,
)
from srtctl.frontends.dynamo import DynamoFrontend
from srtctl.frontends.sgl_router import SGLRouterFrontend
from srtctl.frontends.sglang import SGLangFrontend
from srtctl.frontends.trtllm_serve import TRTLLMServeFrontend
from srtctl.frontends.vllm import VLLMFrontend
from srtctl.frontends.vllm_router import VLLMRouterFrontend

__all__ = [
    "DynamoFrontend",
    "FrontendProtocol",
    "FrontendType",
    "SGLRouterFrontend",
    "SGLangFrontend",
    "TRTLLMServeFrontend",
    "VLLMFrontend",
    "VLLMRouterFrontend",
    "get_frontend",
]
