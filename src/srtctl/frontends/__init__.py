# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Frontend implementations for routing requests to backend workers.

Supported frontend types:
- dynamo: Dynamo frontend with NATS/etcd communication
- sglang: SGLang native router with direct worker connections
- vllm: Direct vLLM OpenAI server for aggregate jobs
- vllm-router: Official vLLM Router with static aggregate or P/D workers
"""

from srtctl.frontends.atomesh import AtomeshFrontend
from srtctl.frontends.base import (
    FrontendProtocol,
    FrontendType,
    get_frontend,
)
from srtctl.frontends.dynamo import DynamoFrontend
from srtctl.frontends.sglang import SGLangFrontend
from srtctl.frontends.trtllm_serve import TRTLLMServeFrontend
from srtctl.frontends.vllm import VLLMFrontend
from srtctl.frontends.vllm_router import VLLMRouterFrontend

__all__ = [
    "AtomeshFrontend",
    "DynamoFrontend",
    "FrontendProtocol",
    "FrontendType",
    "SGLangFrontend",
    "TRTLLMServeFrontend",
    "VLLMFrontend",
    "VLLMRouterFrontend",
    "get_frontend",
]
