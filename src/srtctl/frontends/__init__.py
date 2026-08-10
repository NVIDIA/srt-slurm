# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Frontend implementations for routing requests to backend workers.

Supported frontend types:
- dynamo: Dynamo frontend with NATS/etcd communication
- sglang: SGLang Model Gateway with direct worker connections
- vllm: Official vLLM Router with direct worker connections
- vllm-direct: Direct vLLM OpenAI server for aggregate jobs
"""

from srtctl.frontends.base import (
    FrontendProtocol,
    FrontendType,
    get_frontend,
)
from srtctl.frontends.dynamo import DynamoFrontend
from srtctl.frontends.sglang import SGLangFrontend
from srtctl.frontends.trtllm_serve import TRTLLMServeFrontend
from srtctl.frontends.vllm import VLLMFrontend
from srtctl.frontends.vllm_direct import VLLMDirectFrontend

__all__ = [
    "DynamoFrontend",
    "FrontendProtocol",
    "FrontendType",
    "SGLangFrontend",
    "TRTLLMServeFrontend",
    "VLLMDirectFrontend",
    "VLLMFrontend",
    "get_frontend",
]
