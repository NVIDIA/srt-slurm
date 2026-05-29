# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backend implementations for different LLM serving frameworks.

Supported backends:
- SGLang: Full support with prefill/decode disaggregation
- TRTLLM: TensorRT-LLM backend with prefill/decode disaggregation
"""

from .base import BackendProtocol, BackendType, SrunConfig
from .mocker import MockerProtocol, MockerServerConfig
from .sglang import SGLangProtocol, SGLangServerConfig
from .trtllm import TRTLLMProtocol, TRTLLMServerConfig
from .vllm import VLLMProtocol, VLLMServerConfig

# Union type for all backend configs
BackendConfig = SGLangProtocol | TRTLLMProtocol | VLLMProtocol | MockerProtocol

__all__ = [
    # Base types
    "BackendProtocol",
    "BackendType",
    "BackendConfig",
    "SrunConfig",
    # SGLang
    "SGLangProtocol",
    "SGLangServerConfig",
    # TRTLLM
    "TRTLLMProtocol",
    "TRTLLMServerConfig",
    # vLLM
    "VLLMProtocol",
    "VLLMServerConfig",
    # Mocker
    "MockerProtocol",
    "MockerServerConfig",
]
