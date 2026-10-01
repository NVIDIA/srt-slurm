# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backend implementations for different LLM serving frameworks.

Supported backends:
- SGLang: Full support with prefill/decode disaggregation
- TRTLLM: TensorRT-LLM backend with prefill/decode disaggregation
"""

from .atom import AtomProtocol
from .base import BackendProtocol, BackendType, RoleSettings, SrunConfig
from .mocker import MockerProtocol
from .sglang import MooncakeKVStoreConfig, SGLangProtocol
from .tilert import TileRTProtocol
from .trtllm import TRTLLMProtocol
from .vllm import VLLMFailoverConfig, VLLMMooncakeKVStoreConfig, VLLMProtocol

# Union type for all backend configs
BackendConfig = AtomProtocol | SGLangProtocol | TileRTProtocol | TRTLLMProtocol | VLLMProtocol | MockerProtocol

__all__ = [
    # ATOM
    "AtomProtocol",
    "BackendConfig",
    # Base types
    "BackendProtocol",
    "BackendType",
    # Mocker
    "MockerProtocol",
    # SGLang
    "MooncakeKVStoreConfig",
    "RoleSettings",
    "SGLangProtocol",
    "SrunConfig",
    # TRTLLM
    "TRTLLMProtocol",
    # TileRT
    "TileRTProtocol",
    # vLLM
    "VLLMFailoverConfig",
    "VLLMMooncakeKVStoreConfig",
    "VLLMProtocol",
]
