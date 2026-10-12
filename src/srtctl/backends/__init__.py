# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backend implementations for different LLM serving frameworks.

Supported backends:
- SGLang: Full support with prefill/decode disaggregation
- TRTLLM: TensorRT-LLM backend with prefill/decode disaggregation
"""

from .atom import AtomBackend
from .base import Backend, BackendType, RoleSettings, SrunConfig
from .mocker import MockerBackend
from .sglang import MooncakeKVStoreConfig, SGLangBackend
from .tilert import TileRTBackend
from .tokenspeed import TokenSpeedBackend
from .trtllm import TRTLLMBackend
from .vllm import VLLMBackend, VLLMFailoverConfig, VLLMMooncakeKVStoreConfig

# Union type for all backend configs
BackendConfig = (
    AtomBackend | SGLangBackend | TileRTBackend | TokenSpeedBackend | TRTLLMBackend | VLLMBackend | MockerBackend
)

__all__ = [
    # ATOM
    "AtomBackend",
    # Base types
    "Backend",
    "BackendConfig",
    "BackendType",
    # Mocker
    "MockerBackend",
    # SGLang
    "MooncakeKVStoreConfig",
    "RoleSettings",
    "SGLangBackend",
    "SrunConfig",
    # TRTLLM
    "TRTLLMBackend",
    # TileRT
    "TileRTBackend",
    # TokenSpeed
    "TokenSpeedBackend",
    # vLLM
    "VLLMBackend",
    "VLLMFailoverConfig",
    "VLLMMooncakeKVStoreConfig",
]
