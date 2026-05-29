# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for sa-bench custom tokenizer adapters."""

from __future__ import annotations

import builtins
import importlib
import sys
import types
from pathlib import Path


SA_BENCH_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "srtctl"
    / "benchmarks"
    / "scripts"
    / "sa-bench"
)


def _import_sa_bench_tokenizer(module_name: str):
    sys.path.insert(0, str(SA_BENCH_DIR))
    try:
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(SA_BENCH_DIR))


def test_vllm_deepseek_v4_adapter_import_does_not_require_vllm(monkeypatch):
    """The module is importable on hosts that do not have vLLM installed."""
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise AssertionError("vLLM should not be imported at module import time")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = _import_sa_bench_tokenizer("sa_bench_tokenizers.vllm_deepseek_v4")

    assert hasattr(module, "VLLMDeepseekV4Tokenizer")


def test_vllm_deepseek_v4_adapter_uses_unwrapped_vllm_renderer(monkeypatch):
    """from_pretrained returns vLLM's rendered tokenizer, not its cache wrapper."""
    module = _import_sa_bench_tokenizer("sa_bench_tokenizers.vllm_deepseek_v4")

    fake_hf_tokenizer = object()
    rendered_tokenizer = object()
    calls = {}

    class FakePreTrainedTokenizerFast:
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            calls["from_pretrained"] = (pretrained_model_name_or_path, kwargs)
            return fake_hf_tokenizer

    def fake_get_deepseek_v4_tokenizer(tokenizer):
        calls["renderer_tokenizer"] = tokenizer
        return rendered_tokenizer

    vllm_pkg = types.ModuleType("vllm")
    vllm_pkg.__path__ = []
    tokenizers_pkg = types.ModuleType("vllm.tokenizers")
    tokenizers_pkg.__path__ = []
    deepseek_module = types.ModuleType("vllm.tokenizers.deepseek_v4")
    deepseek_module.get_deepseek_v4_tokenizer = fake_get_deepseek_v4_tokenizer
    vllm_pkg.tokenizers = tokenizers_pkg
    tokenizers_pkg.deepseek_v4 = deepseek_module

    monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    monkeypatch.setitem(sys.modules, "vllm.tokenizers", tokenizers_pkg)
    monkeypatch.setitem(
        sys.modules,
        "vllm.tokenizers.deepseek_v4",
        deepseek_module,
    )
    monkeypatch.setattr(module, "PreTrainedTokenizerFast", FakePreTrainedTokenizerFast)

    tokenizer = module.VLLMDeepseekV4Tokenizer.from_pretrained(
        "/model",
        trust_remote_code=True,
    )

    assert tokenizer is rendered_tokenizer
    assert calls["renderer_tokenizer"] is fake_hf_tokenizer
    assert calls["from_pretrained"] == ("/model", {"trust_remote_code": True})


def test_benchmark_serving_custom_tokenizer_uses_sa_bench_loader(monkeypatch):
    """Custom tokenizer strings must not be routed through vLLM's wrapper."""
    sys.path.insert(0, str(SA_BENCH_DIR))
    try:
        sys.modules.pop("benchmark_serving", None)
        module = importlib.import_module("benchmark_serving")
    finally:
        sys.path.remove(str(SA_BENCH_DIR))

    calls = []
    custom_tokenizer = object()
    default_tokenizer = object()

    def fake_sa_loader(*args, **kwargs):
        calls.append(("sa-bench", args, kwargs))
        return custom_tokenizer

    def fake_vllm_loader(*args, **kwargs):
        calls.append(("vllm", args, kwargs))
        return default_tokenizer

    monkeypatch.setattr(module, "get_sa_bench_tokenizer", fake_sa_loader)
    monkeypatch.setattr(module, "get_vllm_tokenizer", fake_vllm_loader)

    tokenizer = module.load_tokenizer(
        "/model",
        tokenizer_mode="auto",
        trust_remote_code=True,
        custom_tokenizer="sa_bench_tokenizers.vllm_deepseek_v4.VLLMDeepseekV4Tokenizer",
    )

    assert tokenizer is custom_tokenizer
    assert calls == [
        (
            "sa-bench",
            ("/model",),
            {
                "tokenizer_mode": "auto",
                "trust_remote_code": True,
                "custom_tokenizer": "sa_bench_tokenizers.vllm_deepseek_v4.VLLMDeepseekV4Tokenizer",
            },
        )
    ]

    tokenizer = module.load_tokenizer(
        "/model",
        tokenizer_mode="auto",
        trust_remote_code=True,
        custom_tokenizer=None,
    )

    assert tokenizer is default_tokenizer
    assert calls[-1] == (
        "vllm",
        ("/model",),
        {
            "tokenizer_mode": "auto",
            "trust_remote_code": True,
            "custom_tokenizer": None,
        },
    )
