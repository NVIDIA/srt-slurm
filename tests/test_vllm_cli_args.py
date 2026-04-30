# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for vLLM backend CLI flag generation."""

from srtctl.backends.vllm import _config_to_cli_args


def test_dotted_config_flags_preserve_nested_key_spelling() -> None:
    """Dotted vLLM config flags keep nested underscores intact."""
    args = _config_to_cli_args({"attention_config.use_fp4_indexer_cache": "True"})

    assert args == ["--attention-config.use_fp4_indexer_cache", "True"]
    assert "--attention-config.use-fp4-indexer-cache" not in args
