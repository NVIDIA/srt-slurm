# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SA-Bench random dataset cache."""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import sys
from pathlib import Path

import pytest

SA_BENCH_DIR = Path(__file__).resolve().parents[1] / "src" / "srtctl" / "benchmarks" / "scripts" / "sa-bench"


def _import_sa_bench_module(module_name: str):
    sys.path.insert(0, str(SA_BENCH_DIR))
    try:
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(SA_BENCH_DIR))


@pytest.fixture
def sa_bench():
    return _import_sa_bench_module("benchmark_serving")


@pytest.fixture
def counting_builder(sa_bench, monkeypatch):
    """Replace prompt generation with a counter so cache hits are observable."""
    calls = []

    def fake_sample_random_requests(**kwargs):
        calls.append(kwargs)
        return [(f"prompt-{i}", kwargs["input_len"], kwargs["output_len"], None) for i in range(kwargs["num_prompts"])]

    monkeypatch.setattr(sa_bench, "sample_random_requests", fake_sample_random_requests)
    return calls


def _load(sa_bench, cache_dir, **overrides):
    params = {
        "model_name": "my-model-fp8",
        "tokenizer_id": "/model",
        "seed": 0,
        "prefix_len": 0,
        "input_len": 8192,
        "output_len": 1,
        "num_prompts": 32,
        "range_ratio": 1.0,
        "tokenizer": None,
        "use_chat_template": True,
    }
    params.update(overrides)
    return sa_bench.load_or_build_random_requests(
        cache_dir,
        params.pop("model_name"),
        params.pop("tokenizer_id"),
        params.pop("seed"),
        **params,
    )


def test_main_call_site_matches_signature(sa_bench):
    """The call inside main() must bind, including no argument passed twice.

    Every other test calls the loader directly, so only a static check of the
    real call site catches a mismatch between it and the signature.
    """
    source = (SA_BENCH_DIR / "benchmark_serving.py").read_text()
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_or_build_random_requests"
    ]

    assert len(calls) == 1, "expected exactly one call site in main()"
    call = calls[0]
    positional = [inspect.Parameter.empty] * len(call.args)
    keywords = {kw.arg: inspect.Parameter.empty for kw in call.keywords if kw.arg is not None}

    inspect.signature(sa_bench.load_or_build_random_requests).bind(*positional, **keywords)


def test_no_cache_dir_always_rebuilds(sa_bench, counting_builder):
    """Without a cache dir the original generate-every-time behavior is kept."""
    _load(sa_bench, None)
    _load(sa_bench, None)

    assert len(counting_builder) == 2


def test_second_run_hits_the_cache(sa_bench, counting_builder, tmp_path):
    first = _load(sa_bench, str(tmp_path))
    second = _load(sa_bench, str(tmp_path))

    assert len(counting_builder) == 1
    assert second == first


def test_generation_is_bracketed_by_timestamps(sa_bench, counting_builder, tmp_path, capsys):
    """A miss reports when prompt generation started and how long it took."""
    _load(sa_bench, str(tmp_path))

    lines = capsys.readouterr().out.splitlines()
    stamped = [line for line in lines if re.match(r"^\[\d{2}:\d{2}:\d{2}\] \[cache\] ", line)]

    assert any("miss:" in line for line in stamped)
    assert any(re.search(r"generating 32 prompts \(isl=8192, osl=1\)", line) for line in stamped)
    assert any(re.search(r"generated 32 prompts in \d+\.\d+s", line) for line in stamped)
    assert any("saved dataset to" in line for line in stamped)


def test_cache_hit_reports_timestamp_and_full_path(sa_bench, counting_builder, tmp_path, capsys):
    """A hit is traceable: when it happened and exactly which file was reused."""
    _load(sa_bench, str(tmp_path))
    (cache_file,) = list(tmp_path.glob("*.pkl"))
    capsys.readouterr()

    _load(sa_bench, str(tmp_path))

    out = capsys.readouterr().out
    assert re.search(rf"^\[\d{{2}}:\d{{2}}:\d{{2}}\] \[cache\] hit: loaded 32 prompts from {cache_file}", out, re.M)
    assert "generating" not in out


def test_cache_file_name_is_human_readable(sa_bench, counting_builder, tmp_path):
    """Model, ISL, OSL and prompt count stay legible for manual cleanup."""
    _load(sa_bench, str(tmp_path))

    (cache_file,) = list(tmp_path.glob("*.pkl"))

    assert re.fullmatch(r"my-model-fp8_isl8192_osl1_n32_[0-9a-f]{8}\.pkl", cache_file.name)


@pytest.mark.parametrize(
    "override",
    [
        {"num_prompts": 64},
        {"output_len": 8},
        {"input_len": 4096},
        {"seed": 1},
        {"range_ratio": 0.8},
        {"use_chat_template": False},
    ],
)
def test_differing_parameters_do_not_share_a_cache_entry(sa_bench, counting_builder, tmp_path, override):
    """Every input that shifts the seeded RNG must key a separate file."""
    _load(sa_bench, str(tmp_path))
    _load(sa_bench, str(tmp_path), **override)

    assert len(counting_builder) == 2
    assert len(list(tmp_path.glob("*.pkl"))) == 2


def test_changed_tokenizer_invalidates_cache(sa_bench, counting_builder, tmp_path):
    """A model swapped in at the same path must not silently reuse prompts."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tokenizer_json = model_dir / "tokenizer.json"
    tokenizer_json.write_text('{"vocab": "v1"}')
    cache_dir = str(tmp_path / "cache")

    _load(sa_bench, cache_dir, tokenizer_id=str(model_dir))
    tokenizer_json.write_text('{"vocab": "v2"}')
    _load(sa_bench, cache_dir, tokenizer_id=str(model_dir))

    assert len(counting_builder) == 2


def test_corrupt_cache_file_is_regenerated(sa_bench, counting_builder, tmp_path):
    """A truncated file must degrade to a rebuild, never crash the benchmark."""
    _load(sa_bench, str(tmp_path))
    (cache_file,) = list(tmp_path.glob("*.pkl"))
    cache_file.write_bytes(b"not a pickle")

    requests = _load(sa_bench, str(tmp_path))

    assert len(counting_builder) == 2
    assert len(requests) == 32


def test_unwritable_cache_dir_does_not_fail_the_run(sa_bench, counting_builder, tmp_path):
    """Caching is best effort: a read-only dir still yields a usable dataset."""
    cache_dir = tmp_path / "read-only"
    cache_dir.mkdir(mode=0o500)

    requests = _load(sa_bench, str(cache_dir))

    assert len(requests) == 32
