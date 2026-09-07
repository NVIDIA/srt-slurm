# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-type benchmark field split: schema 2 rejects stray fields, schema 1 warns."""

from __future__ import annotations

import dataclasses
import logging

import pytest
import yaml

from srtctl import benchmarks
from srtctl.benchmarks.base import SHARED_BENCHMARK_FIELDS, benchmark_config_fields
from srtctl.core.schema import BenchmarkConfig, SrtConfig

HEAD = """
name: fields-test
model:
  path: /m
  container: /c.sqsh
  precision: bf16
resources:
  gpu_type: h100
  gpus_per_node: 8
  agg_nodes: 1
  agg_workers: 1
backend:
  type: sglang
"""


def _load(benchmark: str, schema: int | None = 2) -> SrtConfig:
    text = HEAD + benchmark
    if schema is not None:
        text = f"schema: {schema}\n" + text
    return SrtConfig.Schema().load(yaml.safe_load(text))


def test_every_declared_field_exists_on_benchmark_config() -> None:
    names = {f.name for f in dataclasses.fields(BenchmarkConfig)}
    assert names >= SHARED_BENCHMARK_FIELDS
    for btype in benchmarks.list_benchmarks():
        assert benchmark_config_fields(btype) <= names, btype
    # manual has no runner: shared fields only
    assert benchmark_config_fields("manual") == SHARED_BENCHMARK_FIELDS


def test_each_type_accepts_the_fields_its_recipes_use() -> None:
    # Field sets seen across the historical recipe corpus, per type.
    _load(
        "benchmark:\n  type: sa-bench\n  isl: 1024\n  osl: 1024\n  concurrencies: '4x8'\n  req_rate: inf\n"
        "  random_range_ratio: 0.8\n  num_prompts_mult: 10\n  num_warmup_mult: 2\n  custom_tokenizer: a.B\n"
        "  use_chat_template: false\n  slow_down_sleep_time: 0.1\n  slow_down_wait_time: 5\n"
    )
    _load(
        "benchmark:\n  type: gsm8k\n  num_examples: 100\n  max_tokens: 256\n  num_threads: 8\n  num_shots: 5\n"
        "  temperature: 0.0\n  top_p: 1.0\n  top_k: 1\n"
    )
    _load("benchmark:\n  type: gpqa\n  num_examples: 10\n  max_tokens: 64\n  repeat: 2\n  num_threads: 4\n")
    _load(
        "benchmark:\n  type: mooncake-router\n  mooncake_workload: conversation\n  ttft_threshold_ms: 2000\n"
        "  itl_threshold_ms: 25\n"
    )
    _load(
        "benchmark:\n  type: trace-replay\n  trace_file: /t.jsonl\n  concurrencies: '4'\n  ttft_threshold_ms: 1\n"
        "  itl_threshold_ms: 1\n  aiperf_package: aiperf>=0.7\n"
    )
    _load("benchmark:\n  type: custom\n  command: echo hi\n  env:\n    A: b\n")
    _load(
        "benchmark:\n  type: agentperf\n  concurrencies: '4'\n  agentperf_client_dir: /c\n  agentperf_config: /c.yaml\n"
    )


def test_shared_fields_are_accepted_for_every_type() -> None:
    config = _load(
        "benchmark:\n  type: gsm8k\n  num_examples: 5\n  client_placement: last_decode\n  export_node_metrics: true\n"
        "  aiperf_args:\n    workers-max: 8\n"
    )
    assert config.benchmark.client_placement == "last_decode"


def test_schema_2_rejects_a_field_the_type_does_not_use() -> None:
    with pytest.raises(ValueError, match="benchmark.type 'gsm8k' does not use isl, osl") as exc:
        _load("benchmark:\n  type: gsm8k\n  num_examples: 5\n  isl: 1024\n  osl: 128\n")
    assert "fields it accepts" in str(exc.value)
    with pytest.raises(ValueError, match="'sa-bench' does not use num_shots"):
        _load("benchmark:\n  type: sa-bench\n  isl: 1\n  osl: 1\n  concurrencies: '1'\n  num_shots: 5\n")
    with pytest.raises(ValueError, match="'manual' does not use concurrencies, isl, osl"):
        _load("benchmark:\n  type: manual\n  isl: 1\n  osl: 1\n  concurrencies: '1'\n")


def test_schema_1_only_warns(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="srtctl.core.schema"):
        config = _load("benchmark:\n  type: gsm8k\n  num_examples: 5\n  isl: 1024\n", schema=None)
    assert config.benchmark.isl == 1024
    assert "benchmark.type 'gsm8k' does not use isl" in caplog.text
    assert "schema: 2 recipe would be rejected" in caplog.text


def test_defaults_never_count_as_set() -> None:
    # req_rate defaults to "inf" and use_chat_template to True; leaving them alone is not "setting" them.
    config = _load("benchmark:\n  type: gsm8k\n  num_examples: 5\n")
    assert config.benchmark.req_rate == "inf"
    assert config.benchmark.use_chat_template is True
