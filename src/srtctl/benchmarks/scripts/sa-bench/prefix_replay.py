#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed and exactly replay long prompts to isolate aggregate decode work."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp
import numpy as np
from backend_request_func import (
    RequestFuncInput,
    RequestFuncOutput,
    async_request_dynamo_completions,
    create_dynamo_session,
)
from benchmark_serving import load_tokenizer, sample_random_requests


@dataclass
class WaveSummary:
    name: str
    duration_s: float
    completed: int
    failed: int
    input_tokens: int
    cached_input_tokens_reported: int | None
    cached_input_tokens_reported_requests: int
    output_tokens: int
    request_throughput: float
    output_token_throughput: float
    logical_total_token_throughput: float
    logical_total_token_throughput_per_gpu: float
    uncached_total_token_throughput: float | None
    uncached_total_token_throughput_per_gpu: float | None
    output_token_throughput_per_gpu: float
    median_ttft_ms: float | None
    median_tpot_ms: float | None
    interactivity_tokens_per_s_per_user: float | None
    errors: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--isl", type=int, required=True)
    parser.add_argument("--osl", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--seed-osl", type=int, default=1)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--total-gpus", type=int, required=True)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    return parser.parse_args()


def _percentile_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


async def _scrape_metrics() -> dict[str, float]:
    raw_urls = os.environ.get("AIPERF_SERVER_METRICS_URLS", "")
    urls = [url.strip() for url in raw_urls.split(",") if url.strip()]
    totals: dict[str, float] = {}
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
        for url in urls:
            try:
                async with session.get(url) as response:
                    text = await response.text()
                    if response.status != 200:
                        continue
            except (aiohttp.ClientError, TimeoutError):
                continue
            for line in text.splitlines():
                if not line or line.startswith("#") or "prefix_cache" not in line:
                    continue
                fields = line.rsplit(None, 1)
                if len(fields) != 2:
                    continue
                try:
                    value = float(fields[1])
                except ValueError:
                    continue
                name = fields[0].split("{", 1)[0]
                totals[name] = totals.get(name, 0.0) + value
    return totals


def _metrics_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: value - before.get(key, 0.0) for key, value in after.items() if value != before.get(key, 0.0)}


def _one_token_filler(tokenizer) -> str:
    """Find a boring string whose repetitions remain one token apiece."""
    for candidate in (" hello", " the", " a", " x", "0", "\n"):
        if len(tokenizer.encode(candidate, add_special_tokens=False)) != 1:
            continue
        if len(tokenizer.encode(candidate * 64, add_special_tokens=False)) == 64:
            return candidate
    raise RuntimeError("could not find a stable one-token filler for exact-ISL prompts")


def _force_exact_prompt_length(prompt: str, tokenizer, target: int, filler: str) -> str:
    """Round-trip a generated prompt until its actual token count is exact."""
    for _ in range(32):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        difference = target - len(token_ids)
        if difference == 0:
            return prompt
        if difference < 0:
            prompt = tokenizer.decode(token_ids[:target])
        else:
            prompt += filler * difference
    actual = len(tokenizer.encode(prompt, add_special_tokens=False))
    raise RuntimeError(f"could not normalize prompt to ISL={target}; final length={actual}")


def _profile_endpoints() -> list[str]:
    return [endpoint for endpoint in os.environ.get("PROFILE_AGG_ENDPOINTS", "").split(",") if endpoint]


async def _profile_request(action: str) -> None:
    if os.environ.get("PROFILE_TYPE", "none") in ("", "none"):
        return
    start = int(os.environ.get("PROFILE_AGG_START_STEP", "0"))
    stop = int(os.environ.get("PROFILE_AGG_STOP_STEP", "50"))
    frontend = os.environ.get("SRTCTL_FRONTEND_TYPE", "")
    path = f"/engine/control/{action}_profile" if frontend == "dynamo" else f"/{action}_profile"
    payload: dict[str, object] = {}
    if action == "start":
        payload = {
            "output_dir": f"{os.environ['PROFILE_OUTPUT_DIR']}/agg",
            "start_step": start,
            "num_steps": max(stop - start, 1),
            "activities": ["CUDA_PROFILER"],
        }
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
        for endpoint in _profile_endpoints():
            url = f"http://{endpoint}{path}"
            async with session.post(url, json=payload) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"profiling request failed: {url} -> {response.status}: {body[:500]}")


async def _run_wave(
    name: str,
    prompts: list[tuple[str, int, int, None]],
    output_len: int,
    endpoint: str,
    model: str,
    dp_size: int,
    total_gpus: int,
) -> tuple[WaveSummary, list[RequestFuncOutput]]:
    api_url = endpoint.rstrip("/") + "/v1/completions"
    async with create_dynamo_session() as session:
        tasks = []
        start = time.perf_counter()
        for index, (prompt, prompt_len, _, _) in enumerate(prompts):
            headers = {"X-data-parallel-rank": str(index % dp_size)} if dp_size > 1 else None
            request = RequestFuncInput(
                prompt=prompt,
                api_url=api_url,
                prompt_len=prompt_len,
                output_len=output_len,
                model=model,
                model_name=model,
                ignore_eos=True,
                extra_headers=headers,
            )
            tasks.append(
                asyncio.create_task(async_request_dynamo_completions(request_func_input=request, session=session))
            )
        outputs = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start

    completed_outputs = [output for output in outputs if output.success]
    output_tokens = sum(output.output_tokens or len(output.itl) + 1 for output in completed_outputs)
    input_tokens = sum(output.prompt_len for output in completed_outputs)
    reported_cached = [
        output.usage_cached_tokens for output in completed_outputs if output.usage_cached_tokens is not None
    ]
    cached_input_tokens = sum(reported_cached) if reported_cached else None
    # Only derive physical/uncached input throughput when every successful
    # response reported cached-token usage; a partial sum would be misleading.
    fully_reported_cache_usage = len(reported_cached) == len(completed_outputs) and bool(completed_outputs)
    uncached_total_tokens = (
        input_tokens - cached_input_tokens + output_tokens
        if fully_reported_cache_usage and cached_input_tokens is not None
        else None
    )
    ttfts = [output.ttft * 1000 for output in completed_outputs]
    tpots = [output.tpot * 1000 for output in completed_outputs if output.tpot > 0]
    if not tpots:
        tpots = [
            (output.latency - output.ttft) * 1000 / (max((output.output_tokens or len(output.itl) + 1) - 1, 1))
            for output in completed_outputs
            if output.latency > output.ttft
        ]
    median_tpot = _percentile_median(tpots)
    errors = [output.error[-1000:] for output in outputs if not output.success]
    summary = WaveSummary(
        name=name,
        duration_s=duration,
        completed=len(completed_outputs),
        failed=len(outputs) - len(completed_outputs),
        input_tokens=input_tokens,
        cached_input_tokens_reported=cached_input_tokens,
        cached_input_tokens_reported_requests=len(reported_cached),
        output_tokens=output_tokens,
        request_throughput=len(completed_outputs) / duration,
        output_token_throughput=output_tokens / duration,
        logical_total_token_throughput=(input_tokens + output_tokens) / duration,
        logical_total_token_throughput_per_gpu=(input_tokens + output_tokens) / duration / total_gpus,
        uncached_total_token_throughput=(
            uncached_total_tokens / duration if uncached_total_tokens is not None else None
        ),
        uncached_total_token_throughput_per_gpu=(
            uncached_total_tokens / duration / total_gpus if uncached_total_tokens is not None else None
        ),
        output_token_throughput_per_gpu=output_tokens / duration / total_gpus,
        median_ttft_ms=_percentile_median(ttfts),
        median_tpot_ms=median_tpot,
        interactivity_tokens_per_s_per_user=1000 / median_tpot if median_tpot else None,
        errors=errors,
    )
    return summary, outputs


async def main() -> int:
    args = parse_args()
    np.random.seed(42)
    tokenizer = load_tokenizer(args.tokenizer, "auto", True, None)
    prompts = sample_random_requests(
        prefix_len=0,
        input_len=args.isl,
        output_len=args.osl,
        num_prompts=args.concurrency,
        range_ratio=1.0,
        tokenizer=tokenizer,
        use_chat_template=False,
        num_workers=8,
        tokenizer_id=args.tokenizer,
        trust_remote_code=True,
    )
    filler = _one_token_filler(tokenizer)
    prompts = [
        (_force_exact_prompt_length(prompt, tokenizer, args.isl, filler), args.isl, output_len, multimodal)
        for prompt, _, output_len, multimodal in prompts
    ]
    actual_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt, _, _, _ in prompts]
    if any(length != args.isl for length in actual_lengths):
        raise RuntimeError(f"generated prompts are not exact ISL={args.isl}: {actual_lengths}")

    prompt_hashes = [hashlib.sha256(prompt.encode()).hexdigest() for prompt, _, _, _ in prompts]
    before = await _scrape_metrics()
    seed_summary, _ = await _run_wave(
        "seed", prompts, args.seed_osl, args.endpoint, args.model, args.dp_size, args.total_gpus
    )
    after_seed = await _scrape_metrics()
    if seed_summary.failed:
        raise RuntimeError(f"cache-seeding wave failed {seed_summary.failed} request(s): {seed_summary.errors[:3]}")

    await asyncio.sleep(args.settle_seconds)
    await _profile_request("start")
    try:
        replay_summary, _ = await _run_wave(
            "replay", prompts, args.osl, args.endpoint, args.model, args.dp_size, args.total_gpus
        )
    finally:
        await _profile_request("stop")
    after_replay = await _scrape_metrics()

    result = {
        "config": vars(args),
        "prompt_hashes": prompt_hashes,
        "prompt_lengths": actual_lengths,
        "seed": asdict(seed_summary),
        "replay": asdict(replay_summary),
        "prefix_cache_metric_delta_seed": _metrics_delta(before, after_seed),
        "prefix_cache_metric_delta_replay": _metrics_delta(after_seed, after_replay),
        "notes": {
            "logical_total_token_throughput": "Includes cached input tokens; do not treat it as physical GPU token work.",
            "uncached_total_token_throughput": (
                "Input tokens not reported as cached plus output tokens; emitted only when every completed request "
                "reports cached-token usage."
            ),
            "decode_comparison": "Use replay output throughput, TPOT/interactivity, iteration logs, and nsys kernels.",
        },
    }
    output_dir = Path("/logs") / f"prefix-replay-isl{args.isl}-osl{args.osl}-c{args.concurrency}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"Prefix-replay results: {output_path}", flush=True)
    return 1 if replay_summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
