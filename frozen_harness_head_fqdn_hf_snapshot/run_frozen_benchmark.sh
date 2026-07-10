#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <point_id> <concurrency> <rounds> <logical_gpus>" >&2
    exit 2
fi

point_id=$1
concurrency=$2
rounds=$3
logical_gpus=$4
model=nvidia/Qwen3.5-397B-A17B-NVFP4
host=${SRT_BENCHMARK_HOST:-$(hostname -f)}
port=8000
tokenizer=/model
if [[ ! -d "$tokenizer" ]]; then
    echo "tokenizer snapshot does not exist: $tokenizer" >&2
    exit 2
fi
num_prompts=$((concurrency * rounds))
num_warmups=$((concurrency * 2))
wrapper_sha256=$(sha256sum "$0" | awk '{print $1}')
client_sha256=$(sha256sum /harness/frozen_client/benchmark_serving.py | awk '{print $1}')

ulimit -n 65536

python3 - "$point_id" "$concurrency" "$rounds" "$logical_gpus" "$num_prompts" "$num_warmups" "$wrapper_sha256" "$client_sha256" "$tokenizer" <<'PY'
import json
import os
import platform
import sys
from pathlib import Path

(
    point_id,
    concurrency,
    rounds,
    logical_gpus,
    num_prompts,
    num_warmups,
    wrapper_sha256,
    client_sha256,
    tokenizer,
) = sys.argv[1:]
payload = {
    "point_id": point_id,
    "concurrency": int(concurrency),
    "multi_round": int(rounds),
    "logical_gpus": int(logical_gpus),
    "num_prompts": int(num_prompts),
    "num_warmups": int(num_warmups),
    "hostname": platform.node(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "image_path": os.environ.get("CAMPAIGN_IMAGE_PATH"),
    "image_sha256": os.environ.get("CAMPAIGN_IMAGE_SHA256"),
    "trtllm_source": "/work",
    "benchmark_client": "/harness/frozen_client/benchmark_serving.py",
    "benchmark_wrapper_sha256": wrapper_sha256,
    "benchmark_client_sha256": client_sha256,
    "tokenizer": tokenizer,
}
Path("/logs/benchmark_identity.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "Pre-benchmark inference probe"
curl --silent --show-error --max-time 900 \
    "http://${host}:${port}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"nvidia/Qwen3.5-397B-A17B-NVFP4","prompt":"Once upon a time, there was a small town","max_tokens":50}' \
    > /logs/pre_benchmark_probe.json

echo "Starting frozen benchmark: point=${point_id} concurrency=${concurrency} rounds=${rounds} prompts=${num_prompts}"
python /harness/frozen_client/benchmark_serving.py \
    --model "$model" \
    --tokenizer "$tokenizer" \
    --backend openai \
    --host "$host" \
    --port "$port" \
    --dataset-name random \
    --num-prompts "$num_prompts" \
    --max-concurrency "$concurrency" \
    --num-warmups "$num_warmups" \
    --ignore-eos \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --random-range-ratio 0.8 \
    --save-result \
    --use-chat-template \
    --result-dir /logs \
    --result-filename result.json \
    --percentile-metrics ttft,tpot,itl,e2el

python3 - "$point_id" "$logical_gpus" <<'PY'
import json
import sys
from pathlib import Path

point_id = sys.argv[1]
logical_gpus = int(sys.argv[2])
result_path = Path("/logs/result.json")
data = json.loads(result_path.read_text())
detailed_keys = {
    "input_lens",
    "output_lens",
    "ttfts",
    "itls",
    "generated_texts",
    "errors",
    "acc_lengths",
}
detailed = {key: data.pop(key) for key in detailed_keys if key in data}
Path("/logs/result_detailed.json").write_text(json.dumps(detailed) + "\n")
result_path.write_text(json.dumps(data, indent=2) + "\n")

expected = int(data.get("num_prompts", 0) or 0)
completed = int(data.get("completed", 0) or 0)
completion_pct = completed / expected * 100.0 if expected else 0.0
median_tpot_ms = float(data.get("median_tpot_ms", 0.0) or 0.0)
total_token_throughput = float(data.get("total_token_throughput", 0.0) or 0.0)
summary = {
    "point_id": point_id,
    "expected": expected,
    "completed": completed,
    "completion_pct": completion_pct,
    "accepted_completion": expected > 0 and completion_pct >= 99.0,
    "logical_gpus": logical_gpus,
    "tps_per_user": 1000.0 / median_tpot_ms if median_tpot_ms > 0 else None,
    "total_tps_per_logical_gpu": total_token_throughput / logical_gpus if logical_gpus > 0 else None,
}
Path("/logs/campaign_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
if not summary["accepted_completion"]:
    raise SystemExit(
        f"completion gate failed: {completed}/{expected} ({completion_pct:.6f}%) is below 99%"
    )
PY

if ! curl --silent --show-error --max-time 900 \
    "http://${host}:${port}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"nvidia/Qwen3.5-397B-A17B-NVFP4","prompt":"Once upon a time, there was a small town","max_tokens":50}' \
    > /logs/post_benchmark_probe.json; then
    echo "Post-benchmark inference probe failed after an accepted >=99% run" >&2
    printf '%s\n' '{"probe_ok":false}' > /logs/post_benchmark_probe_status.json
else
    printf '%s\n' '{"probe_ok":true}' > /logs/post_benchmark_probe_status.json
fi

echo "Benchmark accepted at >=99% completion"
