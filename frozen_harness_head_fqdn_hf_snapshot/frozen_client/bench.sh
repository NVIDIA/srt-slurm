concurrency_levels=(4 8 16 32 64 128 256 512)

echo "test name"

for concurrency in "${concurrency_levels[@]}"
do
    num_prompts=$((concurrency * 5))
    echo "Running benchmark with concurrency: $concurrency and num-prompts: $num_prompts"

    python3 benchmark_serving.py \
        --model nvidia/Llama-3.3-70B-Instruct-FP8 \
        --dataset-name random \
        --num-prompts "$num_prompts" \
        --random-input-len 1024 \
        --random-output-len 1024 \
        --random-range-ratio 0.8 \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$concurrency"

    echo "Completed benchmark with concurrency: $concurrency"
    echo "-----------------------------------------"
done
