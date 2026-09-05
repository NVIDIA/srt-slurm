#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <eval-factory-template.yaml> <task-name>" >&2
    exit 2
fi

template=$1
task_name=$2
target_url="http://${SRT_FRONTEND_HOST:?}:${SRT_FRONTEND_PORT:?}/v1/chat/completions"
output_dir="/logs/accuracy/${task_name}"
resolved_config="${output_dir}/config_ef.resolved.yaml"

if [ ! -r "$template" ]; then
    echo "NEL eval-factory template is not readable: $template" >&2
    exit 2
fi

mkdir -p "$output_dir"
sed \
    -e "s|__SRT_TARGET_URL__|${target_url}|g" \
    -e "s|__NEL_OUTPUT_DIR__|${output_dir}|g" \
    "$template" > "$resolved_config"
cp "$template" "${output_dir}/config_ef.template.yaml"

echo "NEL task: ${task_name}"
echo "Target: ${target_url}"
echo "Resolved config: ${resolved_config}"

# AA-LCR uses an external judge.  Read the credential from the ephemeral
# GitLab Secure File staged by AIB; never place it in the recipe, command line,
# stdout, or result artifacts.  A real chat request catches authorization and
# model-name errors before the expensive generation phase begins.
if [ "$task_name" = "ns_aa_lcr" ]; then
    : "${JUDGE_API_KEY_FILE:?AA-LCR requires JUDGE_API_KEY_FILE}"
    if [ ! -r "$JUDGE_API_KEY_FILE" ]; then
        echo "Judge key file is not readable" >&2
        exit 2
    fi
    INFERENCE_API_KEY=$(<"$JUDGE_API_KEY_FILE")
    export INFERENCE_API_KEY
    if [ -z "$INFERENCE_API_KEY" ]; then
        echo "Judge key file is empty" >&2
        exit 2
    fi

    judge_probe="${output_dir}/judge-probe.json"
    judge_status=$(curl --silent --show-error \
        --output "$judge_probe" \
        --write-out '%{http_code}' \
        --connect-timeout 30 \
        --max-time 120 \
        --header "Authorization: Bearer ${INFERENCE_API_KEY}" \
        --header 'Content-Type: application/json' \
        --data '{"model":"nvidia/qwen/eccn-qwen-235b","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":2,"temperature":0}' \
        'https://inference-api.nvidia.com/v1/chat/completions')
    if [ "$judge_status" != "200" ]; then
        echo "Qwen-235B judge authorization probe failed with HTTP ${judge_status}" >&2
        exit 1
    fi
    echo "Qwen-235B judge authorization probe passed"
fi

evaluator=$(command -v nemo-evaluator || command -v eval-factory)
"$evaluator" run_eval --run_config "$resolved_config"
