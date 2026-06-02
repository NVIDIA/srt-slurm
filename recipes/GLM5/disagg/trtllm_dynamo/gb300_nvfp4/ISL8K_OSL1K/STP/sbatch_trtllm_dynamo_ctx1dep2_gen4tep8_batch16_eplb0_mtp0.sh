#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dynamo TRTLLM variant — ISL8K/OSL1K, ctx1dep2 gen4tep8 batch16 STP, lm-eval
#
#SBATCH --job-name=glm5_nvfp4_dynamo_ISL8K_OSL1K_ctx1dep2_gen4tep8_batch16_eplb0_mtp0
#SBATCH --nodes=10
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --segment=10
#SBATCH --account=restricted
#SBATCH --time=0
#SBATCH --output=/data/home/rihuo/srt-slurm/outputs/%j/logs/sweep_%j.log
#SBATCH --partition=batch_2

set -euo pipefail

SRTCTL_SOURCE="/data/home/rihuo/srt-slurm"
OUTPUT_BASE="/data/home/rihuo/srt-slurm/outputs"
OUTPUT_DIR="${OUTPUT_BASE}/${SLURM_JOB_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"
CONTAINER_IMAGE="/data/home/rihuo/dynamo-trtllm-arm64-tot-8c830c9.sqsh"
EVAL_CONTAINER_IMAGE="/data/home/rihuo/sglang-v0.5.10.post1-cu130.sqsh"
MODEL_PATH="/data/home/rihuo/nvidia_GLM-5-NVFP4"
MODEL_NAME="nvidia_GLM-5-NVFP4"
INFMAX_WORKSPACE="/data/home/rihuo/InferenceMAX"
SCRIPT_MOUNTS="${LOG_DIR}:/logs,${MODEL_PATH}:/model,${SRTCTL_SOURCE}/configs:/configs,${SRTCTL_SOURCE}/src/srtctl/benchmarks/scripts:/srtctl-benchmarks,${INFMAX_WORKSPACE}:/infmax-workspace"
TRTLLM_COMMON_ENV="export ENROOT_ALLOW_DEV=yes && export MIMALLOC_PURGE_DELAY=0 && export NCCL_GRAPH_MIXING_SUPPORT=0 && export TLLM_LOG_LEVEL=INFO && export TRTLLM_ENABLE_PDL=1 && export TRTLLM_SERVER_DISABLE_GC=1 && export TRTLLM_WORKER_DISABLE_GC=1"

# Dynamo ports
ETCD_PORT=2379
NATS_PORT=4222
FRONTEND_PORT=8000

# DYN_SYSTEM_PORT assignments (one per worker)
DYN_SYS_PORT_CTX0=8081
DYN_SYS_PORT_GEN0=8082
DYN_SYS_PORT_GEN1=8083
DYN_SYS_PORT_GEN2=8084
DYN_SYS_PORT_GEN3=8085

mkdir -p "${LOG_DIR}"
exec 2>&1

mapfile -t ALL_NODES < <(scontrol show hostnames "${SLURM_NODELIST}")
if [ "${#ALL_NODES[@]}" -lt 10 ]; then
    echo "ERROR: expected at least 10 nodes, got ${#ALL_NODES[@]}"
    exit 1
fi

HEAD_NODE="${ALL_NODES[0]}"
PREFILL_NODE="${ALL_NODES[1]}"
DECODE_NODES_0=("${ALL_NODES[2]}" "${ALL_NODES[3]}")
DECODE_NODES_1=("${ALL_NODES[4]}" "${ALL_NODES[5]}")
DECODE_NODES_2=("${ALL_NODES[6]}" "${ALL_NODES[7]}")
DECODE_NODES_3=("${ALL_NODES[8]}" "${ALL_NODES[9]}")
DECODE_NODELIST_0="$(IFS=,; echo "${DECODE_NODES_0[*]}")"
DECODE_NODELIST_1="$(IFS=,; echo "${DECODE_NODES_1[*]}")"
DECODE_NODELIST_2="$(IFS=,; echo "${DECODE_NODES_2[*]}")"
DECODE_NODELIST_3="$(IFS=,; echo "${DECODE_NODES_3[*]}")"

SRUN_PIDS=()

cleanup() {
    local exit_code=$?

    if [ "${#SRUN_PIDS[@]}" -gt 0 ]; then
        echo ""
        echo "Cleaning up ${#SRUN_PIDS[@]} background srun steps..."
        kill "${SRUN_PIDS[@]}" 2>/dev/null || true
        wait "${SRUN_PIDS[@]}" 2>/dev/null || true
    fi

    if [ $exit_code -eq 0 ]; then
        echo "✓ Sweep completed successfully"
    else
        echo "✗ Sweep failed (exit code: $exit_code)"
    fi
    echo "End: $(date)"
}
trap cleanup EXIT

start_bg() {
    "$@" &
    SRUN_PIDS+=($!)
}

require_alive() {
    local pid="$1"
    local name="$2"
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: ${name} exited unexpectedly"
        wait "$pid" || true
        exit 1
    fi
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local timeout="${3:-300}"
    local deadline=$((SECONDS + timeout))

    while [ "$SECONDS" -lt "$deadline" ]; do
        if bash -c "echo > /dev/tcp/${host}/${port}" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done

    echo "ERROR: port ${host}:${port} is not open after waiting ${timeout} seconds"
    return 1
}

wait_for_dynamo_workers() {
    local host="$1"
    local port="$2"
    local expected_prefill="$3"
    local expected_decode="$4"
    local timeout="${5:-2700}"
    local report_interval="${6:-60}"
    local deadline=$((SECONDS + timeout))
    local last_report=$SECONDS

    echo "Polling http://${host}:${port}/health every 10s for ${expected_prefill} prefills and ${expected_decode} decodes"

    while [ "$SECONDS" -lt "$deadline" ]; do
        local response
        if response=$(curl -fsS --max-time 5 "http://${host}:${port}/health" 2>/dev/null); then
            local prefill_count decode_count
            prefill_count=$(echo "${response}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(sum(1 for i in data.get('instances', [])
          if i.get('endpoint') == 'generate' and i.get('component') == 'prefill'))
" 2>/dev/null || echo "0")
            decode_count=$(echo "${response}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(sum(1 for i in data.get('instances', [])
          if i.get('endpoint') == 'generate' and i.get('component') in ('decode', 'tensorrt_llm', 'backend')))
" 2>/dev/null || echo "0")

            if [ "${prefill_count}" -ge "${expected_prefill}" ] && [ "${decode_count}" -ge "${expected_decode}" ]; then
                echo "All workers ready: ${prefill_count} prefills, ${decode_count} decodes"
                return 0
            fi

            if [ $((SECONDS - last_report)) -ge "${report_interval}" ]; then
                echo "Waiting: have ${prefill_count}/${expected_prefill} prefills, ${decode_count}/${expected_decode} decodes"
                last_report=$SECONDS
            fi
        else
            if [ $((SECONDS - last_report)) -ge "${report_interval}" ]; then
                echo "Waiting: frontend not responding yet"
                last_report=$SECONDS
            fi
        fi
        sleep 10
    done

    echo "ERROR: workers did not register within ${timeout}s"
    return 1
}

verify_model_ready() {
    local host="$1"
    local port="$2"
    local timeout="${3:-120}"
    local deadline=$((SECONDS + timeout))

    echo "Verifying model is ready via /v1/models..."

    while [ "$SECONDS" -lt "$deadline" ]; do
        local response
        if response=$(curl -fsS --max-time 5 "http://${host}:${port}/v1/models" 2>/dev/null); then
            local model_count
            model_count=$(echo "${response}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('data', [])
for m in models:
    print(f\"  model: {m.get('id', 'unknown')}\", file=sys.stderr)
print(len(models))
" 2>/dev/null || echo "0")

            if [ "${model_count}" -gt 0 ]; then
                echo "Model is serving (${model_count} model(s) available)"
                return 0
            fi
        fi
        sleep 5
    done

    echo "ERROR: /v1/models did not return any models within ${timeout}s"
    return 1
}

write_prefill_config() {
    local output_path="$1"
    cat > "${output_path}" <<'EOF'
tensor_parallel_size: 2
moe_expert_parallel_size: 2
pipeline_parallel_size: 1
enable_attention_dp: true
disable_overlap_scheduler: true
trust_remote_code: true
max_batch_size: 2
max_num_tokens: 16640
max_seq_len: 8232
print_iter_log: true
cuda_graph_config: null
moe_config:
  backend: CUTEDSL
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.6
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 16384
EOF
}

write_decode_config() {
    local output_path="$1"
    cat > "${output_path}" <<'EOF'
allreduce_strategy: MNNVL
tensor_parallel_size: 8
moe_expert_parallel_size: 8
pipeline_parallel_size: 1
enable_attention_dp: false
enable_lm_head_tp_in_adp: false
trust_remote_code: true
max_batch_size: 16
max_num_tokens: 16
max_seq_len: 9256
print_iter_log: true
stream_interval: 100
num_postprocess_workers: 4
cuda_graph_config:
  enable_padding: true
  batch_sizes:
    - 1
    - 2
    - 4
    - 8
    - 16
moe_config:
  backend: TRTLLM
  use_low_precision_moe_combine: true
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.9
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 16384
nvfp4_gemm_config:
  allowed_backends:
    - cutlass
    - cublaslt
    - cutedsl
    - cuda_core
EOF
}

echo "=========================================="
echo "Dynamo TRTLLM Disaggregated Eval"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Container: ${CONTAINER_IMAGE}"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "Head node (infra + frontend): ${HEAD_NODE}"
echo "Prefill node: ${PREFILL_NODE}"
echo "Decode nodes: ${DECODE_NODELIST_0}, ${DECODE_NODELIST_1}, ${DECODE_NODELIST_2}, ${DECODE_NODELIST_3}"
echo ""

write_prefill_config "${LOG_DIR}/trtllm_prefill.yaml"
write_decode_config "${LOG_DIR}/trtllm_decode.yaml"

# ==============================================================================
# Stage 1: Start infrastructure services (etcd + nats) on head node
# ==============================================================================
echo "Starting etcd and nats on head node ${HEAD_NODE}..."

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --nodes 1 \
    --ntasks 1 \
    --nodelist "${HEAD_NODE}" \
    --output "${LOG_DIR}/${HEAD_NODE}_infra.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c '
        set -euo pipefail

        rm -rf /tmp/etcd /tmp/nats
        mkdir -p /tmp/etcd /tmp/nats

        HOST_IP=$(hostname -I | awk "{print \$1}")
        echo "Infra node IP: ${HOST_IP}"

        echo "Starting nats-server..."
        nats-server -js -sd /tmp/nats &
        NATS_PID=$!

        echo "Starting etcd..."
        etcd \
            --data-dir /tmp/etcd \
            --listen-client-urls http://0.0.0.0:'"${ETCD_PORT}"' \
            --advertise-client-urls http://${HOST_IP}:'"${ETCD_PORT}"' &
        ETCD_PID=$!

        # Wait for both services to be ready
        for i in $(seq 1 300); do
            if echo > /dev/tcp/localhost/'"${NATS_PORT}"' 2>/dev/null && \
               echo > /dev/tcp/localhost/'"${ETCD_PORT}"' 2>/dev/null; then
                echo "etcd and nats are ready"
                break
            fi
            sleep 1
        done

        echo "Infrastructure services running (nats PID: ${NATS_PID}, etcd PID: ${ETCD_PID})"

        # Keep running until killed
        wait
    '
INFRA_PID="${SRUN_PIDS[-1]}"
require_alive "${INFRA_PID}" "INFRA_PID"

echo "Waiting for nats (port ${NATS_PORT}) on ${HEAD_NODE}..."
wait_for_port "${HEAD_NODE}" "${NATS_PORT}" 300

echo "Waiting for etcd (port ${ETCD_PORT}) on ${HEAD_NODE}..."
wait_for_port "${HEAD_NODE}" "${ETCD_PORT}" 300
echo "Infrastructure services are ready"

# ==============================================================================
# Stage 2: Start TRTLLM workers via dynamo.trtllm
# ==============================================================================
DYNAMO_WORKER_ENV="export ETCD_ENDPOINTS=http://${HEAD_NODE}:${ETCD_PORT} && export NATS_SERVER=nats://${HEAD_NODE}:${NATS_PORT} && export DYN_REQUEST_PLANE=nats"

echo "Starting prefill worker (1x TP2/EP2)"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 1 \
    --ntasks 2 \
    --nodelist "${PREFILL_NODE}" \
    --output "${LOG_DIR}/${PREFILL_NODE}_prefill_w0.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && ${DYNAMO_WORKER_ENV} && export DYN_SYSTEM_PORT=${DYN_SYS_PORT_CTX0} && export CUDA_VISIBLE_DEVICES=0,1 && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /model --served-model-name ${MODEL_NAME} --disaggregation-mode prefill --extra-engine-args /logs/trtllm_prefill.yaml --request-plane nats"
CTX0_PID="${SRUN_PIDS[-1]}"

echo "Starting decode workers (4x TP8/EP8 MNNVL, 2 nodes each)"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 2 \
    --ntasks 8 \
    --nodelist "${DECODE_NODELIST_0}" \
    --output "${LOG_DIR}/${DECODE_NODES_0[0]}_decode_w0.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && ${DYNAMO_WORKER_ENV} && export DYN_SYSTEM_PORT=${DYN_SYS_PORT_GEN0} && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /model --served-model-name ${MODEL_NAME} --disaggregation-mode decode --extra-engine-args /logs/trtllm_decode.yaml --request-plane nats"
GEN0_PID="${SRUN_PIDS[-1]}"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 2 \
    --ntasks 8 \
    --nodelist "${DECODE_NODELIST_1}" \
    --output "${LOG_DIR}/${DECODE_NODES_1[0]}_decode_w1.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && ${DYNAMO_WORKER_ENV} && export DYN_SYSTEM_PORT=${DYN_SYS_PORT_GEN1} && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /model --served-model-name ${MODEL_NAME} --disaggregation-mode decode --extra-engine-args /logs/trtllm_decode.yaml --request-plane nats"
GEN1_PID="${SRUN_PIDS[-1]}"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 2 \
    --ntasks 8 \
    --nodelist "${DECODE_NODELIST_2}" \
    --output "${LOG_DIR}/${DECODE_NODES_2[0]}_decode_w2.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && ${DYNAMO_WORKER_ENV} && export DYN_SYSTEM_PORT=${DYN_SYS_PORT_GEN2} && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /model --served-model-name ${MODEL_NAME} --disaggregation-mode decode --extra-engine-args /logs/trtllm_decode.yaml --request-plane nats"
GEN2_PID="${SRUN_PIDS[-1]}"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 2 \
    --ntasks 8 \
    --nodelist "${DECODE_NODELIST_3}" \
    --output "${LOG_DIR}/${DECODE_NODES_3[0]}_decode_w3.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${TRTLLM_COMMON_ENV} && ${DYNAMO_WORKER_ENV} && export DYN_SYSTEM_PORT=${DYN_SYS_PORT_GEN3} && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /model --served-model-name ${MODEL_NAME} --disaggregation-mode decode --extra-engine-args /logs/trtllm_decode.yaml --request-plane nats"
GEN3_PID="${SRUN_PIDS[-1]}"

for pid_name in CTX0_PID GEN0_PID GEN1_PID GEN2_PID GEN3_PID; do
    require_alive "${!pid_name}" "${pid_name}"
done

# ==============================================================================
# Stage 3: Start dynamo frontend on head node
# ==============================================================================
echo "Starting dynamo frontend on ${HEAD_NODE}"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 1 \
    --ntasks 1 \
    --nodelist "${HEAD_NODE}" \
    --output "${LOG_DIR}/${HEAD_NODE}_frontend.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "export ETCD_ENDPOINTS=http://${HEAD_NODE}:${ETCD_PORT} && export NATS_SERVER=nats://${HEAD_NODE}:${NATS_PORT} && export DYN_REQUEST_PLANE=nats && python3 -m dynamo.frontend --http-port ${FRONTEND_PORT}"
FRONTEND_PID="${SRUN_PIDS[-1]}"
require_alive "${FRONTEND_PID}" "FRONTEND_PID"

# ==============================================================================
# Stage 4: Wait for all workers to register, then verify model is serving
# ==============================================================================
EXPECTED_PREFILL=1
EXPECTED_DECODE=4

echo "Waiting for ${EXPECTED_PREFILL} prefill and ${EXPECTED_DECODE} decode workers to register..."
if ! wait_for_dynamo_workers "${HEAD_NODE}" "${FRONTEND_PORT}" "${EXPECTED_PREFILL}" "${EXPECTED_DECODE}" 2700 60; then
    echo "ERROR: workers did not become healthy"
    exit 1
fi

if ! verify_model_ready "${HEAD_NODE}" "${FRONTEND_PORT}" 120; then
    echo "ERROR: model is not serving"
    exit 1
fi

echo "All workers healthy and model is serving - starting post eval (lm-eval)"

# ==============================================================================
# Stage 5: Run post eval (lm-eval)
# ==============================================================================
srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --nodes 1 \
    --ntasks 1 \
    --nodelist "${HEAD_NODE}" \
    --output "${LOG_DIR}/eval.out" \
    --container-image "${EVAL_CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    --export="ALL,MODEL_NAME=${MODEL_NAME},EVAL_CONC=84,RUN_EVAL=true,IS_MULTINODE=true,FRAMEWORK=trtllm,PRECISION=fp4,MODEL=/model,PREFILL_TP=2,PREFILL_EP=2,PREFILL_DP_ATTN=true,PREFILL_NUM_WORKERS=1,DECODE_TP=8,DECODE_EP=8,DECODE_DP_ATTN=false,DECODE_NUM_WORKERS=4" \
    bash -c "bash /srtctl-benchmarks/lm-eval/bench.sh http://localhost:${FRONTEND_PORT} /infmax-workspace"
