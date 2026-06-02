#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dynamo TRTLLM variant of sbatch_trtllm_eval.sh
# Uses etcd + nats + dynamo frontend instead of trtllm-serve disaggregated
#
# Topology: 10 prefill (TP4/EP4) + 1 decode (TP16/EP16) + 1 head
#
#SBATCH --job-name=dsr1_fp8_dynamo_ISL8K_OSL1K_ctx10dep4_gen1dep16_batch64_eplb0_mtp1
#SBATCH --nodes=15
#SBATCH --ntasks=60
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --account=coreai_comparch_aarwlt
#SBATCH --time=10:00:00
#SBATCH --output=/home/rihuo/srt-slurm/outputs/%j/logs/sweep_%j.log
#SBATCH --partition=batch_long

set -euo pipefail

SRTCTL_SOURCE="/home/rihuo/srt-slurm"
OUTPUT_BASE="/home/rihuo/srt-slurm/outputs"
OUTPUT_DIR="${OUTPUT_BASE}/${SLURM_JOB_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"
CONTAINER_IMAGE="/lustre/fsw/portfolios/coreai/projects/coreai_comparch_aarwlt/users/rihuo/dynamo-trtllm-rihuo-arm64-tot-8c830c9.sqsh"
EVAL_CONTAINER_IMAGE="/lustre/fsw/portfolios/coreai/projects/coreai_comparch_aarwlt/users/rihuo/dynamo-trtllm-rihuo-arm64-tot-8c830c9.sqsh"
MODEL_PATH="/lustre/fsw/portfolios/coreai/projects/coreai_comparch_aarwlt/users/rihuo/deepseek-ai_DeepSeek-R1-0528"
MODEL_NAME="deepseek-ai/DeepSeek-R1-0528"
INFMAX_WORKSPACE="/home/rihuo/InferenceMAX"
SCRIPT_MOUNTS="${LOG_DIR}:/logs,${MODEL_PATH}:/model,${SRTCTL_SOURCE}/configs:/configs,${SRTCTL_SOURCE}/src/srtctl/benchmarks/scripts:/srtctl-benchmarks,${INFMAX_WORKSPACE}:/infmax-workspace"

# Environment variables from config
PREFILL_ENV="export TLLM_OVERRIDE_LAYER_NUM=61 && export TLLM_LOG_LEVEL=INFO && export TRTLLM_SERVER_DISABLE_GC=1 && export TRTLLM_WORKER_DISABLE_GC=1 && export TRTLLM_ENABLE_PDL=1 && export ENROOT_ALLOW_DEV=yes && export NCCL_GRAPH_MIXING_SUPPORT=0 && export TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED && export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp"
DECODE_ENV="export TLLM_OVERRIDE_LAYER_NUM=61 && export TLLM_LOG_LEVEL=INFO && export TRTLLM_SERVER_DISABLE_GC=1 && export TRTLLM_WORKER_DISABLE_GC=1 && export TRTLLM_ENABLE_PDL=1 && export ENROOT_ALLOW_DEV=yes && export NCCL_GRAPH_MIXING_SUPPORT=0 && export TRTLLM_FORCE_COMM_METHOD=NVLINK_TWO_SIDED && export ENABLE_CONFIGURABLE_MOE=1 && export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp"

# Dynamo ports
ETCD_PORT=2379
NATS_PORT=4222
FRONTEND_PORT=8000

# DYN_SYSTEM_PORT assignments (one per worker)
DYN_SYS_PORT_BASE_CTX=8081
DYN_SYS_PORT_GEN=8091

NUM_PREFILL=10
NUM_DECODE=1

mkdir -p "${LOG_DIR}"
exec 2>&1

mapfile -t ALL_NODES < <(scontrol show hostnames "${SLURM_NODELIST}")
TOTAL_NODES_NEEDED=$((1 + NUM_PREFILL + 4))  # 1 head + 10 prefill + 4 decode
if [ "${#ALL_NODES[@]}" -lt "${TOTAL_NODES_NEEDED}" ]; then
    echo "ERROR: expected at least ${TOTAL_NODES_NEEDED} nodes, got ${#ALL_NODES[@]}"
    exit 1
fi

HEAD_NODE="${ALL_NODES[0]}"

PREFILL_NODES=()
for i in $(seq 1 ${NUM_PREFILL}); do
    PREFILL_NODES+=("${ALL_NODES[$i]}")
done

DECODE_NODES=()
for i in $(seq $((NUM_PREFILL + 1)) $((NUM_PREFILL + 4))); do
    DECODE_NODES+=("${ALL_NODES[$i]}")
done
DECODE_NODELIST="$(IFS=,; echo "${DECODE_NODES[*]}")"

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
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 16384
cuda_graph_config: null
disable_overlap_scheduler: true
enable_attention_dp: true
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.1
max_batch_size: 2
max_num_tokens: 16384
max_seq_len: 8232
moe_config:
  backend: DEEPGEMM
moe_expert_parallel_size: 4
pipeline_parallel_size: 1
print_iter_log: true
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 1
tensor_parallel_size: 4
EOF
}

write_decode_config() {
    local output_path="$1"
    cat > "${output_path}" <<'EOF'
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 16384
cuda_graph_config:
  batch_sizes:
    - 1
    - 2
    - 4
    - 8
    - 16
    - 24
    - 32
    - 40
    - 48
    - 56
    - 64
  enable_padding: true
enable_attention_dp: true
enable_lm_head_tp_in_adp: true
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.7
max_batch_size: 64
max_num_tokens: 128
max_seq_len: 9256
moe_config:
  backend: DEEPGEMM
  use_low_precision_moe_combine: true
moe_expert_parallel_size: 16
num_postprocess_workers: 4
pipeline_parallel_size: 1
print_iter_log: true
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 1
stream_interval: 100
tensor_parallel_size: 16
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
echo "Prefill nodes (${NUM_PREFILL}x TP4/EP4): ${PREFILL_NODES[*]}"
echo "Decode nodes (1x TP16/EP16): ${DECODE_NODELIST}"
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

echo "Starting ${NUM_PREFILL} prefill workers (TP4/EP4, 1 node each)"

CTX_PIDS=()
for i in $(seq 0 $((NUM_PREFILL - 1))); do
    port=$((DYN_SYS_PORT_BASE_CTX + i))
    node="${PREFILL_NODES[$i]}"

    start_bg srun \
        --jobid "${SLURM_JOB_ID}" \
        --overlap \
        --mpi pmix \
        --nodes 1 \
        --ntasks 4 \
        --nodelist "${node}" \
        --output "${LOG_DIR}/${node}_prefill_w${i}.out" \
        --container-image "${CONTAINER_IMAGE}" \
        --no-container-entrypoint \
        --no-container-mount-home \
        --container-mounts "${SCRIPT_MOUNTS}" \
        bash -c "${PREFILL_ENV} && ${DYNAMO_WORKER_ENV} && export DYN_SYSTEM_PORT=${port} && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /model --served-model-name ${MODEL_NAME} --disaggregation-mode prefill --extra-engine-args /logs/trtllm_prefill.yaml --request-plane nats"
    CTX_PIDS+=("${SRUN_PIDS[-1]}")
done

echo "Starting 1 decode worker (TP16/EP16, 4 nodes)"

start_bg srun \
    --jobid "${SLURM_JOB_ID}" \
    --overlap \
    --mpi pmix \
    --nodes 4 \
    --ntasks 16 \
    --nodelist "${DECODE_NODELIST}" \
    --output "${LOG_DIR}/${DECODE_NODES[0]}_decode_w0.out" \
    --container-image "${CONTAINER_IMAGE}" \
    --no-container-entrypoint \
    --no-container-mount-home \
    --container-mounts "${SCRIPT_MOUNTS}" \
    bash -c "${DECODE_ENV} && ${DYNAMO_WORKER_ENV} && export DYN_SYSTEM_PORT=${DYN_SYS_PORT_GEN} && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path /model --served-model-name ${MODEL_NAME} --disaggregation-mode decode --extra-engine-args /logs/trtllm_decode.yaml --request-plane nats"
GEN0_PID="${SRUN_PIDS[-1]}"

for pid in "${CTX_PIDS[@]}"; do
    require_alive "${pid}" "PREFILL_PID"
done
require_alive "${GEN0_PID}" "GEN0_PID"

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
EXPECTED_PREFILL=${NUM_PREFILL}
EXPECTED_DECODE=${NUM_DECODE}

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
    --export="ALL,MODEL_NAME=${MODEL_NAME},EVAL_CONC=1229,RUN_EVAL=true,IS_MULTINODE=true,FRAMEWORK=trtllm,PRECISION=fp8,MODEL=/model,PREFILL_TP=4,PREFILL_EP=4,PREFILL_DP_ATTN=true,PREFILL_NUM_WORKERS=${NUM_PREFILL},DECODE_TP=16,DECODE_EP=16,DECODE_DP_ATTN=true,DECODE_NUM_WORKERS=${NUM_DECODE}" \
    bash -c "bash /srtctl-benchmarks/lm-eval/bench.sh http://localhost:${FRONTEND_PORT} /infmax-workspace"
