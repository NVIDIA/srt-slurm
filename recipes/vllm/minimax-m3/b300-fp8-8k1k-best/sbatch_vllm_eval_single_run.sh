#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dynamo vLLM disaggregated serving (single-run / manual test variant).
# Mirrors recipes/vllm/minimax-m3/b300-fp8-8k1k-best/5p2d-dep2-tep8-8k1k.yaml:
#   - 5 prefill workers, each TP1 / DP2 (data-parallel attention) -> 2 single-GPU procs
#   - 2 decode workers,  each TP8 / EP8 (expert parallel)         -> 1 proc per node
# Launches dynamo.vllm workers (one srun per process), etcd + nats + dynamo.frontend.
# Does NOT run lm-eval -- it brings the endpoint up and holds it so you can run
# manual curl tests against the frontend, then exits when the job time limit hits
# or you scancel it.
#
#SBATCH --job-name=minimax_m3_vllm_5p2d_dep2_tep8_single_run
#SBATCH --nodes=5
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=8
#SBATCH --segment=5
#SBATCH --account=core_dlfw_ci
#SBATCH --time=04:00:00
#SBATCH --output=/home/rihuo/srt-slurm/outputs/%j/logs/sweep_%j.log
#SBATCH --partition=batch

set -euo pipefail

SRTCTL_SOURCE="/home/rihuo/srt-slurm"
OUTPUT_BASE="/home/rihuo/srt-slurm/outputs"
OUTPUT_DIR="${OUTPUT_BASE}/${SLURM_JOB_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"
CONTAINER_IMAGE="/lustre/fsw/coreai_comparch_inferencex/rihuo/vllm-openai-minimax-m3.sqsh"
EVAL_CONTAINER_IMAGE="/lustre/fsw/coreai_comparch_inferencex/rihuo/vllm-openai-minimax-m3.sqsh"
MODEL_PATH="/lustre/fsw/coreai_comparch_inferencex/rihuo/MiniMaxAI_MiniMax-M3-MXFP8"
MODEL_NAME="MiniMaxAI/MiniMax-M3-MXFP8"
# Directory containing the minimax dynamo wheel(s) installed into each worker (dynamo.install=false).
WHEELS_PATH="/lustre/fsw/coreai_comparch_inferencex/rihuo/minimax_wheels"
INFMAX_WORKSPACE="/home/rihuo/InferenceX"
SCRIPT_MOUNTS="${LOG_DIR}:/logs,${MODEL_PATH}:/model,${WHEELS_PATH}:/wheels,${SRTCTL_SOURCE}/configs:/configs,${SRTCTL_SOURCE}/src/srtctl/benchmarks/scripts:/srtctl-benchmarks,${INFMAX_WORKSPACE}:/infmax-workspace"

# setup_script: install_minimax_dynamo_wheel.sh -- the container ships without dynamo.vllm,
# so each worker/frontend installs the wheel before launching.
WHEEL_INSTALL="echo 'Installing nvidia-cutlass-dsl...' && pip install --break-system-packages --force-reinstall 'nvidia-cutlass-dsl[cu13]==4.5.2' && echo 'Installing minimax dynamo wheel from /wheels...' && pip install --break-system-packages --force-reinstall /wheels/*.whl"

# vLLM worker environment (prefill_environment / decode_environment in the recipe)
VLLM_COMMON_ENV="export VLLM_FLOAT32_MATMUL_PRECISION=high && export UCX_NET_DEVICES=all && export UCX_TLS=rc,cuda_ipc,cuda_copy,sm,self,tcp"

# KV transfer connector (recipe: connector=null at top-level, nixl on each worker via kv-transfer-config)
KV_TRANSFER_CONFIG='{"kv_connector": "NixlConnector", "kv_role": "kv_both"}'

# Static vLLM CLI args per mode (from vllm_config in the recipe).
# Per-process flags (--data-parallel-rank/-address/-rpc-port, CUDA_VISIBLE_DEVICES) are added at launch.
PREFILL_VLLM_ARGS="--model /model --served-model-name ${MODEL_NAME} --disaggregation-mode prefill --request-plane nats --block-size 128 --data-parallel-size 2 --gpu-memory-utilization 0.9 --kv-transfer-config '${KV_TRANSFER_CONFIG}' --language-model-only --max-cudagraph-capture-size 2048 --max-model-len 9472 --max-num-batched-tokens 16384 --no-enable-prefix-caching --stream-interval 32 --tensor-parallel-size 1 --trust-remote-code"
DECODE_VLLM_ARGS="--model /model --served-model-name ${MODEL_NAME} --disaggregation-mode decode --request-plane nats --block-size 128 --enable-expert-parallel --gpu-memory-utilization 0.9 --kv-transfer-config '${KV_TRANSFER_CONFIG}' --language-model-only --max-cudagraph-capture-size 4096 --max-model-len 9472 --max-num-batched-tokens 16384 --max-num-seqs 1024 --no-enable-prefix-caching --stream-interval 32 --tensor-parallel-size 8 --trust-remote-code"

# Dynamo ports
ETCD_PORT=2379
NATS_PORT=4222
FRONTEND_PORT=8000

mkdir -p "${LOG_DIR}"
exec 2>&1

mapfile -t ALL_NODES < <(scontrol show hostnames "${SLURM_NODELIST}")
if [ "${#ALL_NODES[@]}" -lt 5 ]; then
    echo "ERROR: expected at least 5 nodes, got ${#ALL_NODES[@]}"
    exit 1
fi

HEAD_NODE="${ALL_NODES[0]}"        # infra (etcd + nats) + dynamo frontend
PREFILL_NODE_A="${ALL_NODES[1]}"   # prefill workers 0-3 (8 GPUs)
PREFILL_NODE_B="${ALL_NODES[2]}"   # prefill worker 4 (2 GPUs)
DECODE_NODE_0="${ALL_NODES[3]}"    # decode worker 0 (TP8, 8 GPUs)
DECODE_NODE_1="${ALL_NODES[4]}"    # decode worker 1 (TP8, 8 GPUs)

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

# Launch one DP rank of a vLLM prefill worker (TP1/DP2, pinned to a single GPU).
# Both ranks of an endpoint share the same data-parallel-address (the node IP)
# and rpc-port; they differ by --data-parallel-rank and GPU/port assignments.
launch_prefill_rank() {
    local node="$1" gpu="$2" sys_port="$3" kv_evt_port="$4" nixl_port="$5" dp_rank="$6" rpc_port="$7" logname="$8"

    start_bg srun \
        --jobid "${SLURM_JOB_ID}" \
        --overlap \
        --nodes 1 \
        --ntasks 1 \
        --nodelist "${node}" \
        --output "${LOG_DIR}/${logname}.out" \
        --container-image "${CONTAINER_IMAGE}" \
        --no-container-entrypoint \
        --no-container-mount-home \
        --container-mounts "${SCRIPT_MOUNTS}" \
        bash -c "${WHEEL_INSTALL} && ${DYNAMO_WORKER_ENV} && ${VLLM_COMMON_ENV} && export DYN_SYSTEM_PORT=${sys_port} && export DYN_VLLM_KV_EVENT_PORT=${kv_evt_port} && export VLLM_NIXL_SIDE_CHANNEL_PORT=${nixl_port} && export CUDA_VISIBLE_DEVICES=${gpu} && HOST_IP=\$(hostname -I | awk '{print \$1}') && export VLLM_NIXL_SIDE_CHANNEL_HOST=\${HOST_IP} && python3 -m dynamo.vllm ${PREFILL_VLLM_ARGS} --data-parallel-rank ${dp_rank} --data-parallel-address \${HOST_IP} --data-parallel-rpc-port ${rpc_port}"
}

# Launch a vLLM decode worker (TP8/EP8, uses all 8 GPUs on the node).
launch_decode_worker() {
    local node="$1" sys_port="$2" kv_evt_port="$3" nixl_port="$4" logname="$5"

    start_bg srun \
        --jobid "${SLURM_JOB_ID}" \
        --overlap \
        --nodes 1 \
        --ntasks 1 \
        --nodelist "${node}" \
        --output "${LOG_DIR}/${logname}.out" \
        --container-image "${CONTAINER_IMAGE}" \
        --no-container-entrypoint \
        --no-container-mount-home \
        --container-mounts "${SCRIPT_MOUNTS}" \
        bash -c "${WHEEL_INSTALL} && ${DYNAMO_WORKER_ENV} && ${VLLM_COMMON_ENV} && export DYN_SYSTEM_PORT=${sys_port} && export DYN_VLLM_KV_EVENT_PORT=${kv_evt_port} && export VLLM_NIXL_SIDE_CHANNEL_PORT=${nixl_port} && HOST_IP=\$(hostname -I | awk '{print \$1}') && export VLLM_NIXL_SIDE_CHANNEL_HOST=\${HOST_IP} && python3 -m dynamo.vllm ${DECODE_VLLM_ARGS}"
}

echo "=========================================="
echo "Dynamo vLLM Disaggregated Eval (5P TP1/DP2 + 2D TP8/EP8)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Container: ${CONTAINER_IMAGE}"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "Head node (infra + frontend): ${HEAD_NODE}"
echo "Prefill nodes: ${PREFILL_NODE_A} (workers 0-3), ${PREFILL_NODE_B} (worker 4)"
echo "Decode nodes: ${DECODE_NODE_0}, ${DECODE_NODE_1}"
echo ""

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

        # The vLLM container does not ship etcd/nats. srt-slurm vendors the
        # binaries under configs/ (downloaded by "make setup", gitignored) and
        # runs them from the mounted /configs dir -- same as cli/setup_head.py.
        for bin in /configs/nats-server /configs/etcd; do
            if [ ! -x "${bin}" ]; then
                echo "ERROR: ${bin} not found. Run \"make setup ARCH=aarch64\" to download the NATS/etcd binaries into configs/."
                exit 1
            fi
        done

        rm -rf /tmp/etcd /tmp/nats
        mkdir -p /tmp/etcd /tmp/nats

        HOST_IP=$(hostname -I | awk "{print \$1}")
        echo "Infra node IP: ${HOST_IP}"

        echo "Starting nats-server..."
        /configs/nats-server -js -sd /tmp/nats &
        NATS_PID=$!

        echo "Starting etcd..."
        /configs/etcd \
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
# Stage 2: Start vLLM workers via dynamo.vllm
# ==============================================================================
DYNAMO_WORKER_ENV="export ETCD_ENDPOINTS=http://${HEAD_NODE}:${ETCD_PORT} && export NATS_SERVER=nats://${HEAD_NODE}:${NATS_PORT} && export DYN_REQUEST_PLANE=nats"

echo "Starting 5 prefill workers (TP1/DP2 -> 10 single-GPU procs)"
# Prefill node A: endpoints 0-3 (GPUs 0-7), one rpc-port per endpoint
launch_prefill_rank "${PREFILL_NODE_A}" 0 8081 20000 21000 0 13345 "${PREFILL_NODE_A}_prefill_ep0_r0"
launch_prefill_rank "${PREFILL_NODE_A}" 1 8082 20001 21000 1 13345 "${PREFILL_NODE_A}_prefill_ep0_r1"
launch_prefill_rank "${PREFILL_NODE_A}" 2 8083 20002 21002 0 13346 "${PREFILL_NODE_A}_prefill_ep1_r0"
launch_prefill_rank "${PREFILL_NODE_A}" 3 8084 20003 21002 1 13346 "${PREFILL_NODE_A}_prefill_ep1_r1"
launch_prefill_rank "${PREFILL_NODE_A}" 4 8085 20004 21004 0 13347 "${PREFILL_NODE_A}_prefill_ep2_r0"
launch_prefill_rank "${PREFILL_NODE_A}" 5 8086 20005 21004 1 13347 "${PREFILL_NODE_A}_prefill_ep2_r1"
launch_prefill_rank "${PREFILL_NODE_A}" 6 8087 20006 21006 0 13348 "${PREFILL_NODE_A}_prefill_ep3_r0"
launch_prefill_rank "${PREFILL_NODE_A}" 7 8088 20007 21006 1 13348 "${PREFILL_NODE_A}_prefill_ep3_r1"
# Prefill node B: endpoint 4 (GPUs 0-1)
launch_prefill_rank "${PREFILL_NODE_B}" 0 8089 20008 21008 0 13345 "${PREFILL_NODE_B}_prefill_ep4_r0"
launch_prefill_rank "${PREFILL_NODE_B}" 1 8090 20009 21008 1 13345 "${PREFILL_NODE_B}_prefill_ep4_r1"

echo "Starting 2 decode workers (TP8/EP8, 1 node each)"
launch_decode_worker "${DECODE_NODE_0}" 8091 20010 21010 "${DECODE_NODE_0}_decode_ep0"
launch_decode_worker "${DECODE_NODE_1}" 8092 20011 21011 "${DECODE_NODE_1}_decode_ep1"

for pid in "${SRUN_PIDS[@]:1}"; do
    require_alive "${pid}" "vllm_worker"
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
    bash -c "${WHEEL_INSTALL} && export ETCD_ENDPOINTS=http://${HEAD_NODE}:${ETCD_PORT} && export NATS_SERVER=nats://${HEAD_NODE}:${NATS_PORT} && export DYN_REQUEST_PLANE=nats && python3 -m dynamo.frontend --http-port ${FRONTEND_PORT}"
FRONTEND_PID="${SRUN_PIDS[-1]}"
require_alive "${FRONTEND_PID}" "FRONTEND_PID"

# ==============================================================================
# Stage 4: Wait for all workers to register, then verify model is serving
# ==============================================================================
# Each DP rank registers independently: 5 prefill workers x DP2 = 10 prefill instances.
EXPECTED_PREFILL=10
EXPECTED_DECODE=2

echo "Waiting for ${EXPECTED_PREFILL} prefill and ${EXPECTED_DECODE} decode workers to register..."
if ! wait_for_dynamo_workers "${HEAD_NODE}" "${FRONTEND_PORT}" "${EXPECTED_PREFILL}" "${EXPECTED_DECODE}" 2700 60; then
    echo "ERROR: workers did not become healthy"
    exit 1
fi

if ! verify_model_ready "${HEAD_NODE}" "${FRONTEND_PORT}" 120; then
    echo "ERROR: model is not serving"
    exit 1
fi

echo "All workers healthy and model is serving"

# ==============================================================================
# Stage 5: Hold the endpoint open for manual curl tests (no eval)
# ==============================================================================
FRONTEND_IP=$(getent hosts "${HEAD_NODE}" | awk '{print $1}')
[ -z "${FRONTEND_IP}" ] && FRONTEND_IP="${HEAD_NODE}"

echo "=========================================="
echo "Endpoint is UP - ready for manual testing"
echo "=========================================="
echo "Frontend: http://${HEAD_NODE}:${FRONTEND_PORT}  (IP: http://${FRONTEND_IP}:${FRONTEND_PORT})"
echo ""
echo "From a login/head node, open a shell on ${HEAD_NODE} and curl localhost, e.g.:"
echo ""
echo "  srun --jobid ${SLURM_JOB_ID} --overlap -w ${HEAD_NODE} --pty bash"
echo ""
echo "  curl http://localhost:${FRONTEND_PORT}/v1/models"
echo ""
echo "  curl http://localhost:${FRONTEND_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"${MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 64}'"
echo ""
echo "Health: curl http://localhost:${FRONTEND_PORT}/health"
echo "=========================================="
echo "Holding until the SLURM time limit or scancel ${SLURM_JOB_ID}. Logs: ${LOG_DIR}"
echo "=========================================="

# Keep the allocation (and all workers/frontend) alive. Exit early if the
# frontend dies so the job doesn't sit idle on a dead endpoint.
while true; do
    if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
        echo "ERROR: frontend exited - shutting down"
        wait "${FRONTEND_PID}" || true
        exit 1
    fi
    sleep 30
done
