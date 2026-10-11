#!/usr/bin/env bash
#SBATCH --job-name=gpu-availability
#SBATCH --account=root
#SBATCH --partition=batch-r41
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=gpu-availability-%j.out

# Find nodes whose GPUs fail CUDA init with cudaErrorDevicesUnavailable
# ("CUDA-capable device(s) is/are busy or unavailable").
#
#   sbatch configs/raplab-gpu-availability.sbatch
#   sbatch --nodes=2 --nodelist=vr-r41-cn01,vr-r41-cn02 configs/raplab-gpu-availability.sbatch
#
# Each node runs one CUDA process per GPU at the same time, as the TRT-LLM
# workers do: set_device, a 256 MiB allocation and a matmul. It also records
# compute mode, fabric (MNNVL) state and any compute process already on the
# GPU. Nothing on the host is changed. Override the image with CONTAINER_IMAGE.
set -uo pipefail

CONTAINER_IMAGE=${CONTAINER_IMAGE:-/home/rihuo/tensorrt-llm-release-arm64-f1bdd25-user_imant_glm5-rubin-30428.sqsh}
OUT=${OUT:-$PWD/gpu-availability-$SLURM_JOB_ID}
mkdir -p "$OUT"

cat > "$OUT/check_node.sh" <<'NODE'
#!/usr/bin/env bash
set -uo pipefail
node=$(hostname -s)
log=$OUT/$node.log
{
    echo "== nvidia-smi summary"
    timeout 60 nvidia-smi --query-gpu=index,pci.bus_id,compute_mode,persistence_mode,memory.used --format=csv
    echo "== compute processes already on the GPUs"
    timeout 60 nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv
    echo "== fabric"
    timeout 60 nvidia-smi -q | grep -A3 -E '^\s+Fabric$'
} > "$log" 2>&1

ngpu=$(timeout 60 nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
pids=()
for ((gpu = 0; gpu < ngpu; gpu++)); do
    timeout 180 python3 - "$gpu" > "$OUT/$node.gpu$gpu.result" 2>&1 <<'PY' &
import sys, traceback
gpu = int(sys.argv[1])
try:
    import torch
    torch.cuda.set_device(gpu)
    a = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    m = torch.randn(4096, 4096, device="cuda")
    (m @ m).sum().item()
    torch.cuda.synchronize()
    print(f"OK gpu={gpu}")
except Exception as e:
    print(f"FAIL gpu={gpu} {type(e).__name__}: {str(e).splitlines()[0]}", flush=True)
    traceback.print_exc(file=sys.stdout)
PY
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || true; done

for ((gpu = 0; gpu < ngpu; gpu++)); do
    # Anything other than an OK line is a failure (timeout, crash, Python error).
    result=$(grep -m1 -E '^(OK|FAIL) gpu=' "$OUT/$node.gpu$gpu.result" 2>/dev/null)
    [[ $result == OK* ]] || result="FAIL gpu=$gpu ${result#FAIL gpu=$gpu }"
    [[ $result == "FAIL gpu=$gpu " ]] && result="FAIL gpu=$gpu no result line (timeout or crash); see $node.gpu$gpu.result"
    echo "$node $result"
done
[[ $ngpu -gt 0 ]] || echo "$node FAIL nvidia-smi sees no GPUs"
NODE
chmod +x "$OUT/check_node.sh"

# One task per node. A failing node must not cancel the others.
srun --kill-on-bad-exit=0 --ntasks-per-node=1 \
    --container-image="$CONTAINER_IMAGE" --container-mounts="$OUT:$OUT" \
    --export=ALL,OUT="$OUT" \
    bash "$OUT/check_node.sh" | sort -V | tee "$OUT/summary.txt"

echo
echo "Nodes with a failing GPU:"
grep FAIL "$OUT/summary.txt" | awk '{print $1}' | sort -uV || true
reported=$(awk '{print $1}' "$OUT/summary.txt" | sort -u)
for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    grep -qx "$node" <<< "$reported" || echo "$node (no output: container or srun failed on this node)"
done
echo "Per-node details: $OUT/<node>.log and $OUT/<node>.gpu<N>.result"
