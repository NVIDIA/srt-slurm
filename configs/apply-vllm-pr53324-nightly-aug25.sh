#!/usr/bin/env bash
# Apply the Kimi-K3 DCP/PP/DSpark/Mooncake runtime patch stack to the
# Aug 25 nightly. InferenceMAX PR #247 already refers to this setup-script
# name, so keep the filename stable as the patch stack grows.

set -euo pipefail

readonly SITE_PACKAGES="${VLLM_SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"
readonly VLLM_ROOT="${SITE_PACKAGES}/vllm"
readonly VERSION_FILE="${VLLM_ROOT}/_version.py"
readonly PR53324_PATCH_FILE="${VLLM_PR53324_PATCH_FILE:-/configs/patches/vllm-pr53324-runtime-574d2e4-on-a9a17e7.patch}"
readonly K3_AGENT_PATCH_FILE="${VLLM_K3_AGENT_PATCH_FILE:-/configs/patches/vllm-k3-agent-all-missing-on-a9a17e7.patch}"
readonly PR53682_PATCH_FILE="${VLLM_PR53682_PATCH_FILE:-/configs/patches/vllm-pr53682-cudagraph-profiling-pool-on-a9a17e7.patch}"
readonly PR53773_PATCH_FILE="${VLLM_PR53773_PATCH_FILE:-/configs/patches/vllm-pr53773-kimi-mamba-profiling-state-on-a9a17e7.patch}"
readonly K3_TAIL_GATE_PATCH_FILE="${VLLM_K3_TAIL_GATE_PATCH_FILE:-/configs/patches/vllm-k3-latent-tail-env-gate-on-a9a17e7.patch}"
readonly K3_DEFERRED_FINALIZE_GATE_PATCH_FILE="${VLLM_K3_DEFERRED_FINALIZE_GATE_PATCH_FILE:-/configs/patches/vllm-k3-deferred-moe-finalize-env-gate-on-a9a17e7.patch}"
readonly K3_INTERNAL_CHECKPOINT_GATE_PATCH_FILE="${VLLM_K3_INTERNAL_CHECKPOINT_GATE_PATCH_FILE:-/configs/patches/vllm-k3-internal-prefill-checkpoints-env-gate-on-a9a17e7.patch}"
readonly PR53324_MARKER_FILE="${VLLM_ROOT}/.pr53324_574d2e4_on_a9a17e709"
readonly K3_AGENT_MARKER_FILE="${VLLM_ROOT}/.k3_agent_all_728d3ad_on_a9a17e709"
readonly PR53682_MARKER_FILE="${VLLM_ROOT}/.pr53682_e6cc089_on_a9a17e709"
readonly PR53773_MARKER_FILE="${VLLM_ROOT}/.pr53773_a447955_on_a9a17e709"
readonly K3_TAIL_GATE_MARKER_FILE="${VLLM_ROOT}/.k3_latent_tail_env_gate_on_a9a17e709"
readonly K3_DEFERRED_FINALIZE_GATE_MARKER_FILE="${VLLM_ROOT}/.k3_deferred_moe_finalize_env_gate_on_a9a17e709"
readonly K3_INTERNAL_CHECKPOINT_GATE_MARKER_FILE="${VLLM_ROOT}/.k3_internal_prefill_checkpoints_env_gate_on_a9a17e709"

if [[ ! -r "${VERSION_FILE}" ]] || ! grep -q "a9a17e709" "${VERSION_FILE}"; then
  echo "Refusing to patch: expected vLLM nightly commit a9a17e709." >&2
  echo "Version file: ${VERSION_FILE}" >&2
  exit 1
fi

apply_patch_once() {
  local label="$1"
  local patch_file="$2"
  local marker_file="$3"

  if [[ ! -r "${patch_file}" ]]; then
    echo "Missing ${label} runtime patch: ${patch_file}" >&2
    exit 1
  fi

  if [[ -f "${marker_file}" ]]; then
    echo "${label} runtime patch is already applied."
  elif patch --batch --forward --dry-run -d "${SITE_PACKAGES}" -p1 \
    < "${patch_file}" >/dev/null; then
    patch --batch --forward -d "${SITE_PACKAGES}" -p1 < "${patch_file}"
    touch "${marker_file}"
    echo "Applied ${label} to nightly commit a9a17e709."
  elif patch --batch --reverse --dry-run -d "${SITE_PACKAGES}" -p1 \
    < "${patch_file}" >/dev/null; then
    touch "${marker_file}"
    echo "${label} runtime content is already present."
  else
    echo "${label} neither applies cleanly nor appears already applied." >&2
    exit 1
  fi
}

# PR #53324 supersedes the older Mooncake/DCP commits in
# xinli-sw/vllm:k3-agent-all. The supplemental patch carries only the runtime
# changes still missing from a9a17e709: PP speculative decoding and its DSpark
# loading fixes, DCP dummy-batch sequence lengths, hybrid load-failure recovery,
# and hybrid cache-accounting safeguards. Changes already merged upstream are
# deliberately not replayed.
apply_patch_once \
  "vLLM PR #53324 head 574d2e4" \
  "${PR53324_PATCH_FILE}" \
  "${PR53324_MARKER_FILE}"
apply_patch_once \
  "xinli-sw/vllm:k3-agent-all supplemental changes" \
  "${K3_AGENT_PATCH_FILE}" \
  "${K3_AGENT_MARKER_FILE}"
apply_patch_once \
  "vLLM PR #53682 merged commit e6cc089" \
  "${PR53682_PATCH_FILE}" \
  "${PR53682_MARKER_FILE}"
apply_patch_once \
  "vLLM PR #53773 merged commit a447955" \
  "${PR53773_PATCH_FILE}" \
  "${PR53773_MARKER_FILE}"
apply_patch_once \
  "Kimi-K3 latent-MoE tail environment gate" \
  "${K3_TAIL_GATE_PATCH_FILE}" \
  "${K3_TAIL_GATE_MARKER_FILE}"
apply_patch_once \
  "Kimi-K3 deferred MoE-finalize environment gate" \
  "${K3_DEFERRED_FINALIZE_GATE_PATCH_FILE}" \
  "${K3_DEFERRED_FINALIZE_GATE_MARKER_FILE}"
# vLLM commit 9eb9d9d395 enables internal prefill checkpoints by default, but
# sustained Kimi-K3 DCP/PP traffic can hit an illegal memory access in that
# path. Keep the feature opt-in until the upstream checkpoint implementation
# is fixed. Prefix caching itself remains enabled when this optimization is off.
apply_patch_once \
  "Kimi-K3 internal prefill-checkpoint environment gate" \
  "${K3_INTERNAL_CHECKPOINT_GATE_PATCH_FILE}" \
  "${K3_INTERNAL_CHECKPOINT_GATE_MARKER_FILE}"

python3 -m compileall -q \
  "${VLLM_ROOT}/config/speculative.py" \
  "${VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/mooncake/store" \
  "${VLLM_ROOT}/model_executor/models/interfaces.py" \
  "${VLLM_ROOT}/models/kimi_k3/nvidia" \
  "${VLLM_ROOT}/v1/core/kv_cache_coordinator.py" \
  "${VLLM_ROOT}/v1/core/kv_cache_utils.py" \
  "${VLLM_ROOT}/v1/core/sched/scheduler.py" \
  "${VLLM_ROOT}/v1/simple_kv_offload/manager.py" \
  "${VLLM_ROOT}/v1/worker/gpu/cudagraph_utils.py" \
  "${VLLM_ROOT}/v1/worker/gpu/model_runner.py" \
  "${VLLM_ROOT}/v1/worker/gpu/pp_utils.py" \
  "${VLLM_ROOT}/v1/worker/gpu/spec_decode"
