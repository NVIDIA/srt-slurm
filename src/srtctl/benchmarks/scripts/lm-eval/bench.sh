#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Self-contained lm-eval accuracy evaluation for OpenAI-compatible chat APIs.
# Expects: endpoint

set -euo pipefail

ENDPOINT=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/logs}"
WORK_DIR="${EVAL_WORK_DIR:-${LOG_DIR}/lm-eval-work}"

HOST=$(echo "$ENDPOINT" | sed -E 's|https?://||; s|:.*||')
PORT=$(echo "$ENDPOINT" | sed -E 's|.*:([0-9]+).*|\1|')

echo "lm-eval Config: endpoint=${ENDPOINT}; host=${HOST}; port=${PORT}; work_dir=${WORK_DIR}"

if [[ -z "${MODEL_NAME:-}" ]]; then
    DISCOVERED_MODEL=$(curl -sf "${ENDPOINT}/v1/models" 2>/dev/null \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || true)
    if [[ -n "$DISCOVERED_MODEL" ]]; then
        export MODEL_NAME="$DISCOVERED_MODEL"
        echo "Auto-discovered MODEL_NAME from /v1/models: ${MODEL_NAME}"
    else
        export MODEL_NAME="${MODEL:-}"
        echo "WARNING: Could not discover model name from /v1/models; using MODEL_NAME=${MODEL_NAME:-unset}"
    fi
else
    echo "Using MODEL_NAME from environment: ${MODEL_NAME}"
fi

_install_lm_eval_deps() {
    _pip_install() {
        local pip_args=(-q --no-cache-dir)
        pip_args+=(--index-url "${EVAL_PIP_INDEX_URL:-https://pypi.org/simple}")
        if [[ -n "${EVAL_PIP_EXTRA_INDEX_URL:-}" ]]; then
            pip_args+=(--extra-index-url "${EVAL_PIP_EXTRA_INDEX_URL}")
        fi
        if python3 -m pip install --help 2>/dev/null | grep -q -- "--break-system-packages"; then
            pip_args+=(--break-system-packages)
        fi
        python3 -m pip install "${pip_args[@]}" "$@"
    }

    _pip_install "lm-eval[api]" "huggingface-hub<1.0" || true
    local lm_eval_ref="b315ef3b05176acc9732bb7fdec116abe1ecc476"
    if command -v git >/dev/null 2>&1; then
        if ! _pip_install --no-deps --force-reinstall \
            "git+https://github.com/EleutherAI/lm-evaluation-harness.git@${lm_eval_ref}"; then
            _pip_install --no-deps --force-reinstall \
                "https://github.com/EleutherAI/lm-evaluation-harness/archive/${lm_eval_ref}.tar.gz" || true
        fi
    else
        _pip_install --no-deps --force-reinstall \
            "https://github.com/EleutherAI/lm-evaluation-harness/archive/${lm_eval_ref}.tar.gz" || true
    fi
}

_patch_lm_eval() {
    local patch_dir
    patch_dir="$(mktemp -d /tmp/lm_eval_patch-XXXXXX)"
    cat > "${patch_dir}/sitecustomize.py" <<'PY'
import json

from lm_eval.models.openai_completions import LocalChatCompletion as _LCC

try:
    from lm_eval.filters import extraction as ex

    if hasattr(ex, "get_match"):
        _orig_get_match = ex.get_match

        def _patched_get_match(
            regex,
            doc_to_choice,
            text,
            group_select,
            ignore_case=False,
            ignore_punctuation=False,
            regexes_to_ignore=None,
        ):
            match = _orig_get_match(
                regex,
                doc_to_choice,
                text,
                group_select,
                ignore_case,
                ignore_punctuation,
                regexes_to_ignore,
            )
            if match is None and isinstance(group_select, int) and group_select < 0:
                flags = __import__("re").IGNORECASE if ignore_case else 0
                pattern = regex.pattern if hasattr(regex, "pattern") else regex
                matches = list(__import__("re").finditer(pattern, text, flags))
                if matches:
                    groups = matches[-1].groups()
                    for candidate in reversed(groups):
                        if candidate:
                            return candidate
            return match

        ex.get_match = _patched_get_match
except Exception:
    pass


class _JsonChatStr(str):
    pass


_orig_template_call = _LCC._create_payload


def _patched_create_payload(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None, **kwargs):
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            pass
    payload = _orig_template_call(self, messages, generate=generate, gen_kwargs=gen_kwargs, seed=seed, eos=eos, **kwargs)
    if isinstance(payload.get("messages"), str):
        try:
            payload["messages"] = json.loads(payload["messages"])
        except Exception:
            pass
    return payload


_LCC._create_payload = _patched_create_payload

try:
    from lm_eval.models import api_models as _api_models

    _TemplateAPI = _api_models.TemplateAPI

    def _patched_apply_chat_template(self, chat_history, add_generation_prompt=True):
        if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        elif self.tokenizer_backend == "remote" and self.tokenized_requests:
            return chat_history
        else:
            return _JsonChatStr(json.dumps([{**item} for item in chat_history], ensure_ascii=False))

    _TemplateAPI.apply_chat_template = _patched_apply_chat_template
except Exception:
    pass
PY
    export PYTHONPATH="${patch_dir}:${PYTHONPATH:-}"
}

get_native_max_context_length() {
    local model_path="$1"
    if [[ -n "${MODEL_PATH:-}" && -d "${MODEL_PATH}" ]]; then
        model_path="${MODEL_PATH}"
    fi
    python3 -c "
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained('${model_path}', trust_remote_code=True)
    for attr in ['max_position_embeddings', 'max_sequence_length', 'seq_length', 'n_positions']:
        if hasattr(config, attr):
            print(getattr(config, attr))
            break
    else:
        print(0)
except Exception:
    print(0)
"
}

compute_eval_context_length() {
    local model="$1"
    local benchmark_ctx="${2:-0}"
    local native_max
    native_max=$(get_native_max_context_length "$model")
    native_max="${native_max:-0}"

    if [[ "$benchmark_ctx" -eq 0 ]] 2>/dev/null; then
        benchmark_ctx="${native_max:-0}"
    fi
    local eval_ctx=$(( benchmark_ctx * 1 ))
    if [[ "$native_max" -gt 0 ]] 2>/dev/null && [[ "$eval_ctx" -gt "$native_max" ]]; then
        eval_ctx="$native_max"
    fi
    if [[ "$eval_ctx" -le 0 ]] 2>/dev/null; then
        echo "WARN: could not determine context length for $model" >&2
        eval_ctx="${MAX_MODEL_LEN:-16384}"
    fi
    EVAL_MAX_MODEL_LEN="$eval_ctx"
    echo "$eval_ctx"
}

resolve_tasks_file() {
    local tasks="${EVAL_TASKS_DIR:-${SCRIPT_DIR}/gsm8k.yaml}"
    if [[ -f "$tasks" ]]; then
        echo "$tasks"
    elif [[ -f "${SCRIPT_DIR}/$(basename "$tasks")" ]]; then
        echo "${SCRIPT_DIR}/$(basename "$tasks")"
    else
        echo "$tasks"
    fi
}

append_lm_eval_summary() {
    local results_dir="${EVAL_RESULT_DIR}"
    if [[ -z "${results_dir}" || ! -d "${results_dir}" ]]; then
        echo "WARN: EVAL_RESULT_DIR='${results_dir:-}' is missing; skipping artifact collection" >&2
        return 1
    fi

    local meta_json="${results_dir}/meta_env.json"
    local is_multinode_json="false"
    [[ "${IS_MULTINODE:-false}" == "true" ]] && is_multinode_json="true"

    local prefill_tp="${PREFILL_TP:-${TP:-1}}"
    local prefill_ep="${PREFILL_EP:-${EP_SIZE:-1}}"
    local prefill_num_workers="${PREFILL_NUM_WORKERS:-1}"
    local decode_tp="${DECODE_TP:-${TP:-1}}"
    local decode_ep="${DECODE_EP:-${EP_SIZE:-1}}"
    local decode_num_workers="${DECODE_NUM_WORKERS:-1}"

    local dp_json="false"
    [[ "${DP_ATTENTION:-false}" == "true" ]] && dp_json="true"
    local prefill_dp_json="false"
    [[ "${PREFILL_DP_ATTENTION:-${DP_ATTENTION:-false}}" == "true" ]] && prefill_dp_json="true"
    local decode_dp_json="false"
    [[ "${DECODE_DP_ATTENTION:-${DP_ATTENTION:-false}}" == "true" ]] && decode_dp_json="true"

    cat > "${meta_json}" <<META
{
  "is_multinode": ${is_multinode_json},
  "framework": "${FRAMEWORK:-unknown}",
  "precision": "${PRECISION:-unknown}",
  "spec_decoding": "${SPEC_DECODING:-none}",
  "tp": ${TP:-1},
  "conc": ${CONC:-1},
  "ep": ${EP_SIZE:-1},
  "dp_attention": ${dp_json},
  "prefill_tp": ${prefill_tp},
  "prefill_ep": ${prefill_ep},
  "prefill_dp_attention": ${prefill_dp_json},
  "prefill_num_workers": ${prefill_num_workers},
  "decode_tp": ${decode_tp},
  "decode_ep": ${decode_ep},
  "decode_dp_attention": ${decode_dp_json},
  "decode_num_workers": ${decode_num_workers},
  "model": "${MODEL_NAME:-${MODEL:-}}",
  "infmax_model_prefix": "${MODEL_PREFIX:-unknown}",
  "hw": "${RUNNER_TYPE:-unknown}",
  "isl": "${ISL:-0}",
  "osl": "${OSL:-0}"
}
META

    if [[ -d "${results_dir}" ]]; then
        while IFS= read -r -d '' jf; do
            mv -f "$jf" ./ || echo "WARN: failed to move ${jf}" >&2
        done < <(find "${results_dir}" -type f \( -name "*.json" -o -name "*.jsonl" \) -print0 2>/dev/null)
    fi
    rm -rf --one-file-system "${results_dir}" 2>/dev/null || rm -rf "${results_dir}" || true
}

mkdir -p "${WORK_DIR}" "${LOG_DIR}/eval_results"
cd "${WORK_DIR}"

if [[ "${EVAL_SKIP_DEP_INSTALL:-false}" == "true" ]]; then
    echo "Skipping lm-eval dependency install because EVAL_SKIP_DEP_INSTALL=true"
else
    _install_lm_eval_deps
fi
_patch_lm_eval

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC:-${EVAL_CONCURRENT_REQUESTS:-256}}"
export IS_MULTINODE="${IS_MULTINODE:-true}"
export TP="${TP:-${PREFILL_TP:-1}}"
export CONC="${CONC:-${EVAL_CONC:-${EVAL_CONCURRENT_REQUESTS:-1}}}"
export EP_SIZE="${EP_SIZE:-${PREFILL_EP:-1}}"
export DP_ATTENTION="${DP_ATTENTION:-${PREFILL_DP_ATTN:-false}}"
export PREFILL_DP_ATTENTION="${PREFILL_DP_ATTENTION:-${PREFILL_DP_ATTN:-${DP_ATTENTION:-false}}}"
export DECODE_DP_ATTENTION="${DECODE_DP_ATTENTION:-${DECODE_DP_ATTN:-${DP_ATTENTION:-false}}}"

if [[ -z "${EVAL_MAX_MODEL_LEN:-}" ]]; then
    compute_eval_context_length "${MODEL:-${MODEL_NAME}}" "${MAX_MODEL_LEN:-0}" >/dev/null
fi

TASKS_FILE="$(resolve_tasks_file)"
RESULTS_DIR="${EVAL_RESULT_DIR:-$(mktemp -d /tmp/eval_out-XXXXXX)}"
export EVAL_RESULT_DIR="${RESULTS_DIR}"

eval_context_len="${EVAL_MAX_MODEL_LEN:-16384}"
max_output_tokens=$(( eval_context_len > 4096 ? eval_context_len - 4096 : eval_context_len / 2 ))
if [[ "$max_output_tokens" -gt 16384 ]]; then
    max_output_tokens=16384
fi

echo "Running lm-eval with tasks=${TASKS_FILE}; concurrent-requests=${EVAL_CONCURRENT_REQUESTS}"
echo "Eval budget: eval_context_len=${eval_context_len}, max_output_tokens=${max_output_tokens}"

eval_rc=0
set -x
limit_args=()
if [[ -n "${EVAL_LIMIT:-}" ]]; then
    limit_args+=(--limit "${EVAL_LIMIT}")
fi
if [[ -n "${EVAL_NUM_FEWSHOT:-}" ]]; then
    limit_args+=(--num_fewshot "${EVAL_NUM_FEWSHOT}")
fi

python3 -m lm_eval --model local-chat-completions --apply_chat_template \
    --tasks "${TASKS_FILE}" \
    --output_path "${RESULTS_DIR}" \
    --log_samples \
    "${limit_args[@]}" \
    --model_args "model=${MODEL_NAME},base_url=http://${HOST}:${PORT}/v1/chat/completions,api_key=${OPENAI_API_KEY},eos_string=</s>,max_retries=5,num_concurrent=${EVAL_CONCURRENT_REQUESTS},timeout=1800,tokenized_requests=False,max_length=${eval_context_len}" \
    --gen_kwargs "max_tokens=${max_output_tokens},temperature=0,top_p=1" || eval_rc=$?
set +x

append_lm_eval_summary || true

echo "Copying eval artifacts to ${LOG_DIR}/eval_results/..."
cp -v meta_env.json "${LOG_DIR}/eval_results/" 2>/dev/null || true
cp -v results*.json "${LOG_DIR}/eval_results/" 2>/dev/null || true
cp -v sample*.jsonl "${LOG_DIR}/eval_results/" 2>/dev/null || true

if [[ "${VALIDATE_EVAL_SCORES:-false}" == "true" ]]; then
    echo "Validating eval scores..."
    python3 "${SCRIPT_DIR}/validate_scores.py" \
        --thresholds "${EVAL_THRESHOLDS:-${SCRIPT_DIR}/thresholds.json}" \
        --results-glob "results*.json"
fi

if [[ "$eval_rc" -ne 0 ]]; then
    echo "lm-eval evaluation failed with exit code ${eval_rc}"
    exit "$eval_rc"
fi

echo "lm-eval evaluation complete"
