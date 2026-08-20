#!/usr/bin/env bash
# Resolve the chat template from the already-prefetched Hugging Face snapshot.
# Every local TRT-LLM rank runs this preamble, so serialize the tiny symlink
# update instead of racing on /tmp.
set -euo pipefail

cache_root="${HF_HOME:?HF_HOME must be set}/hub/models--nvidia--MiniMax-M3-NVFP4"
ref_file="${cache_root}/refs/main"
template_link="/tmp/minimax-m3-chat-template.jinja"

if [[ ! -s "${ref_file}" ]]; then
  echo "Missing prefetched MiniMax-M3 Hugging Face ref: ${ref_file}" >&2
  exit 1
fi

(
  flock 9
  revision="$(tr -d '\r\n' < "${ref_file}")"
  template="${cache_root}/snapshots/${revision}/chat_template.jinja"
  if [[ ! -s "${template}" ]]; then
    echo "Missing MiniMax-M3 chat template: ${template}" >&2
    exit 1
  fi
  ln -sfn "${template}" "${template_link}"
) 9>/tmp/minimax-m3-chat-template.lock

test -s "${template_link}"
