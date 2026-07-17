#!/usr/bin/env bash
set -euo pipefail

# Setup and patch set for the 515d6e9 GB300 AgentX image family. The pinned
# Dynamo v1.2.1 base install does not install its optional vLLM dependency, so
# these vLLM patches remain intact when Dynamo is installed after this script.

bash /configs/patches/vllm-container-deps.sh

# Fail open instead of leaving a request indefinitely in
# WAITING_FOR_REMOTE_KVS when a Mooncake external-KV GET wedges.
python3 /configs/patches/mooncake_load_watchdog.py
echo "mooncake-load-watchdog: applied"

# Restore the pre-assert scale-packing behavior used by the validated DSV4
# AMXFP4 path on GB300.
python3 /configs/patches/deepgemm_smxx_sf_assert_patch.py
echo "deepgemm-smxx-sf-assert-off: applied"
