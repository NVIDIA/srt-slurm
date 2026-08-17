#!/usr/bin/env bash
set -euo pipefail

# Combines bind-b300-prefill-cpus.sh (dense B300 prefill CPU affinity) and
# install_dynamo_wheel.sh (dynamo wheel install) for recipes that need both.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/bind-b300-prefill-cpus.sh"
"${SCRIPT_DIR}/install_dynamo_wheel.sh"
