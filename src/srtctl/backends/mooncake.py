# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared mooncake_master constants used by both SGLang and vLLM backends.

Kept in a dedicated module (rather than re-exported from one backend) so
neither backend has to import from the other just to reach the port numbers.
"""

# RPC port the master listens on. Workers reach it via MOONCAKE_MASTER.
MOONCAKE_MASTER_PORT = 50051

# Port for the master's embedded HTTP metadata server (enabled with
# --enable_http_metadata_server=true). Workers point MOONCAKE_TE_META_DATA_SERVER
# at /metadata on this port so no separate metadata service is required.
MOONCAKE_HTTP_METADATA_PORT = 8080

# Port for the master's admin HTTP server. Matches the upstream default in
# mooncake-store/src/master.cpp (--metrics_port=9003). Always listens once the
# master is up — --enable_metric_reporting only toggles a periodic stdout log
# thread, not the HTTP endpoints. Exposes /metrics (Prometheus text),
# /metrics/summary, /health, /role, /ha_status, /leader, /query_key.
MOONCAKE_METRICS_PORT = 9003
