# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized default ports used by srt-slurm runtime components.

Fixed ports are plain constants. Per-process ports are ``PortKind`` ranges at
the bottom of this module, handed out by ``srtctl.core.topology.NodePortAllocator``.
"""

from dataclasses import dataclass

# Shared infrastructure services.
ETCD_CLIENT_PORT = 2379
ETCD_PEER_PORT = 2380
NATS_PORT = 4222

# Frontend service ports.
FRONTEND_PUBLIC_PORT = 8000
FRONTEND_INTERNAL_PORT = 8180

# Shared worker endpoint ports.
# SGLang uses this for --kv-events-config; vLLM uses it for
# DYN_VLLM_KV_EVENT_PORT.
KV_EVENTS_PORT_BASE = 5200

# SGLang backend ports.
SGLANG_HTTP_PORT_BASE = 6100
SGLANG_HTTP_PORT_STRIDE = 32
SGLANG_BOOTSTRAP_PORT_BASE = 7200
SGLANG_DIST_INIT_PORT_BASE = 8300
# One per physical SGLang server process. This is used for SGLang's local TP
# rendezvous; a deterministic assignment avoids concurrent free-port races.
SGLANG_NCCL_PORT_BASE = 17500
# SGLang Model Gateway (sglang_router) Prometheus listener; the router's own default.
# Only started when --prometheus-port is passed, which srtctl does so tachometer can scrape it.
SGLANG_ROUTER_METRICS_PORT = 29000

# TRT-LLM torch.distributed bootstrap, one port per MPI endpoint.
TRTLLM_DIST_INIT_PORT_BASE = 29500

# Mooncake transfer-engine ports (shared by SGLang and vLLM backends).
MOONCAKE_MASTER_PORT = 8700
MOONCAKE_HTTP_METADATA_PORT = 8701
# Master's admin HTTP server (Prometheus metrics + /health, /role, /query_key, …).
# Mooncake's compile-time default is 9003; we pass --metrics_port explicitly so
# the master lives entirely inside our consolidated 8700-range.
MOONCAKE_METRICS_PORT = 8702

# vLLM backend ports.
VLLM_NIXL_PORT_BASE = 5400
# vLLM Router discovery endpoint (frontend.type: vllm-router with a discovery
# connector): the one ZMQ listener on the router node that MoRI-IO workers
# register their HTTP and transfer addresses with (--vllm-discovery-address).
VLLM_DISCOVERY_PORT = 36367
# MoRI-IO adds rank offsets to both bases. Keep these blocks below Linux's
# default ephemeral range (32768-60999), used by its other bind(0) listeners.
VLLM_MORIIO_HANDSHAKE_PORT_BASE = 26000
VLLM_MORIIO_NOTIFY_PORT_BASE = 27000
VLLM_DATA_PARALLEL_RPC_PORT = 8400
VLLM_PORT_BASE = 20000
VLLM_PORT_STRIDE = 50
# torch.distributed rendezvous of a multi-node vLLM engine (--master-port; vLLM's own
# default). Under backend.failover every engine of a worker needs its own TCPStore, so
# shadow engine k listens on BASE + k * STRIDE, the same stagger the Dynamo operator uses.
VLLM_MASTER_PORT_BASE = 29500
VLLM_MASTER_PORT_STRIDE = 100

# Dynamo runtime and connector ports.
DYN_SYSTEM_PORT_BASE = 7500
KVBM_ZMQ_PORT_BASE = 5600

# Ray cluster (services[].type: ray): GCS on the head, dashboard (also the job
# submission API) on the head. Ray's own defaults; options.port / dashboard_port move them.
RAY_GCS_PORT = 6379
RAY_DASHBOARD_PORT = 8265

# Dynamo sidecar gRPC listener next to a native engine (dynamo.sidecar_port).
DYNAMO_SIDECAR_GRPC_PORT = 50051


@dataclass(frozen=True)
class PortKind:
    """One kind of listener a worker process binds, allocated by ``NodePortAllocator``.

    ``base`` is the first port handed out, ``stride`` the distance between
    consecutive allocations (more than one when the engine scans or offsets a
    range of its own from the port it is given), and ``per_node`` whether the
    counter restarts on every node (the port is only bound there) or runs
    across the whole job (a side channel that peers on other nodes address).
    Every per-process port is allocated once in ``endpoints_to_processes`` and
    carried on ``Process``; nothing derives a port from another port.
    """

    name: str
    base: int
    stride: int = 1
    per_node: bool = False


# Bound on every worker process.
SYS_PORTS = PortKind("sys", DYN_SYSTEM_PORT_BASE)
HTTP_PORTS = PortKind("http", SGLANG_HTTP_PORT_BASE, SGLANG_HTTP_PORT_STRIDE, per_node=True)
BOOTSTRAP_PORTS = PortKind("bootstrap", SGLANG_BOOTSTRAP_PORT_BASE, per_node=True)
KV_EVENTS_PORTS = PortKind("kv_events", KV_EVENTS_PORT_BASE)
NIXL_PORTS = PortKind("nixl", VLLM_NIXL_PORT_BASE)
DP_RPC_PORTS = PortKind("dp_rpc", VLLM_DATA_PARALLEL_RPC_PORT, per_node=True)
# KVBM leader ZMQ pair: pub at the port, ack at the port + 1.
KVBM_ZMQ_PORTS = PortKind("kvbm_zmq", KVBM_ZMQ_PORT_BASE, 2)
SIDECAR_GRPC_PORTS = PortKind("sidecar_grpc", DYNAMO_SIDECAR_GRPC_PORT)
# Engine-specific: the backend allocates these for its own processes.
NCCL_PORTS = PortKind("nccl", SGLANG_NCCL_PORT_BASE)
DIST_INIT_PORTS = PortKind("dist_init", SGLANG_DIST_INIT_PORT_BASE, per_node=True)
VLLM_SCAN_PORTS = PortKind("vllm_scan", VLLM_PORT_BASE, VLLM_PORT_STRIDE)
# vLLM discovery-connector (MoRI-IO) workers: one port per local rank in each block.
MORIIO_HANDSHAKE_PORTS = PortKind("moriio_handshake", VLLM_MORIIO_HANDSHAKE_PORT_BASE)
MORIIO_NOTIFY_PORTS = PortKind("moriio_notify", VLLM_MORIIO_NOTIFY_PORT_BASE)
TRTLLM_DIST_INIT_PORTS = PortKind("trtllm_dist_init", TRTLLM_DIST_INIT_PORT_BASE)
