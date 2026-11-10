# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized default ports used by srt-slurm runtime components.

Fixed ports are plain constants. Per-process ports are ``PortKind`` ranges at
the bottom of this module, handed out by ``srtctl.core.topology.NodePortAllocator``.

Every port srtctl chooses sits in 1024-8999, below ``EPHEMERAL_FLOOR``: some
clusters set net.ipv4.ip_local_port_range as low as 9000, and a port inside that
range can already be held by an unrelated outbound socket on the node when the
listener binds (EADDRINUSE). ``tests/test_port_layout.py`` checks the blocks
stay below the floor and apart; the few that cannot are listed there with why.
"""

from dataclasses import dataclass

EPHEMERAL_FLOOR = 9000

# Shared infrastructure services.
ETCD_CLIENT_PORT = 2379
ETCD_PEER_PORT = 2380
NATS_PORT = 4222
OTEL_GRPC_PORT = 4317

# Frontend service ports.
FRONTEND_PUBLIC_PORT = 8000
FRONTEND_INTERNAL_PORT = 8180

# Shared worker endpoint ports.
# SGLang uses this for --kv-events-config; vLLM uses it for
# DYN_VLLM_KV_EVENT_PORT.
KV_EVENTS_PORT_BASE = 3000

# SGLang backend ports.
SGLANG_HTTP_PORT_BASE = 6100
SGLANG_HTTP_PORT_STRIDE = 32
SGLANG_BOOTSTRAP_PORT_BASE = 7200
SGLANG_DIST_INIT_PORT_BASE = 8300
# One per physical SGLang server process. This is used for SGLang's local TP
# rendezvous; a deterministic assignment avoids concurrent free-port races.
SGLANG_NCCL_PORT_BASE = 4400
# HTTP sidecar of an SGLang gRPC-mode server (--grpc-http-sidecar-port): /metrics and
# the profiler routes, which the gRPC listener does not serve.
SGLANG_GRPC_HTTP_PORT_BASE = 6700
# SGLang Model Gateway (sglang_router) Prometheus listener.
# Only started when --prometheus-port is passed, which srtctl does so tachometer can scrape it.
SGLANG_ROUTER_METRICS_PORT = 8190

# Shepherd Model Gateway (frontend.type: smg) Prometheus listener on each router node,
# passed as --prometheus-port so srtctl, not upstream's default, owns it.
SMG_METRICS_PORT = 8191

# llm-d router-node listeners, passed explicitly; Envoy admin serves /ready.
LLM_D_EPP_GRPC_PORT = 8210
LLM_D_EPP_HEALTH_PORT = 8211
LLM_D_EPP_METRICS_PORT = 8212
LLM_D_ENVOY_ADMIN_PORT = 8213
# EPP binds this ZMQ subscriber; all KV-event publishers connect to it.
LLM_D_EPP_KV_EVENTS_PORT = 5557
# Per-worker proxy listeners, including llm-d decode sidecars; see PROXY_PORTS.
WORKER_PROXY_PORT_BASE = 7300

# TRT-LLM torch.distributed bootstrap, one port per MPI endpoint.
TRTLLM_DIST_INIT_PORT_BASE = 2050
# TRT-LLM DWDP rendezvous (TRTLLM_DWDP_MASTER_PORT) on the rank-0 worker's node.
TRTLLM_DWDP_MASTER_PORT = 8220

# TokenSpeed --port, one per process: a follower node's health listener and the base
# of the engine's own NCCL free-port scan, which starts 100 to 1000 ports above it.
TOKENSPEED_PORT_BASE = 10000
TOKENSPEED_PORT_STRIDE = 1024
TOKENSPEED_ZMQ_PORT_BASE = 2200
TOKENSPEED_METRICS_PORT_BASE = 6800

# Mooncake transfer-engine ports (shared by SGLang and vLLM backends).
MOONCAKE_MASTER_PORT = 8700
MOONCAKE_HTTP_METADATA_PORT = 8701
# Master's admin HTTP server (Prometheus metrics + /health, /role, /query_key, …).
# Mooncake's compile-time default is 9003; we pass --metrics_port explicitly so
# the master lives entirely inside our consolidated 8700-range.
MOONCAKE_METRICS_PORT = 8702

# LMCache multiprocess server (services[].type: lmcache-server), one per worker node,
# reached by that node's vLLM ranks over localhost. LMCache's own defaults (5555, 8080)
# sit inside ranges the NIXL side channel and the frontend can reach.
LMCACHE_SERVER_PORT = 8750
LMCACHE_HTTP_PORT = 8751

# vLLM backend ports.
VLLM_NIXL_PORT_BASE = 6820
# vLLM Router discovery endpoint (frontend.type: vllm-router with a discovery
# connector): the one ZMQ listener on the router node that MoRI-IO workers
# register their HTTP and transfer addresses with (--vllm-discovery-address).
VLLM_DISCOVERY_PORT = 8200
# MoRI-IO adds rank offsets to both bases. Keep these blocks out of the ephemeral
# range, used by its other bind(0) listeners.
VLLM_MORIIO_HANDSHAKE_PORT_BASE = 4900
VLLM_MORIIO_NOTIFY_PORT_BASE = 5560
VLLM_DATA_PARALLEL_RPC_PORT = 7400
VLLM_PORT_BASE = 20000
VLLM_PORT_STRIDE = 50
# torch.distributed rendezvous of a multi-node vLLM engine (--master-port; vLLM's own
# default). Under backend.failover every engine of a worker needs its own TCPStore, so
# shadow engine k listens on BASE + k * STRIDE, the same stagger the Dynamo operator uses.
VLLM_MASTER_PORT_BASE = 29500
VLLM_MASTER_PORT_STRIDE = 100

# Dynamo runtime and connector ports.
DYN_SYSTEM_PORT_BASE = 7500
KVBM_ZMQ_PORT_BASE = 1024

# Ray cluster (services[].type: ray): GCS on the head, dashboard (also the job
# submission API) on the head. Ray's own defaults; options.port / dashboard_port move them.
RAY_GCS_PORT = 6379
RAY_DASHBOARD_PORT = 8265

# Dynamo sidecar gRPC listener next to a native engine (dynamo.sidecar_port), one
# per sidecar process across the job. Outside the ephemeral range on purpose: a
# loopback connect to a port inside it can be handed that port as its source and
# connect to itself, and the TIME_WAIT socket it leaves makes the engine's bind fail.
DYNAMO_SIDECAR_GRPC_PORT = 2400

# Prometheus exporters (observability.tachometer), offset from the conventional
# 9400/9100 because managed clusters may run host exporters there.
DCGM_EXPORTER_PORT = 9401
NODE_EXPORTER_PORT = 9101
PROCESS_EXPORTER_PORT = 9256
CPU_POWER_EXPORTER_PORT = 9405


def rank_offset_subscriber_port(port: int, rank: int) -> int:
    """Undo vLLM's global DP-rank offset to reach a shared KV-event subscriber.

    Tested image's offset_endpoint_port:
    https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/distributed/kv_events.py#L495-L523
    """
    return port - rank


@dataclass(frozen=True)
class PortKind:
    """One kind of listener a worker process binds, allocated by ``NodePortAllocator``.

    ``base`` is the first port handed out, ``stride`` the distance between
    consecutive allocations (more than one when the engine scans or offsets a
    range of its own from the port it is given), and ``per_node`` whether the
    counter restarts on every node (the port is only bound there) or runs
    across the whole job (a side channel that peers on other nodes address).
    ``cap`` is the most slots one counter hands out, so the block ends at
    ``base + (cap - 1) * stride`` and never runs into the next one.
    Every per-process port is allocated once in ``endpoints_to_processes`` and
    carried on ``Process``; nothing derives a port from another port.
    """

    name: str
    base: int
    stride: int = 1
    per_node: bool = False
    cap: int = 1


# Bound on every worker process.
SYS_PORTS = PortKind("sys", DYN_SYSTEM_PORT_BASE, cap=500)
HTTP_PORTS = PortKind("http", SGLANG_HTTP_PORT_BASE, SGLANG_HTTP_PORT_STRIDE, per_node=True, cap=8)
BOOTSTRAP_PORTS = PortKind("bootstrap", SGLANG_BOOTSTRAP_PORT_BASE, per_node=True, cap=64)
KV_EVENTS_PORTS = PortKind("kv_events", KV_EVENTS_PORT_BASE, cap=1200)
NIXL_PORTS = PortKind("nixl", VLLM_NIXL_PORT_BASE, cap=380)
DP_RPC_PORTS = PortKind("dp_rpc", VLLM_DATA_PARALLEL_RPC_PORT, per_node=True, cap=64)
# KVBM leader ZMQ pair: pub at the port, ack at the port + 1.
KVBM_ZMQ_PORTS = PortKind("kvbm_zmq", KVBM_ZMQ_PORT_BASE, 2, cap=500)
SIDECAR_GRPC_PORTS = PortKind("sidecar_grpc", DYNAMO_SIDECAR_GRPC_PORT, cap=500)
# Allocated on each worker's node for Frontend.proxied_worker_modes.
PROXY_PORTS = PortKind("proxy", WORKER_PROXY_PORT_BASE, per_node=True, cap=64)
# Engine-specific: the backend allocates these for its own processes.
NCCL_PORTS = PortKind("nccl", SGLANG_NCCL_PORT_BASE, cap=500)
DIST_INIT_PORTS = PortKind("dist_init", SGLANG_DIST_INIT_PORT_BASE, per_node=True, cap=400)
GRPC_HTTP_PORTS = PortKind("grpc_http", SGLANG_GRPC_HTTP_PORT_BASE, per_node=True, cap=32)
VLLM_SCAN_PORTS = PortKind("vllm_scan", VLLM_PORT_BASE, VLLM_PORT_STRIDE, cap=900)
# vLLM discovery-connector (MoRI-IO) workers: one port per local rank in each block.
MORIIO_HANDSHAKE_PORTS = PortKind("moriio_handshake", VLLM_MORIIO_HANDSHAKE_PORT_BASE, cap=512)
MORIIO_NOTIFY_PORTS = PortKind("moriio_notify", VLLM_MORIIO_NOTIFY_PORT_BASE, cap=512)
TRTLLM_DIST_INIT_PORTS = PortKind("trtllm_dist_init", TRTLLM_DIST_INIT_PORT_BASE, cap=128)
TOKENSPEED_PORTS = PortKind("tokenspeed", TOKENSPEED_PORT_BASE, TOKENSPEED_PORT_STRIDE, per_node=True, cap=8)
# SMG binds one handshake per endpoint on the frontend node, including remote workers.
TOKENSPEED_ZMQ_PORTS = PortKind("tokenspeed_zmq", TOKENSPEED_ZMQ_PORT_BASE, cap=128)
TOKENSPEED_METRICS_PORTS = PortKind("tokenspeed_metrics", TOKENSPEED_METRICS_PORT_BASE, per_node=True, cap=16)
