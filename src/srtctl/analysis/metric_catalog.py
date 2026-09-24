# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presentation semantics for captured metric families, independent of run data.

Tachometer does not preserve Prometheus HELP/TYPE/unit metadata. Known counter
families and the Prometheus _total naming convention are differentiated, with
explicit historical exceptions; other families remain stored values.
Classification never supplies topology, capacities, timings, or measurements.

Taxonomy ported from NVIDIA/srt-slurm PR #447, commit
b2509c17c68f4b2326f7656b3e33738770e1d575, with its dashboard subgroup order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

COMPONENTS = ("Frontend", "Router", "Workers", "GPU", "Host")

GROUP_ORDER = {
    "Frontend": (
        "Requests and latency",
        "Native HTTP requests",
        "Native prefill client",
        "Native decode client",
        "Tokenization",
        "Tokenizer cache",
        "Worker dispatch and streaming",
        "Detokenization",
        "Tokio runtime",
        "Advertised model metadata",
        "Native HTTP metadata",
        "Request lifecycle",
        "First response and transport",
        "Component lifecycle",
        "Other captured metrics",
    ),
    "Router": (
        "Routing decisions",
        "Queue and backpressure",
        "Worker selection and feedback",
        "KV matching and cache",
        "KV index and events",
        "Other captured metrics",
    ),
    "Workers": (
        "Engine scheduling",
        "Engine iterations",
        "Engine KV cache",
        "Engine KV transfer",
        "Engine requests and latency",
        "Engine memory",
        "Engine speculative decoding",
        "Engine KV events",
        "Request lifecycle",
        "Admission and queues",
        "Work-handler pool",
        "First response and transport",
        "Engine configuration",
        "Component lifecycle",
        "Other captured metrics",
    ),
    "GPU": (
        "Utilization",
        "Framebuffer memory",
        "Power, clocks and temperature",
        "NVLink and PCIe",
        "Hardware events",
        "Other captured metrics",
    ),
    "Host": (
        "Process CPU and scheduling",
        "Host scheduling",
        "Host memory",
        "Process memory",
        "NUMA memory and locality",
        "Processes and threads",
        "Paging and reclaim",
        "Network and InfiniBand",
        "Collection health",
        "Other captured metrics",
    ),
}

# These exporters expose counters whose historical filtered names omit _total.
_COUNTERS = frozenset(
    {
        "dynamo_frontend_disconnected_clients",
        "dynamo_frontend_detokenize_token",
        "dynamo_frontend_detokenize_total_us",
        "dynamo_component_kv_cache_events_applied",
        "dynamo_component_kv_cache_event_warnings",
        "FI_DEV_TOTAL_ENERGY_CONSUMPTION",
        "FI_DEV_PCIE_REPLAY_COUNTER",
        "go_gc_duration_seconds_count",
        "go_gc_duration_seconds_sum",
        "namedprocess_scrape_errors",
        "namedprocess_scrape_partial_errors",
        "namedprocess_scrape_procread_errors",
        # Linux include/linux/vm_event_item.h defines these event counters.
        # Other vmstat fields can be current state (for example nr_free_pages).
        "vmstat_pgpgin",
        "vmstat_pgpgout",
        "vmstat_pswpin",
        "vmstat_pswpout",
        "vmstat_pgfault",
        "vmstat_pgmajfault",
        "vmstat_oom_kill",
    }
)

_TITLES = {
    "dynamo_frontend_requests_total": "Completed requests",
    "dynamo_frontend_requests_started_total": "Started requests",
    "dynamo_frontend_output_tokens_total": "Output tokens",
    "dynamo_frontend_time_to_first_token_seconds": "Time to first token",
    "dynamo_frontend_inter_token_latency_seconds": "Inter-token latency",
    "dynamo_frontend_request_duration_seconds": "Request duration",
    "dynamo_frontend_inflight_requests": "Inflight requests",
    "dynamo_frontend_active_requests": "Active requests",
    "dynamo_frontend_queued_requests": "Requests awaiting a first token",
    "dynamo_frontend_stage_requests": "Requests by frontend stage",
    "dynamo_frontend_stage_duration_seconds": "Frontend stage durations",
    "dynamo_frontend_template_seconds": "Template rendering",
    "dynamo_frontend_tokenize_seconds": "Tokenization latency",
    "dynamo_frontend_tokenizer_latency_ms": "Tokenizer operations",
    "dynamo_frontend_tokenizer_cache_hits_total": "Tokenizer cache hits",
    "dynamo_frontend_tokenizer_cache_misses_total": "Tokenizer cache misses",
    "dynamo_frontend_tokenizer_cache_cached_tokens_total": "Tokenizer cached tokens",
    "dynamo_frontend_tokenizer_cache_uncached_tokens_total": "Tokenizer uncached tokens",
    "dynamo_frontend_detokenize_token": "Detokenized tokens",
    "dynamo_frontend_detokenize_total_us": "Accumulated detokenization time",
    "dynamo_frontend_event_loop_delay_seconds": "Event-loop delay",
    "dynamo_frontend_event_loop_stall_total": "Event-loop stalls",
    "dynamo_tokio_worker_mean_poll_time_ns": "Tokio mean poll time",
    "dynamo_tokio_worker_busy_ratio": "Tokio poll-time proxy",
    "dynamo_tokio_global_queue_depth": "Tokio global queue",
    "dynamo_tokio_worker_local_queue_depth": "Tokio worker queues",
    "dynamo_tokio_alive_tasks": "Tokio live tasks",
    "dynamo_request_plane_inflight_requests": "Dispatch inflight requests",
    "dynamo_request_plane_queue_seconds": "Request-envelope preparation",
    "dynamo_request_plane_send_seconds": "Request send duration",
    "dynamo_request_plane_roundtrip_ttft_seconds": "Dispatch to first response",
    "dynamo_frontend_router_queue_pending_requests": "Router pending requests",
    "dynamo_frontend_router_queue_backpressure_total": "Router backpressure events",
    "dynamo_router_overhead_total_ms": "Routing decision time",
    "dynamo_component_router_kv_hit_rate": "Predicted KV overlap ratio",
    "dynamo_component_router_kv_transfer_estimated_latency_seconds": "Prefill completion to first decode token",
    "dynamo_frontend_worker_active_decode_blocks": "Estimated active decode blocks",
    "dynamo_frontend_worker_active_prefill_tokens": "Estimated active prefill tokens",
    "dynamo_component_inflight_requests": "Worker inflight requests",
    "dynamo_component_requests_total": "Worker received requests",
    "dynamo_component_errors_total": "Worker errors",
    "dynamo_component_cancellation_total": "Worker cancellations",
    "dynamo_component_request_duration_seconds": "Worker request lifetime",
    "dynamo_engine_request": "Requests admitted to the worker",
    "dynamo_request_queue": "Worker admission queue",
    "dynamo_work_handler_pool_active_tasks": "Active handler tasks",
    "dynamo_work_handler_pool_capacity": "Handler pool capacity",
    "dynamo_work_handler_queue_depth": "Handler queue depth",
    "dynamo_work_handler_queue_capacity": "Handler queue capacity",
    "dynamo_work_handler_permit_wait_seconds": "Concurrency permit wait",
    "dynamo_work_handler_time_to_first_response_seconds": "Handler setup to first prologue",
    "dynamo_work_handler_network_transit_seconds": "Handler network transit",
    "gpu_util": "GPU utilization",
    "gpu_power": "GPU power",
    "gpu_temp": "GPU temperature",
    "gpu_mem_total": "Historical memory-related alias",
    "FI_DEV_FB_USED": "Framebuffer used",
    "FI_DEV_FB_FREE": "Framebuffer free",
    "FI_DEV_SM_CLOCK": "SM clock",
    "FI_DEV_MEM_CLOCK": "Memory clock",
    "FI_DEV_MEM_COPY_UTIL": "Memory-controller utilization",
    "FI_DEV_NVLINK_BANDWIDTH_TOTAL": "NVLink combined receive and transmit rate",
    "FI_DEV_TOTAL_ENERGY_CONSUMPTION": "Accumulated GPU energy",
    "FI_DEV_XID_ERRORS": "GPU XID error code",
    "namedprocess_namegroup_cpu_seconds_total": "Process-group CPU",
    "namedprocess_namegroup_thread_cpu_seconds_total": "Thread-group CPU",
    "namedprocess_namegroup_memory_bytes": "Process-group memory",
    "namedprocess_namegroup_context_switches_total": "Process context switches",
    "namedprocess_namegroup_num_threads": "Process-group threads",
    "namedprocess_namegroup_threads_wchan": "Thread wait channels",
    "memory_MemAvailable_bytes": "Available host memory",
    "memory_MemFree_bytes": "Free host memory",
    "memory_MemTotal_bytes": "Total host memory",
    "memory_numa_MemUsed": "NUMA memory used",
    "context_switches_total": "Host context switches",
    "procs_running": "Runnable processes",
    "procs_blocked": "Blocked processes",
    "infiniband_port_data_received_bytes_total": "InfiniBand received data",
    "infiniband_port_data_transmitted_bytes_total": "InfiniBand transmitted data",
    "infiniband_port_transmit_wait_total": "InfiniBand transmit wait counter",
    "process_cpu_seconds_total": "Exporter CPU",
    "process_resident_memory_bytes": "Exporter resident memory",
    "scrape_collector_success": "Collector success",
    "scrape_collector_duration_seconds": "Collector duration",
}

_DISPLAY_ORDER = (
    "dynamo_frontend_requests_total",
    "dynamo_frontend_time_to_first_token_seconds",
    "dynamo_frontend_output_tokens_total",
    "dynamo_frontend_inter_token_latency_seconds",
    "dynamo_frontend_inflight_requests",
    "dynamo_frontend_tokenize_seconds",
    "dynamo_frontend_template_seconds",
    "dynamo_request_plane_roundtrip_ttft_seconds",
    "dynamo_router_overhead_total_ms",
    "dynamo_frontend_router_queue_pending_requests",
    "dynamo_component_inflight_requests",
    "dynamo_component_requests_total",
    "gpu_util",
    "FI_DEV_MEM_COPY_UTIL",
    "FI_DEV_FB_USED",
    "gpu_power",
    "gpu_temp",
    "namedprocess_namegroup_cpu_seconds_total",
    "context_switches_total",
    "procs_running",
    "memory_MemAvailable_bytes",
    "namedprocess_namegroup_memory_bytes",
)

_FEATURED = frozenset(
    {
        "dynamo_frontend_requests_total",
        "dynamo_frontend_time_to_first_token_seconds",
        "dynamo_frontend_inter_token_latency_seconds",
        "dynamo_frontend_inflight_requests",
        "dynamo_frontend_output_tokens_total",
        "dynamo_frontend_stage_requests",
        "dynamo_frontend_tokenize_seconds",
        "dynamo_frontend_template_seconds",
        "dynamo_frontend_tokenizer_cache_hits_total",
        "dynamo_frontend_tokenizer_cache_misses_total",
        "dynamo_frontend_detokenize_token",
        "dynamo_request_plane_queue_seconds",
        "dynamo_request_plane_roundtrip_ttft_seconds",
        "dynamo_frontend_event_loop_delay_seconds",
        "dynamo_tokio_global_queue_depth",
        "dynamo_tokio_alive_tasks",
        "dynamo_frontend_router_queue_pending_requests",
        "dynamo_frontend_router_queue_backpressure_total",
        "dynamo_router_overhead_total_ms",
        "dynamo_component_router_kv_hit_rate",
        "dynamo_frontend_worker_active_decode_blocks",
        "dynamo_component_kv_cache_events_applied",
        "dynamo_component_inflight_requests",
        "dynamo_component_requests_total",
        "dynamo_component_errors_total",
        "dynamo_work_handler_permit_wait_seconds",
        "dynamo_work_handler_pool_active_tasks",
        "dynamo_work_handler_queue_depth",
        "dynamo_work_handler_time_to_first_response_seconds",
        "dynamo_work_handler_network_transit_seconds",
        "gpu_util",
        "FI_DEV_FB_USED",
        "FI_DEV_MEM_COPY_UTIL",
        "gpu_power",
        "gpu_temp",
        "FI_DEV_SM_CLOCK",
        "FI_DEV_NVLINK_BANDWIDTH_TOTAL",
        "FI_DEV_XID_ERRORS",
        "namedprocess_namegroup_cpu_seconds_total",
        "namedprocess_namegroup_memory_bytes",
        "namedprocess_namegroup_num_threads",
        "memory_MemAvailable_bytes",
        "memory_numa_MemUsed",
        "context_switches_total",
        "procs_running",
        "vmstat_pgmajfault",
        "infiniband_port_data_received_bytes_total",
        "infiniband_port_data_transmitted_bytes_total",
        "scrape_collector_success",
    }
)


_NATIVE_HTTP_METRICS = frozenset(
    {
        "total_requests_total",
        "stream_requests_total",
        "nonstream_requests_total",
        "validation_exceptions_total",
        "http_exceptions_total",
        "internal_errors_total",
        "total_responses_total",
        "error_requests_total",
        "retry_requests_total",
        "completed_requests_total",
        "queue_latency_seconds",
        "first_token_latency_seconds",
        "complete_latency_seconds",
        "per_token_latency_seconds",
    }
)
_ENGINE_REQUEST_METRICS = frozenset(
    {
        "trtllm_request_success_total",
        "trtllm_request_error_total",
        "trtllm_e2e_request_latency_seconds",
        "trtllm_time_to_first_token_seconds",
        "trtllm_time_per_output_token_seconds",
        "trtllm_request_queue_time_seconds",
        "trtllm_request_prefill_time_seconds",
        "trtllm_request_decode_time_seconds",
        "trtllm_request_inference_time_seconds",
        "trtllm_prompt_tokens_total",
        "trtllm_generation_tokens_total",
    }
)


def _native_http_name(name: str) -> tuple[str, str] | None:
    role = ""
    for prefix, label in (("ctx_", "Prefill"), ("gen_", "Decode"), ("mme_", "Multimodal")):
        if name.startswith(prefix):
            role, name = label, name[len(prefix) :]
            break
    normalized = name.removesuffix("_created")
    if name.endswith("_created") and normalized not in _NATIVE_HTTP_METRICS:
        normalized += "_total"
    if normalized in _NATIVE_HTTP_METRICS:
        return role, normalized
    return None


def _canonical(name: str) -> str:
    if name.startswith("DCGM_"):
        return name[5:]
    if name.startswith("node_"):
        return name[5:]
    return name


def is_counter(name: str) -> bool:
    """Apply known semantics or _total convention; never infer from sample trends."""
    name = _canonical(name)
    if name in {"FI_DEV_NVLINK_BANDWIDTH_TOTAL", "gpu_mem_total"} or name.startswith("cpu_time_"):
        return False
    return name in _COUNTERS or name.endswith("_total")


def _placement(name: str, endpoints: Sequence[str]) -> tuple[str, str]:
    if name.startswith("trtllm_"):
        if name.endswith(("_config_info", "_created")):
            return "Workers", "Engine configuration"
        if "kv_event" in name:
            return "Workers", "Engine KV events"
        if "kv_transfer" in name:
            return "Workers", "Engine KV transfer"
        if "kv_cache" in name or "prompt_cached" in name:
            return "Workers", "Engine KV cache"
        if "spec_decode" in name:
            return "Workers", "Engine speculative decoding"
        if "memory_usage" in name:
            return "Workers", "Engine memory"
        if name in _ENGINE_REQUEST_METRICS:
            return "Workers", "Engine requests and latency"
        if "num_" in name or "batch_size" in name:
            return "Workers", "Engine scheduling"
        return "Workers", "Engine iterations"
    if native := _native_http_name(name):
        role, _ = native
        if name.endswith("_created"):
            return "Frontend", "Native HTTP metadata"
        return "Frontend", f"Native {role.lower()} client" if role else "Native HTTP requests"
    if name.startswith(("go_", "process_", "promhttp_", "scrape_collector_", "namedprocess_scrape_")) or name in {
        "exporter_build_info",
    }:
        return "Host", "Collection health"
    if name.startswith("namedprocess_namegroup_"):
        if "cpu_" in name or "context_switch" in name:
            return "Host", "Process CPU and scheduling"
        if "memory" in name:
            return "Host", "Process memory"
        return "Host", "Processes and threads"
    if name.startswith(("FI_DEV_", "FI_PROF_", "gpu_")):
        if "NVLINK" in name or "PCIE" in name:
            return "GPU", "NVLink and PCIe"
        if "FB_" in name or "mem_total" in name:
            return "GPU", "Framebuffer memory"
        if any(s in name for s in ("CLOCK", "POWER", "ENERGY", "TEMP", "power", "temp")):
            return "GPU", "Power, clocks and temperature"
        if any(s in name for s in ("ERROR", "REMAPPED", "REMAP", "LICENSE")):
            return "GPU", "Hardware events"
        return "GPU", "Utilization"
    if name.startswith(("dynamo_router_", "dynamo_component_router_", "dynamo_frontend_router_")):
        if "queue" in name or "backpressure" in name:
            return "Router", "Queue and backpressure"
        if "overhead" in name:
            return "Router", "Routing decisions"
        if any(s in name for s in ("cache", "overlap", "kv_hit")):
            return "Router", "KV matching and cache"
        return "Router", "Worker selection and feedback"
    if name.startswith(("dynamo_component_kv_cache_", "dynamo_component_ckf_")):
        return "Router", "KV index and events"
    if name.startswith("dynamo_frontend_worker_"):
        return "Router", "Worker selection and feedback"
    if name.startswith("dynamo_tokio_") or "event_loop" in name:
        return "Frontend", "Tokio runtime"
    if name.startswith(("dynamo_request_plane_", "dynamo_transport_")):
        return "Frontend", "Worker dispatch and streaming"
    if name.startswith("dynamo_work_handler_"):
        if "pool" in name:
            return "Workers", "Work-handler pool"
        if "response" in name or "transit" in name:
            return "Workers", "First response and transport"
        return "Workers", "Admission and queues"
    if name in {"dynamo_engine_request", "dynamo_request_queue", "dynamo_rejection_request_total"}:
        return "Workers", "Admission and queues"
    if name.startswith("dynamo_component_"):
        component = "Frontend" if endpoints and all(e.startswith("frontend") for e in endpoints) else "Workers"
        if "load_time" in name or "uptime" in name:
            return component, "Component lifecycle"
        if "bytes" in name:
            return component, "First response and transport"
        return component, "Request lifecycle"
    if name.startswith("dynamo_frontend_"):
        if "detokenize" in name:
            return "Frontend", "Detokenization"
        if "tokenizer_cache" in name:
            return "Frontend", "Tokenizer cache"
        if "tokeniz" in name or "template" in name:
            return "Frontend", "Tokenization"
        if "model_" in name and "cancellation" not in name:
            return "Frontend", "Advertised model metadata"
        if "stage" in name:
            return "Frontend", "Worker dispatch and streaming"
        return "Frontend", "Requests and latency"
    if name.startswith("infiniband_"):
        return "Host", "Network and InfiniBand"
    if name.startswith("memory_numa_"):
        return "Host", "NUMA memory and locality"
    if name.startswith("memory_"):
        return "Host", "Host memory"
    if name.startswith("vmstat_"):
        return "Host", "Paging and reclaim"
    if name.startswith(("cpu_", "procs_", "context_", "intr_", "forks_", "boot_")):
        return "Host", "Host scheduling"
    if name.startswith("processes_"):
        return "Host", "Processes and threads"
    if endpoints and all(e.startswith("frontend") for e in endpoints):
        return "Frontend", "Other captured metrics"
    if endpoints and all(e.startswith("backend") for e in endpoints):
        return "Workers", "Other captured metrics"
    return "Host", "Other captured metrics"


def _unit(name: str) -> str:
    if name.endswith("_created"):
        return "epoch s"
    if native := _native_http_name(name):
        _, base = native
        if base.endswith("_seconds"):
            return "s"
        if "responses" in base:
            return "responses"
        if any(word in base for word in ("exceptions", "errors")):
            return "errors"
        return "requests"
    if name.startswith("trtllm_"):
        if name.endswith("_config_info"):
            return "info"
        if name in {
            "trtllm_kv_cache_utilization",
            "trtllm_kv_cache_host_utilization",
            "trtllm_kv_cache_hit_rate",
            "trtllm_kv_cache_iter_reuse_rate",
            "trtllm_prefill_batch_occupancy",
        }:
            return "ratio"
        if name in {"trtllm_total_context_tokens", "trtllm_avg_decoded_tokens_per_iter"}:
            return "tokens/iteration"
        if name == "trtllm_kv_cache_tokens_per_block":
            return "tokens/block"
        if "kv_event" in name and "seconds" not in name:
            return "events"
        if "blocks" in name:
            return "blocks"
        if (
            "requests" in name
            or "batch_size" in name
            or name in {"trtllm_request_success_total", "trtllm_request_error_total"}
        ):
            return "requests"
    exact = {
        "trtllm_kv_transfer_speed_gb_s": "GB/s",
        "trtllm_kv_transfer_success_total": "transfers",
        "gpu_util": "%",
        "gpu_power": "W",
        "gpu_temp": "°C",
        "FI_DEV_FB_USED": "MiB",
        "FI_DEV_FB_FREE": "MiB",
        "FI_DEV_SM_CLOCK": "MHz",
        "FI_DEV_MEM_CLOCK": "MHz",
        "FI_DEV_MEM_COPY_UTIL": "%",
        "FI_DEV_DEC_UTIL": "%",
        "FI_DEV_ENC_UTIL": "%",
        "FI_DEV_NVLINK_BANDWIDTH_TOTAL": "MB/s",
        "FI_DEV_TOTAL_ENERGY_CONSUMPTION": "mJ",
        "dynamo_frontend_detokenize_total_us": "µs",
        "go_gc_duration_seconds_count": "observations",
        "dynamo_frontend_requests_total": "requests",
        "dynamo_frontend_requests_started_total": "requests",
        "dynamo_frontend_inflight_requests": "requests",
        "dynamo_frontend_active_requests": "requests",
        "dynamo_frontend_queued_requests": "requests",
        "dynamo_frontend_stage_requests": "requests",
        "dynamo_frontend_router_queue_pending_requests": "requests",
        "dynamo_frontend_router_queue_backpressure_total": "events",
        "dynamo_frontend_tokenizer_cache_hits_total": "hits",
        "dynamo_frontend_tokenizer_cache_misses_total": "misses",
        "dynamo_frontend_event_loop_stall_total": "stalls",
        "dynamo_frontend_disconnected_clients": "disconnects",
        "dynamo_component_inflight_requests": "requests",
        "dynamo_component_requests_total": "requests",
        "dynamo_component_errors_total": "errors",
        "dynamo_component_cancellation_total": "cancellations",
        "dynamo_component_kv_cache_events_applied": "events",
        "dynamo_component_kv_cache_event_warnings": "warnings",
        "dynamo_engine_request": "requests",
        "dynamo_request_queue": "requests",
        "dynamo_request_plane_inflight_requests": "requests",
        "dynamo_work_handler_pool_active_tasks": "tasks",
        "dynamo_work_handler_pool_capacity": "tasks",
        "dynamo_tokio_alive_tasks": "tasks",
        "dynamo_tokio_global_queue_depth": "tasks",
        "dynamo_tokio_worker_local_queue_depth": "tasks",
        "namedprocess_namegroup_num_procs": "processes",
        "namedprocess_namegroup_num_threads": "threads",
        "namedprocess_namegroup_threads_wchan": "threads",
        "namedprocess_namegroup_context_switches_total": "switches",
        "context_switches_total": "switches",
        "procs_running": "processes",
        "procs_blocked": "processes",
        "FI_DEV_GPU_UTIL": "%",
    }
    if name in exact:
        return exact[name]
    if name in {"gpu_mem_total", "dynamo_tokio_worker_busy_ratio"} or name.startswith("cpu_time_"):
        return "stored value"
    if "bytes" in name:
        return "bytes"
    if name.endswith("_ms"):
        return "ms"
    if name.endswith("_ns"):
        return "ns"
    if "seconds" in name:
        return "s"
    if "tokens" in name or name == "dynamo_frontend_detokenize_token":
        return "tokens"
    if "hit_rate" in name or "fd_ratio" in name:
        return "ratio"
    return "stored value"


def describe_metric(name: str, endpoints: Sequence[str] = ()) -> dict[str, Any]:
    """Return presentation metadata, not measurements or missing-label guesses."""
    n = _canonical(name)
    component, group = _placement(n, endpoints)
    quality = ""
    description = "Captured values retain their endpoint, inline labels, and capture metadata."
    if n.startswith("trtllm_"):
        description = (
            "TRT-LLM engine measurements at the captured endpoint. Preserve worker role, process and rank labels; "
            "engine observations are not frontend/client measurements."
        )
        if group in {"Engine scheduling", "Engine iterations", "Engine memory", "Engine KV cache"}:
            description += " Scraped gauges report the latest published engine statistics, not every iteration."
        if n in _ENGINE_REQUEST_METRICS:
            description += " Request statistics and token counts are recorded when a request finishes."
    elif native := _native_http_name(n):
        role, base = native
        description = (
            f"Native serving HTTP {'upstream ' + role.lower() + ' client' if role else 'server'} "
            "measurements at this endpoint; keep server and worker identities separate."
        )
        if base == "per_token_latency_seconds":
            description += " Observations are gaps between streamed responses, which need not each contain one token."
        if base == "first_token_latency_seconds":
            description += " Time to the first upstream streamed response, not whole-request completion."
    elif group == "Collection health":
        description = "Collector/exporter instrumentation; process and Go statistics here describe the exporter itself."
    elif n.startswith("namedprocess_namegroup_"):
        description = (
            "Observed process groups selected by the exporter. Use groupname and endpoint labels to identify a process."
        )
        if "thread" in n:
            description += " Thread names identify exported groups, not guaranteed individual threads or runtime pools."
    elif group == "Tokio runtime":
        description = (
            "Runtime activity observed at the selected frontend endpoint; it does not isolate HTTP from router work."
        )
    elif group == "Worker selection and feedback":
        description = "Router-side estimates and worker feedback; these are not engine occupancy measurements."
    elif group == "Requests and latency":
        description = "Server-observed request measurements. Outcomes use the server's own definitions."
    if n == "trtllm_iteration_latency_seconds":
        quality = "Latest published iteration latency gauge. Its scrape distribution is not an all-iteration latency histogram."
    elif n == "trtllm_total_context_tokens":
        quality = "Context tokens in the reported iteration; not a cumulative token counter or tokens-per-second rate."
    elif n == "trtllm_time_per_output_token_seconds":
        quality = (
            "Distribution of per-request average decode time per output token; not individual inter-token latency."
        )
    elif n == "trtllm_kv_cache_hit_rate":
        quality = "Engine-reported cache reuse ratio; not the router's predicted KV overlap."
    elif n in {"trtllm_spec_decode_acceptance_length", "trtllm_spec_decode_draft_overhead"}:
        quality = (
            "Engine-reported scalar; units are not encoded in the capture and this producer version is not verified."
        )
    elif n.startswith("trtllm_kv_transfer_") and not n.endswith("_created"):
        description = (
            "TRT-LLM request transfer timing, size or speed published by Dynamo. "
            "Transfer speed uses decimal GB/s; this timer does not include the full prefill-to-decode handoff."
        )
    elif n.endswith("_created"):
        quality = "Metric creation timestamp in epoch seconds, not a latency or event count."
    elif n == "dynamo_tokio_worker_busy_ratio":
        quality = "Historical poll-time proxy, not CPU utilization or a true busy ratio. Prefer mean poll time and queue metrics."
    elif n.startswith("cpu_time_"):
        quality = (
            "Quantile of cumulative per-CPU counters. Differences are not CPU utilization quantiles or node CPU totals."
        )
    elif n == "gpu_mem_total":
        quality = "Historical broad memory-field rename; cannot identify GPU capacity from this stored alias."
    elif n in {"dynamo_frontend_stage_duration_seconds", "dynamo_frontend_tokenizer_latency_ms"}:
        quality = "Historical attached histogram sums/counts can cross label sets; use recorded bucket samples."
    elif n == "dynamo_engine_request":
        quality = "Handler-admitted request gauge; does not measure scheduled engine requests or engine occupancy."
    elif n == "dynamo_work_handler_pool_active_tasks":
        description = "Active handler tasks, including requests that may be waiting inside a backend."
    elif n == "dynamo_work_handler_time_to_first_response_seconds":
        description = (
            "Worker handler setup to its first prologue response; not engine TTFT or frontend time to first token."
        )
    elif n == "dynamo_component_router_kv_transfer_estimated_latency_seconds":
        quality = (
            "Reported handoff interval can include admission/slot waiting; it is not an isolated KV-transfer timer."
        )
    elif n == "dynamo_component_router_time_to_first_token_seconds":
        quality = "Historical router timing can observe the prefill leg more than once; not frontend/client TTFT."
    elif n.startswith("dynamo_frontend_model_") and "cancellation" not in n:
        quality = "Advertised model gauge can be overwritten by different worker types; do not infer per-role engine capacity."
    elif n == "FI_DEV_NVLINK_BANDWIDTH_TOTAL":
        description = "Already a combined receive/transmit rate; not a counter and not isolated KV-transfer traffic."
    elif n == "FI_DEV_MEM_COPY_UTIL":
        description = "Memory-controller busy percentage; does not measure achieved memory bandwidth."
    elif n == "dynamo_frontend_disconnected_clients":
        quality = "Historically exported as a gauge but incremented cumulatively for disconnect events."
    elif n == "dynamo_request_plane_queue_seconds":
        description = "Request-envelope preparation duration before send; does not measure waiting in a dispatch queue."
    elif n == "dynamo_component_requests_total":
        description = "Requests received at worker handler entry; does not establish successful completion."
    elif n == "dynamo_frontend_queued_requests":
        description = "Requests awaiting a first token; does not identify which internal queue they occupy."
    elif n == "dynamo_frontend_stage_requests":
        description = "Concurrent requests by observed frontend stage, including dispatch awaiting a first token."
    engine_titles = {
        "trtllm_num_requests_running": "Engine active requests",
        "trtllm_num_requests_waiting": "Engine queued requests",
        "trtllm_num_scheduled_requests": "Engine scheduled requests",
        "trtllm_num_context_requests": "Engine prefill requests",
        "trtllm_num_generation_requests": "Engine decode requests",
        "trtllm_num_paused_requests": "Engine paused requests",
        "trtllm_iteration_latency_seconds": "Published engine iteration latency",
        "trtllm_total_context_tokens": "Context tokens per iteration",
        "trtllm_avg_decoded_tokens_per_iter": "Average decoded tokens per iteration",
        "trtllm_kv_cache_utilization": "Engine KV cache used fraction",
        "trtllm_kv_cache_hit_rate": "Engine KV cache hit ratio",
        "trtllm_kv_cache_free_blocks": "Engine KV cache free blocks",
        "trtllm_kv_cache_used_blocks": "Engine KV cache used blocks",
        "trtllm_kv_cache_max_blocks": "Engine KV cache block capacity",
        "trtllm_time_to_first_token_seconds": "Engine request time to first token",
        "trtllm_time_per_output_token_seconds": "Per-request average time per output token",
        "trtllm_e2e_request_latency_seconds": "Engine request lifetime",
        "trtllm_request_queue_time_seconds": "Engine request queue time",
        "trtllm_request_prefill_time_seconds": "Engine request prefill time",
        "trtllm_request_decode_time_seconds": "Engine request decode time",
        "trtllm_request_inference_time_seconds": "Engine request inference time",
        "trtllm_prompt_tokens_total": "Prompt tokens in finished requests",
        "trtllm_generation_tokens_total": "Output tokens in finished requests",
    }
    title = _TITLES.get(n) or engine_titles.get(n)
    if title is None and (native := _native_http_name(n)):
        role, base = native
        labels = {
            "total_requests_total": "Requests started",
            "stream_requests_total": "Streaming requests",
            "nonstream_requests_total": "Non-streaming requests",
            "validation_exceptions_total": "Request validation errors",
            "http_exceptions_total": "HTTP errors",
            "internal_errors_total": "Internal errors",
            "total_responses_total": "Responses",
            "error_requests_total": "Request errors",
            "retry_requests_total": "Request retries",
            "completed_requests_total": "Requests finished",
            "queue_latency_seconds": "HTTP request queue latency",
            "first_token_latency_seconds": "First streamed response latency",
            "complete_latency_seconds": "Upstream request lifetime",
            "per_token_latency_seconds": "Streamed response interval",
        }
        title = (role + " client: " if role else "Native HTTP: ") + labels[base]
        if n.endswith("_created"):
            title = (role + " client " if role else "Native HTTP ") + "metric creation time"
    if title is None:
        short = n
        for prefix in (
            "dynamo_frontend_",
            "dynamo_component_",
            "dynamo_",
            "trtllm_",
            "namedprocess_namegroup_",
            "FI_DEV_",
        ):
            if short.startswith(prefix):
                short = short[len(prefix) :]
                break
        title = short.replace("_", " ").capitalize()
    unit = _unit(n)
    counter = is_counter(n)
    rate_unit = "cores" if n.endswith("cpu_seconds_total") else f"{unit}/s"
    return {
        "component": component,
        "group": group,
        "group_order": GROUP_ORDER[component].index(group) if group in GROUP_ORDER[component] else 100,
        "title": title,
        "unit": unit,
        "rate_unit": rate_unit,
        "counter": counter,
        "description": description,
        "quality": quality,
        "featured": n in _FEATURED,
        "order": _DISPLAY_ORDER.index(n) if n in _DISPLAY_ORDER else 100,
    }
