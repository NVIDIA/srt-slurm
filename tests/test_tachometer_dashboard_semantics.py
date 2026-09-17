# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prevent known scrape semantics from becoming misleading dashboard claims.

These regressions follow the Hecate capture audit: the tests distinguish the
measurement's scope from the endpoint exporting it and from its historical name.
"""

from __future__ import annotations

import pytest

from srtctl.analysis.tachometer_dashboard.catalog import describe_metric, is_counter


def _text(metric: dict) -> str:
    return " ".join(str(metric.get(key, "")) for key in ("title", "group", "description", "quality")).lower()


def test_exporter_cpu_is_distinct_from_workload_process_cpu() -> None:
    exporter = describe_metric("process_cpu_seconds_total", ("process_exporter_node1",))
    workload = describe_metric("namedprocess_namegroup_cpu_seconds_total", ("process_exporter_node1",))

    assert exporter["group"] != workload["group"]
    assert "exporter" in _text(exporter)
    assert workload["component"] == "Host"
    assert exporter["counter"] and workload["counter"]


@pytest.mark.parametrize("endpoint", ["node_exporter_node1", "process_exporter_node1"])
def test_exporter_runtime_is_not_the_frontend_runtime(endpoint: str) -> None:
    exporter = describe_metric("go_goroutines", (endpoint,))
    frontend = describe_metric("dynamo_tokio_alive_tasks", ("frontend0",))

    assert "exporter" in _text(exporter)
    assert frontend["component"] == "Frontend"
    assert (exporter["component"], exporter["group"]) != (frontend["component"], frontend["group"])


def test_frontend_dispatch_worker_admission_and_router_estimates_stay_separate() -> None:
    dispatch = describe_metric("dynamo_frontend_stage_requests", ("frontend0",))
    handler = describe_metric("dynamo_work_handler_pool_active_tasks", ("backend_decode0_rank0",))
    estimate = describe_metric("dynamo_frontend_worker_active_decode_blocks", ("frontend0",))

    assert dispatch["component"] == "Frontend"
    assert handler["component"] == "Workers"
    assert estimate["component"] == "Router"
    assert any(word in _text(estimate) for word in ("estimate", "estimated", "belief", "predict"))


def test_generic_component_uptime_follows_the_measured_component() -> None:
    frontend = describe_metric("dynamo_component_uptime_seconds", ("frontend0",))
    worker = describe_metric("dynamo_component_uptime_seconds", ("backend_prefill0_rank0",))

    assert frontend["component"] == "Frontend"
    assert worker["component"] == "Workers"
    assert not frontend["counter"] and not worker["counter"]


def test_model_cancellations_remain_request_outcomes() -> None:
    cancellation = describe_metric("dynamo_frontend_model_cancellation_total", ("frontend0",))
    capacity = describe_metric("dynamo_frontend_model_max_num_seqs", ("frontend0",))

    assert cancellation["component"] == "Frontend"
    assert cancellation["counter"]
    assert (cancellation["component"], cancellation["group"]) != (capacity["component"], capacity["group"])


@pytest.mark.parametrize(
    "name",
    ["dynamo_engine_request", "dynamo_work_handler_pool_active_tasks"],
)
def test_handler_activity_is_not_presented_as_engine_batch_occupancy(name: str) -> None:
    metric = describe_metric(name, ("backend_decode0_rank0",))

    assert metric["component"] == "Workers"
    assert any(word in _text(metric) for word in ("handler", "worker-admitted", "admitted to the worker"))
    assert not metric["counter"]


def test_tokio_busy_proxy_is_not_a_cpu_percentage() -> None:
    metric = describe_metric("dynamo_tokio_worker_busy_ratio", ("frontend0",))

    assert metric["component"] == "Frontend"
    assert metric["quality"]
    assert "poll" in _text(metric)
    assert metric["unit"].lower() not in {"%", "percent", "percentage"}
    assert not metric["counter"]


def test_historical_gpu_mem_total_is_not_usable_framebuffer_capacity() -> None:
    metric = describe_metric("gpu_mem_total", ("dcgm_node1",))

    assert metric["component"] == "GPU"
    assert metric["quality"]
    assert any(word in _text(metric) for word in ("temperature", "alias", "rename"))
    assert metric["unit"].lower() not in {"b", "bytes", "mib", "gib", "mb", "gb"}
    assert not metric["counter"]


@pytest.mark.parametrize("name", ["cpu_time_min", "cpu_time_p10", "cpu_time_p90", "cpu_time_max"])
def test_cumulative_cpu_quantiles_cannot_claim_node_utilization(name: str) -> None:
    metric = describe_metric(name, ("node_exporter_node1",))

    assert metric["component"] == "Host"
    assert metric["quality"]
    assert any(word in _text(metric) for word in ("quantile", "cumulative", "percentile"))
    assert metric["unit"].lower() not in {"%", "percent", "percentage", "cores"}
    assert not metric["counter"]
    assert not is_counter(name)


@pytest.mark.parametrize("name", ["FI_DEV_NVLINK_BANDWIDTH_TOTAL", "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL"])
def test_nvlink_bandwidth_is_already_a_rate_despite_its_name(name: str) -> None:
    metric = describe_metric(name, ("dcgm_node1",))

    assert metric["component"] == "GPU"
    assert not metric["counter"]
    assert not is_counter(name)


@pytest.mark.parametrize(
    "name",
    [
        "dynamo_frontend_detokenize_token",
        "dynamo_frontend_detokenize_total_us",
        "dynamo_frontend_disconnected_clients",
        "vmstat_oom_kill",
        "namedprocess_scrape_errors",
        "namedprocess_scrape_partial_errors",
        "namedprocess_scrape_procread_errors",
    ],
)
def test_known_counters_do_not_depend_on_a_total_suffix(name: str) -> None:
    assert is_counter(name)


def test_summary_quantile_is_not_a_counter() -> None:
    assert not is_counter("go_gc_duration_seconds")


def test_worker_request_counter_measures_receipts_not_completions() -> None:
    metric = describe_metric("dynamo_component_requests_total", ("backend_decode0_rank0",))

    assert metric["counter"]
    assert "complet" not in metric["title"].lower()
    assert any(word in _text(metric) for word in ("receiv", "receipt", "arrival", "start"))


def test_request_plane_queue_histogram_describes_envelope_preparation() -> None:
    metric = describe_metric("dynamo_request_plane_queue_seconds", ("frontend0",))

    assert "queue wait" not in metric["title"].lower()
    assert any(word in _text(metric) for word in ("envelope", "preparation", "prepare", "build"))


def test_summary_observation_count_does_not_inherit_seconds_unit() -> None:
    metric = describe_metric("go_gc_duration_seconds_count", ("process_exporter_node1",))

    assert metric["counter"]
    assert metric["unit"].lower() not in {"s", "ms", "seconds"}


@pytest.mark.parametrize(
    ("name", "unit", "counter"),
    [
        ("dynamo_frontend_requests_total", "requests", True),
        ("dynamo_component_requests_total", "requests", True),
        ("dynamo_component_inflight_requests", "requests", False),
        ("dynamo_component_errors_total", "errors", True),
        ("dynamo_work_handler_pool_active_tasks", "tasks", False),
        ("dynamo_tokio_global_queue_depth", "tasks", False),
        ("context_switches_total", "switches", True),
        ("procs_running", "processes", False),
        ("namedprocess_namegroup_num_threads", "threads", False),
    ],
)
def test_known_count_units_make_gauges_and_rates_interpretable(name: str, unit: str, counter: bool) -> None:
    metric = describe_metric(name)

    assert metric["unit"] == unit
    assert metric["counter"] is counter
    if counter:
        assert metric["rate_unit"] == f"{unit}/s"


# Metric-name-only snapshot of job4_metric_inventory.md (2026-09-09 capture).
# This is a classification acceptance fixture, never dashboard measurement input.
_JOB4_INVENTORY_NAMES = frozenset(
    {
        "FI_DEV_CORRECTABLE_REMAPPED_ROWS",
        "FI_DEV_DEC_UTIL",
        "FI_DEV_ENC_UTIL",
        "FI_DEV_FB_FREE",
        "FI_DEV_FB_USED",
        "FI_DEV_MEM_CLOCK",
        "FI_DEV_MEM_COPY_UTIL",
        "FI_DEV_NVLINK_BANDWIDTH_TOTAL",
        "FI_DEV_PCIE_REPLAY_COUNTER",
        "FI_DEV_ROW_REMAP_FAILURE",
        "FI_DEV_SM_CLOCK",
        "FI_DEV_TOTAL_ENERGY_CONSUMPTION",
        "FI_DEV_UNCORRECTABLE_REMAPPED_ROWS",
        "FI_DEV_VGPU_LICENSE_STATUS",
        "FI_DEV_XID_ERRORS",
        "boot_time_seconds",
        "context_switches_total",
        "cpu_guest_seconds_total",
        "cpu_time_max",
        "cpu_time_min",
        "cpu_time_p10",
        "cpu_time_p90",
        "dynamo_component_cancellation_total",
        "dynamo_component_ckf_mutation_total",
        "dynamo_component_errors_total",
        "dynamo_component_inflight_requests",
        "dynamo_component_kv_cache_event_warnings",
        "dynamo_component_kv_cache_events_applied",
        "dynamo_component_model_load_time_seconds",
        "dynamo_component_request_bytes_total",
        "dynamo_component_request_duration_seconds_bucket",
        "dynamo_component_request_duration_seconds_count",
        "dynamo_component_request_duration_seconds_sum",
        "dynamo_component_requests_total",
        "dynamo_component_response_bytes_total",
        "dynamo_component_router_input_sequence_tokens_bucket",
        "dynamo_component_router_input_sequence_tokens_count",
        "dynamo_component_router_input_sequence_tokens_sum",
        "dynamo_component_router_inter_token_latency_seconds_bucket",
        "dynamo_component_router_inter_token_latency_seconds_count",
        "dynamo_component_router_inter_token_latency_seconds_sum",
        "dynamo_component_router_kv_hit_rate_bucket",
        "dynamo_component_router_kv_hit_rate_count",
        "dynamo_component_router_kv_hit_rate_sum",
        "dynamo_component_router_kv_transfer_estimated_latency_seconds_bucket",
        "dynamo_component_router_kv_transfer_estimated_latency_seconds_count",
        "dynamo_component_router_kv_transfer_estimated_latency_seconds_sum",
        "dynamo_component_router_non_max_overlap_selections_total",
        "dynamo_component_router_output_sequence_tokens_bucket",
        "dynamo_component_router_output_sequence_tokens_count",
        "dynamo_component_router_output_sequence_tokens_sum",
        "dynamo_component_router_overlap_blocks_lost_bucket",
        "dynamo_component_router_overlap_blocks_lost_count",
        "dynamo_component_router_overlap_blocks_lost_sum",
        "dynamo_component_router_requests_started_total",
        "dynamo_component_router_requests_total",
        "dynamo_component_router_shared_cache_beyond_blocks_bucket",
        "dynamo_component_router_shared_cache_beyond_blocks_count",
        "dynamo_component_router_shared_cache_beyond_blocks_sum",
        "dynamo_component_router_shared_cache_hit_rate_bucket",
        "dynamo_component_router_shared_cache_hit_rate_count",
        "dynamo_component_router_shared_cache_hit_rate_sum",
        "dynamo_component_router_time_to_first_token_seconds_bucket",
        "dynamo_component_router_time_to_first_token_seconds_count",
        "dynamo_component_router_time_to_first_token_seconds_sum",
        "dynamo_component_router_worker_registered",
        "dynamo_component_uptime_seconds",
        "dynamo_engine_request",
        "dynamo_frontend_active_requests",
        "dynamo_frontend_audio_per_request_bucket",
        "dynamo_frontend_audio_per_request_count",
        "dynamo_frontend_audio_per_request_sum",
        "dynamo_frontend_cached_tokens_bucket",
        "dynamo_frontend_cached_tokens_count",
        "dynamo_frontend_cached_tokens_sum",
        "dynamo_frontend_detokenize_token",
        "dynamo_frontend_detokenize_total_us",
        "dynamo_frontend_disconnected_clients",
        "dynamo_frontend_event_loop_delay_seconds_bucket",
        "dynamo_frontend_event_loop_delay_seconds_count",
        "dynamo_frontend_event_loop_delay_seconds_sum",
        "dynamo_frontend_event_loop_stall_total",
        "dynamo_frontend_image_tokens_per_request_bucket",
        "dynamo_frontend_image_tokens_per_request_count",
        "dynamo_frontend_image_tokens_per_request_sum",
        "dynamo_frontend_images_per_request_bucket",
        "dynamo_frontend_images_per_request_count",
        "dynamo_frontend_images_per_request_sum",
        "dynamo_frontend_inflight_requests",
        "dynamo_frontend_input_sequence_tokens_bucket",
        "dynamo_frontend_input_sequence_tokens_count",
        "dynamo_frontend_input_sequence_tokens_sum",
        "dynamo_frontend_inter_token_latency_seconds_bucket",
        "dynamo_frontend_inter_token_latency_seconds_count",
        "dynamo_frontend_inter_token_latency_seconds_sum",
        "dynamo_frontend_model_cancellation_total",
        "dynamo_frontend_model_context_length",
        "dynamo_frontend_model_kv_cache_block_size",
        "dynamo_frontend_model_max_num_batched_tokens",
        "dynamo_frontend_model_max_num_seqs",
        "dynamo_frontend_model_migration_limit",
        "dynamo_frontend_model_total_kv_blocks",
        "dynamo_frontend_output_sequence_tokens_bucket",
        "dynamo_frontend_output_sequence_tokens_count",
        "dynamo_frontend_output_sequence_tokens_sum",
        "dynamo_frontend_output_tokens_total",
        "dynamo_frontend_queued_requests",
        "dynamo_frontend_request_duration_seconds_bucket",
        "dynamo_frontend_request_duration_seconds_count",
        "dynamo_frontend_request_duration_seconds_sum",
        "dynamo_frontend_requests_started_total",
        "dynamo_frontend_requests_total",
        "dynamo_frontend_router_queue_backpressure_total",
        "dynamo_frontend_router_queue_pending_cached_tokens",
        "dynamo_frontend_router_queue_pending_isl_tokens",
        "dynamo_frontend_router_queue_pending_requests",
        "dynamo_frontend_stage_duration_seconds_bucket",
        "dynamo_frontend_stage_duration_seconds_count",
        "dynamo_frontend_stage_duration_seconds_sum",
        "dynamo_frontend_stage_requests",
        "dynamo_frontend_template_seconds_bucket",
        "dynamo_frontend_template_seconds_count",
        "dynamo_frontend_template_seconds_sum",
        "dynamo_frontend_time_to_first_token_seconds_bucket",
        "dynamo_frontend_time_to_first_token_seconds_count",
        "dynamo_frontend_time_to_first_token_seconds_sum",
        "dynamo_frontend_tokenize_seconds_bucket",
        "dynamo_frontend_tokenize_seconds_count",
        "dynamo_frontend_tokenize_seconds_sum",
        "dynamo_frontend_tokenizer_cache_cached_tokens_total",
        "dynamo_frontend_tokenizer_cache_hits_total",
        "dynamo_frontend_tokenizer_cache_misses_total",
        "dynamo_frontend_tokenizer_cache_uncached_tokens_total",
        "dynamo_frontend_tokenizer_latency_ms_bucket",
        "dynamo_frontend_tokenizer_latency_ms_count",
        "dynamo_frontend_tokenizer_latency_ms_sum",
        "dynamo_frontend_videos_per_request_bucket",
        "dynamo_frontend_videos_per_request_count",
        "dynamo_frontend_videos_per_request_sum",
        "dynamo_frontend_worker_active_decode_blocks",
        "dynamo_frontend_worker_active_prefill_tokens",
        "dynamo_frontend_worker_last_input_sequence_tokens",
        "dynamo_frontend_worker_last_inter_token_latency_seconds",
        "dynamo_frontend_worker_last_time_to_first_token_seconds",
        "dynamo_rejection_request_total",
        "dynamo_request_plane_inflight_requests",
        "dynamo_request_plane_queue_seconds_bucket",
        "dynamo_request_plane_queue_seconds_count",
        "dynamo_request_plane_queue_seconds_sum",
        "dynamo_request_plane_roundtrip_ttft_seconds_bucket",
        "dynamo_request_plane_roundtrip_ttft_seconds_count",
        "dynamo_request_plane_roundtrip_ttft_seconds_sum",
        "dynamo_request_plane_send_seconds_bucket",
        "dynamo_request_plane_send_seconds_count",
        "dynamo_request_plane_send_seconds_sum",
        "dynamo_request_queue",
        "dynamo_router_overhead_block_hashing_ms_bucket",
        "dynamo_router_overhead_block_hashing_ms_count",
        "dynamo_router_overhead_block_hashing_ms_sum",
        "dynamo_router_overhead_indexer_find_matches_ms_bucket",
        "dynamo_router_overhead_indexer_find_matches_ms_count",
        "dynamo_router_overhead_indexer_find_matches_ms_sum",
        "dynamo_router_overhead_scheduling_ms_bucket",
        "dynamo_router_overhead_scheduling_ms_count",
        "dynamo_router_overhead_scheduling_ms_sum",
        "dynamo_router_overhead_seq_hashing_ms_bucket",
        "dynamo_router_overhead_seq_hashing_ms_count",
        "dynamo_router_overhead_seq_hashing_ms_sum",
        "dynamo_router_overhead_shared_cache_query_ms_bucket",
        "dynamo_router_overhead_shared_cache_query_ms_count",
        "dynamo_router_overhead_shared_cache_query_ms_sum",
        "dynamo_router_overhead_total_ms_bucket",
        "dynamo_router_overhead_total_ms_count",
        "dynamo_router_overhead_total_ms_sum",
        "dynamo_router_shared_cache_errors_total",
        "dynamo_tokio_alive_tasks",
        "dynamo_tokio_blocking_idle_threads",
        "dynamo_tokio_blocking_queue_depth",
        "dynamo_tokio_blocking_threads",
        "dynamo_tokio_budget_forced_yield_total",
        "dynamo_tokio_global_queue_depth",
        "dynamo_tokio_worker_busy_ratio",
        "dynamo_tokio_worker_local_queue_depth",
        "dynamo_tokio_worker_mean_poll_time_ns",
        "dynamo_tokio_worker_overflow_count_total",
        "dynamo_tokio_worker_park_count_total",
        "dynamo_tokio_worker_steal_count_total",
        "dynamo_transport_tcp_bytes_received_total",
        "dynamo_transport_tcp_bytes_sent_total",
        "dynamo_transport_tcp_errors_total",
        "dynamo_work_handler_enqueue_rejected_total",
        "dynamo_work_handler_network_transit_seconds_bucket",
        "dynamo_work_handler_network_transit_seconds_count",
        "dynamo_work_handler_network_transit_seconds_sum",
        "dynamo_work_handler_permit_wait_seconds_bucket",
        "dynamo_work_handler_permit_wait_seconds_count",
        "dynamo_work_handler_permit_wait_seconds_sum",
        "dynamo_work_handler_pool_active_tasks",
        "dynamo_work_handler_pool_capacity",
        "dynamo_work_handler_queue_capacity",
        "dynamo_work_handler_queue_depth",
        "dynamo_work_handler_time_to_first_response_seconds_bucket",
        "dynamo_work_handler_time_to_first_response_seconds_count",
        "dynamo_work_handler_time_to_first_response_seconds_sum",
        "exporter_build_info",
        "forks_total",
        "go_gc_duration_seconds",
        "go_gc_duration_seconds_count",
        "go_gc_duration_seconds_sum",
        "go_goroutines",
        "go_info",
        "go_memstats_alloc_bytes",
        "go_memstats_alloc_bytes_total",
        "go_memstats_buck_hash_sys_bytes",
        "go_memstats_frees_total",
        "go_memstats_gc_sys_bytes",
        "go_memstats_heap_alloc_bytes",
        "go_memstats_heap_idle_bytes",
        "go_memstats_heap_inuse_bytes",
        "go_memstats_heap_objects",
        "go_memstats_heap_released_bytes",
        "go_memstats_heap_sys_bytes",
        "go_memstats_last_gc_time_seconds",
        "go_memstats_lookups_total",
        "go_memstats_mallocs_total",
        "go_memstats_mcache_inuse_bytes",
        "go_memstats_mcache_sys_bytes",
        "go_memstats_mspan_inuse_bytes",
        "go_memstats_mspan_sys_bytes",
        "go_memstats_next_gc_bytes",
        "go_memstats_other_sys_bytes",
        "go_memstats_stack_inuse_bytes",
        "go_memstats_stack_sys_bytes",
        "go_memstats_sys_bytes",
        "go_threads",
        "gpu_mem_total",
        "gpu_power",
        "gpu_temp",
        "gpu_util",
        "infiniband_excessive_buffer_overrun_errors_total",
        "infiniband_info",
        "infiniband_link_downed_total",
        "infiniband_link_error_recovery_total",
        "infiniband_local_link_integrity_errors_total",
        "infiniband_multicast_packets_received_total",
        "infiniband_multicast_packets_transmitted_total",
        "infiniband_physical_state_id",
        "infiniband_port_constraint_errors_received_total",
        "infiniband_port_constraint_errors_transmitted_total",
        "infiniband_port_data_received_bytes_total",
        "infiniband_port_data_transmitted_bytes_total",
        "infiniband_port_discards_transmitted_total",
        "infiniband_port_errors_received_total",
        "infiniband_port_packets_received_total",
        "infiniband_port_packets_transmitted_total",
        "infiniband_port_receive_remote_physical_errors_total",
        "infiniband_port_receive_switch_relay_errors_total",
        "infiniband_port_transmit_wait_total",
        "infiniband_rate_bytes_per_second",
        "infiniband_state_id",
        "infiniband_symbol_error_total",
        "infiniband_unicast_packets_received_total",
        "infiniband_unicast_packets_transmitted_total",
        "infiniband_vl15_dropped_total",
        "intr_total",
        "memory_Active_anon_bytes",
        "memory_Active_bytes",
        "memory_Active_file_bytes",
        "memory_AnonHugePages_bytes",
        "memory_AnonPages_bytes",
        "memory_Balloon_bytes",
        "memory_Bounce_bytes",
        "memory_Buffers_bytes",
        "memory_Cached_bytes",
        "memory_CmaFree_bytes",
        "memory_CmaTotal_bytes",
        "memory_CommitLimit_bytes",
        "memory_Committed_AS_bytes",
        "memory_Dirty_bytes",
        "memory_FileHugePages_bytes",
        "memory_FilePmdMapped_bytes",
        "memory_HardwareCorrupted_bytes",
        "memory_HugePages_Free",
        "memory_HugePages_Rsvd",
        "memory_HugePages_Surp",
        "memory_HugePages_Total",
        "memory_Hugepagesize_bytes",
        "memory_Hugetlb_bytes",
        "memory_Inactive_anon_bytes",
        "memory_Inactive_bytes",
        "memory_Inactive_file_bytes",
        "memory_KReclaimable_bytes",
        "memory_KernelStack_bytes",
        "memory_Mapped_bytes",
        "memory_MemAvailable_bytes",
        "memory_MemFree_bytes",
        "memory_MemTotal_bytes",
        "memory_Mlocked_bytes",
        "memory_NFS_Unstable_bytes",
        "memory_PageTables_bytes",
        "memory_Percpu_bytes",
        "memory_SReclaimable_bytes",
        "memory_SUnreclaim_bytes",
        "memory_SecPageTables_bytes",
        "memory_ShadowCallStack_bytes",
        "memory_ShmemHugePages_bytes",
        "memory_ShmemPmdMapped_bytes",
        "memory_Shmem_bytes",
        "memory_Slab_bytes",
        "memory_SwapCached_bytes",
        "memory_SwapFree_bytes",
        "memory_SwapTotal_bytes",
        "memory_Unevictable_bytes",
        "memory_VmallocChunk_bytes",
        "memory_VmallocTotal_bytes",
        "memory_VmallocUsed_bytes",
        "memory_WritebackTmp_bytes",
        "memory_Writeback_bytes",
        "memory_Zswap_bytes",
        "memory_Zswapped_bytes",
        "memory_numa_Active",
        "memory_numa_Active_anon",
        "memory_numa_Active_file",
        "memory_numa_AnonHugePages",
        "memory_numa_AnonPages",
        "memory_numa_Bounce",
        "memory_numa_Dirty",
        "memory_numa_FileHugePages",
        "memory_numa_FilePages",
        "memory_numa_FilePmdMapped",
        "memory_numa_HugePages_Free",
        "memory_numa_HugePages_Surp",
        "memory_numa_HugePages_Total",
        "memory_numa_Inactive",
        "memory_numa_Inactive_anon",
        "memory_numa_Inactive_file",
        "memory_numa_KReclaimable",
        "memory_numa_KernelStack",
        "memory_numa_Mapped",
        "memory_numa_MemFree",
        "memory_numa_MemTotal",
        "memory_numa_MemUsed",
        "memory_numa_Mlocked",
        "memory_numa_NFS_Unstable",
        "memory_numa_PageTables",
        "memory_numa_SReclaimable",
        "memory_numa_SUnreclaim",
        "memory_numa_SecPageTables",
        "memory_numa_ShadowCallStack",
        "memory_numa_Shmem",
        "memory_numa_ShmemHugePages",
        "memory_numa_ShmemPmdMapped",
        "memory_numa_Slab",
        "memory_numa_SwapCached",
        "memory_numa_Unevictable",
        "memory_numa_Writeback",
        "memory_numa_WritebackTmp",
        "memory_numa_interleave_hit_total",
        "memory_numa_local_node_total",
        "memory_numa_numa_foreign_total",
        "memory_numa_numa_hit_total",
        "memory_numa_numa_miss_total",
        "memory_numa_other_node_total",
        "namedprocess_namegroup_context_switches_total",
        "namedprocess_namegroup_cpu_seconds_total",
        "namedprocess_namegroup_major_page_faults_total",
        "namedprocess_namegroup_memory_bytes",
        "namedprocess_namegroup_minor_page_faults_total",
        "namedprocess_namegroup_num_procs",
        "namedprocess_namegroup_num_threads",
        "namedprocess_namegroup_oldest_start_time_seconds",
        "namedprocess_namegroup_open_filedesc",
        "namedprocess_namegroup_read_bytes_total",
        "namedprocess_namegroup_states",
        "namedprocess_namegroup_thread",
        "namedprocess_namegroup_thread_context_switches_total",
        "namedprocess_namegroup_thread_cpu_seconds_total",
        "namedprocess_namegroup_thread_io_bytes_total",
        "namedprocess_namegroup_thread_major_page_faults_total",
        "namedprocess_namegroup_thread_minor_page_faults_total",
        "namedprocess_namegroup_threads_wchan",
        "namedprocess_namegroup_worst_fd_ratio",
        "namedprocess_namegroup_write_bytes_total",
        "namedprocess_scrape_errors",
        "namedprocess_scrape_partial_errors",
        "namedprocess_scrape_procread_errors",
        "process_cpu_seconds_total",
        "process_exporter_build_info",
        "process_max_fds",
        "process_open_fds",
        "process_resident_memory_bytes",
        "process_start_time_seconds",
        "process_virtual_memory_bytes",
        "process_virtual_memory_max_bytes",
        "processes_max_processes",
        "processes_max_threads",
        "processes_pids",
        "processes_state",
        "processes_threads",
        "processes_threads_state",
        "procs_blocked",
        "procs_running",
        "promhttp_metric_handler_errors_total",
        "promhttp_metric_handler_requests_in_flight",
        "promhttp_metric_handler_requests_total",
        "scrape_collector_duration_seconds",
        "scrape_collector_success",
        "vmstat_oom_kill",
        "vmstat_pgfault",
        "vmstat_pgmajfault",
        "vmstat_pgpgin",
        "vmstat_pgpgout",
        "vmstat_pgsteal_anon",
        "vmstat_pgsteal_direct",
        "vmstat_pgsteal_file",
        "vmstat_pgsteal_khugepaged",
        "vmstat_pgsteal_kswapd",
        "vmstat_pgsteal_proactive",
        "vmstat_pswpin",
        "vmstat_pswpout",
    }
)


def _inventory_family(name: str) -> str:
    # Collapse only an explicitly recorded histogram triplet. Go summary
    # quantiles and their sum/count are distinct raw families, not histograms.
    for suffix in ("_bucket", "_sum", "_count"):
        if name.endswith(suffix):
            base = name.removesuffix(suffix)
            if base + "_bucket" in _JOB4_INVENTORY_NAMES:
                return base
    return name


def test_all_job4_inventory_names_have_a_named_component_subgroup() -> None:
    assert len(_JOB4_INVENTORY_NAMES) == 420
    families = {_inventory_family(name) for name in _JOB4_INVENTORY_NAMES}
    assert len(families) == 346
    unresolved = []
    for name in sorted(families):
        metric = describe_metric(name)
        if metric["component"] not in {"Frontend", "Router", "Workers", "GPU", "Host"}:
            unresolved.append((name, "component", metric["component"]))
        if not metric["group"] or metric["group"] == "Other captured metrics":
            unresolved.append((name, "group", metric["group"]))
    assert not unresolved


def test_inventory_reconciliation_does_not_reinterpret_go_summary_quantiles() -> None:
    for name in ("go_gc_duration_seconds", "go_gc_duration_seconds_count", "go_gc_duration_seconds_sum"):
        assert _inventory_family(name) == name
    for suffix in ("_bucket", "_count", "_sum"):
        assert _inventory_family("dynamo_frontend_time_to_first_token_seconds" + suffix) == (
            "dynamo_frontend_time_to_first_token_seconds"
        )


def test_engine_iteration_latency_is_a_sampled_gauge() -> None:
    metric = describe_metric("trtllm_iteration_latency_seconds", ("backend_decode0_rank0",))

    assert metric["component"] == "Workers"
    assert metric["group"] == "Engine iterations"
    assert metric["unit"] == "s"
    assert not metric["counter"]
    assert "gauge" in _text(metric)
    assert "not an all-iteration" in _text(metric)


def test_context_tokens_are_iteration_work_not_a_throughput_counter() -> None:
    metric = describe_metric("trtllm_total_context_tokens")

    assert metric["component"] == "Workers"
    assert metric["unit"] == "tokens/iteration"
    assert not metric["counter"]
    assert "not a cumulative" in _text(metric)


@pytest.mark.parametrize("endpoint", ["frontend0", "backend_prefill0_rank0", "backend_decode0_rank0"])
def test_engine_occupancy_keeps_its_semantics_at_every_exporter(endpoint: str) -> None:
    metric = describe_metric("trtllm_num_requests_running", (endpoint,))
    admission = describe_metric("dynamo_engine_request", (endpoint,))

    assert metric["component"] == "Workers"
    assert metric["group"] == "Engine scheduling"
    assert metric["unit"] == "requests"
    assert not metric["counter"]
    assert metric["group"] != admission["group"]


@pytest.mark.parametrize("name", ["trtllm_kv_cache_utilization", "trtllm_kv_cache_hit_rate"])
def test_engine_cache_ratios_are_not_raw_percentages_or_router_predictions(name: str) -> None:
    metric = describe_metric(name)

    assert metric["unit"] == "ratio"
    assert not metric["counter"]
    assert (metric["component"], metric["group"]) == ("Workers", "Engine KV cache")


def test_native_http_stream_latency_and_engine_average_tpot_are_distinct() -> None:
    native = describe_metric("gen_per_token_latency_seconds", ("frontend0",))
    engine = describe_metric("trtllm_time_per_output_token_seconds", ("backend_decode0_rank0",))

    assert native["component"] == "Frontend"
    assert "streamed responses" in _text(native)
    assert engine["component"] == "Workers"
    assert "per-request average" in _text(engine)
    assert "not individual inter-token" in _text(engine)


@pytest.mark.parametrize("name", ["trtllm_prompt_tokens_total", "trtllm_generation_tokens_total"])
def test_engine_token_counters_retain_completion_accounting(name: str) -> None:
    metric = describe_metric(name)

    assert metric["counter"]
    assert metric["rate_unit"] == "tokens/s"
    assert "request finishes" in _text(metric)


@pytest.mark.parametrize(
    "name",
    ["total_requests_total", "ctx_completed_requests_total", "gen_retry_requests_total"],
)
def test_native_http_request_counts_are_frontend_counters(name: str) -> None:
    metric = describe_metric(name, ("frontend0",))

    assert metric["component"] == "Frontend"
    assert metric["unit"] == "requests"
    assert metric["counter"]


@pytest.mark.parametrize("name", ["gen_total_requests_created", "ctx_first_token_latency_seconds_created"])
def test_metric_creation_timestamps_do_not_become_latency_or_request_panels(name: str) -> None:
    metric = describe_metric(name, ("frontend0",))

    assert metric["group"] == "Native HTTP metadata"
    assert metric["unit"] == "epoch s"
    assert not metric["counter"]


def test_transfer_speed_is_a_per_transfer_distribution_not_a_bandwidth_counter() -> None:
    metric = describe_metric("trtllm_kv_transfer_speed_gb_s", ("backend_decode0_rank0",))

    assert (metric["component"], metric["group"]) == ("Workers", "Engine KV transfer")
    assert metric["unit"] == "GB/s"
    assert not metric["counter"]
    assert "full prefill-to-decode handoff" in _text(metric)
