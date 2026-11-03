# RDMA NIC Traffic

Every job that spans more than one node records each RDMA NIC's transmit and receive bytes once per second, and post-processing reports the mean and peak Gbit/s of every NIC and every node over each benchmark concurrency window. Use it to see how much KV cache, collective or other RDMA traffic a disaggregated or multi-node deployment moves, and how bursty it is.

## Table of Contents

- [Enabling It](#enabling-it)
- [How It Runs](#how-it-runs)
- [Counter Sources](#counter-sources)
- [Artifacts](#artifacts)
- [Window Summary](#window-summary)
- [Running the Sampler by Hand](#running-the-sampler-by-hand)
- [Limitations](#limitations)

## Enabling It

Capture needs no configuration. A recipe whose engine roles and service pools use more than one node (`SrtConfig.total_nodes > 1`) captures by default. It is independent of `observability.enabled` and `telemetry.enabled`.

```yaml
observability:
  nic_traffic:
    enabled: false               # unset: on for multi-node jobs; true: also on a single node
    sample_interval_seconds: 1.0 # read period on each node; at most 3
    storage_subdir: nic_traffic  # below the run log directory
```

Fields: [NicTrafficConfig](schema-reference.md#nictrafficconfig). `storage_subdir` must not equal or nest with the Tachometer, GPU power or host CPU power directory, because the power report reads any `samples.csv` below those directories as power. `srtctl dry-run` shows a `nic_traffic` row when capture is on. The [example recipe](https://github.com/NVIDIA/srt-slurm/blob/main/examples/features/nic-traffic.yaml) runs SGLang prefill and decode on separate nodes so every KV transfer crosses the network.

## How It Runs

After the workers, the frontend and the `after_frontend` services start, and before the benchmark runs, srtctl starts `srtctl.core.nic_traffic` once per node with the orchestrator's Python interpreter, directly on the bare host and outside the model container. The sampler is one `srun` step named `telemetry_nic_traffic`, or one step per component in a heterogeneous job. The nodes are the engine worker nodes plus every service pool node, as for the CPU power collector.

At startup the sampler lists `/sys/class/infiniband/*`, picks each device's counter source from its driver, and writes `nodes/<host>.metadata.json` with the devices it will sample and the ones it cannot. It then writes a ready marker and appends one row per port per interval to `nodes/<host>.csv`. A node with no readable RDMA counter writes an error marker instead and exits 0.

At teardown, before process cleanup, the head node stops the samplers with SIGTERM, merges the node CSVs into `nic_traffic.csv`, and writes `nic_manifest.json`. Capture is best-effort throughout: a launch failure, a missing node or a sampler that exits is logged and recorded in the manifest, and never fails or skips the benchmark.

Sample timestamps come from each node's wall clock and window boundaries from the benchmark client's, so the summary assumes NTP-synchronized nodes, as [GPU power telemetry](power-telemetry.md) does.

## Counter Sources

The driver bound to `/sys/class/infiniband/<device>/device` selects the counter files. Each row of the table in `srtctl.core.nic_traffic.COUNTER_SOURCES` names them; a driver without a row uses the standard counters.

| Driver | Transmit / receive files under `ports/<port>/` | Bytes per counter unit |
| --- | --- | --- |
| `ionic` (AMD Pollara) | `hw_counters/tx_rdma_ucast_bytes`, `hw_counters/rx_rdma_ucast_bytes` | 1 |
| any other (`mlx5_core` for NVIDIA ConnectX, and other IB and RoCE drivers) | `counters/port_xmit_data`, `counters/port_rcv_data` | 4 |

**Standard counters.** The kernel documents `port_xmit_data` and `port_rcv_data` as "data octets, divided by 4 (lanes)" and as 64-bit counters ([Documentation/ABI/stable/sysfs-class-infiniband, Linux v6.8](https://github.com/torvalds/linux/blob/e8f897f4afef0031fe618a8e94127a0934896aba/Documentation/ABI/stable/sysfs-class-infiniband#L150-L154)). The IB core serves them from the port's performance management counters ([drivers/infiniband/core/sysfs.c#L655-L665](https://github.com/torvalds/linux/blob/e8f897f4afef0031fe618a8e94127a0934896aba/drivers/infiniband/core/sysfs.c#L655-L665)), and creates the `counters/` directory only for drivers that implement `process_mad` ([#L1252-L1253](https://github.com/torvalds/linux/blob/e8f897f4afef0031fe618a8e94127a0934896aba/drivers/infiniband/core/sysfs.c#L1252-L1253)).

**AMD Pollara.** The ionic RDMA driver implements no `process_mad`, so its ports have no `counters/` directory. Its `hw_counters/` names come from the device firmware rather than the driver source: the upstream driver copies each name from the firmware's counter descriptors ([ionic_hw_stats.c, Linux v6.18](https://github.com/torvalds/linux/blob/7d0a66e4bb9081d75c82ec4957c50034cb0ea449/drivers/infiniband/hw/ionic/ionic_hw_stats.c#L26-L37)). The RDMA device's parent is the PCI function ([ionic_ibdev.c#L329](https://github.com/torvalds/linux/blob/7d0a66e4bb9081d75c82ec4957c50034cb0ea449/drivers/infiniband/hw/ionic/ionic_ibdev.c#L329)), whose driver is the `ionic` Ethernet driver; that is the name the table keys on.

The unicast byte counters were chosen by measurement. On an MI355X node with eight Pollara 400G ports (`rdma0` to `rdma7`, Linux 6.8), a 10-second `ib_write_bw` from `rdma0` to `rdma1` ran at 391.17 Gbit/s, about 489 GB of payload:

| Counter | Change during the test |
| --- | --- |
| `rdma0 hw_counters/tx_rdma_ucast_bytes` | +490.015 GB |
| `rdma1 hw_counters/rx_rdma_ucast_bytes` | +490.015 GB |
| netdev `statistics/tx_bytes`, `rx_bytes`, ethtool `tx_bytes` | 0 (RDMA bypasses the netdev) |
| ethtool MAC `octets_tx_ok` | +310 GB read right after the test (it lags) |

The unicast counters matched the payload and are readable by an unprivileged job user. The other byte counters (`*_rdma_mcast_bytes`, `*_rdma_ccl_cts_bytes`, `tx_rdma_retx_bytes`) are not added: their overlap with unicast is undocumented, and retransmits would count the same payload twice.

The node-exporter `infiniband` collector that Tachometer runs cannot read ionic. node-exporter v1.8.2, the built-in image, parses sysfs with procfs v0.14.0, which reads `counters/` for every device not named `irdma*` and fails the port when that directory is missing ([sysfs/class_infiniband.go#L282-L289](https://github.com/prometheus/procfs/blob/6bec248f1af7b94f7b57c4b2cefd7c2a8b80e40a/sysfs/class_infiniband.go#L282-L289), [#L305-L312](https://github.com/prometheus/procfs/blob/6bec248f1af7b94f7b57c4b2cefd7c2a8b80e40a/sysfs/class_infiniband.go#L305-L312)). Running the sampler on its own also keeps the capture independent of `observability.tachometer`, which a recipe may turn off.

## Artifacts

```text
<log_dir>/<storage_subdir>/
├── nic_traffic.csv           # merged samples, all nodes, in time order
├── nic_manifest.json
├── nic_traffic_summary.csv   # per-window rates, written by post-processing
├── nodes/
│   ├── <host>.csv
│   └── <host>.metadata.json
└── ready/
    └── <host>.ready.json | <host>.error.json
```

No file is named `samples.csv` or `manifest.json`: the power energy report and downstream tools treat files with those names as power evidence.

`nic_traffic.csv` has the header `timestamp_unix,sample_seq,hostname,device,port,netdev,driver,tx_bytes,rx_bytes`. Each row holds the raw cumulative counters, already converted to bytes, for one port at one sampler pass. Rates are never stored.

- `sample_seq` numbers a node's sampler passes from 0. Every port read in a pass carries the same value.
- `timestamp_unix` is the midpoint of that port's own transmit and receive reads, not the start of the pass. Counter reads are not instant: on the MI355X node above, one ionic `hw_counters` read took a median of 0.4 ms but up to 98 ms, and a pass over its 16 counter files took up to 456 ms. With one timestamp per pass, a port read late in a slow pass was stamped early, and 1-second peaks of a 391 Gbit/s transfer read up to 437 Gbit/s, above the 400G line rate.

`nic_manifest.json` records `status` (`running` while capturing, then `complete` or `incomplete`), `expected_nodes`, `observed_nodes`, `missing_nodes`, `node_errors`, `sample_row_count`, `exited_processes` and, under `nodes`, each node's metadata: the sampled `devices` with their `driver`, `counter_source`, counter files and `bytes_per_unit`, and the `unsupported` devices with the reason. `reason_codes` lists every reason the capture is incomplete:

| Reason | Meaning |
| --- | --- |
| `nic_counters_unavailable` | At least one node found no RDMA port with readable counters (see `node_errors`). |
| `nic_node_samples_missing` | At least one expected node has no rows (see `missing_nodes`). |
| `nic_samples_malformed` | A node CSV had an unexpected header or a row of the wrong width; the rest merged. |
| `nic_sampler_exited` | A sampler step had exited before teardown. |
| `interrupted` | The job was stopped by a signal or a critical process failure. |

## Window Summary

After the run, post-processing writes `nic_traffic_summary.csv` whenever `nic_traffic.csv` exists, whether or not power telemetry ran. The windows are the ones the power energy report uses: sa-bench's recorded start and end time per concurrency, or the first request start and last request end of AIPerf's profiling records. A run without benchmark windows (serve-only, eval-only) gets no summary.

For each window and NIC, the window is snapped to the NIC's nearest sample on each side. A boundary more than 3 seconds from any sample, or a window shorter than one sample interval, leaves that NIC out of the window, and the sweep log names it. Each pair of consecutive samples of a NIC is an interval whose rate is the counter delta over the time between that NIC's own timestamps. An interval in which either counter decreased spans a counter reset and is dropped. Over the remaining intervals:

- `mean_*_gbps` is the summed byte delta times 8, over the summed interval time, in 10^9 bit/s;
- `peak_*_gbps` is the highest single-interval rate, so its resolution is `sample_interval_seconds`.

Each window has one `nic` row per covered NIC and one `node` row per host. A node's mean is the sum of its NICs' means. Its peak sums the rates of the NIC intervals that end at the same `sample_seq` and takes the highest; NICs read in one pass have different timestamps, so the pass, not the time, aligns them. The columns are `concurrency, window_start_unix, window_end_unix, scope, hostname, device, port, netdev, driver, intervals, dropped_intervals, mean_tx_gbps, mean_rx_gbps, peak_tx_gbps, peak_rx_gbps`.

## Running the Sampler by Hand

The sampler imports only the Python standard library, so a copy of `src/srtctl/core/nic_traffic.py` runs on any node with `python3`. It samples until SIGTERM:

```bash
timeout -s TERM 60 python3 nic_traffic.py --output-dir /tmp/nic --interval-seconds 1
```

This writes `/tmp/nic/<host>.csv`, `<host>.metadata.json` and the ready or error marker. `--sysfs-root` points it at another sysfs tree; `--ready-dir` moves the marker.

## Limitations

- Pollara traffic is unicast RDMA payload only. Multicast, retransmitted and other firmware-counted bytes are not included.
- A device whose port performance counters are only 32 bits wide exposes saturating counters. Once saturated, such a port reports no further traffic.
- A port that fails a read during a pass misses that row; its next interval then spans two passes, which lowers the peak resolution for that interval.
- Peaks shorter than the sample interval are averaged over it.
