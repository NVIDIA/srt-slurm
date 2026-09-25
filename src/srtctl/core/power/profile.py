# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU power exporter profiles: which Prometheus metric and labels carry a GPU's watts.

The power collector is exporter-agnostic. Every exporter it can scrape is one
row in :data:`POWER_PROFILES`, keyed by the ``power_profile`` name a cluster or
recipe puts on its exporter block. A row names the power metric, the labels
that identify a device, the optional utilization riders, and the launch and
tachometer defaults for that exporter. The parser, session, manifest,
validator, and telemetry stage read the row; none of them compares a vendor or
exporter name.

Adding an exporter is adding a row. The artifact contract (``samples.csv``
columns, manifest keys, reason codes) does not change per row; the manifest
records the row's name, metric, and scope so a reader can interpret the watts.
"""

from __future__ import annotations

from dataclasses import dataclass

from srtctl.core.power.contract import POWER_METRIC, POWER_SCOPE, UTILIZATION_METRICS, UtilizationMetric

# Power telemetry's DCGM template: 100ms NVML sampling is its purpose (dense power
# curves inside sa-bench measurement windows). Never used for tachometer.
DCGM_EXPORTER_COMMAND_TEMPLATE = "dcgm-exporter --collect-interval=100 --address :{port}"

# The rocm/device-metrics-exporter image's ENTRYPOINT starts the ``gpuagent``
# daemon the exporter reads from and then execs the exporter. Pyxis runs the
# command it is given rather than the image entrypoint, so the entrypoint script
# is the command. The exporter takes no port flag: it listens on 5000 unless a
# ``/etc/metrics/config.json`` sets ``ServerPort``.
AMD_DEVICE_METRICS_EXPORTER_COMMAND_TEMPLATE = "/home/amd/tools/entrypoint.sh"


@dataclass(frozen=True)
class PowerMetricProfile:
    """One exporter's mapping onto the fixed GPU power artifact.

    ``gpu_index_label`` must carry the node-local device index srt-slurm
    allocates by (the same index the worker sees in its visible-devices mask).
    ``gpu_identity_label`` must be stable for one physical device across the run
    and distinct between devices; it fills the ``gpu_uuid`` column.
    ``instance_labels`` mark samples for logical sub-devices (MIG instances,
    partitions) that the artifact cannot represent; such samples are dropped
    with ``mig_instance_unsupported``. ``utilization_metrics`` must be a subset
    of the contract's columns.
    """

    name: str
    power_metric: str
    power_scope: str
    gpu_index_label: str
    gpu_identity_label: str
    default_command_template: str
    utilization_metrics: tuple[UtilizationMetric, ...] = ()
    instance_labels: tuple[str, ...] = ()
    # What tachometer applies when it also scrapes this exporter alongside the
    # power collector (``TelemetryStageMixin._power_exporter_targets``).
    tachometer_filter: str = "passthrough"
    tachometer_gpu_metadata: bool = False

    def __post_init__(self) -> None:
        contract_columns = {metric.column: metric for metric in UTILIZATION_METRICS}
        for metric in self.utilization_metrics:
            spec = contract_columns.get(metric.column)
            if spec is None:
                raise ValueError(f"power profile {self.name!r} maps unknown artifact column {metric.column!r}")
            if (metric.unit, metric.max_value) != (spec.unit, spec.max_value):
                raise ValueError(f"power profile {self.name!r} changes the contract of column {metric.column!r}")
        columns = [metric.column for metric in self.utilization_metrics]
        metrics = [self.power_metric, *(metric.metric for metric in self.utilization_metrics)]
        if len(set(columns)) != len(columns) or len(set(metrics)) != len(metrics):
            raise ValueError(f"power profile {self.name!r} repeats a column or metric")
        if self.gpu_index_label == self.gpu_identity_label:
            raise ValueError(f"power profile {self.name!r} needs distinct index and identity labels")


DCGM_POWER_PROFILE = PowerMetricProfile(
    name="dcgm",
    power_metric=POWER_METRIC,
    power_scope=POWER_SCOPE,
    gpu_index_label="gpu",
    gpu_identity_label="UUID",
    default_command_template=DCGM_EXPORTER_COMMAND_TEMPLATE,
    utilization_metrics=UTILIZATION_METRICS,
    instance_labels=("GPU_I_ID", "GPU_I_PROFILE"),
    tachometer_filter="dcgm",
    tachometer_gpu_metadata=True,
)

# rocm/device-metrics-exporter (https://github.com/ROCm/device-metrics-exporter).
# Metric and label names are lowercase in its exposition. ``gpu_power_usage`` is
# the per-device draw the exporter reports for MI2xx/MI3xx on bare metal; the
# socket figures (``gpu_package_power``) are a different boundary and not used.
# ``serial_number`` is exported by default; ``gpu_uuid`` is not, so the serial is
# the device identity. Partitioned GPUs share one serial and report 0 W past the
# first partition, which surfaces as ``gpu_uuid_changed``: partitioning is not
# supported by this profile, just as MIG is not by the DCGM one.
AMD_DEVICE_METRICS_POWER_PROFILE = PowerMetricProfile(
    name="amd-device-metrics",
    power_metric="gpu_power_usage",
    power_scope="gpu_device_power_as_reported_by_amd_device_metrics_exporter",
    gpu_index_label="gpu_id",
    gpu_identity_label="serial_number",
    default_command_template=AMD_DEVICE_METRICS_EXPORTER_COMMAND_TEMPLATE,
    utilization_metrics=(
        UtilizationMetric(column="gpu_util_pct", metric="gpu_gfx_activity", unit="percent", max_value=100.0),
    ),
)

POWER_PROFILES: dict[str, PowerMetricProfile] = {
    profile.name: profile for profile in (DCGM_POWER_PROFILE, AMD_DEVICE_METRICS_POWER_PROFILE)
}
DEFAULT_POWER_PROFILE = DCGM_POWER_PROFILE


def get_power_profile(name: str | None) -> PowerMetricProfile:
    """The profile an exporter block names, or the DCGM default when it names none."""
    if name is None:
        return DEFAULT_POWER_PROFILE
    try:
        return POWER_PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(POWER_PROFILES))
        raise ValueError(f"unknown power_profile {name!r}; known profiles: {known}") from None
