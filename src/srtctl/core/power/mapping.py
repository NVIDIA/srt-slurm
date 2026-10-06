# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which Prometheus metric and labels carry a GPU's watts in an exporter's scrape.

The power collector is exporter-agnostic. An exporter block's ``gpu_labels`` and
``gpu_metrics`` resolve to one :class:`PowerMetricMapping`; unset, they are DCGM
(:data:`DCGM_POWER_MAPPING`). The parser, session, manifest, and telemetry stage
read the mapping; none of them compares a vendor or exporter name. The artifact
contract (``samples.csv`` columns, manifest keys, reason codes) is the same for
every mapping; the manifest records the metric and scope so a reader can
interpret the watts.
"""

from __future__ import annotations

from dataclasses import dataclass

from srtctl.core.power.contract import (
    POWER_METRIC,
    POWER_SCOPE,
    TEMPERATURE_METRIC,
    UTILIZATION_METRICS,
    UtilizationMetric,
)

# Power telemetry's DCGM template: 100ms NVML sampling is its purpose (dense power
# curves inside sa-bench measurement windows). Never used for tachometer.
DCGM_EXPORTER_COMMAND_TEMPLATE = "dcgm-exporter --collect-interval=100 --address :{port}"


@dataclass(frozen=True)
class PowerMetricMapping:
    """One exporter's mapping onto the fixed GPU power artifact.

    ``gpu_index_label`` must carry the node-local device index srt-slurm
    allocates by (the same index the worker sees in its visible-devices mask).
    ``gpu_identity_label`` must be stable for one physical device across the run
    and distinct between devices; it fills the ``gpu_uuid`` column.
    ``instance_labels`` mark samples for logical sub-devices (MIG instances,
    partitions) that the artifact cannot represent; such samples are dropped
    with ``mig_instance_unsupported``. ``utilization_metrics`` fill contract
    columns with the contract's unit and range. ``temperature_metric``, when
    set, fills the optional ``temperature_c`` column in Celsius.
    """

    power_metric: str
    power_scope: str
    gpu_index_label: str
    gpu_identity_label: str
    utilization_metrics: tuple[UtilizationMetric, ...] = ()
    instance_labels: tuple[str, ...] = ()
    temperature_metric: str | None = None
    # What tachometer applies when it also scrapes this exporter alongside the
    # power collector (``TelemetryStageMixin._power_exporter_targets``).
    tachometer_filter: str = "passthrough"
    tachometer_gpu_metadata: bool = False


DCGM_POWER_MAPPING = PowerMetricMapping(
    power_metric=POWER_METRIC,
    power_scope=POWER_SCOPE,
    gpu_index_label="gpu",
    gpu_identity_label="UUID",
    utilization_metrics=UTILIZATION_METRICS,
    instance_labels=("GPU_I_ID", "GPU_I_PROFILE"),
    temperature_metric=TEMPERATURE_METRIC,
    tachometer_filter="dcgm",
    tachometer_gpu_metadata=True,
)
