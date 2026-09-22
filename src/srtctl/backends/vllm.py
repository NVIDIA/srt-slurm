# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
vLLM backend configuration.

Implements BackendProtocol for vLLM inference serving with prefill/decode disaggregation.
Supports Dynamo's vLLM integration module or direct ``vllm serve`` behind
either the public direct-vLLM frontend or official vLLM Router.
"""

from __future__ import annotations

import builtins
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass as stdlib_dataclass
from dataclasses import field, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
)

from marshmallow import Schema, ValidationError
from marshmallow_dataclass import dataclass

from srtctl.backends.sidecar import build_sidecar_launch_command, get_dynamo_sidecar_config, sidecar_grpc_port
from srtctl.ports import (
    BOOTSTRAP_PORTS,
    DP_RPC_PORTS,
    DYN_SYSTEM_PORT_BASE,
    HTTP_PORTS,
    KV_EVENTS_PORTS,
    KVBM_ZMQ_PORTS,
    MOONCAKE_HTTP_METADATA_PORT,
    MOONCAKE_MASTER_PORT,
    NIXL_PORTS,
    SIDECAR_GRPC_PORTS,
    SYS_PORTS,
    VLLM_DATA_PARALLEL_RPC_PORT,
    VLLM_MASTER_PORT_BASE,
    VLLM_MASTER_PORT_STRIDE,
    VLLM_SCAN_PORTS,
)

if TYPE_CHECKING:
    from srtctl.backends.base import SrunConfig
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import DynamoConfig, ProfilingConfig
    from srtctl.core.topology import Endpoint, NodePortAllocator, Process

# Type alias for worker modes
WorkerMode = Literal["prefill", "decode", "agg"]
DPLaunchMode = Literal["per_gpu", "per_node"]

logger = logging.getLogger(__name__)

# vLLM CLI flags srtslurm derives for the direct vLLM frontend. Recipes may
# still set these for backward compatibility; dry-run warns and direct-vLLM
# command building strips them so user values cannot override allocation.
_VLLM_ORCHESTRATION_FLAGS = frozenset(
    {
        "headless",
        "host",
        "port",
        "master-addr",
        "nnodes",
        "node-rank",
    }
)


# vLLM CLI flags that only apply where the OpenAI API server runs. Headless node
# ranks host engine workers only, and vLLM rejects a positive --api-server-count
# there instead of ignoring it.
_VLLM_API_SERVER_ONLY_FLAGS = frozenset({"api-server-count"})


def normalize_vllm_config_key(key: str) -> str:
    """Normalize a vllm_config dict key to kebab-case CLI flag form."""
    return str(key).replace("_", "-")


def _pop_flags(config: dict[str, Any], flags: frozenset[str]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for key in list(config.keys()):
        normalized = normalize_vllm_config_key(key)
        if normalized in flags:
            found[normalized] = config.pop(key)
    return found


def pop_vllm_orchestration_flags(config: dict[str, Any]) -> dict[str, Any]:
    """Remove topology-managed vLLM flags from a mode config dict.

    Returns the normalized flag names that were present, mapped to the value the
    recipe asked for, so callers can report what they overrode.
    """
    return _pop_flags(config, _VLLM_ORCHESTRATION_FLAGS)


def pop_vllm_api_server_flags(config: dict[str, Any]) -> dict[str, Any]:
    """Remove API-server-only vLLM flags from a mode config dict.

    Returns the normalized flag names that were present, mapped to the recipe value.
    """
    return _pop_flags(config, _VLLM_API_SERVER_ONLY_FLAGS)


def _log_overridden_recipe_flags(
    overridden: dict[str, Any],
    effective: dict[str, str],
    node: str,
) -> None:
    """Report recipe flags srtslurm took over, alongside the values it used.

    Without this the override is invisible at runtime: the flag simply vanishes
    from the worker command and the recipe still claims otherwise.
    """
    if not overridden:
        return

    changes = ", ".join(
        f"--{flag}={value!s} -> {effective.get(flag, 'not passed')}" for flag, value in overridden.items()
    )
    logger.warning(
        "Overriding topology-managed vllm_config flags on %s: %s. srtslurm derives these from the allocation.",
        node,
        changes,
    )


def find_vllm_orchestration_recipe_flags(backend: VLLMProtocol) -> list[tuple[str, str]]:
    """Return ``(mode, flag)`` pairs set in recipe ``vllm_config``."""
    if backend.vllm_config is None:
        return []

    findings: list[tuple[str, str]] = []
    for mode_name, mode_config in (
        ("prefill", backend.vllm_config.prefill),
        ("decode", backend.vllm_config.decode),
        ("aggregated", backend.vllm_config.aggregated),
    ):
        if not mode_config:
            continue
        for key in mode_config:
            normalized = normalize_vllm_config_key(key)
            if normalized in _VLLM_ORCHESTRATION_FLAGS:
                findings.append((mode_name, normalized))
    return findings


# Filename for the mooncake-store JSON config srtslurm writes to log_dir at job
# start. log_dir is mounted into every worker at /logs, so workers read the JSON
# from MOONCAKE_STORE_CONFIG_CONTAINER_PATH.
MOONCAKE_STORE_CONFIG_FILENAME = "mooncake_store_config.json"
MOONCAKE_STORE_CONFIG_CONTAINER_PATH = f"/logs/{MOONCAKE_STORE_CONFIG_FILENAME}"


@dataclass(frozen=True)
class VLLMMooncakeKVStoreConfig:
    """Mooncake KV store config for the vLLM backend.

    When present, srtslurm launches ``mooncake_master`` on the infra node
    (co-located with etcd/nats) using the shared SGLang launch command and
    injects on every vLLM worker::

        MOONCAKE_MASTER              = <infra_ip>:8700
        MOONCAKE_TE_META_DATA_SERVER = http://<infra_ip>:8701/metadata
        MOONCAKE_LOCAL_HOSTNAME      = <worker_ip>
        MOONCAKE_CONFIG_PATH         = /logs/mooncake_store_config.json

    The JSON file referenced by ``MOONCAKE_CONFIG_PATH`` is generated by
    srtslurm at job start (see ``do_sweep._write_mooncake_store_config``) from
    ``store_config`` below. vLLM's ``MooncakeStoreConnector`` reads this
    file via ``MooncakeStoreConfig.load_from_env()``.

    ``env:`` is injected on every vLLM worker (alongside the auto-stamped
    ``MOONCAKE_*`` vars above), not on the standalone ``mooncake_master``
    daemon — the master srun command passes no env. Use this for
    in-process Mooncake C++ libraries linked into the worker:
    ``MC_*`` knobs read by the transfer engine / store client
    (e.g. ``MC_ENABLE_DEST_DEVICE_AFFINITY``, ``MC_STORE_CLIENT_METRIC``,
    ``MC_TE_METRIC``), and any ``MOONCAKE_*`` overrides the connector
    consults.

    Example YAML::

        backend:
          type: vllm
          mooncake_kv_store:
            container: inferactinc/public:mk-int-20260507  # optional
            env:                              # injected on every worker
              MC_ENABLE_DEST_DEVICE_AFFINITY: "1"   # in-process Mooncake C++ knobs
              MC_STORE_CLIENT_METRIC: "1"
            store_config:                     # MooncakeStoreConfig JSON keys
              metadata_server: "P2PHANDSHAKE"
              global_segment_size: "100GB"
              local_buffer_size: "4GB"
              protocol: "rdma"
              device_name: ""
              # master_server_address: srtslurm auto-fills from infra IP
    """

    container: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    master_extra_args: list[str] = field(default_factory=list)
    # ``store_config`` values are JSON-serialized into MOONCAKE_CONFIG_PATH and
    # parsed by vLLM's ``MooncakeStoreConfig`` dataclass — fields are a mix of
    # str (e.g. ``protocol``), int (e.g. ``port``), and human-readable sizes
    # (e.g. ``"4GB"``). Type as ``dict[str, Any]`` to avoid forcing users to
    # quote numeric values.
    store_config: dict[str, Any] | None = None
    # Optional physical-GPU-indexed HCA names. Each launched process receives
    # a JSON config restricted to the devices assigned to its physical GPUs.
    # This does not assign different configs to nested vLLM TP ranks.
    device_names_by_gpu: list[str] = field(default_factory=list)

    Schema: ClassVar[builtins.type[Schema]] = Schema

    def validate_device_mapping(self, gpus_per_node: int) -> None:
        """Validate an opt-in physical GPU map before allocation and at launch."""
        devices = self.device_names_by_gpu
        if not devices:
            return
        if len(devices) != gpus_per_node:
            raise ValueError("mooncake device_names_by_gpu must have one entry per physical GPU on each node")
        if any(not isinstance(d, str) or not d or any(c.isspace() for c in d) or "," in d for d in devices):
            raise ValueError(
                "mooncake device_names_by_gpu entries must be single nonempty device names without whitespace"
            )


# Name of the flock file the engines of one worker elect the serving engine with.
FAILOVER_LOCK_FILENAME = "failover.lock"


@dataclass(frozen=True)
class VLLMFailoverConfig:
    """Shadow engine recovery for vLLM workers (Dynamo GPU Memory Service).

    Every worker runs ``1 + shadow_engines`` ``dynamo.vllm`` engines on the same
    GPUs next to a GPU Memory Service (GMS) that owns the weights (the implied
    ``gms`` service, one instance per worker), so a shadow maps the one copy
    already in HBM instead of loading its own. The engines elect the serving one
    with a ``flock`` on a shared file: when the active engine dies the kernel
    drops the lock, a shadow acquires it, materializes its KV cache and registers
    with the frontend within seconds. ``roles.<role>.restart`` relaunches the dead
    engine in place as the new shadow. This is the Kubernetes intra-pod failover
    layout (``experimental.failover``) without DRA: on SLURM the steps of one
    worker share the node's GPUs natively. See ``docs/shadow-engine-recovery.md``.

    Attributes:
        shadow_engines: Standby engines per worker.
        shared_dir: Node-local host directory that every container on a node sees.
            The GMS sockets and the lock file of a worker live under
            ``<shared_dir>/srtctl-<job_id>/<role>_<index>``. enroot bind-mounts the
            host's ``/dev/shm`` into every container; ``/tmp`` is a fresh tmpfs per
            container and does not work.
    """

    shadow_engines: int = 1
    shared_dir: str = "/dev/shm"

    Schema: ClassVar[builtins.type[Schema]] = Schema

    def __post_init__(self) -> None:
        if self.shadow_engines < 1:
            raise ValidationError(f"engine.failover.shadow_engines must be at least 1, got {self.shadow_engines}")
        if not self.shared_dir.startswith("/") or self.shared_dir == "/":
            raise ValidationError(
                f"engine.failover.shared_dir must be an absolute directory below /, got {self.shared_dir!r}"
            )

    @property
    def engines_per_worker(self) -> int:
        return 1 + self.shadow_engines


def failover_root(shared_dir: str, job_id: str) -> str:
    """Per-job directory under ``shared_dir`` holding every worker's sockets and lock file."""
    return f"{shared_dir.rstrip('/')}/srtctl-{job_id}"


def failover_worker_dir(shared_dir: str, job_id: str, process: Process) -> str:
    """The directory one worker's engines and GMS sidecar share on a node (the same path on every node)."""
    return f"{failover_root(shared_dir, job_id)}/{process.endpoint_mode}_{process.endpoint_index}"


@dataclass(frozen=True)
class VLLMServerConfig:
    """vLLM server CLI configuration per mode (prefill/decode/aggregated).

    Each mode can have its own configuration dict that gets converted
    to CLI flags when starting the worker. These are passed directly to
    vLLM's AsyncEngineArgs.
    """

    prefill: dict[str, Any] | None = None
    decode: dict[str, Any] | None = None
    aggregated: dict[str, Any] | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class VLLMProtocol:
    """vLLM protocol - implements BackendProtocol.

    This frozen dataclass both holds configuration AND implements the
    BackendProtocol methods for process allocation and launching.

    dynamo 1.0.0+: ``--connector`` was removed; the ``connector`` field is now
    translated to ``--kv-transfer-config`` with the appropriate JSON payload.

    Example YAML:
        backend:
          type: vllm
          connector: nixl  # translated to --kv-transfer-config JSON
          allow_prefill_decode_colocation: true  # pack P/D on one node when all workers fit
          allow_prefill_decode_colocation_across_nodes: true  # continue packing on later nodes
          prefill_environment:
            PYTHONUNBUFFERED: "1"
          vllm_config:
            prefill:
              tensor-parallel-size: 2
              gpu-memory-utilization: 0.9
              connector: lmcache  # override connector for prefill
            decode:
              tensor-parallel-size: 2
              gpu-memory-utilization: 0.85
              # uses default connector (nixl)
    """

    type: Literal["vllm"] = "vllm"

    # Environment variables per mode
    prefill_environment: dict[str, str] = field(default_factory=dict)
    decode_environment: dict[str, str] = field(default_factory=dict)
    aggregated_environment: dict[str, str] = field(default_factory=dict)

    # vLLM server CLI config per mode
    vllm_config: VLLMServerConfig | None = None

    # Use an environment mask instead of the engine's --device-ids option.
    set_visible_devices: bool = False

    # Default KV connector: "nixl", "lmcache", "kvbm", or a raw JSON string for --kv-transfer-config.
    # Can be overridden per role by setting "connector" in roles.<role>.args; connector_for_mode resolves it.
    # dynamo 1.0.0+: translated to --kv-transfer-config (--connector was removed).
    connector: str | None = "nixl"

    # Mooncake KV store — when set, srtslurm launches mooncake_master on the
    # infra node and auto-injects MOONCAKE_MASTER / MOONCAKE_TE_META_DATA_SERVER
    # / MOONCAKE_LOCAL_HOSTNAME on every vLLM worker.
    mooncake_kv_store: VLLMMooncakeKVStoreConfig | None = None

    # Shadow engine recovery: when set, every worker runs shadow_engines standby engines
    # on its GPUs next to an implied `gms` service that owns the weights. Dynamo frontend only.
    failover: VLLMFailoverConfig | None = None

    # KV events config - enables --kv-events-config with auto-allocated ports.
    # Required for Dynamo's event-driven KV-aware routing.
    # Global true enables defaults for prefill and decode workers.
    # Per-mode: {"prefill": true, "decode": {"topic": "custom"}}
    kv_events_config: bool | dict[str, Any] | None = None

    # Allow prefill and decode workers to share one node when the combined GPU
    # request fits within gpus_per_node. Defaults off to preserve existing P/D
    # node separation.
    allow_prefill_decode_colocation: bool = False

    # Extend P/D colocation to multi-node topologies. When enabled together
    # with allow_prefill_decode_colocation, workers are packed contiguously
    # across the minimum number of nodes instead of reserving separate P/D
    # node pools. Defaults off to preserve the original one-node-only policy.
    allow_prefill_decode_colocation_across_nodes: bool = False

    # DP process layout. Per-node lets vLLM manage the node-local portion of a
    # DP x TP x PP topology in one CUDA namespace and derives cross-node TP/PP
    # rendezvous when a replica is larger than the node-local GPU allocation.
    # Per-GPU remains available as a deprecated compatibility layout.
    dp_launch_mode: DPLaunchMode = "per_node"

    # Executable used by direct aggregate frontend.type=vllm jobs. This can be
    # set to vllm-rs (or its absolute path) to use the Rust OpenAI frontend.
    vllm_serve_binary: str = "vllm"

    Schema: ClassVar[builtins.type[Schema]] = Schema

    def find_dp_modes(self) -> list[tuple[str, dict[str, Any]]]:
        """Return modes whose configured data-parallel size is greater than one."""
        if self.vllm_config is None:
            return []

        dp_mode_configs: list[tuple[str, dict[str, Any]]] = []
        for mode_name, mode_config in (
            ("prefill", self.vllm_config.prefill),
            ("decode", self.vllm_config.decode),
            ("aggregated", self.vllm_config.aggregated),
        ):
            configured_dp_size = next(
                (
                    value
                    for key, value in (mode_config or {}).items()
                    if str(key).replace("_", "-") == "data-parallel-size"
                ),
                None,
            )
            if configured_dp_size is None:
                continue
            try:
                normalized_dp_size = int(configured_dp_size)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"vllm_config.{mode_name}.data-parallel-size must be a positive integer; got {configured_dp_size!r}"
                ) from exc
            if normalized_dp_size < 1:
                raise ValidationError(
                    f"vllm_config.{mode_name}.data-parallel-size must be a positive integer; got {configured_dp_size!r}"
                )
            if normalized_dp_size > 1:
                assert mode_config is not None
                dp_mode_configs.append((mode_name, mode_config))
        return dp_mode_configs

    def __post_init__(self) -> None:
        """Validate flags whose behavior is fixed by the per-node launch topology."""
        dp_mode_configs = self.find_dp_modes()

        # The per_gpu advisory lives in SrtConfig, which can tell whether the
        # setting applies at all: a direct vLLM frontend ignores it.
        if not dp_mode_configs or self.dp_launch_mode == "per_gpu":
            return

        hybrid_lb_modes: list[str] = []
        headless_modes: list[str] = []
        for mode_name, mode_config in dp_mode_configs:
            normalized_keys = {str(key).replace("_", "-") for key in mode_config}
            if "headless" in normalized_keys:
                headless_modes.append(mode_name)
            if "data-parallel-hybrid-lb" in normalized_keys:
                hybrid_lb_modes.append(mode_name)

        if headless_modes:
            fields = ", ".join(f"vllm_config.{mode}.headless" for mode in headless_modes)
            raise ValidationError(
                f"{fields} cannot be set when vLLM uses per-node DP. "
                "srtslurm derives the head/headless layout; remove headless."
            )

        if hybrid_lb_modes:
            fields = ", ".join(f"vllm_config.{mode}.data-parallel-hybrid-lb" for mode in hybrid_lb_modes)
            logger.warning(
                "%s is unnecessary when dp_launch_mode=per_node; "
                "srtslurm derives --data-parallel-hybrid-lb from the topology and ignores the configured value",
                fields,
            )

    # =========================================================================
    # BackendProtocol Implementation
    # =========================================================================

    def get_srun_config(self) -> SrunConfig:
        """vLLM launches one srun step for each generated process."""
        from srtctl.backends.base import SrunConfig

        return SrunConfig(mpi=None, oversubscribe=False, launch_per_endpoint=False)

    def get_config_for_mode(self, mode: WorkerMode) -> dict[str, Any]:
        """Get merged config dict for a worker mode."""
        if not self.vllm_config:
            return {}

        if mode == "prefill":
            return dict(self.vllm_config.prefill or {})
        elif mode == "decode":
            return dict(self.vllm_config.decode or {})
        elif mode == "agg":
            return dict(self.vllm_config.aggregated or {})
        return {}

    def get_kv_events_config_for_mode(self, mode: WorkerMode) -> dict[str, Any] | None:
        """Get --kv-events-config payload for a worker mode."""
        if not self.kv_events_config:
            return None

        if self.kv_events_config is True:
            if mode in ("prefill", "decode"):
                return {
                    "publisher": "zmq",
                    "topic": "kv-events",
                    "enable_kv_cache_events": True,
                }
            return None

        if isinstance(self.kv_events_config, dict):
            mode_cfg = self.kv_events_config.get("aggregated") if mode == "agg" else self.kv_events_config.get(mode)
            if mode_cfg is None:
                return None
            if mode_cfg is True:
                return {
                    "publisher": "zmq",
                    "topic": "kv-events",
                    "enable_kv_cache_events": True,
                }
            if isinstance(mode_cfg, dict):
                result = {
                    "publisher": "zmq",
                    "topic": "kv-events",
                    "enable_kv_cache_events": True,
                }
                result.update(mode_cfg)
                return result

        return None

    def get_environment_for_mode(self, mode: WorkerMode) -> dict[str, str]:
        """Get environment variables for a worker mode."""
        if mode == "prefill":
            environment = dict(self.prefill_environment)
        elif mode == "decode":
            environment = dict(self.decode_environment)
        elif mode == "agg":
            environment = dict(self.aggregated_environment)
        else:
            environment = {}

        return environment

    def connector_for_mode(self, mode: WorkerMode) -> str | None:
        """The KV connector a worker mode runs: ``roles.<mode>.args.connector``, else ``engine.connector``."""
        mode_connector = self.get_config_for_mode(mode).get("connector")
        return mode_connector if mode_connector is not None else self.connector

    def kv_connector_for_mode(self, mode: WorkerMode) -> KVConnector | None:
        """The connector table row for a mode; None for no connector or a raw JSON ``--kv-transfer-config``."""
        return kv_connector_row(self.connector_for_mode(mode))

    def discovers_workers(self) -> bool:
        """Whether the prefill/decode workers register with the router over its discovery endpoint.

        True when either role runs a discovery connector; the router then learns
        its workers by registration instead of from URLs on its command line.
        """
        return any(row is not None and row.discovery for row in map(self.kv_connector_for_mode, ("prefill", "decode")))

    def kv_transfer_config(self, mode: WorkerMode) -> str | None:
        """``--kv-transfer-config`` JSON for a worker mode, or None when the mode has no connector.

        Table connectors expand to their preset for the mode; a raw JSON string
        from the recipe passes through unchanged.
        """
        connector = self.connector_for_mode(mode)
        if not connector or connector.lower() in ("null", "none"):
            return None
        return _connector_to_kv_transfer_config(connector, mode)

    def get_process_environment(self, process: Process) -> dict[str, str]:
        """Get process-specific environment variables for vLLM workers.

        vLLM with dynamo requires unique ports for each worker:
        - DYN_VLLM_KV_EVENT_PORT: ZMQ port for KV events publishing
        - VLLM_NIXL_SIDE_CHANNEL_PORT: Port for NIXL side channel transfers
        - VLLM_NIXL_SIDE_CHANNEL_HOST: Routable IP for NIXL side channel
          (vLLM defaults to ``0.0.0.0`` / ``localhost`` which breaks the
          multi-node NIXL handshake)
        - VLLM_PORT: private base for vLLM's get_open_port() scans, unique per
          process so co-located workers don't race for the same rendezvous port
          (see the notes on VLLM_PORT_BASE in srtctl.ports)
        """
        from srtctl.core.slurm import get_hostname_ip

        env: dict[str, str] = {}
        if process.kv_events_port is not None:
            env["DYN_VLLM_KV_EVENT_PORT"] = str(process.kv_events_port)
        if process.nixl_port is not None:
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(process.nixl_port)
            env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = get_hostname_ip(process.node)
        # Unique per-process VLLM_PORT base to avoid EADDRINUSE rendezvous races
        # when endpoints are co-located on a node. E.g. PD 4xDEP2+1xDEP8 on
        # 4xGB200 nodes: each prefill endpoint is DEP2 (uses 2 of the 4 GPUs), so
        # two endpoints share one physical node and would otherwise scan
        # overlapping get_open_port() ranges. The allocator spaces the ranges a
        # full stride apart.
        if process.vllm_scan_port is not None:
            env["VLLM_PORT"] = str(process.vllm_scan_port)
        return env

    def get_mooncake_worker_env(self, infra_node_ip: str, local_hostname: str) -> dict[str, str]:
        """Get mooncake env vars to inject on a specific vLLM worker.

        Returns ``{}`` when ``mooncake_kv_store`` is unset. Otherwise:

        - ``MOONCAKE_MASTER`` and ``MOONCAKE_TE_META_DATA_SERVER`` are always
          stamped by srtslurm (the user can't know the infra IP at config time).
        - ``MOONCAKE_LOCAL_HOSTNAME`` defaults to the worker's resolved IP for
          multi-node peer transfers, but a value in ``mooncake_kv_store.env``
          wins (use this to pin to a specific RDMA NIC IP).
        - ``MOONCAKE_CONFIG_PATH`` points to the JSON file srtslurm writes at
          job start (mounted into the container at ``/logs``). vLLM's
          ``MooncakeStoreConnector`` requires this — it does not read the
          ``MOONCAKE_*`` env vars directly.
        """
        if self.mooncake_kv_store is None:
            return {}
        return {
            "MOONCAKE_LOCAL_HOSTNAME": local_hostname,
            **self.mooncake_kv_store.env,
            "MOONCAKE_MASTER": f"{infra_node_ip}:{MOONCAKE_MASTER_PORT}",
            "MOONCAKE_TE_META_DATA_SERVER": (f"http://{infra_node_ip}:{MOONCAKE_HTTP_METADATA_PORT}/metadata"),
            "MOONCAKE_CONFIG_PATH": MOONCAKE_STORE_CONFIG_CONTAINER_PATH,
        }

    def build_mooncake_store_config(self, infra_node_ip: str) -> dict[str, Any]:
        """Build the JSON payload for vLLM's ``MooncakeStoreConfig.load_from_env()``.

        Pass-through of ``mooncake_kv_store.store_config`` with
        ``master_server_address`` force-overridden to the infra node IP
        (the user can't know that at config time). srtslurm intentionally
        does not provide defaults for the other fields — values like
        ``global_segment_size``, ``protocol``, and ``device_name`` are
        hardware-specific and silently using a default is worse than
        failing loudly. Set them explicitly in YAML; see
        ``docs/mooncake-kv-store.md``.
        """
        result: dict[str, Any] = {}
        if self.mooncake_kv_store is not None and self.mooncake_kv_store.store_config:
            result.update(self.mooncake_kv_store.store_config)
        result["master_server_address"] = f"{infra_node_ip}:{MOONCAKE_MASTER_PORT}"
        return result

    def build_mooncake_process_config(
        self, process: Process, infra_node_ip: str, gpus_per_node: int
    ) -> tuple[str, dict[str, Any]] | None:
        """Return an opt-in process-local filename/payload, using physical GPU IDs.

        GPU numbering restarts on each node; filenames are reusable across nodes
        with the same mapping. Multi-GPU processes receive the HCA subset, not a
        per-nested-rank binding. The shared-config default is unchanged.
        """
        if self.mooncake_kv_store is None or not self.mooncake_kv_store.device_names_by_gpu:
            return None
        self.mooncake_kv_store.validate_device_mapping(gpus_per_node)
        devices = self.mooncake_kv_store.device_names_by_gpu
        gpu_ids = sorted(process.gpu_indices)
        if not gpu_ids or any(gpu < 0 or gpu >= len(devices) for gpu in gpu_ids):
            raise ValueError(f"mooncake device_names_by_gpu does not cover physical GPUs {gpu_ids}")
        # Sharing an HCA between physical GPUs is legal. Avoid duplicating its
        # name when a process spans multiple GPUs mapped to the same HCA.
        process_devices = list(dict.fromkeys(devices[gpu] for gpu in gpu_ids))
        payload = self.build_mooncake_store_config(infra_node_ip)
        payload["device_name"] = ",".join(process_devices)
        filename = "mooncake_store_config_gpu" + "-".join(map(str, gpu_ids)) + ".json"
        return filename, payload

    def get_served_model_name(self, default: str) -> str:
        """Get served model name from vLLM config, or return default."""
        if self.vllm_config:
            for cfg in [self.vllm_config.prefill, self.vllm_config.aggregated, self.vllm_config.decode]:
                if cfg:
                    name = cfg.get("served-model-name") or cfg.get("served_model_name")
                    if name:
                        return name
        return default

    def should_colocate_prefill_decode(
        self,
        *,
        num_prefill: int,
        num_decode: int,
        num_agg: int,
        gpus_per_prefill: int,
        gpus_per_decode: int,
        gpus_per_agg: int,
        gpus_per_node: int,
    ) -> bool:
        """Whether prefill and decode workers should be packed contiguously."""
        if not self.allow_prefill_decode_colocation:
            return False
        if num_prefill <= 0 or num_decode <= 0 or gpus_per_node <= 0:
            return False

        total_worker_gpus = num_prefill * gpus_per_prefill + num_decode * gpus_per_decode + num_agg * gpus_per_agg
        return total_worker_gpus <= gpus_per_node or self.allow_prefill_decode_colocation_across_nodes

    def allocate_endpoints(
        self,
        num_prefill: int,
        num_decode: int,
        num_agg: int,
        gpus_per_prefill: int,
        gpus_per_decode: int,
        gpus_per_agg: int,
        gpus_per_node: int,
        available_nodes: Sequence[str],
        spread_workers: bool = False,
    ) -> list[Endpoint]:
        """Allocate endpoints to nodes."""
        from srtctl.core.topology import allocate_endpoints

        return allocate_endpoints(
            num_prefill=num_prefill,
            num_decode=num_decode,
            num_agg=num_agg,
            gpus_per_prefill=gpus_per_prefill,
            gpus_per_decode=gpus_per_decode,
            gpus_per_agg=gpus_per_agg,
            gpus_per_node=gpus_per_node,
            available_nodes=available_nodes,
            spread_workers=spread_workers,
            allow_prefill_decode_colocation=self.should_colocate_prefill_decode(
                num_prefill=num_prefill,
                num_decode=num_decode,
                num_agg=num_agg,
                gpus_per_prefill=gpus_per_prefill,
                gpus_per_decode=gpus_per_decode,
                gpus_per_agg=gpus_per_agg,
                gpus_per_node=gpus_per_node,
            ),
        )

    def _is_dp_mode(self, mode: WorkerMode) -> bool:
        """Check if this mode uses Data Parallel + Expert Parallel pattern.

        DP+EP mode is detected when data-parallel-size is set in the mode's config.
        ``dp_launch_mode`` controls whether a process owns one rank or all local ranks.
        """
        dp_size = self._get_dp_size(mode)
        return dp_size is not None and int(dp_size) > 1

    def _get_dp_size(self, mode: WorkerMode) -> int | None:
        """Get the data-parallel-size for a mode, or None if not in DP mode."""
        config = self.get_config_for_mode(mode)
        if "data-parallel-size" in config:
            return config["data-parallel-size"]
        return config.get("data_parallel_size")

    def _get_tp_size(self, mode: WorkerMode) -> int:
        """Get the tensor-parallel-size for a mode, defaulting to 1."""
        config = self.get_config_for_mode(mode)
        return config.get("tensor-parallel-size") or config.get("tensor_parallel_size") or 1

    def _get_pp_size(self, mode: WorkerMode) -> int:
        """Get the pipeline-parallel-size for a mode, defaulting to 1."""
        config = self.get_config_for_mode(mode)
        return config.get("pipeline-parallel-size") or config.get("pipeline_parallel_size") or 1

    def _get_parallel_size(self, mode: WorkerMode, name: str) -> int:
        """Return a normalized vLLM parallel dimension, defaulting to one."""
        config = self.get_config_for_mode(mode)
        value = config.get(name, config.get(name.replace("-", "_"), 1))
        try:
            size = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"vLLM {mode} {name} must be a positive integer; got {value!r}") from exc
        if size < 1:
            raise ValueError(f"vLLM {mode} {name} must be a positive integer; got {value!r}")
        return size

    def _get_model_parallel_size(self, mode: WorkerMode) -> int:
        """Return GPUs consumed by one DP replica across all vLLM dimensions."""
        return (
            self._get_parallel_size(mode, "tensor-parallel-size")
            * self._get_parallel_size(mode, "pipeline-parallel-size")
            * self._get_parallel_size(mode, "prefill-context-parallel-size")
        )

    def _get_local_dp_size(self, mode: WorkerMode, local_gpu_count: int) -> int:
        """Derive the complete DP replicas owned by one node-local process."""
        model_parallel_size = self._get_model_parallel_size(mode)
        if local_gpu_count % model_parallel_size != 0:
            raise ValueError(
                f"vLLM {mode} node-local allocation has {local_gpu_count} GPUs, which is not divisible by "
                f"TP*PP*PCP={model_parallel_size}"
            )
        return local_gpu_count // model_parallel_size

    def _validate_endpoint_parallelism(self, endpoint: Endpoint) -> tuple[int, int]:
        """Validate vLLM's configured world size against an endpoint allocation."""
        configured_dp_size = self._get_dp_size(endpoint.mode)
        dp_size = int(configured_dp_size) if configured_dp_size is not None else 1
        if dp_size < 1:
            raise ValueError(
                f"vLLM {endpoint.mode} data-parallel-size must be a positive integer; got {configured_dp_size!r}"
            )
        model_parallel_size = self._get_model_parallel_size(endpoint.mode)
        required_gpus = dp_size * model_parallel_size
        if required_gpus != endpoint.total_gpus:
            raise ValueError(
                f"{endpoint.mode} data-parallel-size={dp_size}, "
                f"tensor-parallel-size={self._get_parallel_size(endpoint.mode, 'tensor-parallel-size')}, "
                f"pipeline-parallel-size={self._get_parallel_size(endpoint.mode, 'pipeline-parallel-size')}, "
                "prefill-context-parallel-size="
                f"{self._get_parallel_size(endpoint.mode, 'prefill-context-parallel-size')} require "
                f"{required_gpus} GPUs, but the endpoint has {endpoint.total_gpus} allocated GPUs"
            )
        return dp_size, model_parallel_size

    def _get_int_flag(self, mode: WorkerMode, name: str, default: int = 1) -> int:
        """Read a positive integer CLI flag from the mode config."""
        config = self.get_config_for_mode(mode)
        raw = config.get(name)
        if raw is None:
            raw = config.get(name.replace("-", "_"))
        if raw is None:
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{mode} {name}={raw!r} is not an integer") from exc
        if value < 1:
            raise ValueError(f"{mode} {name}={value} must be >= 1")
        return value

    def _tp_size(self, mode: WorkerMode) -> int:
        return self._get_int_flag(mode, "tensor-parallel-size", 1)

    def _pp_size(self, mode: WorkerMode) -> int:
        return self._get_int_flag(mode, "pipeline-parallel-size", 1)

    def _gpus_per_dp_rank(self, mode: WorkerMode) -> int:
        """GPUs consumed by one DP rank: TP × PP."""
        return self._tp_size(mode) * self._pp_size(mode)

    def _local_dp_size(self, mode: WorkerMode, gpu_count: int) -> int:
        """DP ranks owned by one process that sees ``gpu_count`` GPUs.

        vLLM world size is TP × DP × PP, so a DP rank occupies TP × PP GPUs.
        Those GPUs must fit on the node: a DP rank that spans nodes needs the
        non-DP TP launch path (``--nnodes`` / ``--node-rank``), not DP local ranks.
        """
        tp = self._tp_size(mode)
        pp = self._pp_size(mode)
        gpus_per_rank = tp * pp
        if gpu_count % gpus_per_rank != 0:
            raise ValueError(
                f"{mode} tensor-parallel-size={tp} * pipeline-parallel-size={pp} "
                f"= {gpus_per_rank} GPUs per DP rank does not divide the "
                f"{gpu_count} GPUs allocated on each node"
            )
        return gpu_count // gpus_per_rank

    def _validate_dp_world_size(self, mode: WorkerMode, dp_size: int, total_gpus: int) -> None:
        """Require TP × DP × PP to match the GPUs allocated to the endpoint."""
        tp = self._tp_size(mode)
        pp = self._pp_size(mode)
        world = dp_size * tp * pp
        if world != total_gpus:
            raise ValueError(
                f"{mode} data-parallel-size={dp_size} * tensor-parallel-size={tp} "
                f"* pipeline-parallel-size={pp} = {world} does not match "
                f"the endpoint's {total_gpus} allocated GPUs"
            )

    def _dp_rank_gpu_groups(self, mode: WorkerMode, gpu_indices: frozenset[int]) -> list[frozenset[int]]:
        """Split a node's GPUs into one group per local DP rank."""
        gpus_per_rank = self._gpus_per_dp_rank(mode)
        self._local_dp_size(mode, len(gpu_indices))
        sorted_gpus = sorted(gpu_indices)
        return [
            frozenset(sorted_gpus[offset : offset + gpus_per_rank])
            for offset in range(0, len(sorted_gpus), gpus_per_rank)
        ]

    # =========================================================================
    # Shadow engine recovery (backend.failover)
    # =========================================================================

    @property
    def engines_per_process(self) -> int:
        """Engines launched per (worker, node): 1, or ``1 + shadow_engines`` under ``failover``."""
        return self.failover.engines_per_worker if self.failover is not None else 1

    def failover_worker_dir(self, job_id: str, process: Process) -> str:
        """Node-local directory this worker's engines and GMS sidecar share; see ``VLLMFailoverConfig.shared_dir``."""
        assert self.failover is not None
        return failover_worker_dir(self.failover.shared_dir, job_id, process)

    def get_failover_environment(self, process: Process, job_id: str) -> dict[str, str]:
        """Environment one engine of a failover worker needs (the same names the Dynamo operator injects).

        ``ENGINE_ID`` 0 is the engine that loads the weights into GMS (it takes a
        read-write GMS session, or read-only when the weights are already there
        after a relaunch); every other engine imports them read-only. The lock file
        is per worker, so only that worker's engines contend for it.
        """
        if self.failover is None:
            return {}
        worker_dir = self.failover_worker_dir(job_id, process)
        return {
            "ENGINE_ID": str(process.engine_id),
            "GMS_SOCKET_DIR": worker_dir,
            "FAILOVER_LOCK_PATH": f"{worker_dir}/{FAILOVER_LOCK_FILENAME}",
            "DYN_VLLM_GMS_SHADOW_MODE": "true",
            # /health answers "notready" until the engine holds the lock and serves; a
            # parked shadow then reports healthy so nothing restarts it while it waits.
            "DYN_SYSTEM_STARTING_HEALTH_STATUS": "notready",
        }

    def _failover_flags(self, config: dict[str, Any], process: Process, is_multi_node: bool) -> list[str]:
        """``dynamo.vllm`` flags every engine of a failover worker gets; pops the keys it owns from ``config``."""
        owned = {"load-format", "gms-shadow-mode"}
        master_port: int | None = None
        for key in list(config):
            normalized = normalize_vllm_config_key(key)
            if normalized in owned:
                config.pop(key)
            elif normalized == "master-port":
                master_port = int(config.pop(key))
        # Weights come from the worker's GMS sidecar; the engine parks after
        # initialization and waits for the lock before it registers.
        flags = ["--load-format", "gms", "--gms-shadow-mode"]
        if is_multi_node:
            # Every engine of a multi-node worker is its own torch.distributed job and
            # needs its own TCPStore port on the leader node.
            base = master_port if master_port is not None else VLLM_MASTER_PORT_BASE
            flags.extend(["--master-port", str(base + process.engine_id * VLLM_MASTER_PORT_STRIDE)])
        return flags

    def should_set_visible_devices(self) -> bool:
        """Whether worker launch should set the cluster-configured GPU mask."""
        return self.set_visible_devices

    def endpoints_to_processes(
        self,
        endpoints: list[Endpoint],
        base_sys_port: int = DYN_SYSTEM_PORT_BASE,
        port_allocator: NodePortAllocator | None = None,
        frontend_type: str = "dynamo",
        dynamo_sidecar: bool = False,
    ) -> list[Process]:
        """Convert endpoints to processes.

        Dynamo DP+EP mode uses the configured per-GPU or per-node process layout.
        For direct vLLM aggregate jobs, `vllm serve` manages local DP ranks from
        one process, so keep the standard one-process-per-node topology.
        For standard TP mode, creates one process per node. Every process then
        gets its private ``VLLM_PORT`` scan range from the allocator.
        """
        from srtctl.core.topology import endpoints_to_processes, port_allocator_for
        from srtctl.frontends import get_frontend

        allocator = port_allocator_for(port_allocator, base_sys_port)
        if get_frontend(frontend_type).worker_api_port("agg") == "public":
            # The worker is the public endpoint: one `vllm serve` per node owns
            # its local DP ranks, so the standard topology applies.
            processes = endpoints_to_processes(endpoints, port_allocator=allocator, sidecar_grpc=dynamo_sidecar)
        elif not any(self._is_dp_mode(ep.mode) for ep in endpoints):
            # Standard TP mode: one process per node, or one per engine of the
            # worker under backend.failover (engine 0 plus its shadows).
            processes = endpoints_to_processes(
                endpoints,
                port_allocator=allocator,
                engines_per_process=self.engines_per_process,
                sidecar_grpc=dynamo_sidecar,
            )
        elif self.dp_launch_mode == "per_node":
            processes = self._dp_per_node_endpoints_to_processes(endpoints, allocator, sidecar_grpc=dynamo_sidecar)
        else:
            processes = self._dp_per_gpu_endpoints_to_processes(endpoints, allocator, sidecar_grpc=dynamo_sidecar)
        return [replace(process, vllm_scan_port=allocator.next(VLLM_SCAN_PORTS)) for process in processes]

    def _dp_per_gpu_endpoints_to_processes(
        self,
        endpoints: list[Endpoint],
        allocator: NodePortAllocator,
        *,
        sidecar_grpc: bool,
    ) -> list[Process]:
        """DP+EP mode with one process per DP rank (TP x PP GPUs each)."""
        from srtctl.core.topology import Process

        processes: list[Process] = []
        for endpoint in endpoints:
            if not self._is_dp_mode(endpoint.mode):
                # Non-DP endpoints get standard processing (all modes are normally consistent).
                for node_rank, node in enumerate(endpoint.nodes):
                    is_leader = node_rank == 0
                    processes.append(
                        Process(
                            node=node,
                            gpu_indices=endpoint.gpu_indices,
                            sys_port=allocator.next(SYS_PORTS),
                            http_port=allocator.next(HTTP_PORTS, node) if is_leader else 0,
                            endpoint_mode=endpoint.mode,
                            endpoint_index=endpoint.index,
                            node_rank=node_rank,
                            bootstrap_port=(
                                allocator.next(BOOTSTRAP_PORTS, node)
                                if endpoint.mode == "prefill" and is_leader
                                else None
                            ),
                            kv_events_port=allocator.next(KV_EVENTS_PORTS),
                            nixl_port=allocator.next(NIXL_PORTS),
                            kvbm_zmq_port=allocator.next(KVBM_ZMQ_PORTS),
                            sidecar_grpc_port=allocator.next(SIDECAR_GRPC_PORTS) if sidecar_grpc else None,
                        )
                    )
                continue

            # One process per DP rank. With TP=1 this is one GPU; with TP>1 the
            # process owns the TP x PP GPUs for that rank.
            dp_rank = 0
            dp_rpc_port = allocator.next(DP_RPC_PORTS, endpoint.leader_node)
            dp_size = self._get_dp_size(endpoint.mode) or (endpoint.total_gpus // self._gpus_per_dp_rank(endpoint.mode))
            self._validate_dp_world_size(endpoint.mode, dp_size, endpoint.total_gpus)
            rank_gpu_groups = self._dp_rank_gpu_groups(endpoint.mode, endpoint.gpu_indices)
            # vLLM computes actual_port = base + data_parallel_rank, so every DP
            # rank of the endpoint shares one reserved block.
            nixl_base_port = allocator.next(NIXL_PORTS, size=dp_size)
            for node in endpoint.nodes:
                for rank_gpus in rank_gpu_groups:
                    is_leader = dp_rank == 0
                    processes.append(
                        Process(
                            node=node,
                            gpu_indices=rank_gpus,
                            sys_port=allocator.next(SYS_PORTS),
                            http_port=allocator.next(HTTP_PORTS, node) if is_leader else 0,
                            endpoint_mode=endpoint.mode,
                            endpoint_index=endpoint.index,
                            node_rank=dp_rank,  # dp_rank stored in node_rank for now
                            bootstrap_port=(
                                allocator.next(BOOTSTRAP_PORTS, node)
                                if endpoint.mode == "prefill" and is_leader
                                else None
                            ),
                            kv_events_port=allocator.next(KV_EVENTS_PORTS),
                            nixl_port=nixl_base_port,
                            dp_rpc_port=dp_rpc_port,
                            kvbm_zmq_port=allocator.next(KVBM_ZMQ_PORTS),
                            sidecar_grpc_port=allocator.next(SIDECAR_GRPC_PORTS) if sidecar_grpc else None,
                        )
                    )
                    dp_rank += 1

        return processes

    def _dp_per_node_endpoints_to_processes(
        self,
        endpoints: list[Endpoint],
        allocator: NodePortAllocator,
        *,
        sidecar_grpc: bool,
    ) -> list[Process]:
        """Convert DP endpoints to one process per node.

        ``--data-parallel-size-local`` is GPUs-on-node / (TP x PP x PCP), not the GPU
        count. Start ranks advance by that local DP size.
        """
        from srtctl.core.topology import Process, endpoints_to_processes

        processes: list[Process] = []
        for endpoint in endpoints:
            if not self._is_dp_mode(endpoint.mode):
                processes.extend(
                    endpoints_to_processes([endpoint], port_allocator=allocator, sidecar_grpc=sidecar_grpc)
                )
                continue

            dp_size, replica_size = self._validate_endpoint_parallelism(endpoint)

            local_gpu_count = len(endpoint.gpu_indices)
            spans_nodes = replica_size > local_gpu_count
            if spans_nodes:
                if len(endpoint.nodes) < 2:
                    raise ValueError("cross-node per_node DP requires a multi-node endpoint")
                if replica_size % local_gpu_count != 0:
                    raise ValueError(
                        f"{endpoint.mode} TP x PP replica size {replica_size} does not divide evenly across "
                        f"nodes with {local_gpu_count} allocated GPUs each"
                    )
                nodes_per_dp_rank = replica_size // local_gpu_count
                if len(endpoint.nodes) != dp_size * nodes_per_dp_rank:
                    raise ValueError(
                        f"{endpoint.mode} requires {nodes_per_dp_rank} nodes per DP rank and "
                        f"{dp_size * nodes_per_dp_rank} nodes total, but the endpoint has {len(endpoint.nodes)}"
                    )
                processes.extend(
                    endpoints_to_processes([endpoint], port_allocator=allocator, sidecar_grpc=sidecar_grpc)
                )
                continue

            if local_gpu_count % replica_size != 0:
                raise ValueError(
                    f"{endpoint.mode} TP x PP replica size {replica_size} does not divide the node's "
                    f"{local_gpu_count} allocated GPUs"
                )

            local_dp_size = self._get_local_dp_size(endpoint.mode, local_gpu_count)
            dp_rpc_port = allocator.next(DP_RPC_PORTS, endpoint.leader_node)
            nixl_base_port = allocator.next(NIXL_PORTS, size=dp_size)
            dp_start_rank = 0

            for node in endpoint.nodes:
                processes.append(
                    Process(
                        node=node,
                        gpu_indices=endpoint.gpu_indices,
                        sys_port=allocator.next(SYS_PORTS),
                        http_port=allocator.next(HTTP_PORTS, node),
                        endpoint_mode=endpoint.mode,
                        endpoint_index=endpoint.index,
                        node_rank=dp_start_rank,
                        bootstrap_port=(allocator.next(BOOTSTRAP_PORTS, node) if endpoint.mode == "prefill" else None),
                        # One KV-event publisher per local DP rank: reserve the block.
                        kv_events_port=allocator.next(KV_EVENTS_PORTS, size=local_dp_size),
                        nixl_port=nixl_base_port,
                        dp_rpc_port=dp_rpc_port,
                        het_group=endpoint.het_group,
                        kvbm_zmq_port=allocator.next(KVBM_ZMQ_PORTS),
                        sidecar_grpc_port=allocator.next(SIDECAR_GRPC_PORTS) if sidecar_grpc else None,
                    )
                )
                dp_start_rank += local_dp_size

        return processes

    def build_worker_command(
        self,
        process: Process,
        endpoint_processes: list[Process],
        runtime: RuntimeContext,
        frontend_type: str = "dynamo",
        nsys_prefix: list[str] | None = None,
        dump_config_path: Path | None = None,
        profiling: ProfilingConfig | None = None,
    ) -> list[str]:
        """Build the command to start a vLLM worker process.

        Args:
            process: The process to start
            endpoint_processes: All processes for this endpoint (for multi-node)
            runtime: Runtime context with paths and settings
            frontend_type: Frontend type ("dynamo" or direct "vllm")
            nsys_prefix: Optional nsys profiling command prefix
            dump_config_path: Path to dump config JSON
            profiling: Profiling config; drives --profiler-config for iteration-based nsys
        """
        from srtctl.core.slurm import get_hostname_ip
        from srtctl.frontends import get_frontend

        mode = process.endpoint_mode
        config = self.get_config_for_mode(mode)
        # The frontend owns the worker shape: Dynamo registration versus a direct
        # server, which port that server binds, and whether a router expands
        # node-local DP pools. Nothing below compares frontend names.
        frontend = get_frontend(frontend_type)
        direct_workers = frontend.worker_launch == "direct"
        binds_public_port = frontend.worker_api_port(mode) == "public"

        # Determine if multi-node
        endpoint_nodes = list(dict.fromkeys(p.node for p in endpoint_processes))
        is_multi_node = len(endpoint_nodes) > 1

        # Native vLLM rendezvous must use the configured interface, including
        # native engines behind a Dynamo sidecar.
        if direct_workers or get_dynamo_sidecar_config(runtime) is not None:
            leader_ip = get_hostname_ip(endpoint_nodes[0], runtime.network_interface)
        else:
            leader_ip = get_hostname_ip(endpoint_nodes[0])

        # Determine model path: HF model ID or container mount path
        # For HF models (hf:prefix), model_path contains the HF model ID (e.g., "facebook/opt-125m")
        # For local models, model is mounted to /model in the container
        model_arg = str(runtime.model_path) if runtime.is_hf_model else "/model"

        # Get served model name from config or use model path name
        served_model_name = self.get_served_model_name(runtime.model_path.name)

        # Start with nsys prefix if provided
        cmd: list[str] = list(nsys_prefix) if nsys_prefix else []

        if profiling is not None and profiling.is_nsys and not profiling.is_nsys_time:
            phase = profiling._get_phase_config(mode)
            if phase is not None and phase.start_step is not None and phase.stop_step is not None:
                config["profiler-config"] = json.dumps(
                    {
                        "profiler": "cuda",
                        "delay_iterations": phase.vllm_nsys_delay_iterations,
                        "max_iterations": phase.vllm_nsys_max_iterations,
                    }
                )

        sidecar_config = get_dynamo_sidecar_config(runtime)
        if sidecar_config is not None:
            if frontend.worker_launch != "dynamo":
                raise ValueError("vLLM sidecar mode requires frontend.type: dynamo")
            process_ip = get_hostname_ip(process.node, getattr(runtime, "network_interface", None))
            return self._build_sidecar_command(
                process=process,
                endpoint_processes=endpoint_processes,
                config=config,
                model_arg=model_arg,
                served_model_name=served_model_name,
                leader_ip=leader_ip,
                process_ip=process_ip,
                nsys_prefix=nsys_prefix,
                sidecar_config=sidecar_config,
            )

        if direct_workers:
            if binds_public_port and mode != "agg":
                raise ValueError(f"frontend.type: {frontend.type} supports aggregate vLLM jobs only")

            overridden = pop_vllm_orchestration_flags(config)
            config.setdefault("served-model-name", served_model_name)

            # A prefill/decode worker gets its KV connector; an aggregate worker has none.
            config.pop("connector", None)
            if mode in {"prefill", "decode"}:
                kv_transfer_config = self.kv_transfer_config(mode)
                if kv_transfer_config is not None:
                    config.setdefault("kv-transfer-config", kv_transfer_config)

            node_rank = endpoint_nodes.index(process.node)
            # The worker that is itself the public endpoint may run the alternate
            # OpenAI frontend binary (vllm-rs); routed workers run vllm.
            serve_binary = self.vllm_serve_binary if binds_public_port else "vllm"
            cmd.extend([serve_binary, "serve", model_arg])
            # Collected as the command is built so the override report below can
            # name the value srtslurm actually passed for each flag it took over.
            srtslurm_owned: dict[str, str] = {}
            is_dp_mode = self._is_dp_mode(mode)
            replica_size = self._get_model_parallel_size(mode)
            local_gpu_count = len(process.gpu_indices)
            spans_nodes = replica_size > local_gpu_count
            is_router_local_dp = (
                frontend.expands_node_local_dp
                and is_multi_node
                and is_dp_mode
                and self.dp_launch_mode == "per_node"
                and not spans_nodes
            )

            if frontend.expands_node_local_dp and is_dp_mode and self.dp_launch_mode != "per_node":
                raise ValueError(
                    f"frontend.type: {frontend.type} with data-parallel-size requires backend.dp_launch_mode: per_node"
                )

            if node_rank == 0 or is_router_local_dp:
                api_port = runtime.frontend_port if binds_public_port else process.http_port
                cmd.extend(["--host", "0.0.0.0", "--port", str(api_port)])
                srtslurm_owned["host"] = "0.0.0.0"
                srtslurm_owned["port"] = str(api_port)

            if is_router_local_dp:
                for key in list(config):
                    if normalize_vllm_config_key(key) in {
                        "data-parallel-size-local",
                        "data-parallel-start-rank",
                        "data-parallel-address",
                        "data-parallel-rpc-port",
                        "data-parallel-hybrid-lb",
                    }:
                        config.pop(key)
                local_dp_size = self._get_local_dp_size(mode, local_gpu_count)
                cmd.extend(
                    [
                        "--data-parallel-size-local",
                        str(local_dp_size),
                        "--data-parallel-start-rank",
                        str(process.node_rank),
                        "--data-parallel-address",
                        leader_ip,
                        "--data-parallel-rpc-port",
                        str(process.dp_rpc_port or VLLM_DATA_PARALLEL_RPC_PORT),
                        "--data-parallel-hybrid-lb",
                    ]
                )
            elif is_multi_node:
                # vLLM-native multi-node serve (torchrun-style): the leader owns
                # the OpenAI server; other node ranks run headless engine workers.
                cmd.extend(
                    [
                        "--master-addr",
                        leader_ip,
                        "--nnodes",
                        str(len(endpoint_nodes)),
                        "--node-rank",
                        str(node_rank),
                    ]
                )
                srtslurm_owned["master-addr"] = leader_ip
                srtslurm_owned["nnodes"] = str(len(endpoint_nodes))
                srtslurm_owned["node-rank"] = str(node_rank)
                if frontend.expands_node_local_dp and is_dp_mode:
                    for key in list(config):
                        if normalize_vllm_config_key(key) in {
                            "data-parallel-address",
                            "data-parallel-rpc-port",
                        }:
                            config.pop(key)
                    cmd.extend(
                        [
                            "--data-parallel-address",
                            leader_ip,
                            "--data-parallel-rpc-port",
                            str(process.dp_rpc_port or VLLM_DATA_PARALLEL_RPC_PORT),
                        ]
                    )
                if node_rank > 0:
                    cmd.append("--headless")
                    srtslurm_owned["headless"] = "true"
                    dropped = pop_vllm_api_server_flags(config)
                    if dropped:
                        logger.info(
                            "Dropping %s on headless node rank %d (%s); headless ranks run no API server",
                            ", ".join(f"--{flag}={value!s}" for flag, value in dropped.items()),
                            node_rank,
                            process.node,
                        )
            _log_overridden_recipe_flags(overridden, srtslurm_owned, process.node)
            if not self.should_set_visible_devices():
                device_ids = ",".join(str(i) for i in sorted(process.gpu_indices))
                if device_ids:
                    cmd.extend(["--device-ids", device_ids])
            cmd.extend(_config_to_cli_args(config))
            return cmd

        # Base command - use dynamo.vllm module
        cmd.extend(
            [
                "python3",
                "-m",
                "dynamo.vllm",
                "--model",
                model_arg,
                "--served-model-name",
                served_model_name,
            ]
        )

        # Disaggregation mode (dynamo 1.0.0+: --is-prefill-worker/--is-decode-worker are deprecated)
        if mode in ("prefill", "decode"):
            cmd.extend(["--disaggregation-mode", mode])

        # KV connector → --kv-transfer-config (dynamo 1.0.0+: --connector was removed).
        # Pop from config so it doesn't get added again by _config_to_cli_args.
        config.pop("connector", None)
        kv_transfer_cfg = self.kv_transfer_config(mode)
        if kv_transfer_cfg is not None:
            cmd.extend(["--kv-transfer-config", kv_transfer_cfg])

        # Under failover the worker stage pins CUDA_VISIBLE_DEVICES instead: the
        # engines and their GMS sidecar must see the same device list so that
        # "device k" means the same GPU (and the same socket) in all of them.
        if not self.should_set_visible_devices() and self.failover is None:
            device_ids = ",".join(str(i) for i in sorted(process.gpu_indices))
            if device_ids:
                cmd.extend(["--device-ids", device_ids])

        # Check if this is DP+EP mode (data-parallel-size set)
        is_dp_mode = self._is_dp_mode(mode)
        if is_dp_mode and self.dp_launch_mode == "per_node":
            rpc_port_kebab = config.pop("data-parallel-rpc-port", None)
            rpc_port_snake = config.pop("data_parallel_rpc_port", None)
            config_dp_rpc_port = rpc_port_kebab or rpc_port_snake
            dp_rpc_port = process.dp_rpc_port or config_dp_rpc_port or VLLM_DATA_PARALLEL_RPC_PORT

            # These values are derived from the allocated process topology. Hybrid LB
            # is required so every node-local Dynamo runtime registers with the frontend.
            config.pop("data-parallel-size-local", None)
            config.pop("data_parallel_size_local", None)
            config.pop("data-parallel-start-rank", None)
            config.pop("data_parallel_start_rank", None)
            config.pop("data-parallel-hybrid-lb", None)
            config.pop("data_parallel_hybrid_lb", None)
            config.pop("headless", None)

            replica_size = self._get_tp_size(mode) * self._get_pp_size(mode)
            local_gpu_count = len(process.gpu_indices)
            spans_nodes = replica_size > local_gpu_count

            if spans_nodes:
                if not is_multi_node:
                    raise ValueError("cross-node per_node DP requires a multi-node endpoint")
                node_rank = endpoint_nodes.index(process.node)
                cmd.extend(
                    [
                        "--master-addr",
                        leader_ip,
                        "--nnodes",
                        str(len(endpoint_nodes)),
                        "--node-rank",
                        str(node_rank),
                        "--data-parallel-address",
                        leader_ip,
                        "--data-parallel-rpc-port",
                        str(dp_rpc_port),
                    ]
                )
                # A single global API/DPLB leader owns the shared DP address.
                if node_rank > 0:
                    cmd.append("--headless")
            else:
                local_dp_size = local_gpu_count // replica_size
                cmd.extend(
                    [
                        "--data-parallel-size-local",
                        str(local_dp_size),
                        "--data-parallel-start-rank",
                        str(process.node_rank),
                        "--data-parallel-address",
                        leader_ip,
                        "--data-parallel-rpc-port",
                        str(dp_rpc_port),
                        "--data-parallel-hybrid-lb",
                    ]
                )
        elif is_dp_mode:
            # DP+EP per_gpu: each process is one DP rank (TP×PP GPUs).
            # process.node_rank is the dp_rank (set in endpoints_to_processes)
            dp_rank = process.node_rank
            # Use the per-endpoint dp_rpc_port allocated by NodePortAllocator
            # (avoids port collisions when multiple endpoints share a node)
            dp_rpc_port = (
                process.dp_rpc_port
                or config.pop("data-parallel-rpc-port", None)
                or config.pop("data_parallel_rpc_port", VLLM_DATA_PARALLEL_RPC_PORT)
            )
            # Pop from config so it doesn't get added again by _config_to_cli_args
            config.pop("data-parallel-rpc-port", None)
            config.pop("data_parallel_rpc_port", None)

            cmd.extend(
                [
                    "--data-parallel-rank",
                    str(dp_rank),
                    "--data-parallel-address",
                    leader_ip,
                    "--data-parallel-rpc-port",
                    str(dp_rpc_port),
                ]
            )
            # Note: --data-parallel-size is added via _config_to_cli_args from vllm_config
        elif is_multi_node:
            # Standard TP+PP multi-node coordination flags
            node_rank = endpoint_nodes.index(process.node)
            cmd.extend(
                [
                    "--master-addr",
                    leader_ip,
                    "--nnodes",
                    str(len(endpoint_nodes)),
                    "--node-rank",
                    str(node_rank),
                ]
            )

            # Non-leader nodes run headless
            if node_rank > 0:
                cmd.append("--headless")

        if self.failover is not None:
            cmd.extend(self._failover_flags(config, process, is_multi_node))

        # Add request plane
        cmd.extend(["--request-plane", runtime.request_plane])

        # Add config dump path
        if dump_config_path:
            cmd.extend(["--dump-config-to", str(dump_config_path)])

        kv_cfg = self.get_kv_events_config_for_mode(mode)
        if kv_cfg and process.kv_events_port is not None:
            kv_cfg["endpoint"] = f"tcp://*:{process.kv_events_port}"
            cmd.extend(["--kv-events-config", json.dumps(kv_cfg)])

        # Add all config flags from vllm_config
        cmd.extend(_config_to_cli_args(config))

        return cmd

    def _build_sidecar_command(
        self,
        *,
        process: Process,
        endpoint_processes: list[Process],
        config: dict[str, Any],
        model_arg: str,
        served_model_name: str,
        leader_ip: str,
        process_ip: str,
        nsys_prefix: list[str] | None,
        sidecar_config: DynamoConfig,
    ) -> list[str]:
        """Expose local DP frontends or one frontend for a cross-node replica."""
        mode = process.endpoint_mode
        is_dp_mode = self._is_dp_mode(mode)
        endpoint_nodes = list(dict.fromkeys(candidate.node for candidate in endpoint_processes))
        is_multi_node = len(endpoint_nodes) > 1
        multi_node_replica = is_multi_node and (
            not is_dp_mode or self._get_model_parallel_size(mode) > len(process.gpu_indices)
        )
        if multi_node_replica and is_dp_mode:
            raise ValueError(
                "vLLM sidecar mode supports one cross-node TP/PP replica per endpoint; "
                "use data-parallel-size: 1 and separate endpoints for additional replicas"
            )
        if multi_node_replica:
            gpu_counts = {len(candidate.gpu_indices) for candidate in endpoint_processes}
            total_gpus = sum(len(candidate.gpu_indices) for candidate in endpoint_processes)
            if len(gpu_counts) != 1 or total_gpus != self._get_model_parallel_size(mode):
                raise ValueError("vLLM sidecar TP*PP*PCP must match the evenly distributed endpoint GPU allocation")
        node_rank = endpoint_nodes.index(process.node)
        headless = multi_node_replica and node_rank > 0
        hybrid_lb = is_dp_mode and is_multi_node and not multi_node_replica
        if hybrid_lb or multi_node_replica:
            normalized = {key.replace("_", "-"): value for key, value in config.items()}
            layout = "hybrid" if hybrid_lb else "multi-node"
            if any(
                normalized.get(flag)
                for flag in ("grpc", "data-parallel-external-lb", "data-parallel-multi-port-external-lb")
            ):
                raise ValueError(
                    f"vLLM sidecar {layout} mode requires the Rust frontend; "
                    "remove grpc and external load-balancing flags"
                )
            if not headless and normalized.get("api-server-count") not in (None, 1, "1"):
                raise ValueError(f"vLLM sidecar {layout} mode requires api-server-count: 1 on frontend nodes")
        if multi_node_replica:
            executor = _pop_flags(config, frozenset({"distributed-executor-backend"}))
            if executor.get("distributed-executor-backend", "mp") != "mp":
                raise ValueError("vLLM multi-node sidecar requires distributed-executor-backend: mp")
            pop_vllm_orchestration_flags(config)
            _pop_flags(config, frozenset({"master-port", "grpc"}))
            if headless:
                pop_vllm_api_server_flags(config)
        grpc_port = sidecar_grpc_port(process)

        for key in (
            "model",
            "served-model-name",
            "served_model_name",
            "grpc-port",
            "grpc_port",
            "headless",
            "data-parallel-size-local",
            "data_parallel_size_local",
            "data-parallel-start-rank",
            "data_parallel_start_rank",
            "data-parallel-address",
            "data_parallel_address",
            "data-parallel-hybrid-lb",
            "data_parallel_hybrid_lb",
            "data-parallel-rank",
            "data_parallel_rank",
        ):
            config.pop(key, None)

        command: list[str] = list(nsys_prefix or [])
        if hybrid_lb or multi_node_replica:
            # Python coordinates distributed workers and starts Rust only on
            # frontend nodes. Headless followers take its native executor path.
            # The current `vllm-rs serve` launcher does not implement hybrid
            # startup; requests still use the Rust frontend in this path.
            # VLLM_RUST_FRONTEND_PATH, when configured, is inherited unchanged.
            command.extend(["env", "VLLM_USE_RUST_FRONTEND=1", "python3", "-m", "vllm.entrypoints.cli.main"])
        else:
            command.append("vllm-rs")
        command.extend(["serve", model_arg, "--served-model-name", served_model_name])
        if not headless:
            command.extend(
                [
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(process.http_port or grpc_port + 1),
                    "--grpc-port",
                    str(grpc_port),
                ]
            )
        if multi_node_replica:
            leader = next(candidate for candidate in endpoint_processes if candidate.node == endpoint_nodes[0])
            # Use the leader's reserved vLLM port range for this endpoint.
            # worker_stage unsets VLLM_PORT for multi-node groups, leaving this
            # port free for rendezvous, including colocated P/D endpoints.
            master_port = leader.vllm_scan_port
            if master_port is None:
                raise ValueError("multi-node vLLM replica needs the leader's allocated vLLM port range")
            command.extend(
                [
                    "--distributed-executor-backend",
                    "mp",
                    "--nnodes",
                    str(len(endpoint_nodes)),
                    "--node-rank",
                    str(node_rank),
                    "--master-addr",
                    leader_ip,
                    "--master-port",
                    str(master_port),
                ]
            )
            if headless:
                command.append("--headless")

        max_model_len = config.pop("max-model-len", None)
        if max_model_len is None:
            max_model_len = config.pop("max_model_len", None)
        if max_model_len is not None:
            command.extend(["--max-model-len", str(max_model_len)])

        if is_dp_mode:
            dp_size = config.pop("data-parallel-size", None)
            if dp_size is None:
                dp_size = config.pop("data_parallel_size", None)
            if dp_size is None:
                raise ValueError("vLLM sidecar DP mode requires data-parallel-size")
            config_dp_rpc_port = config.pop("data-parallel-rpc-port", None)
            if config_dp_rpc_port is None:
                config_dp_rpc_port = config.pop("data_parallel_rpc_port", None)
            dp_rpc_port = process.dp_rpc_port or config_dp_rpc_port or VLLM_DATA_PARALLEL_RPC_PORT
            command.extend(
                [
                    "--data-parallel-size",
                    str(dp_size),
                    "--data-parallel-size-local",
                    str(self._get_local_dp_size(mode, len(process.gpu_indices))),
                    "--data-parallel-address",
                    leader_ip,
                    "--data-parallel-rpc-port",
                    str(dp_rpc_port),
                ]
            )
            if hybrid_lb:
                command.extend(["--data-parallel-start-rank", str(process.node_rank), "--data-parallel-hybrid-lb"])

        config.pop("connector", None)
        has_explicit_kv = "kv-transfer-config" in config or "kv_transfer_config" in config
        kv_transfer_cfg = None if has_explicit_kv else self.kv_transfer_config(mode)
        if kv_transfer_cfg is not None:
            command.extend(["--kv-transfer-config", kv_transfer_cfg])

        kv_cfg = self.get_kv_events_config_for_mode(mode)
        if kv_cfg and process.kv_events_port is not None:
            kv_cfg["endpoint"] = f"tcp://{process_ip}:{process.kv_events_port}"
            command.extend(["--kv-events-config", json.dumps(kv_cfg)])

        command.extend(_config_to_cli_args(config))
        if headless:
            return build_sidecar_launch_command(
                engine=command,
                sidecar=None,
                grpc_port=grpc_port,
                engine_name="vLLM headless follower",
                startup_timeout=sidecar_config.sidecar_startup_timeout,
            )
        sidecar = (
            [sidecar_config.sidecar_binary]
            if sidecar_config.sidecar_binary is not None
            else ["python3", "-m", "dynamo.vllm.sidecar"]
        )
        sidecar.extend(["--grpc-endpoint", f"127.0.0.1:{grpc_port}"])
        if mode in ("prefill", "decode"):
            sidecar.extend(["--disaggregation-mode", mode])
        if mode == "prefill":
            sidecar.extend(["--component", "prefill"])
        sidecar.extend(sidecar_config.sidecar_args)

        return build_sidecar_launch_command(
            engine=command,
            sidecar=sidecar,
            grpc_port=grpc_port,
            engine_name="vLLM",
            startup_timeout=sidecar_config.sidecar_startup_timeout,
        )


@stdlib_dataclass(frozen=True)
class KVConnector:
    """One row of the KV connector table: the vLLM connector class and how srtctl wires it.

    ``kv_role`` None means the role follows the worker mode (prefill produces,
    decode consumes). ``discovery`` marks a connector whose workers find each
    other through the router's discovery endpoint instead of being listed on the
    router command line (AMD's MoRI-IO behind the vLLM Router); the frontend
    asks ``VLLMProtocol.discovers_workers`` and drops the URLs.
    """

    kv_connector: str
    kv_role: str | None = "kv_both"
    module_path: str | None = None
    discovery: bool = False

    def transfer_config(self, mode: WorkerMode) -> dict[str, Any]:
        """The ``--kv-transfer-config`` payload for a worker mode."""
        payload: dict[str, Any] = {"kv_connector": self.kv_connector}
        if self.module_path is not None:
            payload["kv_connector_module_path"] = self.module_path
        payload["kv_role"] = self.kv_role or ("kv_producer" if mode == "prefill" else "kv_consumer")
        return payload


# Connector shorthands a recipe may name in engine.connector or roles.<role>.args.connector.
_CONNECTOR_MAP: dict[str, KVConnector] = {
    "nixl": KVConnector("NixlConnector"),
    "lmcache": KVConnector("LMCacheConnectorV1"),
    "kvbm": KVConnector("DynamoConnector", module_path="kvbm.vllm_integration.connector"),
}


def kv_connector_row(connector: str | None) -> KVConnector | None:
    """The table row for a connector shorthand; None for no connector or a raw JSON string."""
    if not connector:
        return None
    return _CONNECTOR_MAP.get(connector.lower())


def _connector_to_kv_transfer_config(connector: str, mode: WorkerMode) -> str:
    """Translate a connector shorthand to a --kv-transfer-config JSON string.

    Table connectors expand to their preset for ``mode``; anything else passes
    through as the JSON string the recipe wrote.
    """
    row = kv_connector_row(connector)
    if row is None:
        return connector
    return json.dumps(row.transfer_config(mode))


def _config_to_cli_args(config: dict[str, Any]) -> list[str]:
    """Convert config dict to CLI arguments."""
    args: list[str] = []
    for key, value in sorted(config.items()):
        flag_name = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(f"--{flag_name}")
        elif isinstance(value, list):
            args.append(f"--{flag_name}")
            args.extend(str(v) for v in value)
        elif value is not None:
            args.extend([f"--{flag_name}", str(value)])
    return args
