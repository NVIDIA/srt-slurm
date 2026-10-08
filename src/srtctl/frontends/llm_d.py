# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 SemiAnalysis LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""llm-d router frontend (`frontend.type: llm-d`): the Endpoint Picker behind Envoy.

Envoy accepts HTTP requests and asks the Endpoint Picker (EPP) to select workers
via gRPC ext_proc. It forwards to ``x-gateway-destination-endpoint`` using
ORIGINAL_DST. For P/D, EPP selects a decode sidecar and passes the prefill worker
in ``x-prefiller-host-port``; see ``services/llm_d_sidecar.py``.

srtctl supplies a fixed endpoint list through EPP's file-discovery plugin,
including one endpoint per external-LB DP rank. Startup waits for worker health
when enabled, writes the configurations, then launches EPP and Envoy. Readiness
requires Envoy's ``/ready`` and EPP's ready-endpoint count, based on worker metrics.
Optional ``roles.<role>.kv_events`` feed EPP's precise prefix-cache index.

Upstream, pinned: llm-d-router v0.11.0
(https://github.com/llm-d/llm-d-router/tree/a5cbe600ebade00cf3e9885beaf2bfacddeabce1):
``cmd/epp/runner/runner.go`` (``runWithFileDiscovery``),
``pkg/epp/framework/plugins/datalayer/discovery/file``; the Envoy configuration
follows llm-d's no-Kubernetes guide
(https://github.com/llm-d/llm-d/blob/7fb84b0adf8e1d41eb2cef105fc1aafaf5a9b64f/guides/no-kubernetes-deployment/router/envoy/envoy.yaml).
"""

from __future__ import annotations

import copy
import logging
import shlex
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import requests
import yaml

from srtctl.core.health import WorkerHealthResult
from srtctl.frontends.base import numactl_prefix, register_frontend
from srtctl.frontends.static_router import StaticRouterFrontend
from srtctl.ports import (
    LLM_D_ENVOY_ADMIN_PORT,
    LLM_D_EPP_GRPC_PORT,
    LLM_D_EPP_HEALTH_PORT,
    LLM_D_EPP_KV_EVENTS_PORT,
    LLM_D_EPP_METRICS_PORT,
)
from srtctl.services.config import ServiceConfig, ServicePlacementConfig
from srtctl.services.implicit import EffectiveService
from srtctl.services.llm_d_sidecar import LLM_D_SIDECAR_TYPE, sidecar_kv_connector

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)

# Pool the EPP serves; also the namespace of every endpoint in the endpoints file.
POOL = "srtctl"
DISCOVERY_PLUGIN = "srtctl-file-discovery"
ENDPOINTS_FILE = "llm-d-endpoints.yaml"
EPP_CONFIG_FILE = "llm-d-epp-config.yaml"
ENVOY_CONFIG_FILE = "llm-d-envoy.yaml"
ENVOY_ACCESS_LOG = "llm-d-envoy-access.log"
# The EPP's role label (pkg/epp/framework/plugins/scheduling/filter/bylabel/roles.go):
# prefill-filter and decode-filter select on it; an aggregate worker serves both.
ROLE_LABEL = "llm-d.ai/role"
ROLE_BY_MODE = {"prefill": "prefill", "decode": "decode", "agg": "both"}
# Gauge of endpoints whose metrics the EPP scraped within its staleness window
# (pkg/epp/metrics/llm_d_router_metrics.go).
READY_ENDPOINTS_METRIC = "llm_d_epp_ready_endpoints"
# Precise routing indexes KV-cache events and uses a worker's /v1/*/render endpoints
# to obtain matching token IDs
# (pkg/epp/framework/plugins/requestcontrol/dataproducer/{preciseprefixcache,tokenizer}).
PRECISE_PREFIX_PRODUCER = "precise-prefix-cache-producer"
TOKEN_PRODUCER = "token-producer"
_MANAGED_KV_EVENTS_KEYS = frozenset({"zmqEndpoint", "discoverPods", "podDiscoveryConfig"})
# Token-producer backends other than vLLM's render endpoints.
_OTHER_TOKENIZERS = frozenset({"estimate", "udsTokenizerConfig"})
# EPP flags srtctl sets; frontend.args may not repeat them.
_MANAGED_EPP_FLAGS = frozenset(
    {"pool-name", "pool-namespace", "config-file", "config-text", "grpc-port", "grpc-health-port", "metrics-port"}
)


def _is_pd(config: Any) -> bool:
    topology = config.topology
    return topology.num_prefill > 0 and topology.num_decode > 0


def _routed_port(process: Process) -> int:
    """Route through the worker's proxy when present, otherwise its HTTP API."""
    return process.proxy_port if process.proxy_port is not None else process.http_port


def _container_path(name: str) -> str:
    """Container path for a file in ``log_dir``, mounted at ``/logs``."""
    return str(Path("/logs") / name)


# Defaults from guides/no-kubernetes-deployment/router/epp/config.yaml.
# In P/D mode, prefill uses all scorers; decode uses only load scorers.
_DEFAULT_SCORERS = {
    "queue-scorer": 2,
    "kv-cache-utilization-scorer": 2,
    "prefix-cache-scorer": 3,
    "no-hit-lru-scorer": 2,
}
_DEFAULT_DECODE_SCORERS = ("queue-scorer", "kv-cache-utilization-scorer")


def default_epp_config(pd: bool) -> dict[str, Any]:
    """Default scheduler when ``frontend.epp_config`` is unset, with separate profiles for P/D."""

    def profile(name: str, scorers: Any, role_filter: str | None = None) -> dict[str, Any]:
        refs = [{"pluginRef": scorer, "weight": _DEFAULT_SCORERS[scorer]} for scorer in scorers]
        return {"name": name, "plugins": [{"pluginRef": role_filter}, *refs] if role_filter else refs}

    plugins: list[dict[str, Any]] = [{"type": scorer} for scorer in _DEFAULT_SCORERS]
    if not pd:
        return {"plugins": plugins, "schedulingProfiles": [profile("default", _DEFAULT_SCORERS)]}
    plugins += [
        {"type": "prefill-filter"},
        {"type": "decode-filter"},
        {"type": "always-disagg-pd-decider"},
        {"type": "disagg-profile-handler", "parameters": {"deciders": {"prefill": "always-disagg-pd-decider"}}},
    ]
    return {
        "plugins": plugins,
        "schedulingProfiles": [
            profile("prefill", _DEFAULT_SCORERS, "prefill-filter"),
            profile("decode", _DEFAULT_DECODE_SCORERS, "decode-filter"),
        ],
    }


def epp_config_document(epp_config: dict[str, Any] | None, endpoints_path: str) -> dict[str, Any]:
    """The recipe's ``EndpointPickerConfig`` with srtctl's file discovery added.

    ``dataLayer.discovery.pluginRef`` is the spelling both v0.10 and v0.11 accept
    (v0.11 also reads ``discovery.endpoints.pluginRef``).
    """
    document: dict[str, Any] = {"apiVersion": "llm-d.ai/v1alpha1", "kind": "EndpointPickerConfig"}
    document.update(copy.deepcopy(epp_config or {}))
    document["plugins"] = [
        *(document.get("plugins") or []),
        {
            "name": DISCOVERY_PLUGIN,
            "type": "file-discovery",
            # srtctl writes the file once, from the workers it launched.
            "parameters": {"path": endpoints_path, "watchFile": False},
        },
    ]
    document["dataLayer"] = {**(document.get("dataLayer") or {}), "discovery": {"pluginRef": DISCOVERY_PLUGIN}}
    return document


def _plugins(epp_config: dict[str, Any] | None, plugin_type: str) -> list[dict[str, Any]]:
    return [plugin for plugin in (epp_config or {}).get("plugins") or [] if plugin.get("type") == plugin_type]


def _parameters(plugin: dict[str, Any]) -> dict[str, Any]:
    return plugin.get("parameters") or {}


def _vllm_token_producer(plugin: dict[str, Any]) -> bool:
    return not _OTHER_TOKENIZERS & _parameters(plugin).keys()


def with_kv_events(document: dict[str, Any], render_url: str, model_name: str) -> dict[str, Any]:
    """Configure the KV-event subscriber and the worker URL used for tokenization.

    Workers connect to one EPP socket (``kvEventsConfig.zmqEndpoint``). The
    alternative per-pod mode requires a fixed worker port, which collides when
    workers share a node.
    """
    for plugin in document.get("plugins") or []:
        parameters = _parameters(plugin)
        if plugin.get("type") == PRECISE_PREFIX_PRODUCER:
            kv_events = parameters.get("kvEventsConfig") or {}
            parameters["kvEventsConfig"] = {
                **kv_events,
                "zmqEndpoint": f"tcp://*:{LLM_D_EPP_KV_EVENTS_PORT}",
                "discoverPods": False,
            }
        elif plugin.get("type") == TOKEN_PRODUCER and _vllm_token_producer(plugin):
            parameters.setdefault("modelName", model_name)
            parameters["vllm"] = {**(parameters.get("vllm") or {}), "url": render_url}
        else:
            continue
        plugin["parameters"] = parameters
    return document


def envoy_config(listen_port: int, access_log: str) -> str:
    """Envoy in front of the EPP: ext_proc to it on localhost, then ORIGINAL_DST to the endpoint it picked."""
    from jinja2 import Environment, FileSystemLoader

    templates = Environment(loader=FileSystemLoader(str(Path(__file__).parent.parent / "templates")))
    return templates.get_template("llm_d_envoy.yaml.j2").render(
        admin_port=LLM_D_ENVOY_ADMIN_PORT,
        listen_port=listen_port,
        access_log=access_log,
        epp_port=LLM_D_EPP_GRPC_PORT,
    )


def parse_ready_endpoints(metrics_text: str) -> int | None:
    """The EPP's ready-endpoint gauge from its Prometheus text, or None before it is published."""
    total: float | None = None
    for line in metrics_text.splitlines():
        if line.startswith((f"{READY_ENDPOINTS_METRIC} ", f"{READY_ENDPOINTS_METRIC}{{")):
            total = (total or 0.0) + float(line.rsplit(" ", 1)[1])
    return None if total is None else int(total)


@register_frontend("llm-d")
class LLMDFrontend(StaticRouterFrontend):
    """llm-d Endpoint Picker behind Envoy, in front of direct vLLM workers."""

    type: ClassVar[str] = "llm-d"
    required_backend: ClassVar[str | None] = "vllm"
    executable: ClassVar[tuple[str, ...]] = ("epp",)
    # The EPP takes prefill/decode from its scheduler configuration, not a flag.
    pd_flag: ClassVar[str] = ""
    process_name: ClassVar[str] = "llm-d"
    # Gate startup on worker health unless health checks are disabled.
    wait_for_workers_before_start: ClassVar[bool] = True

    def validate(self, config: Any) -> None:
        frontend = config.frontend
        if frontend.enable_multiple_frontends and config.engine_node_count > 1:
            raise ValueError(
                "frontend.type: llm-d requires Envoy and the Endpoint Picker on the public endpoint's node "
                "for readiness checks; multi-node nginx routing is unsupported, even with "
                "num_additional_frontends: 0. Set frontend.enable_multiple_frontends: false"
            )
        topology = config.topology
        for mode, count in (
            ("prefill", topology.num_prefill),
            ("decode", topology.num_decode),
            ("agg", topology.num_agg),
        ):
            if count and config.backend_for_role(mode).is_grpc_mode(mode):
                raise ValueError(f"frontend.type: llm-d routes HTTP workers; roles.{mode} serves gRPC")
        epp_config = frontend.epp_config or {}
        data_layer = epp_config.get("dataLayer") or {}
        if "discovery" in data_layer or any(
            plugin.get("type") == "file-discovery" for plugin in epp_config.get("plugins") or []
        ):
            raise ValueError(
                "frontend.epp_config must not configure endpoint discovery: srtctl adds the file-discovery "
                "plugin and dataLayer.discovery for the workers it launches"
            )
        self._validate_kv_events(config)
        if _is_pd(config):
            if frontend.epp_config is not None and not epp_config.get("schedulingProfiles"):
                raise ValueError(
                    "frontend.type: llm-d with prefill and decode workers needs frontend.epp_config with "
                    "prefill and decode schedulingProfiles and a disaggregation profile handler"
                )
            sidecar_kv_connector(config.backend_for_role("decode"))

    def _validate_kv_events(self, config: Any) -> None:
        """Workers publish KV-cache events exactly when the EPP has a precise prefix-cache producer to index them."""
        epp_config = config.frontend.epp_config
        topology = config.topology
        publishing = [
            mode
            for mode, count in (
                ("prefill", topology.num_prefill),
                ("decode", topology.num_decode),
                ("agg", topology.num_agg),
            )
            if count and config.backend_for_role(mode).get_kv_events_config_for_mode(mode)
        ]
        producers = _plugins(epp_config, PRECISE_PREFIX_PRODUCER)
        if not producers:
            if publishing:
                raise ValueError(
                    f"roles.{publishing[0]}.kv_events publishes KV-cache events for the EPP to index; "
                    f"add a {PRECISE_PREFIX_PRODUCER} plugin to frontend.epp_config"
                )
            return
        if not publishing:
            raise ValueError(
                f"frontend.epp_config's {PRECISE_PREFIX_PRODUCER} indexes the workers' KV-cache events; "
                "set roles.<role>.kv_events on the roles it scores"
            )
        for producer in producers:
            managed = _MANAGED_KV_EVENTS_KEYS & (_parameters(producer).get("kvEventsConfig") or {}).keys()
            if managed:
                raise ValueError(
                    f"frontend.epp_config sets {PRECISE_PREFIX_PRODUCER} kvEventsConfig.{', '.join(sorted(managed))}, "
                    "which srtctl manages: the EPP binds one KV-event socket the workers connect to"
                )
        token_producers = [plugin for plugin in _plugins(epp_config, TOKEN_PRODUCER) if _vllm_token_producer(plugin)]
        if not token_producers:
            raise ValueError(
                f"{PRECISE_PREFIX_PRODUCER} needs the engine's token IDs: add a {TOKEN_PRODUCER} plugin "
                "to frontend.epp_config (srtctl points it at a worker's render endpoint)"
            )
        if any("url" in (_parameters(plugin).get("vllm") or {}) for plugin in token_producers):
            raise ValueError(f"frontend.epp_config sets {TOKEN_PRODUCER} vllm.url, which srtctl points at a worker")
        frontend = config.frontend
        if (
            frontend.enable_multiple_frontends and config.engine_node_count > 1
        ) or frontend.placement.location != "head":
            raise ValueError(
                "roles.<role>.kv_events: the workers publish to the EPP on the head node; set "
                "frontend.enable_multiple_frontends: false and keep frontend.placement.node: head"
            )
        for mode in publishing:
            backend = config.backend_for_role(mode)
            if backend._is_dp_mode(mode) and not backend.is_external_lb(mode):
                raise ValueError(
                    f"roles.{mode}.kv_events with data-parallel-size: vLLM publishes every DP rank's events on its "
                    "own port, so the EPP must route to the ranks themselves; set "
                    f"roles.{mode}.args.data-parallel-external-lb: true"
                )

    def kv_events_subscriber(self, process: Process, runtime: RuntimeContext, model_name: str) -> tuple[str, int, str]:
        """Return the EPP socket and ``kv@<endpoint>@<model>`` topic.

        Use the routed endpoint, including the sidecar port, to match EPP's cache
        index keys (pkg/kvevents/engineadapter/vllm_adapter.go).
        """
        address = self.resolve_worker_host(process.node, runtime.network_interface)
        return runtime.head_node_ip, LLM_D_EPP_KV_EVENTS_PORT, f"kv@{address}:{_routed_port(process)}@{model_name}"

    def worker_metrics_port(self, process: Process, runtime: RuntimeContext) -> int | None:
        """Scrape each routable worker directly, including external-LB DP ranks."""
        return process.http_port if process.http_port > 0 else None

    def worker_endpoint_port(self, process: Process, config: Any, runtime: RuntimeContext) -> int | None:
        return process.http_port if process.http_port > 0 else None

    def proxied_worker_modes(self, config: Any) -> frozenset[str]:
        """Prefill/decode: the router reaches each decode worker through its P/D sidecar."""
        return frozenset({"decode"}) if _is_pd(config) else frozenset()

    def implied_services(self, config: Any) -> list[EffectiveService]:
        if not _is_pd(config):
            return []
        return [
            EffectiveService(
                ServiceConfig(
                    name=LLM_D_SIDECAR_TYPE,
                    type=LLM_D_SIDECAR_TYPE,
                    placement=ServicePlacementConfig(node="decode", per="worker"),
                    # Nothing on the sidecar's path registers with etcd or NATS.
                    inherit_discovery_env=False,
                ),
                implicit=True,
                reason="frontend.type llm-d (prefill/decode)",
            )
        ]

    def frontend_metrics_port(self, frontend_args: dict[str, Any] | None) -> int | None:
        """The EPP serves Prometheus on its own listener."""
        return LLM_D_EPP_METRICS_PORT

    def health_expectations(self, config: Any, processes: list[Process] | None) -> tuple[int, int, str]:
        """Count routable processes, including individual external-LB DP ranks."""
        if processes is None:
            return super().health_expectations(config, processes)
        routable = [process for process in processes if process.http_port > 0]
        prefill = sum(process.endpoint_mode == "prefill" for process in routable)
        decode = len(routable) - prefill
        return prefill, decode, f"{len(routable)} llm-d endpoints"

    def probe_ready(
        self, host: str, port: int, expected_prefill: int, expected_decode: int, config: Any
    ) -> WorkerHealthResult:
        """Envoy's admin ``/ready``, then the EPP's count of endpoints whose metrics it scrapes."""
        envoy = requests.get(f"http://{host}:{LLM_D_ENVOY_ADMIN_PORT}/ready", timeout=5.0)
        if envoy.status_code != 200:
            return WorkerHealthResult(ready=False, message=f"Envoy /ready returned HTTP {envoy.status_code}")
        metrics = requests.get(f"http://{host}:{LLM_D_EPP_METRICS_PORT}/metrics", timeout=5.0)
        expected = expected_prefill + expected_decode
        ready = parse_ready_endpoints(metrics.text) if metrics.status_code == 200 else None
        if ready is None:
            return WorkerHealthResult(ready=False, message=f"EPP has not published {READY_ENDPOINTS_METRIC} yet")
        return WorkerHealthResult(
            ready=ready >= expected,
            message=f"llm-d EPP reports {ready}/{expected} endpoints ready",
            decode_ready=ready,
            decode_expected=expected,
        )

    def endpoints_document(
        self, backend_processes: list[Process], network_interface: str | None
    ) -> dict[str, list[dict[str, Any]]]:
        """The file-discovery endpoints: every routable worker, at its proxy's port when it has one."""
        endpoints = []
        for process in backend_processes:
            if process.http_port <= 0:
                continue
            endpoints.append(
                {
                    "name": f"{process.endpoint_mode}-{process.endpoint_index}-{process.node_rank}",
                    "namespace": POOL,
                    "address": self.resolve_worker_host(process.node, network_interface),
                    "port": str(_routed_port(process)),
                    "labels": {ROLE_LABEL: ROLE_BY_MODE[process.endpoint_mode]},
                }
            )
        return {"endpoints": endpoints}

    def render_url(self, backend_processes: list[Process], network_interface: str | None) -> str:
        """Direct worker API for tokenization; prefer the first prefill worker."""
        routable = [process for process in backend_processes if process.http_port > 0]
        process = next((p for p in routable if p.endpoint_mode == "prefill"), routable[0])
        return f"http://{self.resolve_worker_host(process.node, network_interface)}:{process.http_port}"

    def epp_command(self, config: Any, epp_config_path: str) -> list[str]:
        user_args = self.get_frontend_args_list(config.frontend.args)
        managed = {str(key).replace("_", "-") for key in (config.frontend.args or {})} & _MANAGED_EPP_FLAGS
        if managed:
            raise ValueError(f"frontend.args sets {', '.join(sorted(managed))}, which srtctl manages for llm-d")
        return [
            *self.executable,
            f"--pool-name={POOL}",
            f"--pool-namespace={POOL}",
            f"--config-file={epp_config_path}",
            f"--grpc-port={LLM_D_EPP_GRPC_PORT}",
            f"--grpc-health-port={LLM_D_EPP_HEALTH_PORT}",
            f"--metrics-port={LLM_D_EPP_METRICS_PORT}",
            # Envoy dials the EPP over plaintext HTTP/2 on localhost.
            "--secure-serving=false",
            *user_args,
        ]

    def start_frontends(
        self,
        topology: Any,
        runtime: RuntimeContext,
        config: Any,
        backend: Any,
        backend_processes: list[Process],
        stop_event: threading.Event | None = None,
    ) -> list[ManagedProcess]:
        from srtctl.core.processes import FRONTEND_TERMINATE_TIMEOUT_SECONDS, ManagedProcess

        self.wait_for_workers(
            self.collect_workers(backend, backend_processes, runtime.network_interface), config, stop_event
        )

        endpoints = self.endpoints_document(backend_processes, runtime.network_interface)
        (runtime.log_dir / ENDPOINTS_FILE).write_text(yaml.safe_dump(endpoints, sort_keys=False))
        epp_config = config.frontend.epp_config
        epp_document = epp_config_document(
            default_epp_config(_is_pd(config)) if epp_config is None else epp_config, _container_path(ENDPOINTS_FILE)
        )
        if _plugins(epp_document, PRECISE_PREFIX_PRODUCER) or _plugins(epp_document, TOKEN_PRODUCER):
            epp_document = with_kv_events(
                epp_document,
                self.render_url(backend_processes, runtime.network_interface),
                backend.get_served_model_name(runtime.model_path.name),
            )
        (runtime.log_dir / EPP_CONFIG_FILE).write_text(yaml.safe_dump(epp_document, sort_keys=False))
        (runtime.log_dir / ENVOY_CONFIG_FILE).write_text(
            envoy_config(topology.frontend_port, _container_path(ENVOY_ACCESS_LOG))
        )
        logger.info("llm-d endpoints (%d): %s", len(endpoints["endpoints"]), endpoints["endpoints"])

        commands = {
            "epp": self.epp_command(config, _container_path(EPP_CONFIG_FILE)),
            # Avoid hot-restart socket collisions between Envoy instances on the host.
            "envoy": ["envoy", "-c", _container_path(ENVOY_CONFIG_FILE), "--disable-hot-restart"],
        }
        container_image = config.frontend.container_image or str(runtime.container_image)
        env = {**runtime.environment, **(config.frontend.env or {})}
        processes: list[ManagedProcess] = []
        for idx, node in enumerate(topology.frontend_nodes):
            for component, command in commands.items():
                cmd = numactl_prefix(config) + command
                step_name = f"{self.process_name}-{component}_{idx}"
                log_file = runtime.log_dir / f"{node}_{self.process_name}-{component}_{idx}.out"
                logger.info("Starting llm-d %s %d on %s: %s", component, idx, node, shlex.join(cmd))
                proc = self.start_process(
                    command=cmd,
                    nodelist=[node],
                    output=str(log_file),
                    container_image=container_image,
                    container_mounts=runtime.container_mounts,
                    env_to_set=env or None,
                    het_group=runtime.nodes.het_group_for(node),
                    step_name=step_name,
                    srun_options=runtime.srun_options,
                )
                processes.append(
                    ManagedProcess(
                        name=step_name,
                        popen=proc,
                        log_file=log_file,
                        node=node,
                        critical=True,
                        terminate_timeout=FRONTEND_TERMINATE_TIMEOUT_SECONDS,
                        step_name=step_name,
                    )
                )
        return processes
