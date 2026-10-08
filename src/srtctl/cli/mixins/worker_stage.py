# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Worker stage mixin for SweepOrchestrator.

Handles starting backend worker processes (prefill/decode/agg).
"""

import logging
import shlex
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from srtctl.backends.base import WorkerMode
from srtctl.backends.vllm import VLLMBackend, VLLMFailoverConfig
from srtctl.core.fingerprint import generate_capture_script
from srtctl.core.health import wait_for_health
from srtctl.core.observability_nsys import wrap_observability_nsys
from srtctl.core.processes import ManagedProcess, NamedProcesses, ProcessRegistry
from srtctl.core.schema import build_otel_env, installs_dynamo
from srtctl.core.slurm import CONTAINER_REMAP_ROOT_EXPORT, get_hostname_ip, start_srun_process
from srtctl.core.supervisor import EndpointKey, WorkerSupervisor
from srtctl.frontends import get_frontend
from srtctl.services.implicit import discovery_env

if TYPE_CHECKING:
    from collections.abc import Iterable

    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Endpoint, Process

logger = logging.getLogger(__name__)

# Engines shut down on SIGTERM (deregister, free GPUs, flush); give them longer than the default 10s.
WORKER_TERMINATE_TIMEOUT_SECONDS = 30.0


def worker_step_name(mode: str, index: int, node: str, attempt: int = 0, engine_id: int = 0) -> str:
    """The Slurm step name (and registry name) of a worker step.

    ``engine_id`` > 0 is a shadow engine of the worker (vLLM ``engine.failover``)
    and adds ``_e<k>``. ``attempt`` is the supervisor's relaunch count; a
    relaunched step carries an ``_r<n>`` suffix so ``squeue --steps`` can never
    confuse it with a lingering step of the life it replaces.
    """
    name = f"{mode}_{index}_{node}"
    if engine_id:
        name = f"{name}_e{engine_id}"
    return f"{name}_r{attempt}" if attempt else name


# Dynamo runtime (Rust) log filter for worker containers; a recipe's roles.<role>.env
# overrides it via the merge below.
_DEFAULT_WORKER_DYN_LOG = "info,dynamo_runtime::pipeline::network::ingress::push_handler=warn"


def _nsys_library_path_preamble(paths: list[str] | None) -> str | None:
    """Prepend Nsight paths after the container environment has been loaded."""
    if not paths:
        return None
    prefix = ":".join(dict.fromkeys(path for path in paths if path))
    if not prefix:
        return None
    return f'export LD_LIBRARY_PATH={shlex.quote(prefix)}"${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"'


def _append_preamble(preamble: str | None, command: str | None) -> str | None:
    """Run ``command`` after an existing shell preamble."""
    if not command:
        return preamble
    return f"{preamble} && {command}" if preamble else command


class WorkerStageMixin:
    """Mixin for worker process startup stage.

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
        self.backend: Backend
        self.backend_processes: list[Process]
    """

    # Type hints for mixin dependencies
    config: "SrtConfig"
    runtime: "RuntimeContext"

    def _apply_mooncake_process_config(self, process: "Process", environment: dict[str, str]) -> None:
        backend = self.config.backend_for_role(process.endpoint_mode)
        if not isinstance(backend, VLLMBackend):
            return
        local_config = backend.build_mooncake_process_config(
            process, self.runtime.infra_node_ip, self.runtime.gpus_per_node
        )
        if local_config is not None:
            filename, payload = local_config
            environment["MOONCAKE_CONFIG_PATH"] = str(self.runtime.container_log_dir / filename)
            logger.info(
                "Mooncake process config: node=%s physical_gpus=%s device_name=%s path=%s",
                process.node,
                sorted(process.gpu_indices),
                payload["device_name"],
                environment["MOONCAKE_CONFIG_PATH"],
            )

    @property
    def backend(self) -> Any:
        """Access the backend config (implements Backend)."""
        return self.config.backend

    @property
    def failover(self) -> "VLLMFailoverConfig | None":
        """``backend.failover`` when the engine runs shadow engine recovery, else None."""
        return self.backend.failover

    @property
    def backend_processes(self) -> list["Process"]:
        """Compute physical process topology from endpoints (cached)."""
        raise NotImplementedError

    @property
    def endpoints(self) -> list["Endpoint"]:
        """Endpoint allocation topology."""
        raise NotImplementedError

    def _build_worker_preamble(self) -> str | None:
        """Build bash preamble for worker processes.

        Runs (in order):
        1. Custom setup script from /configs/ (if config.setup_script set)
        2. Dynamo installation (if frontend type is dynamo)
        """
        parts = []

        # 1. Custom setup script (runs first)
        setup_script = getattr(self.config, "setup_script", None)
        if isinstance(setup_script, str) and setup_script:
            script_name = shlex.quote(setup_script)
            parts.append(
                f"setup_script={script_name} && "
                'script_path="/configs/${setup_script}" && '
                'patch_script_path="/configs/patches/${setup_script}" && '
                'echo "Running setup script: ${script_path} (fallback ${patch_script_path})" && '
                'if [ -f "${script_path}" ]; then bash "${script_path}"; '
                'elif [ -f "${patch_script_path}" ]; then bash "${patch_script_path}"; '
                'else echo "WARNING: ${script_path} or ${patch_script_path} not found"; fi'
            )

        # 2. Dynamo installation (required for dynamo.sglang when using dynamo frontend)
        # Skip if dynamo.install is False (container already has dynamo installed)
        if installs_dynamo(self.config):
            parts.append(self.config.dynamo.get_install_commands())

        if not parts:
            return None

        return " && ".join(parts)

    def _fatal_log_patterns(self, mode: WorkerMode) -> tuple[str, ...]:
        """Log lines that fail a worker whose srun step outlives its engine.

        The backend names the lines its launcher prints once the engine has died
        (``Backend.fatal_log_patterns``); the recipe adds its own through
        ``health_check.extra_fatal_log_patterns`` or switches the watch off with
        ``health_check.fatal_log_markers: false``.
        """
        health_check = self.config.health_check
        if not health_check.fatal_log_markers:
            return ()
        backend = self.config.backend_for_role(mode)
        return tuple(backend.fatal_log_patterns(mode)) + tuple(health_check.extra_fatal_log_patterns)

    def _visible_device_environment(self, process: "Process") -> dict[str, str]:
        """The cluster's GPU mask for a process that owns part of its node, when something must read it.

        The engine asks for the mask through the protocol. The Dynamo vLLM sidecar
        and failover engines get it regardless: the sidecar and the GMS service must
        see the same device list as the engine (see start_gms_sidecar). A process
        that owns the whole node needs no mask.
        """
        backend = self.config.backend_for_role(process.endpoint_mode)
        force_mask = (self.config.dynamo.sidecar and backend.type == "vllm") or backend.failover is not None
        if not (force_mask or backend.should_set_visible_devices()):
            return {}
        if len(process.gpu_indices) >= self.runtime.gpus_per_node:
            return {}
        return {self.runtime.visible_devices_env: process.cuda_visible_devices}

    def _worker_environment_defaults(self, process: "Process") -> dict[str, str]:
        env = {"HEAD_NODE_IP": self.runtime.head_node_ip}
        if get_frontend(self.config.frontend.type).worker_launch == "dynamo":
            env.update(discovery_env(self.config, self.runtime))
            env.update(
                DYN_SYSTEM_PORT=str(process.sys_port),
                DYN_REQUEST_PLANE=self.config.dynamo.request_plane,
                DYN_SKIP_SGLANG_LOG_FORMATTING="1",
                DYN_LOG=_DEFAULT_WORKER_DYN_LOG,
            )
            if self.config.dynamo.event_plane:
                env["DYN_EVENT_PLANE"] = self.config.dynamo.event_plane
        return env

    def _apply_kvbm_endpoint_env(self, env_to_set: dict[str, str], endpoint_processes: list["Process"]) -> None:
        """Fill KVBM leader ZMQ settings for an endpoint.

        KVBM defaults its leader control sockets to 127.0.0.1. That works for
        single-node endpoints, but multi-node endpoints need every worker to
        connect to the leader node. Also assign deterministic per-endpoint ports
        when the user did not set them, so co-located KVBM endpoints do not fight
        over the default KVBM leader ZMQ pair.
        """
        if env_to_set.get("DYN_CONNECTOR", "").lower() != "kvbm" or not endpoint_processes:
            return

        leader = endpoint_processes[0]
        endpoint_nodes = list(dict.fromkeys(p.node for p in endpoint_processes))

        if len(endpoint_nodes) > 1:
            leader_host = get_hostname_ip(leader.node, self.runtime.network_interface)
            env_to_set.setdefault("DYN_KVBM_LEADER_ZMQ_HOST", leader_host)

        if leader.kvbm_zmq_port is None:
            return

        env_to_set.setdefault("DYN_KVBM_LEADER_ZMQ_PUB_PORT", str(leader.kvbm_zmq_port))
        env_to_set.setdefault("DYN_KVBM_LEADER_ZMQ_ACK_PORT", str(leader.kvbm_zmq_port + 1))

    def _get_worker_environment_for_mode(self, mode: WorkerMode) -> dict[str, str]:
        """Return mode environment with engine-specific defaults the recipe can override."""
        backend = self.config.backend_for_role(mode)
        environment = backend.get_environment_for_mode(mode)
        if self.config.dynamo.sidecar and backend.type == "vllm":
            # Installed plugins may replace native engine output types and
            # break the fixed Rust/Python MessagePack contract used by vllm-rs.
            environment.setdefault("VLLM_PLUGINS", "")
        if self.config.dynamo.sidecar and backend.type == "sglang":
            # The sidecar talks to SGLang's native gRPC server, a prebuilt Rust extension. In
            # images that run SGLang from a source checkout (the nightlies), the extension
            # loader's default "auto" mode ignores the bundled .so and tries to rebuild it
            # with cargo, which those images do not ship, so the engine dies before gRPC is up.
            # "never" trusts the bundled build. Mode env and the global environment override.
            # TODO: drop once the SGLang loader prefers a bundled extension over a rebuild
            #       (sglang.srt.utils.load_rust_extension, auto mode in source checkouts).
            environment.setdefault("SGLANG_RUST_BUILD_MODE", "never")
        if backend.type == "sglang":
            # SGLang treats its own exit after SIGTERM as a crash: it drains in a few
            # seconds, then tries py-spy (needs root) and waits 60s for CUDA
            # coredumps that are never produced unless SGLANG_CUDA_COREDUMP=1. That
            # wait is what cleanup would otherwise kill through. Skip it unless the
            # recipe opted into coredumps (mode env or the global environment).
            recipe_env = {**self.runtime.environment, **environment}
            if recipe_env.get("SGLANG_CUDA_COREDUMP", "0").lower() not in ("1", "true"):
                environment.setdefault("SGLANG_CUDA_COREDUMP_BEFORE_CRASH", "0")
            environment.setdefault("SGLANG_PYSPY_DUMP_BEFORE_CRASH", "0")
        return environment

    def _profiling_selects_process(self, process: "Process") -> bool:
        """Whether this physical process should be wrapped for profiling.

        Wall-clock captures retain their existing all-process behavior. TRT-LLM
        also remains endpoint-wide because one MPI launch owns all executor
        ranks and uses ``TLLM_PROFILE_START_STOP`` instead of HTTP control.
        """
        backend = self.config.backend_for_role(process.endpoint_mode)
        profiling = self.config.profiling
        if not profiling.is_nsys or profiling.is_nsys_time or backend.type == "trtllm":
            return True
        return profiling.selects_process(
            process.endpoint_mode,
            process.endpoint_index,
            process.node_rank,
        )

    def start_worker(
        self, process: "Process", endpoint_processes: list["Process"], *, attempt: int = 0
    ) -> ManagedProcess:
        """Start a single worker process (one srun per node, used by SGLang).

        ``attempt`` > 0 is a supervisor relaunch of the same process spec: same
        node, GPUs, ports and log path (the previous life's log has been rotated
        away), only the step name differs.
        """
        mode = process.endpoint_mode
        backend = self.config.backend_for_role(mode)
        index = process.endpoint_index
        failover = backend.failover
        # "" for engine 0, "_e<k>" for a shadow: step name, logs, and dumps stay apart.
        # (getattr: tests drive this stage with plain namespaces standing in for Process.)
        suffix = getattr(process, "engine_suffix", "")

        if attempt:
            logger.info("Relaunching %s worker %d on %s (restart %d)", mode, index, process.node, attempt)
        elif suffix:
            logger.info("Starting %s worker %d shadow engine %d on %s", mode, index, process.engine_id, process.node)
        else:
            logger.info("Starting %s worker %d on %s", mode, index, process.node)

        # Log and config files
        worker_log = self.runtime.log_dir / f"{process.node}_{mode}_w{index}{suffix}.out"
        config_dump = self.runtime.container_log_dir / f"{process.node}_config{suffix}.json"

        # Profiling setup
        profiling = self.config.profiling
        profiling_selects_process = self._profiling_selects_process(process)
        nsys_prefix = None
        if profiling.enabled:
            (self.runtime.log_dir / "profiles" / mode).mkdir(parents=True, exist_ok=True)
        if profiling.is_nsys and profiling_selects_process:
            gpu_label = process.cuda_visible_devices.replace(",", "-")
            nsys_output = (
                f"{self.runtime.container_log_dir}/profiles/{mode}/"
                f"{process.node}_{mode}_w{index}{suffix}_profile_gpu{gpu_label}"
            )
            nsys_prefix = profiling.get_nsys_prefix(
                nsys_output, frontend_type=self.config.frontend.type, backend_type=backend.type
            )

        # Build command using backend's method
        cmd = backend.build_worker_command(
            process=process,
            endpoint_processes=endpoint_processes,
            runtime=self.runtime,
            frontend_type=self.config.frontend.type,
            nsys_prefix=nsys_prefix,
            dump_config_path=config_dump,
            profiling=profiling if profiling_selects_process else None,
        )

        automatic_nsys = getattr(self.config, "observability_nsys_enabled", False) is True
        nsys_env: dict[str, str] = {}
        if automatic_nsys:
            gpu_label = process.cuda_visible_devices.replace(",", "-")
            cmd, nsys_env = wrap_observability_nsys(
                cmd,
                config=self.config,
                log_dir=self.runtime.log_dir,
                report_name=f"{mode}/{process.node}_{mode}_w{index}{suffix}_profile_gpu{gpu_label}",
                ranks=1,
            )

        # Worker environment variables
        env_to_set = self._worker_environment_defaults(process)

        # Add OTEL env vars (before mode-specific env so OTEL_SERVICE_NAME can be overridden)
        env_to_set.update(build_otel_env(self.config.observability, mode))
        env_to_set.update(nsys_env)

        # Add mode-specific environment variables from backend
        # Support simple {node} and {node_id} templating
        # Unknown placeholders are left unchanged (no error thrown)
        node_id = self.runtime.nodes.worker.index(process.node)
        template_vars = {"node": process.node, "node_id": node_id}

        class SafeDict(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"  # Leave unknown placeholders unchanged

        for key, value in self._get_worker_environment_for_mode(mode).items():
            formatted_value = value.format_map(SafeDict(template_vars))
            env_to_set[key] = formatted_value

        # Add config environment variables with same templating support
        for key, value in self.runtime.environment.items():
            formatted_value = value.format_map(SafeDict(template_vars))
            env_to_set[key] = formatted_value

        env_to_set.update(self._visible_device_environment(process))

        # Add backend-specific process environment variables (e.g., unique ports)
        env_to_set.update(backend.get_process_environment(process))
        if failover is not None:
            env_to_set.update(backend.get_failover_environment(process, self.runtime.job_id))

        # Add mooncake worker env vars if configured. Resolve the worker's own IP
        # so MOONCAKE_LOCAL_HOSTNAME is correct for multi-node peer-to-peer
        # transfers (defaulting to "localhost" silently breaks them).
        if backend.mooncake_kv_store is not None:
            # A MOONCAKE_LOCAL_HOSTNAME already in the worker env (roles.*.env) pins a NIC; otherwise the node IP.
            local_hostname = env_to_set.get("MOONCAKE_LOCAL_HOSTNAME") or get_hostname_ip(
                process.node, self.runtime.network_interface
            )
            env_to_set.update(backend.get_mooncake_worker_env(self.runtime.infra_node_ip, local_hostname))

        self._apply_mooncake_process_config(process, env_to_set)

        # Add profiling environment variables last.
        if profiling.enabled and profiling_selects_process:
            profile_dir = str(self.runtime.container_log_dir / "profiles")
            env_to_set.update(profiling.get_env_vars(mode, profile_dir))

        self._apply_kvbm_endpoint_env(env_to_set, endpoint_processes)

        # Log env vars in the format: VAR=value VAR2=value2
        env_str = " ".join(f"{k}={v}" for k, v in sorted(env_to_set.items()))
        logger.info("Env: %s", env_str)
        logger.info("Command: %s", shlex.join(cmd))
        logger.info("Log: %s", worker_log)
        if profiling.enabled:
            logger.info("Profiling: %s mode", profiling.type)

        # Build bash preamble (setup script + dynamo install + fingerprint)
        bash_preamble = self._build_worker_preamble()
        if profiling_selects_process and profiling.is_nsys:
            bash_preamble = _append_preamble(
                bash_preamble,
                _nsys_library_path_preamble(profiling.nsys_library_paths),
            )
        fp_cmd = generate_capture_script(f"{self.runtime.container_log_dir}/fingerprint_{mode}_w{index}{suffix}.json")
        # Keep fingerprint failures non-fatal, but do not let its `|| true`
        # mask failures from setup/dynamo install commands before it.
        fp_cmd = f"( {fp_cmd} )"
        bash_preamble = f"{bash_preamble} && {fp_cmd}" if bash_preamble else fp_cmd

        if failover is not None:
            # The engine creates the lock file itself; its directory (also the GMS
            # socket dir) must exist. The gms service made it, but the engine step
            # should not depend on that after a relaunch.
            assert isinstance(backend, VLLMBackend)
            worker_dir = backend.failover_worker_dir(self.runtime.job_id, process)
            bash_preamble = _append_preamble(bash_preamble, f"mkdir -p {shlex.quote(worker_dir)}")

        # vLLM uses VLLM_PORT as the initial port for its internal message
        # queues. In a multi-node endpoint, concurrent TP ranks inherit the
        # same value and can race while probing and binding remote TCP queues.
        # Let vLLM choose an ephemeral base port instead.
        endpoint_nodes = {endpoint_process.node for endpoint_process in endpoint_processes}
        env_to_unset = ["VLLM_PORT"] if backend.type == "vllm" and len(endpoint_nodes) > 1 else None

        step_name = worker_step_name(mode, index, process.node, attempt, getattr(process, "engine_id", 0))
        proc = start_srun_process(
            command=cmd,
            nodelist=[process.node],
            output=str(worker_log),
            container_image=(
                self.config.worker_container_for_role(mode)
                if mode in self.config.role_containers
                else str(self.runtime.container_image)
            ),
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
            env_to_unset=env_to_unset,
            bash_preamble=bash_preamble,
            srun_options={**self.runtime.srun_options, **self.config.roles[mode].srun_options},
            srun_export_env=CONTAINER_REMAP_ROOT_EXPORT if installs_dynamo(self.config) else None,
            het_group=process.het_group,
            step_name=step_name,
        )

        return ManagedProcess(
            name=step_name,
            popen=proc,
            log_file=worker_log,
            node=process.node,
            # roles.<role>.critical: false keeps the run alive when this worker
            # exits, for probes that kill workers on purpose.
            critical=self.config.topology.worker_critical(mode),
            # SIGTERM reaches the engine through the step so it deregisters and
            # frees the GPUs cleanly; a signalled srun would SIGKILL it instead.
            terminate_timeout=(
                self.config.observability.nsys.terminate_timeout if automatic_nsys else WORKER_TERMINATE_TIMEOUT_SECONDS
            ),
            signal_full=not automatic_nsys,
            step_name=step_name,
            fatal_log_patterns=self._fatal_log_patterns(mode),
        )

    def start_endpoint_worker(self, endpoint_processes: list["Process"], *, attempt: int = 0) -> ManagedProcess:
        """Start a worker using MPI-style launching (one srun per endpoint, used by TRTLLM).

        This launches a single srun command that spans all nodes in the endpoint,
        with ntasks = total GPUs across all nodes. ``attempt`` is the supervisor's
        relaunch count (see ``start_worker``).
        """
        # Use the leader process for metadata
        leader = endpoint_processes[0]
        mode = leader.endpoint_mode
        backend = self.config.backend_for_role(mode)
        index = leader.endpoint_index

        # Collect all unique nodes for this endpoint
        endpoint_nodes = list(dict.fromkeys(p.node for p in endpoint_processes))
        num_nodes = len(endpoint_nodes)
        total_gpus = sum(len(p.gpu_indices) for p in endpoint_processes)
        # TRT-LLM derives local devices from global rank modulo visible GPUs.
        if backend.type == "trtllm":
            rank_offset = 0
            for process in endpoint_processes:
                local_size = len(process.gpu_indices)
                if any(
                    (rank_offset + local_rank) % local_size != local_rank
                    or (rank_offset + local_rank) % len(leader.gpu_indices) != local_rank
                    for local_rank in range(local_size)
                ):
                    raise ValueError("MPI GPU layout is incompatible with TRT-LLM local-rank mapping")
                rank_offset += local_size

        logger.info(
            "%s %s worker %d on %d nodes (%s) with %d total GPUs (MPI mode)%s",
            "Relaunching" if attempt else "Starting",
            mode,
            index,
            num_nodes,
            ",".join(endpoint_nodes),
            total_gpus,
            f" (restart {attempt})" if attempt else "",
        )

        # Log and config files (use leader node in name)
        worker_log = self.runtime.log_dir / f"{leader.node}_{mode}_w{index}.out"
        config_dump = self.runtime.container_log_dir / f"{leader.node}_config.json"

        # Profiling setup
        profiling = self.config.profiling
        profiling_selects_process = self._profiling_selects_process(leader)
        nsys_prefix = None
        if profiling.enabled:
            (self.runtime.log_dir / "profiles" / mode).mkdir(parents=True, exist_ok=True)
        if profiling.is_nsys and profiling_selects_process:
            nsys_output = (
                f"{self.runtime.container_log_dir}/profiles/{mode}/"
                f"{leader.node}_{mode}_w{index}_profile_rank%q{{SLURM_PROCID}}"
            )
            nsys_prefix = profiling.get_nsys_prefix(
                nsys_output, frontend_type=self.config.frontend.type, backend_type=backend.type
            )

        # Build command using backend's method
        cmd = backend.build_worker_command(
            process=leader,
            endpoint_processes=endpoint_processes,
            runtime=self.runtime,
            frontend_type=self.config.frontend.type,
            nsys_prefix=nsys_prefix,
            dump_config_path=config_dump,
            profiling=profiling if profiling_selects_process else None,
        )

        automatic_nsys = getattr(self.config, "observability_nsys_enabled", False) is True
        nsys_env: dict[str, str] = {}
        if automatic_nsys:
            cmd, nsys_env = wrap_observability_nsys(
                cmd,
                config=self.config,
                log_dir=self.runtime.log_dir,
                report_name=f"{mode}/{leader.node}_{mode}_w{index}_profile_rank%q{{SLURM_PROCID}}",
                ranks=total_gpus,
            )

        # Worker environment variables
        env_to_set = self._worker_environment_defaults(leader)

        # Add OTEL env vars (before mode-specific env so OTEL_SERVICE_NAME can be overridden)
        env_to_set.update(build_otel_env(self.config.observability, mode))
        env_to_set.update(nsys_env)

        # Add mode-specific environment variables from backend
        env_to_set.update(self._get_worker_environment_for_mode(mode))

        # Add config environment variables
        env_to_set.update(self.runtime.environment)

        if backend.type == "trtllm" and leader.trtllm_dist_init_port is not None:
            # Enroot may infer rank 0 from the sorted step nodelist, which
            # differs from our rank order for workers sharing a partial node.
            env_to_set.setdefault("MASTER_ADDR", get_hostname_ip(leader.node, self.runtime.network_interface))
            env_to_set.setdefault("MASTER_PORT", str(leader.trtllm_dist_init_port))

        # Native TRT-LLM KV-event subscribers need routable publisher hosts for
        # multi-node endpoints.  Dynamo can otherwise fall back to
        # SLURM_STEP_NODELIST, but that step-scoped variable is not guaranteed to
        # be available inside every container-launch path.  Set the endpoint's
        # nodes explicitly, while preserving a recipe-provided override.
        if (
            backend.type == "trtllm"
            and len(endpoint_nodes) > 1
            and env_to_set.get("DYN_TRTLLM_PUBLISH_KV_EVENTS", "").lower() == "true"
        ):
            env_to_set.setdefault("DYN_TRTLLM_KV_EVENT_HOSTS", ",".join(endpoint_nodes))

        force_mask = self.config.dynamo.sidecar and backend.type == "vllm"
        node_gpu_setup = ""
        if force_mask or backend.should_set_visible_devices():
            if any(p.gpu_indices != leader.gpu_indices for p in endpoint_processes):
                # One srun covers every node of the endpoint, so the mask is chosen per node at exec time.
                mask_env = self.runtime.visible_devices_env
                branches = " ".join(
                    f"{shlex.quote(p.node)}) export {mask_env}={shlex.quote(p.cuda_visible_devices)} ;;"
                    for p in endpoint_processes
                )
                node_gpu_setup = f'case "$SLURMD_NODENAME" in {branches} *) exit 1 ;; esac'
            else:
                env_to_set.update(self._visible_device_environment(leader))

        # Add mooncake worker env vars if configured. For MPI-style endpoint
        # launching we use the leader node's IP: mooncake's per-worker hostname
        # is fundamentally per-process, but TRTLLM-style launching uses one srun
        # for the whole endpoint, so leader IP is the best we can do.
        if backend.mooncake_kv_store is not None:
            local_hostname = env_to_set.get("MOONCAKE_LOCAL_HOSTNAME") or get_hostname_ip(
                leader.node, self.runtime.network_interface
            )
            env_to_set.update(backend.get_mooncake_worker_env(self.runtime.infra_node_ip, local_hostname))

        # Add profiling environment variables after the worker environment.
        if profiling.enabled and profiling_selects_process:
            profile_dir = str(self.runtime.container_log_dir / "profiles")
            env_to_set.update(profiling.get_env_vars(mode, profile_dir))

        self._apply_kvbm_endpoint_env(env_to_set, endpoint_processes)

        # Log env vars in the format: VAR=value VAR2=value2
        env_str = " ".join(f"{k}={v}" for k, v in sorted(env_to_set.items()))
        logger.info("Env: %s", env_str)
        logger.info("Command: %s", shlex.join(cmd))
        logger.info("Log: %s", worker_log)
        if profiling.enabled:
            logger.info("Profiling: %s mode", profiling.type)

        # Build bash preamble (setup script + dynamo install + fingerprint)
        bash_preamble = self._build_worker_preamble()
        if profiling_selects_process and profiling.is_nsys:
            bash_preamble = _append_preamble(
                bash_preamble,
                _nsys_library_path_preamble(profiling.nsys_library_paths),
            )
        fp_cmd = generate_capture_script(f"{self.runtime.container_log_dir}/fingerprint_{mode}_w{index}.json")
        # Keep fingerprint failures non-fatal, but do not let its `|| true`
        # mask failures from setup/dynamo install commands before it.
        fp_cmd = f"( {fp_cmd} )"
        bash_preamble = f"{bash_preamble} && {fp_cmd}" if bash_preamble else fp_cmd

        if node_gpu_setup:
            bash_preamble = f"{node_gpu_setup} && {bash_preamble}" if bash_preamble else node_gpu_setup

        # Repeated hosts preserve each node's exact rank count and ordering.
        task_nodes = [p.node for p in endpoint_processes for _ in p.gpu_indices]
        task_counts = [len(p.gpu_indices) for p in endpoint_processes]
        srun_options = {**self.runtime.srun_options, **self.config.roles[mode].srun_options}
        srun_options["ntasks-per-node"] = str(max(task_counts))
        if num_nodes > 1:
            # Every multi-node endpoint, uniform or not, lays its ranks out explicitly:
            # one host per task, in srt-slurm's node order, under --distribution=arbitrary.
            # A plain --nodelist only names the nodes; Slurm may still order the tasks
            # by its own topology, putting rank 0 on a node other than the leader whose
            # IP is MASTER_ADDR, and the rendezvous then waits forever.
            configured = srun_options.get("distribution")
            if configured not in (None, "arbitrary"):
                logger.warning(
                    "%s worker %d: srun_options.distribution=%r is replaced by 'arbitrary'; "
                    "a multi-node endpoint needs the explicit per-rank host list",
                    mode,
                    index,
                    configured,
                )
            srun_options["distribution"] = "arbitrary"
            endpoint_nodes = task_nodes

        # Get srun config from backend
        srun_config = backend.get_srun_config()
        if srun_config.kill_on_bad_exit:
            # One task exiting non-zero (a follower rank under the rank-zero
            # sidecar, or a launcher whose engine died) ends the whole endpoint
            # step instead of leaving the other ranks up with no engine.
            srun_options["kill-on-bad-exit"] = "1"

        step_name = worker_step_name(mode, index, leader.node, attempt)
        proc = start_srun_process(
            command=cmd,
            nodes=num_nodes,
            ntasks=total_gpus,
            nodelist=endpoint_nodes,
            output=str(worker_log),
            container_image=(
                self.config.worker_container_for_role(mode)
                if mode in self.config.role_containers
                else str(self.runtime.container_image)
            ),
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
            bash_preamble=bash_preamble,
            srun_export_env=CONTAINER_REMAP_ROOT_EXPORT if installs_dynamo(self.config) else None,
            mpi=srun_config.mpi,
            oversubscribe=srun_config.oversubscribe,
            cpu_bind=srun_config.cpu_bind,
            # Endpoint (MPI) workers were the only srun path that dropped the
            # recipe-level srun_options; the per-process worker, benchmark and
            # telemetry paths all forward it. Needed so a cluster can express
            # per-rank CPU/NUMA binding, which srun_config.cpu_bind cannot.
            srun_options=srun_options,
            het_group=leader.het_group,
            step_name=step_name,
        )

        return ManagedProcess(
            name=step_name,
            popen=proc,
            log_file=worker_log,
            node=leader.node,
            critical=self.config.topology.worker_critical(mode),
            # Signal every MPI task; profiler wrappers stop capture before the app.
            terminate_timeout=(
                self.config.observability.nsys.terminate_timeout if automatic_nsys else WORKER_TERMINATE_TIMEOUT_SECONDS
            ),
            signal_full=not automatic_nsys,
            step_name=step_name,
            fatal_log_patterns=self._fatal_log_patterns(mode),
        )

    def _wait_for_worker_ready(self, leader: "Process") -> None:
        """Wait for a single endpoint worker to become ready before starting the next.

        For trtllm_serve: polls the worker's per-process HTTP health endpoint (http_port).
        For dynamo.trtllm: polls the dynamo system status server on sys_port. The dynamo
        runtime starts an axum HTTP server on DYN_SYSTEM_PORT (which we set to sys_port)
        and exposes GET /health → 200 {"status":"ready"} once the model is loaded and the
        NATS/TCP request endpoint is registered.
        """
        health_cfg = self.config.health_check
        # The frontend knows which port a worker reports its own health on:
        # trtllm-serve's OpenAI port, or DYN_SYSTEM_PORT (set to sys_port in
        # start_endpoint_worker) where the Dynamo runtime serves /health.
        port = get_frontend(self.config.frontend.type).worker_ready_port(leader)

        logger.info(
            "Sequential node start: waiting for worker %s:%d to be ready",
            leader.node,
            port,
        )
        if not wait_for_health(
            leader.node,
            port,
            max_attempts=health_cfg.max_attempts,
            interval=health_cfg.interval_seconds,
        ):
            raise RuntimeError(f"Sequential node start: worker on {leader.node}:{port} did not become healthy")

    def worker_endpoint_groups(self, processes: list["Process"] | None = None) -> dict[EndpointKey, list["Process"]]:
        """The physical processes of every restart unit, keyed by ``(mode, index, engine)``, in launch order.

        ``processes`` defaults to every backend process; ``_start_workers`` passes
        one role's share so a per-role engine groups only its own workers. The
        engine is 0 for every process unless the backend runs shadow engines
        (vLLM ``engine.failover``), whose processes carry ``engine_id``; each
        engine of a worker is launched, supervised, and relaunched on its own.
        """
        grouped: dict[EndpointKey, list[Process]] = defaultdict(list)
        for process in self.backend_processes if processes is None else processes:
            grouped[(process.endpoint_mode, process.endpoint_index, getattr(process, "engine_id", 0))].append(process)
        return dict(grouped)

    def relaunch_endpoint(self, endpoint_processes: list["Process"], *, attempt: int) -> "Iterable[ManagedProcess]":
        """Yield each launched step so the supervisor tracks it before the next rank starts."""
        # The role's own engine decides the launch strategy (roles may run different engines).
        backend = self.config.backend_for_role(endpoint_processes[0].endpoint_mode)
        if backend.get_srun_config().launch_per_endpoint:
            yield self.start_endpoint_worker(endpoint_processes, attempt=attempt)
        else:
            for process in endpoint_processes:
                yield self.start_worker(process, endpoint_processes, attempt=attempt)

    def worker_ready_probe(self, endpoint_processes: list["Process"]) -> tuple[str, int] | None:
        """Where a relaunched endpoint answers ``GET /health`` once it serves (WorkerLauncher protocol).

        Under the Dynamo frontend that is the runtime's system status server on
        ``DYN_SYSTEM_PORT`` (200 once the model is loaded and the endpoint is
        registered); every other frontend talks to the engine's own HTTP port.
        """
        leader = endpoint_processes[0]
        frontend = get_frontend(self.config.frontend.type)
        port = (
            frontend.worker_ready_port(leader)
            if frontend.worker_launch == "dynamo"
            else frontend.worker_endpoint_port(leader, self.config, self.runtime)
        )
        if port is None or port <= 0:
            return None
        return get_hostname_ip(leader.node, self.runtime.network_interface), port

    def build_worker_supervisor(self, registry: ProcessRegistry, stop_event: threading.Event) -> WorkerSupervisor:
        """The reconcile loop that relaunches workers per ``roles.<role>.restart``; a no-op until ``track_workers``."""
        health = self.config.health_check
        return WorkerSupervisor(
            registry=registry,
            stop_event=stop_event,
            launcher=self,
            log_dir=self.runtime.log_dir,
            # A relaunched worker gets the same budget to come up as the initial health gate.
            ready_timeout=float(health.max_attempts * health.interval_seconds),
        )

    def track_workers(self, supervisor: WorkerSupervisor, worker_procs: NamedProcesses) -> None:
        """Hand the launched workers to the supervisor; roles with ``restart: never`` are skipped."""
        if not worker_procs:
            return  # nothing launched, nothing to supervise
        supervisor.track(self.worker_endpoint_groups(), worker_procs.keys(), self.config.topology.worker_restart)

    def start_all_workers(self) -> NamedProcesses:
        """Launch each role using its engine's existing launch strategy."""
        if not self.config.role_backends:
            return self._start_workers(self.backend, self.backend_processes)
        result: NamedProcesses = {}
        for role, backend in self.config.active_role_backends():
            processes = [p for p in self.backend_processes if p.endpoint_mode == role]
            result.update(self._start_workers(backend, processes))
        return result

    def _start_workers(self, backend: Any, processes: list["Process"]) -> NamedProcesses:
        logger.info("Starting %s backend workers", backend.type)

        # Check if backend uses MPI-style per-endpoint launching
        srun_config = backend.get_srun_config()
        launch_per_endpoint = srun_config.launch_per_endpoint

        grouped = self.worker_endpoint_groups(processes)

        result: NamedProcesses = {}

        if launch_per_endpoint:
            # MPI-style: one srun per endpoint (TRTLLM)
            concurrency = srun_config.sequential_node_start
            if concurrency:
                # Group endpoints by leader node; start in batches within each node
                # so that model loading on a shared node doesn't cause resource contention.
                # Different nodes start in parallel.
                by_node: dict[str, list[list[Process]]] = defaultdict(list)
                for ep_procs in grouped.values():
                    by_node[ep_procs[0].node].append(ep_procs)

                def start_node_workers(node: str, node_groups: list) -> list:
                    if len(node_groups) == 1:
                        return [self.start_endpoint_worker(node_groups[0])]
                    logger.info(
                        "Sequential node start: %d workers share node %s, starting %d at a time",
                        len(node_groups),
                        node,
                        concurrency,
                    )
                    managed_list = []
                    for batch_start in range(0, len(node_groups), concurrency):
                        batch = node_groups[batch_start : batch_start + concurrency]
                        batch_managed = [self.start_endpoint_worker(ep_procs) for ep_procs in batch]
                        managed_list.extend(batch_managed)
                        if batch_start + concurrency < len(node_groups):
                            for ep_procs in batch:
                                self._wait_for_worker_ready(ep_procs[0])
                    return managed_list

                with ThreadPoolExecutor(max_workers=len(by_node)) as executor:
                    futures = {
                        executor.submit(start_node_workers, node, node_groups): node
                        for node, node_groups in by_node.items()
                    }
                    for future in as_completed(futures):
                        for managed in future.result():
                            result[managed.name] = managed
            else:
                for endpoint_processes in grouped.values():
                    managed = self.start_endpoint_worker(endpoint_processes)
                    result[managed.name] = managed
        else:
            # Per-process: one srun per node (SGLang, vLLM). Under backend.failover a
            # worker's processes on a node are engine 0 and its shadows; the gms
            # service that owns their weights came up in the before_workers phase.
            for endpoint_processes in grouped.values():
                for process in endpoint_processes:
                    managed = self.start_worker(process, endpoint_processes)
                    result[managed.name] = managed

        logger.info("Started %d worker processes", len(result))
        return result
