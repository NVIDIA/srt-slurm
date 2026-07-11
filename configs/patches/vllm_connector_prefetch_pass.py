#!/usr/bin/env python3
"""SRT patch: connector prefetch pass (defer-ahead) for external KV loads.

Root cause (py-spy job 8934, DSV4 PD on GB200): lookup submission, async-load
start and Deferred promotion all live inside the scheduler's waiting loop,
which is gated on ``token_budget > 0``. On a busy prefill engine the running
chunk eats the whole budget most steps, so the connector state machine
advances ~1 request per step: requests sit multi-second in
WAITING_FOR_REMOTE_KVS while recv threads and the wire are idle.

This patch adds a budget-independent prefetch pass that runs right before the
waiting loop: it scans the first K WAITING requests in arrival order, submits
their external-KV lookups, and starts their async loads (allocate + defer)
early so the load overlaps queue wait.

Guardrails (see review 2026-07-10):
- ``VLLM_CONNECTOR_PREFETCH_DEPTH`` (default 0 = patch inert).
- ``VLLM_CONNECTOR_PREFETCH_KV_CAP`` (default 0.5): stop when KV usage is
  above this; with scheduler_reserve_full_isl=true prefetched requests pin
  nearly their full ISL, so this cap protects normal admission.
- ``VLLM_CONNECTOR_PREFETCH_MAX_ISL`` (default 300000): never prefetch giant
  requests; one p99 request must not blockade the pool.
- FCFS policy only; requests with LoRA / encoder / mm inputs are skipped.
- Lookup results are cached in LookupKeyClient with a TTL
  (``VLLM_MOONCAKE_LOOKUP_CACHE_TTL_S``, default 2.0) so re-probing a request
  does not re-issue the ~1MB lookup RPC every step (the future was previously
  deleted on first result; this also fixes the pre-existing re-lookup on
  allocation-failure retries).

Intended to run inside the benchmark container after vLLM is installed
(same convention as vllm_mooncake_store_async_fix.py). ``--root`` allows
offline dry-runs against a source checkout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "SRT patch: connector prefetch pass"


def _module_path(module_name: str, root: Path | None) -> Path:
    if root is not None:
        return root / (module_name.replace(".", "/") + ".py")
    import importlib

    module = importlib.import_module(module_name)
    if module.__file__ is None:
        raise RuntimeError(f"module {module_name} has no __file__")
    return Path(module.__file__)


def _replace_once(path: Path, old: str, new: str, what: str) -> bool:
    text = path.read_text()
    if new in text:
        print(f"already applied: {what}")
        return False
    if old not in text:
        raise RuntimeError(f"anchor not found for {what} in {path}")
    path.write_text(text.replace(old, new, 1))
    print(f"applied: {what}")
    return True


# ---------------------------------------------------------------------------
# 1) scheduler.py: call site — run the prefetch pass before the waiting loop
# ---------------------------------------------------------------------------

CALLSITE_OLD = """        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
"""

CALLSITE_NEW = """        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)

            # SRT patch: connector prefetch pass (defer-ahead) — start async
            # external-KV loads for queued requests independent of the token
            # budget so load latency overlaps queue wait.
            self._connector_prefetch_pass()

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
"""

# ---------------------------------------------------------------------------
# 2) scheduler.py: the prefetch pass method
# ---------------------------------------------------------------------------

METHOD_ANCHOR = """    def _inflight_prefill_reserved_blocks(self) -> int:
"""

METHOD_NEW = '''    def _connector_prefetch_pass(self) -> None:
        """SRT patch: connector prefetch pass (defer-ahead).

        Scan the first K WAITING requests in arrival order and, for those
        with an external KV hit, allocate blocks and start the async load
        now (WAITING_FOR_REMOTE_KVS) instead of waiting for the
        budget-gated loop to reach them. Mirrors the load_kv_async branch
        of the main waiting loop; consumes no token budget.
        """
        import os

        depth = int(os.getenv("VLLM_CONNECTOR_PREFETCH_DEPTH", "0"))
        if (
            depth <= 0
            or self.connector is None
            or not self.waiting
            or self.policy != SchedulingPolicy.FCFS
        ):
            return
        kv_cap = float(os.getenv("VLLM_CONNECTOR_PREFETCH_KV_CAP", "0.5"))
        max_isl = int(os.getenv("VLLM_CONNECTOR_PREFETCH_MAX_ISL", "300000"))

        keep: list[Request] = []
        prefetched: list[Request] = []
        scanned = 0
        while self.waiting and scanned < depth:
            request = self.waiting.pop_request()
            scanned += 1
            eligible = (
                request.status == RequestStatus.WAITING
                and request.num_computed_tokens == 0
                and not (self.lora_config and request.lora_request)
                and not request.has_encoder_inputs
                and not request.mm_features
                and not (0 < max_isl < request.num_tokens)
                and self.kv_cache_manager.usage < kv_cap
            )
            if not eligible:
                keep.append(request)
                continue

            new_computed_blocks, num_new_local_computed_tokens = (
                self.kv_cache_manager.get_computed_blocks(request)
            )
            ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
                request, num_new_local_computed_tokens
            )
            if not ext_tokens or not load_kv_async:
                # None: async lookup submitted, retry next step.
                # 0 / sync: nothing to prefetch, normal path handles it.
                keep.append(request)
                continue

            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                0,
                num_new_computed_tokens=num_new_local_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=(
                    0 if self.use_eagle else self.num_lookahead_tokens
                ),
                num_external_computed_tokens=ext_tokens,
                delay_cache_blocks=True,
                full_sequence_must_fit=self.scheduler_reserve_full_isl,
                reserved_blocks=self._inflight_prefill_reserved_blocks(),
                has_scheduled_reqs=bool(self.running),
            )
            if new_blocks is None:
                # KV pressure — stop prefetching this step, keep queue order.
                keep.append(request)
                break

            self.connector.update_state_after_alloc(
                request,
                self.kv_cache_manager.get_blocks(request.request_id),
                ext_tokens,
            )
            if request.prefill_stats is not None:
                request.prefill_stats.set(
                    num_prompt_tokens=request.num_prompt_tokens,
                    num_local_cached_tokens=num_new_local_computed_tokens,
                    num_external_cached_tokens=ext_tokens,
                )
            request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            request.num_computed_tokens = (
                num_new_local_computed_tokens + ext_tokens
            )
            self._inflight_prefills.add(request)
            prefetched.append(request)

        # Restore arrival order for requests left in the waiting queue;
        # prefetched requests join skipped_waiting like any deferred request.
        for request in reversed(keep):
            self.waiting.prepend_request(request)
        for request in prefetched:
            self.skipped_waiting.add_request(request)

    def _inflight_prefill_reserved_blocks(self) -> int:
'''

# ---------------------------------------------------------------------------
# 3) mooncake store/worker.py: LookupKeyClient resolved-value cache (TTL)
# ---------------------------------------------------------------------------

INIT_OLD = """        # Async lookup support
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="MooncakeLookupClient"
        )
        self.futures: dict[str, Future[int]] = {}
"""

INIT_NEW = """        # Async lookup support
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="MooncakeLookupClient"
        )
        self.futures: dict[str, Future[int]] = {}
        # SRT patch: connector prefetch pass — cache resolved lookups so
        # re-probing a still-waiting request does not re-issue the RPC.
        self._resolved_lookups: dict[str, tuple[float, int]] = {}
        self._resolved_lookups_ttl_s = float(
            os.getenv("VLLM_MOONCAKE_LOOKUP_CACHE_TTL_S", "2.0")
        )
"""

LOOKUP_OLD = """        future = self.futures.get(req_id)
        if future is None:
            future = self.executor.submit(
                self._lookup,
                token_len,
                list(block_hashes),
                session_id,
            )
            self.futures[req_id] = future
        if non_block and not future.done():
            return None
        try:
            return future.result()
        except Exception as e:
            logger.error("Async Mooncake lookup failed for %s: %s", req_id, e)
            return 0
        finally:
            del self.futures[req_id]
"""

LOOKUP_NEW = """        # SRT patch: connector prefetch pass — serve recent results from
        # cache instead of re-issuing the lookup RPC on every re-probe.
        cached = self._resolved_lookups.get(req_id)
        if cached is not None:
            cached_at, cached_value = cached
            if time.monotonic() - cached_at <= self._resolved_lookups_ttl_s:
                return cached_value
            del self._resolved_lookups[req_id]

        future = self.futures.get(req_id)
        if future is None:
            future = self.executor.submit(
                self._lookup,
                token_len,
                list(block_hashes),
                session_id,
            )
            self.futures[req_id] = future
        if non_block and not future.done():
            return None
        try:
            result = future.result()
        except Exception as e:
            logger.error("Async Mooncake lookup failed for %s: %s", req_id, e)
            result = 0
        del self.futures[req_id]
        if len(self._resolved_lookups) > 4096:
            now = time.monotonic()
            self._resolved_lookups = {
                k: v
                for k, v in self._resolved_lookups.items()
                if now - v[0] <= self._resolved_lookups_ttl_s
            }
        self._resolved_lookups[req_id] = (time.monotonic(), result)
        return result
"""

DISCARD_OLD = """    def discard(self, req_id: str) -> None:
        \"\"\"Drop any cached/in-flight lookup for ``req_id`` (e.g. on abort).\"\"\"
        future = self.futures.pop(req_id, None)
        if future is not None:
            future.cancel()
"""

DISCARD_NEW = """    def discard(self, req_id: str) -> None:
        \"\"\"Drop any cached/in-flight lookup for ``req_id`` (e.g. on abort).\"\"\"
        # SRT patch: connector prefetch pass — drop the cached result too.
        self._resolved_lookups.pop(req_id, None)
        future = self.futures.pop(req_id, None)
        if future is not None:
            future.cancel()
"""

# Newer sprint-agentx fast images use a scheduler-local ZMQ lookup subprocess
# instead of the older ThreadPoolExecutor client. The prefetch pass still needs
# the same short-lived resolved-result cache to avoid repeated probe RPCs.
INIT_ZMQ_OLD = """        self._pending_since: dict[str, float] = {}
"""

INIT_ZMQ_NEW = """        self._pending_since: dict[str, float] = {}
        # SRT patch: connector prefetch pass — cache resolved lookups so
        # re-probing a still-waiting request does not re-issue the RPC.
        self._resolved_lookups: dict[str, tuple[float, int]] = {}
        self._resolved_lookups_ttl_s = float(
            os.getenv("VLLM_MOONCAKE_LOOKUP_CACHE_TTL_S", "2.0")
        )
"""

LOOKUP_ZMQ_CACHE_OLD = """        if not block_hashes or token_len <= 0:
            return 0
        if not self._lookup_alive():
"""

LOOKUP_ZMQ_CACHE_NEW = """        if not block_hashes or token_len <= 0:
            return 0
        # SRT patch: connector prefetch pass — serve recent results from
        # cache instead of re-issuing the lookup RPC on every re-probe.
        cached = self._resolved_lookups.get(req_id)
        if cached is not None:
            cached_at, cached_value = cached
            if time.monotonic() - cached_at <= self._resolved_lookups_ttl_s:
                return cached_value
            del self._resolved_lookups[req_id]
        if not self._lookup_alive():
"""

LOOKUP_ZMQ_RESULT_OLD = """        if req_id in self._results:
            return self._results.pop(req_id)
"""

LOOKUP_ZMQ_RESULT_NEW = """        if req_id in self._results:
            result = self._results.pop(req_id)
            if len(self._resolved_lookups) > 4096:
                now = time.monotonic()
                self._resolved_lookups = {
                    k: v
                    for k, v in self._resolved_lookups.items()
                    if now - v[0] <= self._resolved_lookups_ttl_s
                }
            self._resolved_lookups[req_id] = (time.monotonic(), result)
            return result
"""

LOOKUP_ZMQ_FINAL_OLD = """        return self._results.pop(req_id)
"""

LOOKUP_ZMQ_FINAL_NEW = """        result = self._results.pop(req_id)
        if len(self._resolved_lookups) > 4096:
            now = time.monotonic()
            self._resolved_lookups = {
                k: v
                for k, v in self._resolved_lookups.items()
                if now - v[0] <= self._resolved_lookups_ttl_s
            }
        self._resolved_lookups[req_id] = (time.monotonic(), result)
        return result
"""

DISCARD_ZMQ_OLD = """        self._pending.pop(req_id, None)
        self._pending_since.pop(req_id, None)
        self._results.pop(req_id, None)
"""

DISCARD_ZMQ_NEW = """        # SRT patch: connector prefetch pass — drop the cached result too.
        self._resolved_lookups.pop(req_id, None)
        self._pending.pop(req_id, None)
        self._pending_since.pop(req_id, None)
        self._results.pop(req_id, None)
"""


def _patch_worker_lookup_client(worker: Path) -> None:
    text = worker.read_text()
    if INIT_OLD in text or INIT_NEW in text:
        _replace_once(worker, INIT_OLD, INIT_NEW, "lookup client cache init")
        _replace_once(worker, LOOKUP_OLD, LOOKUP_NEW, "lookup resolved cache")
        _replace_once(worker, DISCARD_OLD, DISCARD_NEW, "lookup discard cache drop")
        return

    if INIT_ZMQ_OLD in text or INIT_ZMQ_NEW in text:
        _replace_once(worker, INIT_ZMQ_OLD, INIT_ZMQ_NEW, "lookup client cache init")
        _replace_once(worker, LOOKUP_ZMQ_CACHE_OLD, LOOKUP_ZMQ_CACHE_NEW,
                      "lookup resolved cache probe")
        _replace_once(worker, LOOKUP_ZMQ_RESULT_OLD, LOOKUP_ZMQ_RESULT_NEW,
                      "lookup resolved cache non-block result")
        _replace_once(worker, LOOKUP_ZMQ_FINAL_OLD, LOOKUP_ZMQ_FINAL_NEW,
                      "lookup resolved cache blocking result")
        _replace_once(worker, DISCARD_ZMQ_OLD, DISCARD_ZMQ_NEW,
                      "lookup discard cache drop")
        return

    raise RuntimeError(f"no supported LookupKeyClient shape in {worker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="patch a source tree at this root instead of the installed vllm",
    )
    args = parser.parse_args()

    sched = _module_path("vllm.v1.core.sched.scheduler", args.root)
    _replace_once(sched, CALLSITE_OLD, CALLSITE_NEW, "scheduler call site")
    _replace_once(sched, METHOD_ANCHOR, METHOD_NEW, "scheduler prefetch method")

    worker = _module_path(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker",
        args.root,
    )
    _patch_worker_lookup_client(worker)

    print("vllm_connector_prefetch_pass: all patches applied")


if __name__ == "__main__":
    main()
