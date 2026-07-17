#!/usr/bin/env python3
"""Fail-open watchdog for wedged Mooncake external-KV loads.

``store.batch_get_into_multi_buffers`` has no deadline. Under high in-flight
load a GET can wedge inside the RDMA client, so ``set_finished_request`` never
runs and the request remains in WAITING_FOR_REMOTE_KVS indefinitely.

This patch edits the installed Mooncake store worker to add:

1. ISSUE/BEGIN/DONE ledger logs for each asynchronous load.
2. A deadline in ``get_finished``. Loads older than
   ``VLLM_MOONCAKE_LOAD_DEADLINE_S`` are reported with all blocks invalid so
   the scheduler recomputes the request instead of hanging.
3. Exception hardening in the receive thread so a load failure cannot strand
   a request silently.

The patch is anchor-checked against the vLLM 515d6e9 image family. A late
completion after the deadline is suppressed to avoid reporting it twice.
"""

import py_compile
import sys
from pathlib import Path

MARKER = "MC-LOAD WATCHDOG"
TARGETS = [
    Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"
    ),
    Path(
        "/opt/venv/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"
    ),
]

A1 = (
    "            load_spec.token_len = load_spec.kvpool_cached_tokens\n            self.recv_request_queue.put(request)"
)
A1_NEW = (
    "            load_spec.token_len = load_spec.kvpool_cached_tokens\n"
    "            # [mc-load-watchdog] issue ledger\n"
    '            if not hasattr(self, "_mcwd_inflight"):\n'
    "                self._mcwd_inflight = {}\n"
    "            _mcwd_bids = _mcwd_flatten_block_ids(request.block_ids)\n"
    "            self._mcwd_inflight[request.req_id] = (time.perf_counter(), _mcwd_bids)\n"
    "            _mcwd_unsuppress(request.req_id)\n"
    "            logger.info(\n"
    '                "MC-load ISSUE req=%s blocks=%d qdepth=%d",\n'
    "                request.req_id, len(_mcwd_bids), self.recv_request_queue.qsize(),\n"
    "            )\n"
    "            self.recv_request_queue.put(request)"
)

A2 = (
    "        done_recving: set[str] = set()\n"
    "        if self.load_async:\n"
    "            for recv_thread in self.kv_recv_threads:\n"
    "                done_recving |= recv_thread.get_and_clear_finished_requests()"
)
A2_NEW = A2 + (
    "\n"
    "        # [mc-load-watchdog] fail-open for loads stuck past the deadline\n"
    '        if getattr(self, "_mcwd_inflight", None):\n'
    "            for _rid in done_recving:\n"
    "                self._mcwd_inflight.pop(_rid, None)\n"
    "            _mcwd_now = time.perf_counter()\n"
    "            _mcwd_expired = [\n"
    "                (_rid, _mcwd_now - _t, _bids)\n"
    "                for _rid, (_t, _bids) in self._mcwd_inflight.items()\n"
    "                if _mcwd_now - _t > _MCWD_DEADLINE_S\n"
    "            ]\n"
    "            for _rid, _age, _bids in _mcwd_expired:\n"
    "                del self._mcwd_inflight[_rid]\n"
    "                _mcwd_suppress(_rid)\n"
    "                if self.kv_recv_threads:\n"
    "                    self.kv_recv_threads[0]._add_load_error_block_ids(_bids)\n"
    "                done_recving.add(_rid)\n"
    "                logger.error(\n"
    '                    "MC-load WATCHDOG req=%s stuck %.0fs > %.0fs: force-reported "\n'
    '                    "with %d blocks invalid (request will recompute)",\n'
    "                    _rid, _age, _MCWD_DEADLINE_S, len(_bids),\n"
    "                )"
)

TAIL = """

# ===== MC-LOAD WATCHDOG (fail-open for wedged external KV loads) =====
# Injected by configs/patches/mooncake_load_watchdog.py.
import os as _mcwd_os

_MCWD_DEADLINE_S = float(_mcwd_os.environ.get("VLLM_MOONCAKE_LOAD_DEADLINE_S", "120") or "120")
_MCWD_LOCK = threading.Lock()
_MCWD_SUPPRESSED: set[str] = set()


def _mcwd_suppress(req_id: str) -> None:
    with _MCWD_LOCK:
        _MCWD_SUPPRESSED.add(req_id)


def _mcwd_unsuppress(req_id: str) -> None:
    with _MCWD_LOCK:
        _MCWD_SUPPRESSED.discard(req_id)


def _mcwd_consume(req_id: str) -> bool:
    with _MCWD_LOCK:
        if req_id in _MCWD_SUPPRESSED:
            _MCWD_SUPPRESSED.discard(req_id)
            return True
        return False


def _mcwd_flatten_block_ids(block_ids) -> list:
    if block_ids and isinstance(block_ids[0], (list, tuple)):
        out = []
        for _group in block_ids:
            out.extend(_group)
        return out
    return list(block_ids or [])


_mcwd_orig_recv_handle = KVCacheStoreRecvingThread._handle_request


def _mcwd_recv_handle(self, req_meta):
    req_id = req_meta.req_id
    if _mcwd_consume(req_id):
        logger.warning(
            "MC-load SKIP req=%s: watchdog already force-reported; dropping queued load",
            req_id,
        )
        self.request_queue.task_done()
        return
    logger.info("MC-load BEGIN req=%s thread=%s", req_id, self.name)
    try:
        _mcwd_orig_recv_handle(self, req_meta)
        logger.info("MC-load DONE req=%s thread=%s", req_id, self.name)
    except Exception:
        logger.exception(
            "MC-load FAIL req=%s thread=%s: fail-open reporting with all blocks invalid",
            req_id, self.name,
        )
        try:
            self._add_load_error_block_ids(_mcwd_flatten_block_ids(req_meta.block_ids))
        finally:
            self.set_finished_request(req_id)
            self.request_queue.task_done()


KVCacheStoreRecvingThread._handle_request = _mcwd_recv_handle

_mcwd_orig_recv_set_finished = KVCacheStoreRecvingThread.set_finished_request


def _mcwd_recv_set_finished(self, req_id):
    if _mcwd_consume(req_id):
        logger.warning(
            "MC-load LATE req=%s: completion after watchdog force-report; dropping duplicate",
            req_id,
        )
        return
    _mcwd_orig_recv_set_finished(self, req_id)


KVCacheStoreRecvingThread.set_finished_request = _mcwd_recv_set_finished
"""


def patch(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text()
    if MARKER in text:
        print(f"mc-load watchdog already applied: {path}")
        return True
    for name, anchor in (("A1", A1), ("A2", A2)):
        count = text.count(anchor)
        if count != 1:
            raise RuntimeError(f"anchor {name} matched {count} times (expected 1) in {path}")
    if "class KVCacheStoreRecvingThread" not in text:
        raise RuntimeError(f"KVCacheStoreRecvingThread not found in {path}")
    text = text.replace(A1, A1_NEW, 1)
    text = text.replace(A2, A2_NEW, 1)
    text += TAIL
    path.write_text(text)
    py_compile.compile(str(path), doraise=True)
    print(f"mc-load watchdog applied: {path}")
    return True


if __name__ == "__main__":
    if not any(patch(target) for target in TARGETS):
        print("ERROR: Mooncake store worker.py not found", file=sys.stderr)
        sys.exit(1)
