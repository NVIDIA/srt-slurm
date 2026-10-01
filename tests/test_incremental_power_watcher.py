# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time

from srtctl.analysis.incremental_power import IncrementalPowerWatcher


class FakeEmitter:
    def __init__(self, raises=False):
        self.calls = 0
        self.raises = raises
        self._lock = threading.Lock()

    def poll(self):
        with self._lock:
            self.calls += 1
        if self.raises:
            raise RuntimeError("boom")
        return ()


def test_watcher_polls_repeatedly_then_stops():
    emitter = FakeEmitter()
    watcher = IncrementalPowerWatcher(emitter, tick_seconds=0.02, join_timeout_seconds=3.0)

    watcher.start()
    time.sleep(0.2)
    watcher.stop_and_finalize()
    calls_at_stop = emitter.calls

    assert calls_at_stop >= 2
    time.sleep(0.1)
    assert emitter.calls == calls_at_stop  # thread really stopped


def test_watcher_runs_a_final_poll_on_stop():
    emitter = FakeEmitter()
    watcher = IncrementalPowerWatcher(emitter, tick_seconds=3600.0, join_timeout_seconds=3.0)

    watcher.start()
    watcher.stop_and_finalize()

    # One immediate poll from the loop plus the explicit final pass. NOTE:
    # with tick_seconds=3600, the loop's own first poll already satisfies
    # ">= 1" on its own, so this assertion cannot by itself prove the final
    # pass in stop_and_finalize() runs -- see test_stop_without_start_is_safe
    # below for the deterministic, timing-independent assertion of that.
    assert emitter.calls >= 1


def test_a_raising_emitter_is_absorbed():
    emitter = FakeEmitter(raises=True)
    watcher = IncrementalPowerWatcher(emitter, tick_seconds=0.02, join_timeout_seconds=3.0)

    watcher.start()
    time.sleep(0.1)
    watcher.stop_and_finalize()  # must not raise

    assert emitter.calls >= 2  # kept ticking despite the exception


def test_stop_without_start_is_safe():
    # With no thread ever started (self._thread is None), stop_and_finalize()
    # skips the join and the ONLY poll that can possibly happen is the final
    # pass -- so asserting calls == 1 here deterministically proves that pass
    # runs, with zero dependence on timing or tick_seconds.
    emitter = FakeEmitter()
    watcher = IncrementalPowerWatcher(emitter, tick_seconds=0.02, join_timeout_seconds=3.0)

    watcher.stop_and_finalize()  # must not raise

    assert emitter.calls == 1
