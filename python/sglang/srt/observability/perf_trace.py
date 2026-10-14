"""Per-request timeline probes for PD-disaggregation profiling.

Enabled by SGLANG_PERF_TRACE=1. Each process appends JSON lines to
<SGLANG_PERF_TRACE_DIR or $LOG_DIR/profile>/perf-<host>-<pid>.jsonl.
Wall time locates events; monotonic time measures intervals on the same host.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import socket
import threading
import time
from collections import deque

import numpy as np

from sglang.srt.environ import envs

ENABLED = envs.SGLANG_PERF_TRACE.get()

_lock = threading.Lock()
_state = {"pid": None, "fd": None, "role": "unknown", "rank": -1}
_host = socket.gethostname()
_failed = False
_gc_pauses = deque(maxlen=4096)
_sched_getcpu = None
if ENABLED:
    try:
        import ctypes

        _libc = ctypes.PyDLL(None)  # Keep the GIL for this nonblocking probe.
        _sched_getcpu = _libc.sched_getcpu
        _sched_getcpu.argtypes = []
        _sched_getcpu.restype = ctypes.c_int
    except Exception:
        _sched_getcpu = None


def _cpu_id() -> int | None:
    if _sched_getcpu is None:
        return None
    try:
        cpu = _sched_getcpu()
        return cpu if cpu >= 0 else None
    except Exception:
        return None


def set_role(role: str, rank: int = -1) -> None:
    _state["role"] = role
    _state["rank"] = rank


def runs(indices) -> int:
    """Number of contiguous index runs; 1 means fully contiguous."""
    arr = np.asarray(indices)
    if arr.ndim != 1 or arr.size <= 1:
        return int(arr.size)
    return int(np.count_nonzero(np.diff(arr) != 1) + 1)


def lengths_summary(lengths, repeats: int = 1) -> dict:
    values = np.asarray(lengths, dtype=np.uint64)
    sizes, counts = np.unique(values, return_counts=True)
    return {
        "n_desc": int(values.size) * repeats,
        "bytes": int(values.sum()) * repeats,
        "length_hist": [[int(n), int(c) * repeats] for n, c in zip(sizes[:64], counts[:64])],
        "length_hist_truncated": len(sizes) > 64,
    }


def _fd() -> int:
    pid = os.getpid()
    if _state["pid"] != pid:
        directory = envs.SGLANG_PERF_TRACE_DIR.get() or os.path.join(
            os.environ.get("LOG_DIR", "/opt/sglang-perf"), "profile"
        )
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"perf-{_host}-{pid}.jsonl")
        _state["fd"] = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        _state["pid"] = pid
    return _state["fd"]


def _default(obj):
    item = getattr(obj, "item", None)
    return item() if callable(item) else str(obj)


def _emit(event: str, **fields) -> None:
    global _failed
    if _failed:
        return
    record = {
        "ev": event,
        "t": time.time(),
        "mono_ns": time.monotonic_ns(),
        "thread_cpu_ns": time.thread_time_ns(),
        "host": _host,
        "pid": os.getpid(),
        "role": _state["role"],
        "rank": _state["rank"],
        "tid": threading.get_native_id(),
        "cpu": _cpu_id(),
    }
    record.update(fields)
    if event != "gc.pause":
        while True:
            try:
                pause = _gc_pauses.popleft()
            except IndexError:
                break
            _emit("gc.pause", **pause)
    try:
        line = (json.dumps(record, separators=(",", ":"), default=_default) + "\n").encode()
        with _lock:
            fd = _fd()
            while line:
                written = os.write(fd, line)
                if written <= 0:
                    raise OSError("perf trace write made no progress")
                line = line[written:]
    except Exception:
        _failed = True
        logging.getLogger(__name__).exception("Disabling failed perf trace writer")


if ENABLED:
    emit = _emit
    _gc_starts = {}

    def _gc_trace(phase, info):
        generation = info["generation"]
        if phase == "start":
            _gc_starts[generation] = time.monotonic_ns()
        elif phase == "stop":
            start = _gc_starts.pop(generation, None)
            if start is not None:
                end = time.monotonic_ns()
                _gc_pauses.append(
                    {
                        "t": time.time(),
                        "mono_ns": end,
                        "thread_cpu_ns": time.thread_time_ns(),
                        "tid": threading.get_native_id(),
                        "cpu": _cpu_id(),
                        "start_mono_ns": start,
                        "duration_ns": end - start,
                        "generation": generation,
                        "collected": info["collected"],
                        "uncollectable": info["uncollectable"],
                    }
                )

    gc.callbacks.append(_gc_trace)
else:

    def emit(event: str, **fields) -> None:
        return None
