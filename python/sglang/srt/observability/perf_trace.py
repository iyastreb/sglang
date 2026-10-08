"""Per-request timeline probes for PD-disaggregation profiling.

Enabled by SGLANG_PERF_TRACE=1. Each process appends JSON lines to
<SGLANG_PERF_TRACE_DIR or $LOG_DIR/profile>/perf-<host>-<pid>.jsonl.
Records carry wall-clock seconds so files from several nodes can be merged.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time

import numpy as np

from sglang.srt.environ import envs

ENABLED = envs.SGLANG_PERF_TRACE.get()

_lock = threading.Lock()
_state = {"pid": None, "fd": None, "role": "unknown", "rank": -1}


def set_role(role: str, rank: int = -1) -> None:
    _state["role"] = role
    _state["rank"] = rank


def runs(indices) -> int:
    """Number of contiguous index runs; 1 means fully contiguous."""
    arr = np.asarray(indices)
    if arr.ndim != 1 or arr.size <= 1:
        return int(arr.size)
    return int(np.count_nonzero(np.diff(arr) != 1) + 1)


def _fd() -> int:
    pid = os.getpid()
    if _state["pid"] != pid:
        directory = envs.SGLANG_PERF_TRACE_DIR.get() or os.path.join(
            os.environ.get("LOG_DIR", "/tmp"), "profile"
        )
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"perf-{socket.gethostname()}-{pid}.jsonl")
        _state["fd"] = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        _state["pid"] = pid
    return _state["fd"]


def _default(obj):
    item = getattr(obj, "item", None)
    return item() if callable(item) else str(obj)


def _emit(event: str, **fields) -> None:
    record = {
        "ev": event,
        "t": time.time(),
        "role": _state["role"],
        "rank": _state["rank"],
        "tid": threading.get_native_id(),
    }
    record.update(fields)
    line = (json.dumps(record, separators=(",", ":"), default=_default) + "\n").encode()
    with _lock:
        os.write(_fd(), line)


if ENABLED:
    emit = _emit
else:

    def emit(event: str, **fields) -> None:
        return None
