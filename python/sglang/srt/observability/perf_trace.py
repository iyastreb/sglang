"""Per-request timeline probes for PD-disaggregation profiling.

Enabled by SGLANG_PERF_TRACE=1. Each process appends JSON lines to
<SGLANG_PERF_TRACE_DIR or $LOG_DIR/profile>/perf-<host>-<pid>.jsonl.
Wall time locates events; monotonic time measures intervals on the same host.
"""

from __future__ import annotations

import gc
import functools
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
DIRECT_ENABLED = ENABLED and os.environ.get("SGLANG_DFW_PROFILE_CONTROL") == "direct-spans"
_direct_owner = None
_direct_iteration = 0

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


def _profile_records(batch):
    return [dict(rid=req.rid, room=req.bootstrap_room,
                 input_len=len(req.origin_input_ids), prefix_len=len(req.prefix_indices),
                 host_hit=req.host_hit_length, host_loaded=req.host_loaded_length)
            for req in batch.reqs]


def _profile_eligible(records, page_size):
    return records and all(
        r["input_len"] >= 10000 and r["prefix_len"] >= 9984
        and r["prefix_len"] == (r["input_len"] - 1) // page_size * page_size
        and not r["host_hit"] and not r["host_loaded"] for r in records
    )


def direct_begin():
    if DIRECT_ENABLED and _direct_owner == threading.get_ident():
        return time.monotonic_ns(), time.thread_time_ns(), _direct_iteration
    return None


def direct_end(mark, event, **fields):
    if mark is None:
        return
    end_ns, end_cpu_ns = time.monotonic_ns(), time.thread_time_ns()
    emit(event, start_mono_ns=mark[0], end_mono_ns=end_ns,
         duration_ns=end_ns-mark[0], cpu_duration_ns=end_cpu_ns-mark[1],
         iteration=mark[2], **fields)


def direct_span(event, request_arg=None, batch_arg=None, result_batch=False):
    def decorate(function):
        if not DIRECT_ENABLED:
            return function

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            mark = direct_begin()
            if mark is None:
                return function(*args, **kwargs)
            fields = {}
            if request_arg is not None:
                req = args[request_arg]
                fields = dict(rid=req.rid, room=req.bootstrap_room)
            if batch_arg is not None:
                fields = dict(rooms=[r.bootstrap_room for r in args[batch_arg].reqs])
            result = None
            try:
                result = function(*args, **kwargs)
                return result
            finally:
                if result_batch:
                    batch = result.batch_to_run if result is not None else None
                    fields["rooms"] = [r.bootstrap_room for r in batch.reqs] if batch else []
                direct_end(mark, event, **fields)

        return wrapped
    return decorate


class _BoundedCPUProfile:
    def __init__(self):
        self.profiler = None
        self.finished = False
        self.iterations = 0
        self.batches = []
        self.start_ns = 0
        self.start_cpu_ns = 0
        self.ignored_callbacks = 0

    def tick(self):
        if self.profiler is None:
            return
        if self.iterations >= 8:
            self.finish("iteration_limit")
        elif time.monotonic_ns() - self.start_ns >= 1_000_000_000:
            self.finish("wall_limit_at_iteration_boundary")
        else:
            self.iterations += 1

    def observe_batch(self, batch, page_size):
        if self.finished or not batch.reqs:
            return
        records = _profile_records(batch)
        if self.profiler is None:
            if not _profile_eligible(records, page_size):
                return
            import cProfile

            self.profiler = cProfile.Profile()
            owner_thread = threading.get_ident()
            self.owner_native_tid = threading.get_native_id()

            def scheduler_only(callback):
                def filtered(*args):
                    if threading.get_ident() == owner_thread:
                        return callback(*args)
                    self.ignored_callbacks += 1

                return filtered

            for name in (
                "_pystart_callback",
                "_pyreturn_callback",
                "_ccall_callback",
                "_creturn_callback",
            ):
                setattr(self.profiler, name, scheduler_only(getattr(self.profiler, name)))
            self.start_ns = time.monotonic_ns()
            self.start_cpu_ns = time.thread_time_ns()
            self.iterations = 1
            emit("cpu_profile.start", rooms=[r["room"] for r in records])
            self.profiler.enable()
        self.batches.append(
            dict(iteration=self.iterations, mono_ns=time.monotonic_ns(), requests=records)
        )

    def finish(self, reason):
        if self.profiler is None:
            return
        self.profiler.disable()
        end_ns = time.monotonic_ns()
        end_cpu_ns = time.thread_time_ns()
        directory = envs.SGLANG_PERF_TRACE_DIR.get() or os.path.join(
            os.environ.get("LOG_DIR", "/opt/sglang-perf"), "profile"
        )
        os.makedirs(directory, exist_ok=True)
        stem = os.path.join(
            directory, f"cpu-prefill-{_host}-{os.getpid()}-rank{_state['rank']}"
        )
        self.profiler.dump_stats(stem + ".prof")
        manifest = dict(
            host=_host,
            pid=os.getpid(),
            rank=_state["rank"],
            timer="cProfile default monotonic wall clock",
            time_scope="scheduler call wall time including waits; separate scheduler thread CPU envelope",
            owner_native_tid=self.owner_native_tid,
            callback_filter="scheduler thread only",
            ignored_other_thread_callbacks=self.ignored_callbacks,
            start_mono_ns=self.start_ns,
            end_mono_ns=end_ns,
            thread_cpu_ns=end_cpu_ns - self.start_cpu_ns,
            wall_ns=end_ns - self.start_ns,
            iteration_limit=8,
            wall_limit_ns=1_000_000_000,
            deadline_check="scheduler iteration boundary",
            iterations=self.iterations,
            stop_reason=reason,
            selection="input>=10000; GPU prefix>=9984 and page-aligned maximum; no host hits or loads",
            first_iteration="starts after batch selection, before dispatch",
            batches=self.batches,
        )
        with open(stem + ".json", "w") as f:
            json.dump(manifest, f, separators=(",", ":"))
            f.write("\n")
        emit("cpu_profile.done", wall_ns=manifest["wall_ns"], iterations=self.iterations)
        self.profiler = None
        self.finished = True


class _BoundedDirectSpans(_BoundedCPUProfile):
    def tick(self):
        global _direct_iteration
        super().tick()
        _direct_iteration = self.iterations

    def observe_batch(self, batch, page_size):
        global _direct_owner, _direct_iteration
        if self.finished or not batch.reqs:
            return
        records = _profile_records(batch)
        if self.profiler is None:
            if not _profile_eligible(records, page_size):
                return
            self.profiler = True
            self.start_ns, self.start_cpu_ns = time.monotonic_ns(), time.thread_time_ns()
            self.iterations = _direct_iteration = 1
            _direct_owner = threading.get_ident()
            emit("pf.direct_start", rooms=[r["room"] for r in records],
                 start_mono_ns=self.start_ns, owner_native_tid=threading.get_native_id())
        self.batches.append(dict(iteration=self.iterations, requests=records))

    def finish(self, reason):
        global _direct_owner
        if self.profiler is None:
            return
        end_ns, end_cpu_ns = time.monotonic_ns(), time.thread_time_ns()
        _direct_owner = None
        emit("pf.direct_done", start_mono_ns=self.start_ns, end_mono_ns=end_ns,
             duration_ns=end_ns-self.start_ns, cpu_duration_ns=end_cpu_ns-self.start_cpu_ns,
             iterations=self.iterations, stop_reason=reason, batches=self.batches,
             iteration_limit=8, wall_limit_ns=1_000_000_000,
             first_iteration="after batch selection, before dispatch",
             deadline_check="scheduler iteration boundary")
        self.profiler = None
        self.finished = True


def bounded_cpu_profile():
    if DIRECT_ENABLED:
        return _BoundedDirectSpans()
    if ENABLED and os.environ.get("SGLANG_DFW_PROFILE_CONTROL") == "cpu-profile":
        return _BoundedCPUProfile()
    return None


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
