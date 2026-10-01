import functools
import gc
import inspect
import json
import os
import socket
import time

ENABLED = os.getenv("SGLANG_FLUCT_PROFILE") == "1"
_pid = None
_fd = None
_gc_starts = {}


def emit(event, **fields):
    global _pid, _fd
    if not ENABLED:
        return
    pid = os.getpid()
    if pid != _pid:
        if _fd is not None:
            os.close(_fd)
        directory = os.path.join(os.environ["LOG_DIR"], "profile")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{os.getenv('FLUCT_ROLE', 'unknown')}-{socket.gethostname()}-{pid}.jsonl")
        _fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        _pid = pid
    row = dict(event=event, wall_ns=time.time_ns(), mono_ns=time.perf_counter_ns(), pid=pid, **fields)
    os.write(_fd, (json.dumps(row, separators=(",", ":")) + "\n").encode())


def request_ids(args):
    for obj in args[:3]:
        if getattr(obj, "rid", None):
            return [obj.rid]
        if getattr(obj, "rids", None):
            return list(obj.rids)
        if getattr(obj, "reqs", None):
            return [r.rid for r in obj.reqs]
    return []


def span(fn):
    if not ENABLED:
        return fn

    def finish(start, cpu, ids):
        duration = time.perf_counter_ns() - start
        if duration >= 200000 or "compact" in fn.__name__:
            emit("span", name=fn.__qualname__, duration_ns=duration,
                 cpu_ns=time.thread_time_ns() - cpu, rids=ids)

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapped(*args, **kwargs):
            ids = request_ids(args)
            start, cpu = time.perf_counter_ns(), time.thread_time_ns()
            try:
                return await fn(*args, **kwargs)
            finally:
                finish(start, cpu, ids)
    else:
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            ids = request_ids(args)
            start, cpu = time.perf_counter_ns(), time.thread_time_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                finish(start, cpu, ids)
    return wrapped


def stages(cls):
    if not ENABLED:
        return cls
    for name, method in list(vars(cls).items()):
        if not name.startswith("set_") or not callable(method):
            continue
        def wrap(fn):
            @functools.wraps(fn)
            def wrapped(self, *args, **kwargs):
                result = fn(self, *args, **kwargs)
                emit("stage", name=f"{cls.__name__}.{fn.__name__}",
                     rid=self.profile_rid, role=self.disagg_mode_str(),
                     times={k: v for k, v in vars(self).items()
                            if k.endswith("time") and isinstance(v, (int, float))})
                return result
            return wrapped
        setattr(cls, name, wrap(method))
    return cls


def gc_event(phase, info):
    generation = info["generation"]
    if phase == "start":
        _gc_starts[generation] = time.perf_counter_ns()
    else:
        duration = time.perf_counter_ns() - _gc_starts.pop(generation, time.perf_counter_ns())
        if duration >= 200000 or generation == 2:
            emit("gc", duration_ns=duration, **info)


if ENABLED:
    gc.callbacks.append(gc_event)
