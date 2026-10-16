"""Opt-in state preparation timing, buffered until an explicit drain request."""

import functools
import inspect
import json
import logging
import os
import socket
import threading
import time

DIRECTORY = os.environ.get("SGLANG_NIXL_STATE_TRACE_DIR", "")
ENABLED = bool(DIRECTORY)
_local = threading.local()
_writer = None
_MAX_RECORDS = 250_000


class _Writer:
    def __init__(self, rank):
        self.host = socket.gethostname()
        with open("/proc/sys/kernel/random/boot_id") as stream:
            self.boot_id = stream.read().strip()
        self.pid = os.getpid()
        self.rank = int(rank)
        self.lock = threading.Lock()
        self.records = []
        self.active = self.attempted = self.written = 0
        self.dropped = self.errors = self.inflight = 0
        self.drain_token = ""
        self.stem = os.path.join(DIRECTORY, f"state-{self.host}-{self.pid}")
        os.makedirs(DIRECTORY, exist_ok=True)
        self.status()
        threading.Thread(target=self.run, name="nixl-state-trace", daemon=True).start()

    def enter(self):
        with self.lock:
            self.active += 1

    def submit(self, record):
        with self.lock:
            self.active -= 1
            self.attempted += 1
            record["seq"] = self.attempted
            if len(self.records) >= _MAX_RECORDS:
                self.dropped += 1
            else:
                self.records.append(record)

    def status(self):
        with self.lock:
            status = dict(
                host=self.host,
                boot_id=self.boot_id,
                pid=self.pid,
                rank=self.rank,
                attempted=self.attempted,
                written=self.written,
                buffered=len(self.records),
                active=self.active,
                inflight=self.inflight,
                dropped=self.dropped,
                errors=self.errors,
                drain_token=self.drain_token,
                status_mono_ns=time.monotonic_ns(),
            )
        temporary = self.stem + ".status.tmp"
        with open(temporary, "w") as stream:
            json.dump(status, stream, separators=(",", ":"))
        os.replace(temporary, self.stem + ".status.json")

    def run(self):
        while True:
            time.sleep(0.25)
            try:
                try:
                    with open(os.path.join(DIRECTORY, "drain.request")) as stream:
                        token = stream.read().strip()
                except FileNotFoundError:
                    continue
                if not token:
                    continue
                with self.lock:
                    if self.active:
                        continue
                    records, self.records = self.records, []
                    self.inflight = len(records)
                with open(self.stem + ".jsonl", "a") as stream:
                    for record in records:
                        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                with self.lock:
                    self.written += len(records)
                    self.inflight = 0
                    self.drain_token = token
                self.status()
            except Exception:
                with self.lock:
                    self.errors += 1
                logging.getLogger(__name__).exception("NIXL state trace drain failed")
                try:
                    self.status()
                except Exception:
                    pass


def _api(function, kind):
    @functools.wraps(function)
    def timed(*args, **kwargs):
        record = getattr(_local, "state", None)
        if record is None:
            return function(*args, **kwargs)
        start = time.monotonic_ns()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.monotonic_ns() - start
            if kind == "post":
                record["post_ns"] += elapsed
                record["post_calls"] += 1
            else:
                record["create_ns"] += elapsed
                record["create_calls"] += 1
                record[kind + "_calls"] += 1

    return timed


def install(agent, rank):
    global _writer
    if not ENABLED:
        return
    if _writer is not None:
        raise RuntimeError("NIXL state trace requires one prefill agent per process")
    _writer = _Writer(rank)
    agent.initialize_xfer = _api(agent.initialize_xfer, "initialized")
    agent.make_prepped_xfer = _api(agent.make_prepped_xfer, "prepared")
    agent.transfer = _api(agent.transfer, "post")


def _begin(manager, args, kwargs):
    notif = args[5] if len(args) > 5 else kwargs["notif"]
    peer = args[0] if args else kwargs["peer_name"]
    indices = args[1] if len(args) > 1 else kwargs["prefill_state_indices"]
    plan = getattr(manager, "_prepared_state_plans", {}).get(peer)
    record = dict(
        ev="state",
        host=_writer.host,
        boot_id=_writer.boot_id,
        pid=_writer.pid,
        rank=_writer.rank,
        room=int(notif.split("_", 1)[0]),
        peer=peer,
        src_indices=[len(index) if index is not None else 0 for index in indices],
        prepared_components=sorted(plan.components) if plan is not None else [],
        active_ns=0,
        create_ns=0,
        post_ns=0,
        create_calls=0,
        post_calls=0,
        initialized_calls=0,
        prepared_calls=0,
        error="",
    )
    _writer.enter()
    record["start_ns"] = time.monotonic_ns()
    return record


def _active(record, function, *args, **kwargs):
    previous = getattr(_local, "state", None)
    _local.state = record
    start = time.monotonic_ns()
    try:
        return function(*args, **kwargs)
    finally:
        record["active_ns"] += time.monotonic_ns() - start
        _local.state = previous


def _finish(record):
    record["end_ns"] = time.monotonic_ns()
    record["preparation_ns"] = (
        record["active_ns"] - record["create_ns"] - record["post_ns"]
    )
    _writer.submit(record)


def measure(function):
    if not ENABLED:
        return function
    if inspect.isgeneratorfunction(function):

        @functools.wraps(function)
        def measured_generator(self, *args, **kwargs):
            record = _begin(self, args, kwargs)
            try:
                iterator = _active(record, function, self, *args, **kwargs)
                while True:
                    try:
                        handle = _active(record, next, iterator)
                    except StopIteration:
                        break
                    yield handle
            except BaseException as error:
                record["error"] = type(error).__name__ + ": " + str(error)
                raise
            finally:
                _finish(record)

        return measured_generator

    @functools.wraps(function)
    def measured(self, *args, **kwargs):
        record = _begin(self, args, kwargs)
        try:
            return _active(record, function, self, *args, **kwargs)
        except BaseException as error:
            record["error"] = type(error).__name__ + ": " + str(error)
            raise
        finally:
            _finish(record)

    return measured
