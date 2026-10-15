"""Opt-in host API timings; serialization runs outside transfer workers."""

import json
import logging
import os
import queue
import socket
import threading
import time

DIRECTORY = os.environ.get("SGLANG_NIXL_POSTING_TRACE_DIR", "")
ENABLED = bool(DIRECTORY)
_host = socket.gethostname()
_local = threading.local()
_init_lock = threading.Lock()
_writer = None


class _Writer:
    def __init__(self, role, rank):
        self.pid = os.getpid()
        self.role = role
        self.rank = rank
        self.queue = queue.Queue(maxsize=1024)
        self.lock = threading.Lock()
        self.attempted = self.accepted = self.written = 0
        self.dropped = self.errors = self.inflight = 0
        self.last_enqueue_ns = self.last_drained_ns = 0
        self.failed = False
        self.stem = os.path.join(DIRECTORY, f"posting-{_host}-{self.pid}")
        threading.Thread(target=self.run, name="nixl-posting-trace", daemon=True).start()

    def submit(self, record):
        with self.lock:
            self.attempted += 1
            record["seq"] = self.attempted
            if self.failed:
                self.dropped += 1
                return
            try:
                self.queue.put_nowait(record)
            except queue.Full:
                self.dropped += 1
                return
            self.accepted += 1
            self.last_enqueue_ns = time.monotonic_ns()

    def fail(self):
        with self.lock:
            self.errors += 1
            first = not self.failed
            self.failed = True
        if first:
            logging.getLogger(__name__).exception("NIXL posting trace failed")

    def status(self):
        with self.lock:
            queued = self.queue.qsize()
            if not queued and not self.inflight and self.accepted == self.written:
                self.last_drained_ns = time.monotonic_ns()
            record = dict(
                host=_host, pid=self.pid, role=self.role, rank=self.rank,
                attempted=self.attempted, accepted=self.accepted, written=self.written,
                dropped=self.dropped, errors=self.errors, queued=queued,
                inflight=self.inflight, last_enqueue_ns=self.last_enqueue_ns,
                last_drained_ns=self.last_drained_ns,
                status_mono_ns=time.monotonic_ns(), status_wall_ns=time.time_ns(),
            )
        temporary = self.stem + ".status.tmp"
        with open(temporary, "w") as stream:
            json.dump(record, stream, separators=(",", ":"))
        os.replace(temporary, self.stem + ".status.json")

    def run(self):
        fd = None
        last_status = 0
        while True:
            record = None
            try:
                record = self.queue.get(timeout=0.25)
                with self.lock:
                    self.inflight = 1
                if fd is None:
                    os.makedirs(DIRECTORY, exist_ok=True)
                    fd = os.open(self.stem + ".jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
                payload = (json.dumps(record, separators=(",", ":")) + "\n").encode()
                while payload:
                    count = os.write(fd, payload)
                    if count <= 0:
                        raise OSError("posting trace write made no progress")
                    payload = payload[count:]
                with self.lock:
                    self.written += 1
            except queue.Empty:
                pass
            except Exception:
                self.fail()
            finally:
                if record is not None:
                    with self.lock:
                        self.inflight = 0
                if time.monotonic_ns() - last_status >= 250_000_000:
                    try:
                        self.status()
                    except Exception:
                        self.fail()
                    last_status = time.monotonic_ns()


def emit(event, *, role, rank, **fields):
    global _writer
    if not ENABLED:
        return
    try:
        record = dict(
            ev=event, host=_host, pid=os.getpid(), tid=threading.get_native_id(),
            role=role, rank=int(rank), mono_ns=time.monotonic_ns(), **fields,
        )
        with _init_lock:
            if _writer is None or _writer.pid != os.getpid():
                _writer = _Writer(role, int(rank))
        _writer.submit(record)
    except Exception:
        if _writer is not None:
            _writer.fail()
        else:
            logging.getLogger(__name__).exception("Cannot initialize NIXL posting trace")


def _safe(function):
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            if _writer is not None:
                _writer.fail()
            else:
                logging.getLogger(__name__).exception("NIXL posting trace probe failed")
    return wrapped


@_safe
def begin_chunk(*, rank, room, chunk, last, worker, pages):
    _local.chunk = None
    _local.chunk = dict(
        rank=int(rank), room=int(room), chunk=int(chunk), last=bool(last),
        worker=int(worker), pages=int(pages), start_ns=time.monotonic_ns(),
        transfers=[], waits=[], error="",
    )


@_safe
def transfer(**fields):
    chunk = getattr(_local, "chunk", None)
    if chunk is not None:
        if len(chunk["transfers"]) < 128:
            chunk["transfers"].append(fields)
        else:
            chunk["error"] = "posting trace transfer limit exceeded"
            raise RuntimeError(chunk["error"])


@_safe
def wait(start_ns, end_ns, settled, failed, failure_seen):
    chunk = getattr(_local, "chunk", None)
    if chunk is not None:
        chunk["waits"].append(dict(
            start_ns=start_ns, end_ns=end_ns, settled=settled,
            failed=failed, failure_seen=failure_seen,
        ))


@_safe
def finish_chunk(error="", handles=0):
    chunk = getattr(_local, "chunk", None)
    _local.chunk = None
    if chunk is not None:
        chunk["end_ns"] = time.monotonic_ns()
        chunk["handles"] = handles
        if error:
            chunk["error"] = error
        emit("chunk", role="prefill", **chunk)
