"""Pipelined ordered WebSocket owner for robot control and replay boundaries."""

from __future__ import annotations

import itertools
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future

try:
    from websockets.sync.client import connect
except ImportError:
    connect = None

from rlt_online.protocol import MAX_MESSAGE_BYTES, pack, unpack


class BrainBroker:
    def __init__(self, uri: str, *, timeout=120, capacity=64):
        self.uri, self.timeout, self.capacity = uri, timeout, capacity
        self.pending = queue.Queue(maxsize=capacity)
        self.inflight = deque()
        self.condition = threading.Condition()
        self.counter = itertools.count(time.time_ns())
        self.error = None
        self.stop = threading.Event()
        self.socket = None
        self.receiver = None
        self.thread = threading.Thread(target=self._run, name="brain-send", daemon=True)
        self.thread.start()

    def submit(self, payload: dict) -> Future:
        future = Future()
        if self.error is not None or self.stop.is_set():
            future.set_exception(self.error or RuntimeError("Brain broker is closed"))
            return future
        request = dict(payload, req_id=next(self.counter))
        future.request_id = request["req_id"]
        try:
            self.pending.put_nowait((request, future))
        except queue.Full:
            future.set_exception(RuntimeError("Brain send queue is full; stop robot"))
        return future

    def _fail(self, error):
        with self.condition:
            if self.error is None:
                self.error = error
            self.stop.set()
            while self.inflight:
                _, future = self.inflight.popleft()
                if not future.done():
                    future.set_exception(self.error)
            self.condition.notify_all()
        while True:
            try:
                _, future = self.pending.get_nowait()
                if not future.done():
                    future.set_exception(self.error)
            except queue.Empty:
                return

    def _receive(self, socket):
        try:
            while not self.stop.is_set():
                with self.condition:
                    self.condition.wait_for(lambda: self.inflight or self.stop.is_set(), timeout=.1)
                    if self.stop.is_set():
                        return
                reply = unpack(socket.recv(timeout=self.timeout))
                with self.condition:
                    if not self.inflight or reply["req_id"] != self.inflight[0][0]:
                        raise RuntimeError("Out-of-order brain response; stop robot and reconnect")
                    _, future = self.inflight.popleft()
                if reply.get("ok") is not True:
                    future.set_exception(RuntimeError(reply.get("error", "Brain rejected request")))
                else:
                    future.set_result(reply)
        except Exception as error:
            if not self.stop.is_set():
                self._fail(error)

    def _run(self):
        try:
            if connect is None:
                raise RuntimeError("Install websockets>=14 for binary brain connections")
            with connect(self.uri, max_size=MAX_MESSAGE_BYTES, open_timeout=self.timeout,
                         close_timeout=2) as socket:
                self.socket = socket
                self.receiver = threading.Thread(target=self._receive, args=(socket,),
                                                 name="brain-receive", daemon=True)
                self.receiver.start()
                while not self.stop.is_set():
                    try:
                        request, future = self.pending.get(timeout=.1)
                    except queue.Empty:
                        continue
                    if future.cancelled():
                        continue
                    with self.condition:
                        if self.stop.is_set():
                            future.set_exception(self.error or RuntimeError("Brain broker is closed"))
                            break
                        if len(self.inflight) >= self.capacity:
                            raise RuntimeError("Brain inflight queue is full; stop robot")
                        self.inflight.append((request["req_id"], future))
                        self.condition.notify_all()
                    socket.send(pack(request))
        except Exception as error:
            if not self.stop.is_set():
                self._fail(error)
        finally:
            self.stop.set()
            with self.condition:
                self.condition.notify_all()
            if self.socket is not None:
                self.socket.close()
            if self.receiver is not None:
                self.receiver.join(timeout=3)
            self.socket = None

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        if self.socket is not None:
            self.socket.close()
        self.thread.join(timeout=3)
        self._fail(RuntimeError("Brain broker is closed"))
