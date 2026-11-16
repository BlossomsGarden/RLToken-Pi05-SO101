import numpy as np
import pytest
import json
from pathlib import Path

from rlt_online.action_space import absolute_to_delta, delta_to_absolute, normalize, denormalize
from rlt_online.joint_mapping import JointMapping, JOINTS
from rlt_online.protocol import pack, unpack, jpeg, image, joints
from rlt_online.broker import BrainBroker
from windows.deploy_rlt import GripperHold, safe_action


def test_action_round_trip_and_absolute_gripper():
    state = np.array([10, 20, 30, 40, 50, 90], dtype=np.float32)
    lo = np.array([-5, -5, -5, -5, -5, 10], dtype=np.float32)
    hi = np.array([5, 5, 5, 5, 5, 30], dtype=np.float32)
    model = np.array([[0, 0, 0, 0, 0, 0], [.5, -.5, .5, -.5, 0, 1]], dtype=np.float32)
    absolute = delta_to_absolute(model, state, lo, hi)
    np.testing.assert_allclose(absolute[0], [10, 20, 30, 40, 50, 20])
    np.testing.assert_allclose(absolute_to_delta(absolute, state, lo, hi), model, atol=1e-6)
    np.testing.assert_allclose(denormalize(normalize(absolute, lo - 100, hi + 100), lo - 100, hi + 100), absolute, atol=1e-5)


def test_calibration_round_trip_and_invalid_mapping():
    mapping = JointMapping({"joint_names": JOINTS, "scale": [1 + i / 10 for i in range(6)],
                            "offset": [i * 2 for i in range(6)]})
    values = np.arange(12, dtype=np.float32).reshape(2, 6)
    np.testing.assert_allclose(mapping.to_physical(mapping.to_model(values)), values, atol=1e-6)
    with pytest.raises(ValueError):
        JointMapping({"joint_names": JOINTS, "scale": [0] * 6, "offset": [0] * 6})


def test_binary_protocol_and_request_matching_fields():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    payload = {"req_id": 7, "command": "plan", "front": jpeg(frame), "state": [0.] * 6}
    received = unpack(pack(payload))
    assert received["req_id"] == 7
    assert image(received["front"]).shape == frame.shape
    assert joints(received["state"]).shape == (6,)
    with pytest.raises(ValueError):
        unpack(pack({"command": "plan"}))


def test_broker_matches_fifo_and_rejects_wrong_reply(monkeypatch):
    import rlt_online.broker as broker_module

    class Socket:
        def __init__(self, bad=False):
            self.bad = bad
            self.sent = []
            self.read_index = 0
            import threading
            self.condition = threading.Condition()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def send(self, frame):
            with self.condition:
                self.sent.append(unpack(frame))
                self.condition.notify_all()

        def recv(self, timeout):
            with self.condition:
                assert self.condition.wait_for(lambda: self.read_index < len(self.sent), timeout=timeout)
                req_id = self.sent[self.read_index]["req_id"]
                self.read_index += 1
            return pack({"req_id": req_id + self.bad, "ok": True, "mode": "train"})

        def close(self):
            pass

    socket = Socket()
    monkeypatch.setattr(broker_module, "connect", lambda *args, **kwargs: socket)
    broker = BrainBroker("ws://localhost:1", timeout=1)
    try:
        first = broker.submit({"command": "health"})
        second = broker.submit({"command": "health"})
        assert first.result(timeout=2)["req_id"] < second.result(timeout=2)["req_id"]
    finally:
        broker.close()

    monkeypatch.setattr(broker_module, "connect", lambda *args, **kwargs: Socket(bad=True))
    broker = BrainBroker("ws://localhost:1", timeout=1)
    try:
        with pytest.raises(RuntimeError, match="Out-of-order"):
            broker.submit({"command": "health"}).result(timeout=2)
    finally:
        broker.close()


def test_real_websocket_binary_health_round_trip():
    import threading
    server_module = pytest.importorskip("websockets.sync.server")

    def respond(socket):
        for frame in socket:
            request = unpack(frame)
            socket.send(pack({"req_id": request["req_id"], "ok": True, "mode": "train"}))

    with server_module.serve(respond, "127.0.0.1", 0) as server:
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        broker = BrainBroker(f"ws://127.0.0.1:{server.socket.getsockname()[1]}", timeout=3)
        try:
            assert broker.submit({"command": "health"}).result(timeout=4)["mode"] == "train"
            assert broker.submit({"command": "health"}).result(timeout=4)["mode"] == "train"
        finally:
            broker.close()
            server.shutdown()
            serving.join(timeout=3)


def test_broker_pipelines_requests_before_first_reply(monkeypatch):
    import threading
    import rlt_online.broker as broker_module

    class BlockedSocket:
        def __init__(self):
            self.sent = []
            self.wrote_two = threading.Event()
            self.read_index = 0

        def __enter__(self): return self
        def __exit__(self, *args): return False
        def close(self): pass

        def send(self, frame):
            self.sent.append(unpack(frame))
            if len(self.sent) == 2:
                self.wrote_two.set()

        def recv(self, timeout):
            assert self.wrote_two.wait(timeout=timeout), "sender waited for first reply"
            request = self.sent[self.read_index]
            self.read_index += 1
            return pack({"req_id": request["req_id"], "ok": True})

    socket = BlockedSocket()
    monkeypatch.setattr(broker_module, "connect", lambda *args, **kwargs: socket)
    broker = BrainBroker("ws://localhost:1", timeout=2)
    try:
        first = broker.submit({"command": "health"})
        second = broker.submit({"command": "health"})
        assert socket.wrote_two.wait(timeout=2)
        assert first.result(timeout=2)["req_id"] < second.result(timeout=2)["req_id"]
    finally:
        broker.close()

def test_physical_limits_and_gripper_hold_require_measured_telemetry():
    config = json.loads((Path(__file__).parents[1] / "configs/robot.json").read_text(encoding="utf-8"))
    config["physical_min_deg"] = [-90] * 6
    config["physical_max_deg"] = [90] * 6

    class Bus:
        def sync_read(self, name, num_retry):
            return {joint: 20 if joint == "gripper" else 0 for joint in JOINTS}

        def read(self, *args, **kwargs):
            return 0

    class Robot:
        bus = Bus()
        config = type("Config", (), {"num_read_retries": 1})()

        def send_action(self, goal):
            return goal

    robot = Robot()
    with pytest.raises(RuntimeError, match="physical joint limits"):
        safe_action(robot, np.array([100, 0, 0, 0, 0, 0]), GripperHold(config["gripper_hold"]),
                    config, np.array([0, 0, 0, 0, 0, 20]))
    accepted = safe_action(robot, np.array([3, 0, 0, 0, 0, 10]), GripperHold(config["gripper_hold"]),
                           config, np.array([0, 0, 0, 0, 0, 20]))
    assert accepted[0] == 3
    assert accepted[-1] == 13.5  # The 6.5-degree per-command rate limit still applies.
