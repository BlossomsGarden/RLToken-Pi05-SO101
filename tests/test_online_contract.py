import numpy as np
import pytest
import json
from pathlib import Path

from rlt_online.action_space import absolute_to_delta, delta_to_absolute, normalize, denormalize
from rlt_online.joint_mapping import JointMapping, JOINTS
from rlt_online.protocol import pack, unpack, jpeg, image, joints
from rlt_online.broker import BrainBroker
from windows.deploy_rlt import (GripperHold, Keys, LeaderMirror, abort_episode, actor_requested,
                                apply_actor_patch, next_local_episode, reset_follower_torque,
                                safe_action, wait_settled)


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


def test_leader_mirror_writes_only_volatile_torque_and_clamps_both_limits():
    from types import SimpleNamespace

    class Bus:
        calibration = {name: SimpleNamespace(range_min=1000, range_max=3000) for name in JOINTS}
        motors = {name: SimpleNamespace(model="sts3215") for name in JOINTS}
        model_resolution_table = {"sts3215": 4096}
        writes = []

        def write(self, register, name, value, *, normalize):
            self.writes.append((register, name, value, normalize))

    class Leader:
        bus = Bus()
        last_feedback = None

        def get_action(self):
            return {f"{name}.pos": 87.5 if name != "gripper" else 99 for name in JOINTS}

        def send_feedback(self, action):
            self.last_feedback = action

        def enable_torque(self): pass
        def disable_torque(self): pass

    leader = Leader()
    mirror = LeaderMirror(leader)
    assert len(leader.bus.writes) == 6
    assert all(reg == "Torque_Limit" and value == 500 and normalized is False
               for reg, _, value, normalized in leader.bus.writes)
    mirror.resume()
    mirror.feedback(np.full(6, 500, dtype=np.float32))
    assert leader.last_feedback["shoulder_pan.pos"] == pytest.approx(2000 * 180 / 4095)
    assert leader.last_feedback["gripper.pos"] == 100
    mirror.close()
    assert len(leader.bus.writes) == 6  # No EEPROM restore on shutdown.


def test_chunk_settle_uses_four_stable_samples_and_soft_timeout():
    from types import SimpleNamespace

    class Bus:
        def sync_read(self, name, num_retry):
            return {joint: 1 for joint in JOINTS}

    robot = SimpleNamespace(bus=Bus(), config=SimpleNamespace(num_read_retries=1))
    assert wait_settled(robot, {"settle_tolerance_deg": [.5] * 6,
                                              "settle_timeout_s": 1}) is True
    assert wait_settled(robot, {"settle_tolerance_deg": [.5] * 6,
                                              "settle_timeout_s": .01}) is False


def test_async_actor_patch_applies_only_unexecuted_rows():
    plan = {"actions": np.zeros((50, 6), dtype=np.float32), "actor_start": -1}
    patch = {"start": 10, "actions": [[1.] * 6] * 10}
    assert apply_actor_patch(plan, patch, 9)
    assert np.all(plan["actions"][:10] == 0)
    plan["actions"][10:20] = 0
    assert apply_actor_patch(plan, patch, 17)
    assert np.all(plan["actions"][:17] == 0)
    assert np.all(plan["actions"][17:20] == 1)
    assert not apply_actor_patch(plan, patch, 20)


def test_manual_phase_toggle_remains_visible_during_auto_detection_cooldown(monkeypatch):
    import sys
    from collections import deque
    from types import SimpleNamespace
    import windows.deploy_rlt as deploy

    buttons = deque(["c"])
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(
        kbhit=lambda: bool(buttons), getwch=lambda: buttons.popleft()))
    monkeypatch.setattr(deploy.time, "monotonic", lambda: 100.)
    keys = Keys(auto_critical=True)
    keys.poll(False)
    assert keys.manual_critical is True and keys.manual_override() is True
    monkeypatch.setattr(deploy.time, "monotonic", lambda: 104.)
    assert keys.manual_override() is None
    assert keys.manual_critical is True
    buttons.append("c")
    keys.poll(True)
    assert keys.manual_critical is False and keys.manual_override() is False


def test_actor_key_requires_operator_and_window_replanning(monkeypatch):
    import sys
    from collections import deque
    from types import SimpleNamespace

    buttons = deque(["a"])
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(
        kbhit=lambda: bool(buttons), getwch=lambda: buttons.popleft()))
    keys = Keys(auto_critical=False)
    window = SimpleNamespace(no_replan_every_window=False)
    smooth = SimpleNamespace(no_replan_every_window=True)
    assert not actor_requested(keys, window)
    keys.poll(False)
    assert keys.actor_changed and actor_requested(keys, window)
    assert not actor_requested(keys, smooth)
    assert Keys(auto_critical=False, actor=keys.actor).actor  # Persists across episodes.
    buttons.append("a")
    keys.poll(False)
    assert keys.actor_changed and not actor_requested(keys, window)
    buttons.append("a")
    smooth_keys = Keys(auto_critical=False, actor_mode=False)
    smooth_keys.poll(False)
    assert not smooth_keys.actor and not smooth_keys.actor_changed


def test_reset_holds_measured_position_before_reenabling_torque(monkeypatch):
    from types import SimpleNamespace
    import builtins

    calls = []
    parked = {name: float(i) for i, name in enumerate(JOINTS)}

    class Bus:
        def disable_torque(self): calls.append("disable")
        def sync_read(self, key, num_retry):
            calls.append(("read", key, num_retry))
            return parked
        def sync_write(self, key, value): calls.append(("goal", key, value))
        def enable_torque(self): calls.append("enable")

    monkeypatch.setattr(builtins, "input", lambda prompt: calls.append("enter"))
    reset_follower_torque(SimpleNamespace(bus=Bus(), config=SimpleNamespace(num_read_retries=3)))
    assert calls == ["disable", "enter", ("read", "Present_Position", 3),
                     ("goal", "Goal_Position", parked), "enable"]


def test_abort_discards_replay_and_archives_unlabelled_episode(tmp_path):
    from types import SimpleNamespace

    class Broker:
        requests = []
        def submit(self, payload):
            self.requests.append(payload)
            return SimpleNamespace(result=lambda timeout: {"next_episode": 4})

    broker = Broker()
    record = {"steps": [{"state": [0.] * 6, "action": [0.] * 6,
                         "leader": [0.] * 6, "source": "VLA", "critical": False,
                         "critical_prob": .1, "chunk_id": 11, "step_ms": 30.}],
              "boundaries": []}
    abort_episode(broker, tmp_path, 4, 1, record, 5, RuntimeError("watchdog"))
    assert broker.requests == [{"command": "finish", "episode": 4, "step": 1,
                                "reward": 0, "discard": True}]
    with np.load(next(tmp_path.glob("aborted_ep_0004_*.npz"))) as audit:
        assert audit["aborted"] and audit["reward"] == -1 and len(audit["action"]) == 1
    assert next_local_episode(tmp_path) == 1  # Audit does not consume a training episode id.


def test_abort_still_archives_audit_when_brain_is_disconnected(tmp_path, capsys):
    class DisconnectedBroker:
        def submit(self, payload):
            raise ConnectionError("brain unavailable")

    record = {"steps": [], "boundaries": []}
    abort_episode(DisconnectedBroker(), tmp_path, 4, 0, record, 5,
                  ConnectionError("network lost"))
    with np.load(next(tmp_path.glob("aborted_ep_0004_*.npz"))) as audit:
        assert audit["aborted"] and audit["reward"] == -1
        assert audit["action"].shape == (0, 6)
    assert next_local_episode(tmp_path) == 1
    assert "Brain discard unavailable" in capsys.readouterr().out
