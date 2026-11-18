"""Replay boundaries preserve partial takeover and human actions."""

import numpy as np
import pytest
import torch
import threading
from types import SimpleNamespace

pytest.importorskip("torch")
from rlt_online.serve_rlt_brain import Brain, handle_client
from rlt_online.joint_mapping import JointMapping, JOINTS
from scripts.rlt_core import TransitionSource


def boundary(step):
    return {"state": np.full(6, step, dtype=np.float32),
            "z": np.zeros(2048, dtype=np.float32),
            "proprio": np.zeros(6, dtype=np.float32)}


def test_partial_autonomous_then_human_replay_boundaries():
    brain = object.__new__(Brain)
    brain.reference = np.zeros((50, 6), dtype=np.float32)
    brain.raw_model_actions = torch.zeros((1, 50, 6))
    brain.human_reference = np.full((50, 6), .4, dtype=np.float32)
    brain.human_start_step = 6
    brain.mapping = JointMapping({"joint_names": JOINTS, "scale": [1.] * 6, "offset": [0.] * 6})
    brain.delta_low = np.full(6, -10, dtype=np.float32)
    brain.delta_high = np.full(6, 10, dtype=np.float32)
    brain.critical = True
    brain.plan_start = 0
    brain.last_step = 0
    brain.last_boundary = boundary(0)
    brain.pending_next_ref = None
    brain.transitions = []
    brain.episode = 1
    first = {"step": 6, "actual": [[1.] * 6] * 6,
             "sources": ["VLA"] * 6, "critical_steps": [False] * 5 + [True],
             "interrupted": True}
    brain._record_slice(first, boundary(6))
    assert brain.transitions[-1].executed_steps == 6
    assert sum(brain.transitions[-1].valid_mask) == 6
    assert brain.pending_next_ref is brain.transitions[-1]
    second = {"step": 9, "actual": [[2.] * 6] * 3,
              "sources": ["HUMAN"] * 3, "critical_steps": [True] * 3}
    brain._record_slice(second, boundary(9), human=True)
    assert brain.transitions[0].next_ref_chunk[0, 0] == pytest.approx(.4)
    assert brain.transitions[-1].source == int(TransitionSource.HUMAN)
    assert brain.transitions[-1].intervention_flag
    assert brain.last_step == 9
    assert brain.transitions[-1].ref_chunk[0, 0] == pytest.approx(-.4)
    with pytest.raises(ValueError, match="ordered"):
        brain._record_slice(second, boundary(9), human=True)
    brain._record_slice({"step": 30, "actual": [[2.] * 6] * 4,
                         "sources": ["HUMAN"] * 4,
                         "critical_steps": [True] * 4}, boundary(30), human=True)
    assert brain.last_step == 30 and len(brain.transitions) == 2


@pytest.mark.parametrize("next_critical", [False, True])
def test_chunk_close_uses_new_plan_phase_and_preserves_old_reference(next_critical):
    brain = object.__new__(Brain)
    brain.mapping = JointMapping({"joint_names": JOINTS, "scale": [1.] * 6, "offset": [0.] * 6})
    brain.delta_low = np.full(6, -10, dtype=np.float32)
    brain.delta_high = np.full(6, 10, dtype=np.float32)
    brain.episode, brain.plan_id, brain.plan_start = 1, 101, 0
    brain.last_step, brain.last_boundary = 40, boundary(0)
    brain.critical, brain.critical_since = not next_critical, None
    brain.trainer, brain.plan_interrupted = None, False
    brain.transitions, brain.pending_next_ref = [], None
    brain.deferred_slice = None
    brain.raw_model_actions = torch.full((1, 50, 6), .25)
    brain.detector = SimpleNamespace(last_probability=None)
    brain.lock = threading.RLock()
    brain.model = SimpleNamespace(predict_action_chunk=lambda batch, noise: torch.zeros((1, 50, 6)))
    brain.noise = None
    brain._batch = lambda payload: (np.zeros(6, dtype=np.float32), {})
    brain._encode = lambda state, batch, raw: boundary(0)
    brain._critical = lambda encoded, payload: setattr(brain, "critical", payload["manual_critical"])
    brain.handle({"req_id": 102, "command": "boundary_slice", "episode": 1,
                  "plan_id": 101, "step": 50, "actual": [[1.] * 6] * 10,
                  "sources": ["VLA"] * 10, "critical_steps": [not next_critical] * 10,
                  "interrupted": False, "defer_close": True})
    assert not brain.transitions
    brain.handle({"req_id": 103, "command": "plan", "episode": 1, "step": 50,
                  "actor": False, "manual_critical": next_critical})
    assert brain.deferred_slice is None and brain.last_step == 50
    assert len(brain.transitions) == int(next_critical)
    if next_critical:
        assert brain.transitions[0].ref_chunk[0, 0] == pytest.approx(.25)
        assert brain.transitions[0].next_ref_chunk[0, 0] == pytest.approx(0.)


def test_terminal_snapshot_flushes_deferred_window_without_new_plan():
    brain = object.__new__(Brain)
    brain.mapping = JointMapping({"joint_names": JOINTS, "scale": [1.] * 6, "offset": [0.] * 6})
    brain.delta_low = np.full(6, -10, dtype=np.float32)
    brain.delta_high = np.full(6, 10, dtype=np.float32)
    brain.episode, brain.plan_id, brain.plan_start = 1, 101, 0
    brain.last_step, brain.last_boundary = 40, boundary(0)
    brain.critical, brain.plan_interrupted = True, False
    brain.trainer, brain.pending_next_ref = None, None
    brain.transitions = []
    brain.raw_model_actions = torch.full((1, 50, 6), .25)
    brain.deferred_slice = {"step": 50, "actual": [[1.] * 6] * 10,
                            "sources": ["VLA"] * 10, "critical_steps": [True] * 10}
    brain.lock = threading.RLock()
    brain.model = SimpleNamespace(predict_action_chunk=lambda batch, noise: torch.zeros((1, 50, 6)))
    brain.noise = None
    brain._batch = lambda payload: (np.zeros(6, dtype=np.float32), {})
    brain._encode = lambda state, batch, raw: boundary(0)
    reply = brain.handle({"req_id": 102, "command": "terminal_snapshot",
                          "episode": 1, "plan_id": 101, "step": 50, "actual": []})
    assert reply["req_id"] == 102
    assert brain.last_step == 50 and brain.deferred_slice is None
    assert len(brain.transitions) == 1
    assert brain.transitions[0].next_ref_chunk[0, 0] == pytest.approx(0.)


def test_interrupt_at_exact_boundary_requires_no_fake_action():
    brain = object.__new__(Brain)
    brain.last_req = 0
    brain.trainer = None
    brain.episode = 1
    brain.plan_id = 3
    brain.plan_start = 0
    brain.last_step = 10
    brain.plan_interrupted = False
    brain.critical = False
    brain.transitions = []
    brain.detector = SimpleNamespace(last_probability=None)
    result = brain.handle({"req_id": 1, "command": "interrupt_plan",
                           "episode": 1, "plan_id": 3, "step": 10})
    assert result["req_id"] == 1 and brain.plan_interrupted
    brain.handle({"req_id": 2, "command": "interrupt_plan",
                  "episode": 1, "plan_id": 3, "step": 11})
    assert brain.last_step == 11  # Non-critical actions need no JPEG boundaries.
    with pytest.raises(ValueError, match="acknowledged parent boundary"):
        brain.handle({"req_id": 3, "command": "interrupt_plan",
                       "episode": 1, "plan_id": 3, "step": 9})


def test_noncritical_window_is_not_recorded_and_can_be_skipped():
    brain = object.__new__(Brain)
    brain.mapping = JointMapping({"joint_names": JOINTS, "scale": [1.] * 6, "offset": [0.] * 6})
    brain.plan_start = brain.last_step = 0
    brain.last_boundary = boundary(0)
    brain.critical = False
    brain.pending_next_ref = None
    brain.transitions = []
    brain._record_slice({"step": 10, "actual": [[1.] * 6] * 10,
                         "sources": ["VLA"] * 10, "critical_steps": [False] * 10}, None)
    assert brain.last_step == 10
    assert not brain.transitions
    assert brain.last_boundary["state"][0] == 0


def test_discard_keeps_next_episode_number():
    brain = object.__new__(Brain)
    brain.last_req = 0
    brain.trainer = SimpleNamespace(episode=5, worker_error=None)
    brain.episode = 6
    brain.plan_id = 21
    brain.plan_start = 0
    brain.last_step = 0
    brain.plan_interrupted = False
    brain.transitions = []
    reply = brain.handle({"req_id": 1, "command": "finish", "episode": 6,
                          "step": 0, "reward": 0, "discard": True})
    assert reply["next_episode"] == 6
    assert brain.episode is None


@pytest.mark.parametrize("plan_fails", [False, True])
def test_client_disconnect_or_failed_plan_clears_uncommitted_episode(plan_fails):
    from rlt_online.protocol import pack, unpack

    class DummyBrain:
        episode = None
        discarded = 0

        def handle(self, payload):
            self.episode = payload["episode"]
            if plan_fails:
                raise ValueError("plan inference failed")
            return {"req_id": payload["req_id"], "plan_id": 1}

        def discard_active_episode(self):
            self.discarded += 1
            self.episode = None

    class Socket:
        def __init__(self):
            self.replies = []
        def __iter__(self):
            return iter([pack({"req_id": 1, "command": "plan", "episode": 4})])

        def send(self, response):
            self.replies.append(unpack(response))

    brain, socket = DummyBrain(), Socket()
    handle_client(brain, socket)
    assert brain.episode is None and brain.discarded == 1
    assert socket.replies[0]["ok"] is not plan_fails
