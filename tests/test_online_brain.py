"""Replay boundaries preserve partial takeover and human actions."""

import numpy as np
import pytest
import torch
from types import SimpleNamespace

pytest.importorskip("torch")
from rlt_online.serve_rlt_brain import Brain
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
             "sources": ["VLA"] * 6, "critical_steps": [True] * 6,
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
