"""CPU smoke for online updates, terminal rewards and torn-journal recovery."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot.policies.pi05_rlt.modeling_pi05_rlt")

from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTChunkActor
from rlt_online.learner import Stage2Trainer
from scripts.rlt_core import Transition, TransitionSource


def make_transition(episode, step, human=False):
    return Transition(
        z_rl=np.zeros(8, dtype=np.float32), proprio=np.zeros(6, dtype=np.float32),
        ref_chunk=np.zeros((10, 6), dtype=np.float32),
        action_chunk=np.zeros((10, 6), dtype=np.float32),
        rewards=np.zeros(10, dtype=np.float32), done=False,
        next_z_rl=np.zeros(8, dtype=np.float32), next_proprio=np.zeros(6, dtype=np.float32),
        next_ref_chunk=np.zeros((10, 6), dtype=np.float32),
        source=int(TransitionSource.HUMAN if human else TransitionSource.BASE),
        source_chunk=np.full(10, TransitionSource.HUMAN if human else TransitionSource.BASE, dtype=np.uint8),
        valid_mask=np.ones(10, dtype=np.bool_), executed_steps=10,
        collection_phase="online", intervention_flag=human, episode_id=episode, step_id=step)


def test_cpu_update_and_resume_discards_only_uncommitted_tail(tmp_path):
    config = SimpleNamespace(rlt_hidden_dim=8, state_dim=6, action_dim=6,
                             n_action_steps_rl=10, actor_hidden_dim=16, actor_num_layers=2,
                             critic_hidden_dim=16, critic_num_layers=2, policy_fixed_std=.05)
    actor = RLTChunkActor(config)
    trainer = Stage2Trainer(actor, config, tmp_path, batch_size=16, utd=1,
                            warmup_episodes=1, actor_start_updates=0)
    trainer.add_episode(1, [make_transition(1, 0), make_transition(1, 10, human=True)], 1)
    deadline = time.monotonic() + 10
    while trainer.updates < 2 and not trainer.worker_error and time.monotonic() < deadline:
        time.sleep(.05)
    trainer.close()
    assert trainer.updates >= 2
    assert trainer.replay.records[-1]["rewards"][-1] == 1
    checkpoint = tmp_path / "latest_checkpoint.pt"
    committed = (tmp_path / "replay_journal.pkl").stat().st_size
    with (tmp_path / "replay_journal.pkl").open("ab") as stream:
        stream.write(b"uncommitted_torn_pickle")
    resumed = Stage2Trainer(RLTChunkActor(config), config, tmp_path, resume=checkpoint,
                             batch_size=2, utd=1, warmup_episodes=1)
    try:
        assert resumed.episode == 1
        assert (tmp_path / "replay_journal.pkl").stat().st_size == committed
        assert len(resumed.replay) == 2
        resumed.add_episode(2, [make_transition(2, 0)], 0)
        resumed.save()
        assert (tmp_path / "previous_checkpoint.pt").is_file()
    finally:
        resumed.close()
