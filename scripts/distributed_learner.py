"""GPU-side Stage 2 actor/critic updates using the copied paper MLP heads."""

import copy
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.rlt_core import TransitionSource
from scripts.train_rlt_stage2_pi05 import (
    ReplayBuffer, ReplayManager, discounted_chunk_return, polyak_update,
    sample_fixed_std_action, atomic_torch_save,
)
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTTwinCritic


class OnlineLearner:
    def __init__(self, actor, config, device, output, *, restore_replay=None,
                 actor_lr=3e-4, critic_lr=3e-4, batch_size=256, utd_ratio=5,
                 critic_per_actor=2, bc_weight=10.0, discount=0.99, target_tau=0.005,
                 reference_dropout=0.5):
        self.actor, self.device = actor, device
        self.actor_target = copy.deepcopy(actor).to(device)
        self.critic = RLTTwinCritic(config).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.lock = threading.RLock()
        self.batch_size, self.utd_ratio = batch_size, utd_ratio
        self.critic_per_actor, self.bc_weight = critic_per_actor, bc_weight
        self.discount, self.target_tau = discount, target_tau
        self.reference_dropout = reference_dropout
        self.output = Path(output)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"Use a new empty --output directory: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        self.replay = ReplayManager(ReplayBuffer(100000, sample_strategy="uniform"),
                                    str(self.output / "replay_journal.pkl"), restore=False)
        if restore_replay:
            # Restore a copy, never mutate the user's Stage 4 warmup journal.
            import shutil
            shutil.copyfile(restore_replay, self.output / "replay_journal.pkl")
            self.replay._restore_from_journal()
        self.updates = 0
        self.desired_updates = self.utd_ratio * len(self.replay._buffer)
        self.running = True
        self.worker_error = None
        self.worker = threading.Thread(target=self._run_worker, daemon=True, name="gpu-rl-learner")
        self.worker.start()

    def _run_worker(self):
        try:
            self._work()
        except Exception as exc:
            self.worker_error = exc
            self.running = False

    def raise_if_failed(self):
        if self.worker_error is not None:
            raise RuntimeError("GPU learner stopped after a training error") from self.worker_error

    def add(self, transition):
        self.replay.add_transition(transition)
        self.desired_updates += self.utd_ratio

    def _work(self):
        while self.running:
            if self.updates >= self.desired_updates or len(self.replay._buffer) < self.batch_size:
                time.sleep(0.05)
                continue
            with self.lock:
                batch = {k: v.to(self.device) for k, v in self.replay.sample_batch(self.batch_size).items()}
                with torch.no_grad():
                    mean, std = self.actor_target(batch["next_z_rl"], batch["next_proprio"],
                                                   batch["next_ref_chunk"])
                    next_action = sample_fixed_std_action(mean, std)
                    next_q = self.critic_target.q_min(batch["next_z_rl"], batch["next_proprio"], next_action)
                    returns, bootstrap = discounted_chunk_return(batch["rewards"], batch["valid_mask"], self.discount)
                    target = (returns + bootstrap * (1 - batch["done"]) * next_q).clamp(-2, 2)
                q1, q2 = self.critic(batch["z_rl"], batch["proprio"], batch["action_chunk"])
                loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
                self.critic_opt.step()
                self.updates += 1
                if self.updates % self.critic_per_actor == 0:
                    ref = batch["ref_chunk"]
                    if self.reference_dropout:
                        drop = torch.rand(ref.shape[0], device=self.device) < self.reference_dropout
                        ref = torch.where(drop[:, None, None], torch.zeros_like(ref), ref)
                    mean, std = self.actor(batch["z_rl"], batch["proprio"], ref, training=True)
                    action = sample_fixed_std_action(mean, std)
                    # Retain critic input gradients while freezing its weights.
                    for param in self.critic.parameters():
                        param.requires_grad_(False)
                    q = self.critic.q_min(batch["z_rl"].detach(), batch["proprio"], action).mean()
                    for param in self.critic.parameters():
                        param.requires_grad_(True)
                    human = (batch["source_chunk"] == int(TransitionSource.HUMAN)) | (batch["source_chunk"] == int(TransitionSource.MIXED))
                    bc_target = torch.where(human[:, :, None], batch["action_chunk"], batch["ref_chunk"])
                    errors = (action - bc_target).square().mean(-1)
                    valid = batch["valid_mask"]
                    bc = (errors * valid).sum() / valid.sum().clamp_min(1)
                    self.actor_opt.zero_grad(set_to_none=True)
                    (self.bc_weight * bc - q).backward()
                    self.actor_opt.step()
                    polyak_update(self.actor, self.actor_target, self.target_tau)
                    polyak_update(self.critic, self.critic_target, self.target_tau)

    def save(self, episode, provenance):
        self.raise_if_failed()
        with self.lock:
            path = self.output / "latest_checkpoint.pt"
            atomic_torch_save({
                "schema_version": 3, "actor_contract": "paper_full_output_v1", "rl_chunk_length": 10,
                "episode": episode, "total_updates": self.updates, "runtime_provenance": provenance,
                "actor_state_dict": self.actor.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_opt.state_dict(), "critic_optimizer": self.critic_opt.state_dict(),
                "replay_entries": self.replay.stats()["adds_total"],
            }, path)
            return path

    def stop(self):
        self.running = False
        self.worker.join(timeout=5)
