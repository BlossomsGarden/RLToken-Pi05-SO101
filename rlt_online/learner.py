"""Durable episode replay and asynchronous RLT actor/critic training."""

from __future__ import annotations

import copy
import json
import os
import pickle
import threading
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTTwinCritic
from scripts.rlt_core import COLLECTION_PHASE_ONLINE, COLLECTION_PHASE_WARMUP, TransitionSource


class Replay:
    def __init__(self, path: Path, capacity: int = 100_000):
        self.path, self.capacity = path, capacity
        self.records: list[dict] = []
        self.entries = 0
        self.rng = np.random.default_rng(0)
        if path.exists():
            with path.open("rb") as stream:
                while True:
                    try:
                        record = pickle.load(stream)
                    except EOFError:
                        break
                    self._append_memory(record)
        self._rebuild_pools()

    def _rebuild_pools(self):
        self.all_indices = np.arange(len(self.records), dtype=np.int64)
        if not self.records:
            self.recent_indices = self.warmup_indices = self.human_indices = self.all_indices
            return
        online_episodes = [int(record["episode_id"]) for record in self.records
                           if int(record["collection_phase_id"]) == COLLECTION_PHASE_ONLINE]
        latest_episode = max(online_episodes, default=0)
        self.recent_indices = np.asarray([
            i for i, record in enumerate(self.records)
            if int(record["collection_phase_id"]) == COLLECTION_PHASE_ONLINE
            and int(record["episode_id"]) >= latest_episode - 19], dtype=np.int64)
        self.warmup_indices = np.asarray([
            i for i, record in enumerate(self.records)
            if int(record["collection_phase_id"]) == COLLECTION_PHASE_WARMUP], dtype=np.int64)
        self.human_indices = np.asarray([
            i for i, record in enumerate(self.records)
            if bool(record["intervention_flag"]) or
            np.any(record["source_chunk"] == int(TransitionSource.HUMAN))], dtype=np.int64)

    def _append_memory(self, record):
        if not {"valid_mask", "source_chunk", "next_ref_chunk", "rewards"} <= record.keys():
            raise ValueError("Replay record does not contain the required online action-window fields")
        self.records.append(record)
        if len(self.records) > self.capacity:
            self.records.pop(0)
        self.entries += 1

    def append_episode(self, transitions):
        if not transitions:
            return
        records = [transition.to_numpy() for transition in transitions]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as stream:
            for record in records:
                pickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        for record in records:
            self._append_memory(record)
        self._rebuild_pools()

    def sample(self, size, device):
        if not self.records:
            raise ValueError("Cannot sample from an empty replay")
        picks = self.rng.choice(self.all_indices, size=size, replace=True)
        keys = self.records[0]
        return {key: torch.from_numpy(np.stack([self.records[i][key] for i in picks]).astype(np.float32)).to(device)
                for key in keys}

    def __len__(self):
        return len(self.records)


def chunk_return(rewards, mask, gamma):
    mask = mask.float()
    powers = gamma ** torch.arange(rewards.shape[1], device=rewards.device)
    value = (rewards * mask * powers).sum(1, keepdim=True)
    continuation = gamma ** mask.sum(1, keepdim=True)
    return value, continuation


def polyak(source, target, tau):
    with torch.no_grad():
        for left, right in zip(source.parameters(), target.parameters()):
            right.mul_(1 - tau).add_(left, alpha=tau)


class Stage2Trainer:
    def __init__(self, actor, config, output: Path, *, warmup_journal: Path | None = None,
                 resume: Path | None = None, batch_size=256, utd=5, warmup_episodes=5,
                 actor_start_updates=1000, save_every=20):
        self.actor = actor
        self.device = next(actor.parameters()).device
        self.target_actor = copy.deepcopy(actor)
        self.critic = RLTTwinCritic(config).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.journal = output / "replay_journal.pkl"
        self.metrics = output / "metrics.jsonl"
        if resume and warmup_journal:
            raise ValueError("Resume checkpoint and warmup replay cannot both be supplied")
        if warmup_journal:
            if self.journal.exists():
                raise FileExistsError("Refusing to overwrite an existing replay journal")
            import shutil
            shutil.copyfile(warmup_journal, self.journal)
        # A torn append after the last atomic checkpoint is uncommitted; roll
        # its journal tail back before unpickling and rebuilding the replay.
        if resume:
            committed = torch.load(resume, map_location="cpu", weights_only=False)
            if committed.get("schema_version") != 3 or committed.get("action_space_contract") != "openpi_delta_q01q99":
                raise ValueError("Incompatible Stage 2 checkpoint action space")
            size = int(committed["journal_bytes"])
            if not self.journal.exists() or self.journal.stat().st_size < size:
                raise ValueError("Replay journal is missing committed checkpoint data")
            if self.journal.stat().st_size > size:
                with self.journal.open("r+b") as stream:
                    stream.truncate(size)
                    stream.flush()
                    os.fsync(stream.fileno())
        self.replay = Replay(self.journal)
        self.lock = threading.RLock()   # Also guards GPU inference in the brain.
        self.stop_event = threading.Event()
        self.batch_size, self.utd = batch_size, utd
        self.warmup_episodes, self.actor_start_updates = warmup_episodes, actor_start_updates
        self.save_every = save_every
        self.episode = 0
        self.episodes_completed = 0
        self.updates = 0
        self.worker_error = None
        if resume:
            self._resume(resume)
        elif self.replay.entries and not warmup_journal:
            raise ValueError("Existing journal requires its matching checkpoint")
        self.worker = threading.Thread(target=self._train_loop, name="rlt-learner", daemon=False)
        self.worker.start()

    def _resume(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("schema_version") != 3 or checkpoint.get("action_space_contract") != "openpi_delta_q01q99":
            raise ValueError("Incompatible Stage 2 checkpoint action space")
        if checkpoint.get("replay_entries") != self.replay.entries or checkpoint.get("journal_bytes") != self.journal.stat().st_size:
            raise ValueError("Checkpoint and replay journal must represent the same committed episodes")
        for key, module in (("actor_state_dict", self.actor), ("actor_target_state_dict", self.target_actor),
                            ("critic_state_dict", self.critic), ("critic_target_state_dict", self.target_critic)):
            module.load_state_dict(checkpoint[key], strict=True)
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.updates, self.episode = int(checkpoint["total_updates"]), int(checkpoint["episode"])
        self.episodes_completed = int(checkpoint.get("episodes_completed", len({
            int(record["episode_id"]) for record in self.replay.records
        })))
        self.replay.rng.bit_generator.state = checkpoint["replay_rng_state"]
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    def add_episode(self, episode: int, transitions, reward: int):
        if episode < self.episode + 1 or reward not in (0, 1):
            raise ValueError("Out-of-order episode or invalid binary reward")
        with self.lock:
            if transitions:
                for transition in transitions:
                    transition.done = False
                    transition.rewards[:] = 0
                transitions[-1].done = True
                transitions[-1].rewards[transitions[-1].executed_steps - 1] = reward
                transitions[-1].success = reward
                self.replay.append_episode(transitions)
                self.episodes_completed += 1
            self.episode = episode
            if transitions and self.episodes_completed % self.save_every == 0:
                self.save()
            with self.metrics.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"type": "episode", "episode": episode, "success": reward,
                                         "episodes_completed": self.episodes_completed,
                                         "transitions": len(transitions), "replay": len(self.replay),
                                         "updates": self.updates, "has_replay": bool(transitions),
                                         "discard": False}) + "\n")

    def save(self):
        with self.lock:
            path = self.output / "latest_checkpoint.pt"
            payload = {
                "schema_version": 3, "actor_contract": "paper_full_output_v1", "rl_chunk_length": 10,
                "action_space_contract": "openpi_delta_q01q99",
                "episode": self.episode, "episodes_completed": self.episodes_completed,
                "total_updates": self.updates,
                "actor_state_dict": self.actor.state_dict(), "actor_target_state_dict": self.target_actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(), "critic_target_state_dict": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "replay_entries": self.replay.entries,
                "journal_bytes": self.journal.stat().st_size if self.journal.exists() else 0,
                "replay_rng_state": self.replay.rng.bit_generator.state,
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            previous = self.output / "previous_checkpoint.pt"
            previous_temp = previous.with_name(f".{previous.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("wb") as stream:
                    torch.save(payload, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                if path.exists():
                    import shutil
                    shutil.copyfile(path, previous_temp)
                    with previous_temp.open("r+b") as stream:
                        os.fsync(stream.fileno())
                    os.replace(previous_temp, previous)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
                previous_temp.unlink(missing_ok=True)
            return path

    def _train_loop(self):
        try:
            while not self.stop_event.is_set():
                if (self.episodes_completed < self.warmup_episodes or len(self.replay) < self.batch_size
                        or self.updates >= self.utd * self.replay.entries):
                    self.stop_event.wait(.05)
                    continue
                with self.lock:
                    batch = self.replay.sample(self.batch_size, self.device)
                    with torch.no_grad():
                        mean, std = self.target_actor(batch["next_z_rl"], batch["next_proprio"], batch["next_ref_chunk"])
                        next_action = mean + std * torch.randn_like(mean)
                        q = self.target_critic.q_min(batch["next_z_rl"], batch["next_proprio"], next_action)
                        value, gamma = chunk_return(batch["rewards"], batch["valid_mask"], .99)
                        target = (value + gamma * (1 - batch["done"].reshape(-1, 1)) * q).clamp(-2, 2)
                    q1, q2 = self.critic(batch["z_rl"], batch["proprio"], batch["action_chunk"])
                    loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
                    self.critic_optimizer.step()
                    self.updates += 1
                    if self.updates % 2 == 0:
                        refs = batch["ref_chunk"].clone()
                        drop = torch.rand(refs.shape[0], device=self.device) < .5
                        refs[drop] = 0
                        mean, std = self.actor(batch["z_rl"], batch["proprio"], refs, training=True)
                        actions = mean + std * torch.randn_like(mean)
                        with torch.no_grad():
                            for parameter in self.critic.parameters():
                                parameter.requires_grad_(False)
                        actor_q = self.critic.q_min(batch["z_rl"].detach(), batch["proprio"], actions).mean()
                        for parameter in self.critic.parameters():
                            parameter.requires_grad_(True)
                        human = (batch["source_chunk"] == int(TransitionSource.HUMAN)) | (batch["source_chunk"] == int(TransitionSource.MIXED))
                        bc_target = torch.where(human.unsqueeze(-1), batch["action_chunk"], batch["ref_chunk"])
                        errors = ((actions - bc_target) ** 2).mean(-1)
                        bc = (errors * batch["valid_mask"]).sum() / batch["valid_mask"].sum().clamp_min(1)
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        (10 * bc - actor_q).backward()
                        self.actor_optimizer.step()
                        polyak(self.actor, self.target_actor, .005)
                        polyak(self.critic, self.target_critic, .005)
                    if self.updates % 50 == 0:
                        with self.metrics.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"type": "update", "episode": self.episode,
                                                     "updates": self.updates, "replay": len(self.replay),
                                                     "critic_loss": float(loss.detach().cpu()),
                                                     "actor_loss": float((10 * bc - actor_q).detach().cpu())}) + "\n")
        except Exception as error:
            self.worker_error = error
            self.stop_event.set()

    def close(self):
        self.stop_event.set()
        self.worker.join()
        if self.worker_error:
            raise RuntimeError("Stage 2 learner failed") from self.worker_error
        self.save()
