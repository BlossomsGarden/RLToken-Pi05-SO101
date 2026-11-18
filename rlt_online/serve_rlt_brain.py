"""Frozen pi0.5 planning and asynchronous Stage 2 learning over binary WebSockets."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
from pathlib import Path

import numpy as np
import torch
from websockets.sync.server import serve

from rlt_online.action_space import absolute_to_delta, delta_to_absolute, normalize
from rlt_online.learner import Stage2Trainer
from rlt_online.joint_mapping import JointMapping
from rlt_online.protocol import MAX_MESSAGE_BYTES, SOURCES, TASK, image, joints, pack, unpack
from scripts.rlt_core import Transition, TransitionSource, openpi_quantiles

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def resolve_asset(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def quantiles(path, key):
    values = openpi_quantiles(resolve_asset(path))
    if key not in ("state", "actions"):
        raise ValueError(f"Invalid OpenPI quantile key: {key}")
    return values[:2] if key == "state" else values[2:]


def pad_chunk(rows, start, width=10):
    part = np.asarray(rows[start:start + width], dtype=np.float32)
    if not len(part):
        return np.zeros((width, 6), dtype=np.float32)
    return np.pad(part, ((0, width - len(part)), (0, 0)))


def source_id(sources):
    sources = set(sources)
    if not sources <= SOURCES or not sources:
        raise ValueError("Unknown or empty executed action sources")
    if len(sources) > 1:
        return int(TransitionSource.MIXED)
    return {"VLA": int(TransitionSource.BASE), "ACTOR": int(TransitionSource.RL),
            "HUMAN": int(TransitionSource.HUMAN)}[next(iter(sources))]


class Brain:
    def __init__(self, config):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
        from lerobot.policies.pi05_rlt.critical import CriticalPhaseDetector
        from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTokenEncoder, RLTChunkActor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the GPU brain")
        self.config, self.device = config, torch.device("cuda")
        self.mapping = JointMapping(json.loads(resolve_asset(config["joint_mapping_file"]).read_text(encoding="utf-8")))
        if config.get("require_nonidentity_mapping") and np.allclose(self.mapping.scale, 1) and np.allclose(self.mapping.offset, 0):
            raise ValueError("Measured joint mapping is required for the replacement arm")
        self.model = PI05Policy.from_pretrained(resolve_asset(config["pi05_checkpoint"])).to(self.device).eval()
        if (self.model.config.chunk_size != 50 or self.model.config.max_action_dim != 32
                or not self.model.config.use_relative_actions
                or set(self.model.config.relative_exclude_joints) != {"gripper"}):
            raise ValueError("Expected 50-step delta-arm/absolute-gripper pi0.5 checkpoint")
        self.pre, _ = make_pre_post_processors(self.model.config,
                                                 pretrained_path=resolve_asset(config["pi05_checkpoint"]))
        for weight in self.model.parameters():
            weight.requires_grad_(False)
        self.delta_low, self.delta_high = quantiles(config["openpi_norm_stats"], "actions")
        self.state_low, self.state_high = quantiles(config["openpi_norm_stats"], "state")
        stage1 = torch.load(resolve_asset(config["encoder_checkpoint"]), map_location=self.device, weights_only=False)
        if stage1.get("rlt_architecture") != "paper_v1" or not stage1.get("image_only"):
            raise ValueError("Stage 1 encoder contract does not match image-only RL Token")
        self.rlt_config = PI05RLTConfig(mode="online_rl", state_dim=6, action_dim=6,
                                        n_action_steps_rl=10, action_stride=2)
        self.encoder = RLTokenEncoder(self.rlt_config).to(self.device).eval()
        self.encoder.load_state_dict(stage1["encoder_state_dict"], strict=True)
        for weight in self.encoder.parameters():
            weight.requires_grad_(False)
        self.actor = RLTChunkActor(self.rlt_config).to(self.device).eval()
        mode = config["mode"]
        if mode == "train":
            self.trainer = Stage2Trainer(
                self.actor, self.rlt_config, resolve_asset(config["output"]),
                warmup_journal=resolve_asset(config["warmup_journal"]) if config.get("warmup_journal") else None,
                resume=resolve_asset(config["resume_checkpoint"]) if config.get("resume_checkpoint") else None,
                batch_size=256, utd=5, warmup_episodes=5, actor_start_updates=1000)
        elif mode == "infer":
            self.trainer = None
            checkpoint = torch.load(resolve_asset(config["actor_checkpoint"]), map_location=self.device, weights_only=False)
            if checkpoint.get("schema_version") != 3 or checkpoint.get("action_space_contract") != "openpi_delta_q01q99":
                raise ValueError("Expected Stage 2 checkpoint in OpenPI delta action space")
            self.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        else:
            raise ValueError("mode must be train or infer")
        self.lock = self.trainer.lock if self.trainer else threading.RLock()
        self.detector = CriticalPhaseDetector(
            str(resolve_asset(config["critical_classifier"])), self.device,
            threshold_on=.80, threshold_off=.50, smooth_steps=4, min_on_chunks=3,
        )
        rng = np.random.default_rng(int(config["fixed_noise_seed"]))
        self.noise = torch.from_numpy(rng.standard_normal((1, 50, 32)).astype(np.float32)).to(self.device)
        self.episode, self.plan_id, self.plan_start = None, None, None
        self.plan_actions, self.reference, self.raw_model_actions = None, None, None
        self.last_boundary, self.last_step = None, None
        self.critical, self.critical_since, self.detected_once = False, None, False
        self.transitions = []
        self.pending_next_ref = None
        self.deferred_slice = None
        self.plan_interrupted = False
        self.human_start_step = None
        self.human_reference = None

    def _batch(self, payload):
        from lerobot.policies.utils import prepare_observation_for_inference
        state = self.mapping.to_model(joints(payload["state"]))
        observation = {"observation.state": state,
                       "observation.images.front": image(payload["front"]),
                       "observation.images.side": image(payload["side"])}
        batch = self.pre(prepare_observation_for_inference(observation, self.device, task=TASK))
        return state, batch

    def _encode(self, state, batch, raw_model_actions):
        from lerobot.policies.pi05_rlt.vla_compat import extract_embeddings
        images, masks = self.model._preprocess_images(batch)
        padded = torch.zeros(1, 50, 32, device=self.device, dtype=raw_model_actions.dtype)
        padded[:, :, :6] = raw_model_actions
        embeddings, _, embedding_mask = extract_embeddings(
            self.model, images, masks, batch["observation.language.tokens"],
            batch["observation.language.attention_mask"], padded,
            chunk_size=50, max_action_dim=32, noise=self.noise, image_only=True)
        token = self.encoder(embeddings.float(), mask=embedding_mask)[0].cpu().numpy()
        proprio = normalize(state, self.state_low, self.state_high)
        return {"z": token, "proprio": proprio, "state": state}

    def _critical(self, boundary, payload, *, human=False):
        step = int(payload["step"])
        if payload.get("manual_critical") is not None:
            desired = bool(payload["manual_critical"])
        elif not human and step >= 60:
            desired = self.detector.update(boundary["z"], boundary["state"])
            if desired is None:
                desired = self.critical
            if self.detected_once and not self.critical and desired:
                desired = False
        else:
            desired = self.critical
        if desired != self.critical:
            self.critical_since = step if desired else None
        self.critical = desired
        self.detected_once |= desired

    def _actor_allowed(self, payload, step):
        return (bool(payload.get("actor")) and self.critical
                and self.critical_since is not None and step - self.critical_since >= 20
                and (self.trainer is None or (self.trainer.episodes_completed >= self.trainer.warmup_episodes
                                              and self.trainer.updates >= self.trainer.actor_start_updates)))

    def _actor_patch(self, boundary, start, payload):
        if not self._actor_allowed(payload, self.plan_start + start):
            return None
        chunks = []
        with torch.no_grad(), self.lock:
            z = torch.tensor(boundary["z"], dtype=torch.float32, device=self.device).unsqueeze(0)
            state = torch.tensor(boundary["proprio"], dtype=torch.float32, device=self.device).unsqueeze(0)
            reference = pad_chunk(self.raw_model_actions[0].float().cpu().numpy(), start)
            mean, _ = self.actor(z, state, torch.tensor(reference, device=self.device).unsqueeze(0))
            absolute = delta_to_absolute(np.clip(mean[0].float().cpu().numpy(), -1, 1),
                                         boundary["state"], self.delta_low, self.delta_high)
            chunks.extend(absolute[:min(10, 50 - start)].tolist())
        return {"start": start, "actions": chunks}

    def _record_slice(self, payload, boundary, *, human=False):
        step = int(payload["step"])
        if step <= self.last_step or (not human and step > self.plan_start + 50):
            raise ValueError("Replay requires ordered, at-most-10-step boundary slices")
        actual = self.mapping.to_model(joints(payload["actual"], rows=True, max_rows=10))
        sources = payload["sources"]
        if len(sources) != len(actual):
            raise ValueError("Every actual action requires its own source label")
        if len(payload["critical_steps"]) != len(actual):
            raise ValueError("Every action requires a critical-phase label")
        count = len(actual)
        if count == 0:
            raise ValueError("Boundary slices require executed actions")
        start = self.last_step
        if step - start > 10:
            LOG.warning("Skipping noncontiguous critical window at step %s (previous %s)", step, start)
            self.last_boundary, self.last_step, self.pending_next_ref = boundary, step, None
            return
        if count != step - start:
            raise ValueError("Actual action count must equal elapsed control steps")
        close_critical = bool(payload["critical_steps"][-1])
        record_window = close_critical and self.critical
        if not record_window:
            if boundary is not None:
                self.last_boundary = boundary
            self.last_step = step
            self.pending_next_ref = None
            return
        if boundary is None or self.last_boundary is None:
            raise ValueError("Critical replay requires both closing and opening observations")
        normalized = absolute_to_delta(actual, self.last_boundary["state"], self.delta_low, self.delta_high)
        if human and self.pending_next_ref is not None:
            self.pending_next_ref.next_ref_chunk = pad_chunk(self.human_reference, 0)
            self.pending_next_ref = None
        source_ids = np.asarray([source_id([source]) for source in sources], dtype=np.uint8)
        if human:
            if self.human_reference is None:
                raise ValueError("Human replay requires a fresh VLA snapshot at takeover")
            refs = pad_chunk(self.human_reference, start - self.human_start_step)
        else:
            refs = pad_chunk(self.raw_model_actions[0].float().cpu().numpy(), start - self.plan_start)
        for index, label in enumerate(sources):
            if label == "HUMAN":
                refs[index] = normalized[index]
        self.transitions.append(Transition(
            z_rl=self.last_boundary["z"], proprio=self.last_boundary["proprio"],
            ref_chunk=refs, action_chunk=pad_chunk(normalized, 0),
            rewards=np.zeros(10, dtype=np.float32), done=False,
            next_z_rl=boundary["z"], next_proprio=boundary["proprio"],
            next_ref_chunk=pad_chunk(self.human_reference if human else
                                     self.raw_model_actions[0].float().cpu().numpy(),
                                     step-(self.human_start_step if human else self.plan_start)),
            source=source_id(sources), source_chunk=np.pad(source_ids, (0, 10-count), constant_values=int(TransitionSource.BASE)),
            valid_mask=np.arange(10) < count, executed_steps=count,
            collection_phase=("warmup" if getattr(self, "trainer", None) is not None and
                              self.trainer.episodes_completed < self.trainer.warmup_episodes else "online"),
            intervention_flag="HUMAN" in sources, episode_id=self.episode, step_id=self.last_step))
        if step == self.plan_start + 50 or payload.get("interrupted") or human:
            self.pending_next_ref = self.transitions[-1]
        self.last_boundary, self.last_step = boundary, step

    def handle(self, payload):
        req_id = payload["req_id"]
        command = payload["command"]
        if command == "health":
            return {"req_id": req_id, "mode": self.config["mode"],
                    "next_episode": self.trainer.episode + 1 if self.trainer else 1,
                    "active_episode": self.episode}
        if self.trainer and self.trainer.worker_error:
            raise RuntimeError("Learner thread failed") from self.trainer.worker_error
        episode = int(payload["episode"])
        if command == "plan":
            if self.episode is not None and episode != self.episode:
                raise ValueError("Previous episode must be finished first")
            if self.episode is None:
                if self.trainer and episode < self.trainer.episode + 1:
                    raise ValueError("Episode id precedes resumed learner state")
                self.episode = episode
                self.detector.reset()
                self.critical = self.detected_once = False
                self.critical_since = None
                self.human_reference = self.human_start_step = None
                self.deferred_slice = None
            if self.plan_id is not None and int(payload["step"]) != self.plan_start + 50 and not self.plan_interrupted:
                raise ValueError("Finish the prior plan or interrupt it with an exact boundary slice")
            if self.plan_id is not None and (int(payload["step"]) < self.last_step or
                                             (not self.plan_interrupted and int(payload["step"]) > self.plan_start + 50)):
                raise ValueError("Plan must follow the previous control chunk")
            state, batch = self._batch(payload)
            with torch.no_grad(), self.lock, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model_actions = self.model.predict_action_chunk(batch, noise=self.noise).float()
                if tuple(model_actions.shape) != (1, 50, 6):
                    raise ValueError("Policy did not produce the required 50x6 chunk")
                raw_actions = model_actions.cpu().numpy()
                absolute = delta_to_absolute(raw_actions[0], state, self.delta_low, self.delta_high)
                boundary = self._encode(state, batch, model_actions)
            self._critical(boundary, payload)
            if self.deferred_slice is not None:
                staged = dict(self.deferred_slice)
                staged["critical_steps"] = [*staged["critical_steps"][:-1], self.critical]
                self._record_slice(staged, boundary)
                self.deferred_slice = None
            self.plan_id, self.plan_start = req_id, int(payload["step"])
            self.plan_interrupted = False
            self.reference, self.raw_model_actions = absolute, model_actions
            if self.pending_next_ref is not None and self.pending_next_ref.step_id + self.pending_next_ref.executed_steps == self.plan_start:
                self.pending_next_ref.next_ref_chunk = pad_chunk(raw_actions[0], 0)
            self.pending_next_ref = None
            self.plan_actions = absolute.copy()
            self.last_boundary, self.last_step = boundary, self.plan_start
            patch = self._actor_patch(boundary, 0, payload)
            if patch:
                self.plan_actions[:len(patch["actions"])] = patch["actions"]
                patch = {**patch, "actions": self.mapping.to_physical(patch["actions"]).tolist()}
            return {"req_id": req_id, "plan_id": self.plan_id,
                    "actions": self.mapping.to_physical(self.plan_actions).tolist(),
                    "actor_patch": patch, "critical": self.critical,
                    "critical_prob": self.detector.last_probability,
                    "updates": self.trainer.updates if self.trainer else None}
        if command == "boundary_slice":
            if episode != self.episode or payload["plan_id"] != self.plan_id:
                raise ValueError("Stale episode or plan boundary")
            if int(payload["step"]) > self.plan_start + 50:
                raise ValueError("Slice exceeds the parent VLA chunk")
            if payload.get("defer_close"):
                if (int(payload["step"]) != self.plan_start + 50 or payload.get("interrupted")
                        or self.deferred_slice is not None):
                    raise ValueError("Deferred slice must close exactly one complete parent chunk")
                self.deferred_slice = dict(payload)
                return {"req_id": req_id, "critical": self.critical,
                        "updates": self.trainer.updates if self.trainer else None}
            boundary = None
            if self.critical or any(payload["critical_steps"]) or payload.get("manual_critical"):
                state, batch = self._batch(payload)
                with torch.no_grad(), self.lock, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    boundary = self._encode(state, batch, self.raw_model_actions)
            # The classifier is updated at plan boundaries, not every 10-step slice.
            if payload.get("manual_critical") is not None:
                self._critical(boundary, payload, human=True)
            self._record_slice(payload, boundary)
            self.plan_interrupted |= bool(payload.get("interrupted"))
            offset = int(payload["step"]) - self.plan_start
            patch_start = offset
            patch = self._actor_patch(boundary, patch_start, payload) if patch_start < 50 and not payload.get("interrupted") else None
            if patch:
                patch = {**patch, "actions": self.mapping.to_physical(patch["actions"]).tolist()}
            return {"req_id": req_id, "plan_id": self.plan_id, "patch": patch,
                    "critical": self.critical, "critical_prob": self.detector.last_probability,
                    "updates": self.trainer.updates if self.trainer else None}
        if command == "human_slice":
            if episode != self.episode or self.plan_id is None or payload["plan_id"] != self.plan_id:
                raise ValueError("Human actions require an active parent plan")
            self.plan_interrupted = True
            if set(payload["sources"]) != {"HUMAN"}:
                raise ValueError("Human slices may only contain executed human actions")
            state, batch = self._batch(payload)
            with torch.no_grad(), self.lock, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                boundary = self._encode(state, batch, self.raw_model_actions)
            if payload.get("manual_critical") is not None:
                self._critical(boundary, payload, human=True)
            self._record_slice(payload, boundary, human=True)
            return {"req_id": req_id, "critical": self.critical,
                    "critical_prob": self.detector.last_probability,
                    "updates": self.trainer.updates if self.trainer else None}
        if command == "interrupt_plan":
            if (episode != self.episode or payload["plan_id"] != self.plan_id or
                int(payload["step"]) < self.last_step or int(payload["step"]) > self.plan_start + 50):
                raise ValueError("Interrupt must follow the acknowledged parent boundary")
            if self.last_step < int(payload["step"]):
                self.pending_next_ref = None
            self.last_step = int(payload["step"])
            self.plan_interrupted = True
            if self.transitions and self.transitions[-1].step_id + self.transitions[-1].executed_steps == self.last_step:
                self.pending_next_ref = self.transitions[-1]
            return {"req_id": req_id, "critical": self.critical,
                    "critical_prob": self.detector.last_probability,
                    "updates": self.trainer.updates if self.trainer else None}
        if command == "snapshot":
            if (episode != self.episode or payload["plan_id"] != self.plan_id or
                int(payload["step"]) < self.last_step or
                (not self.plan_interrupted and int(payload["step"]) > self.plan_start + 50)):
                raise ValueError("Snapshot requires the last acknowledged takeover boundary")
            if int(payload["step"]) > self.last_step and self.deferred_slice is None:
                self.pending_next_ref = None
            state, batch = self._batch(payload)
            with torch.no_grad(), self.lock, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw = self.model.predict_action_chunk(batch, noise=self.noise).float()
                if tuple(raw.shape) != (1, 50, 6):
                    raise ValueError("Snapshot must contain a 50x6 VLA proposal")
                boundary = self._encode(state, batch, raw)
            if self.deferred_slice is not None:
                if int(payload["step"]) != int(self.deferred_slice["step"]):
                    raise ValueError("Takeover snapshot must close the staged window")
                self._record_slice(self.deferred_slice, boundary)
                self.deferred_slice = None
            self.last_step = int(payload["step"])
            self.plan_interrupted = True
            self.human_start_step = self.last_step
            self.human_reference = raw[0].cpu().numpy()
            self.last_boundary = boundary
            if self.pending_next_ref is not None and self.pending_next_ref.step_id + self.pending_next_ref.executed_steps == self.last_step:
                self.pending_next_ref.next_ref_chunk = pad_chunk(self.human_reference, 0)
                self.pending_next_ref = None
            return {"req_id": req_id, "critical": self.critical,
                    "updates": self.trainer.updates if self.trainer else None}
        if command == "terminal_snapshot":
            step = int(payload["step"])
            if (episode != self.episode or payload["plan_id"] != self.plan_id
                    or step < self.last_step or
                    (not self.plan_interrupted and step > self.plan_start + 50)):
                raise ValueError("Terminal snapshot must follow the final executed action")
            state, batch = self._batch(payload)
            with torch.no_grad(), self.lock, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw = self.model.predict_action_chunk(batch, noise=self.noise).float()
                boundary = self._encode(state, batch, raw)
            if self.deferred_slice is not None:
                if step != int(self.deferred_slice["step"]) or payload.get("actual"):
                    raise ValueError("Terminal snapshot must close the staged parent window")
                self._record_slice(self.deferred_slice, boundary)
                self.deferred_slice = None
            elif payload.get("actual"):
                self._record_slice(payload, boundary, human=set(payload["sources"]) == {"HUMAN"})
            else:
                self.last_boundary, self.last_step = boundary, step
            if self.transitions and self.transitions[-1].step_id + self.transitions[-1].executed_steps == step:
                self.transitions[-1].next_ref_chunk = pad_chunk(raw[0].cpu().numpy(), 0)
            self.pending_next_ref = None
            self.plan_interrupted = True
            return {"req_id": req_id, "critical": self.critical,
                    "updates": self.trainer.updates if self.trainer else None}
        if command == "finish":
            if episode != self.episode or self.last_step > int(payload["step"]):
                raise ValueError("Finish cannot precede the last acknowledged boundary")
            if not self.plan_interrupted and self.plan_start is not None and int(payload["step"]) > self.plan_start + 50:
                raise ValueError("Finish cannot exceed the parent plan without an interruption")
            if payload.get("discard"):
                self.transitions.clear()
            elif self.trainer:
                self.trainer.add_episode(episode, self.transitions, int(payload["reward"]))
            next_episode = self.trainer.episode + 1 if self.trainer else episode + int(not payload.get("discard"))
            self.discard_active_episode()
            return {"req_id": req_id, "next_episode": next_episode}
        raise ValueError("Unknown brain command")

    def discard_active_episode(self):
        """Forget an uncommitted episode after discard or client disconnect."""
        self.episode = self.plan_id = self.plan_start = None
        self.plan_actions = self.reference = self.raw_model_actions = None
        self.last_boundary = self.last_step = None
        self.critical, self.critical_since, self.detected_once = False, None, False
        self.transitions = []
        self.pending_next_ref = self.deferred_slice = None
        self.plan_interrupted = False
        self.human_reference = self.human_start_step = None

    def close(self):
        if self.trainer:
            self.trainer.close()


def handle_client(brain, socket):
    last_req = -1
    claimed_episode = None
    try:
        for message in socket:
            req_id = None
            payload = None
            try:
                payload = unpack(message)
                req_id = payload["req_id"]
                if req_id <= last_req:
                    raise ValueError("Requests must be strictly ordered on each connection")
                last_req = req_id
                if payload["command"] == "plan" and brain.episode is None:
                    claimed_episode = int(payload["episode"])
                reply = brain.handle(payload)
                if payload["command"] == "finish":
                    claimed_episode = None
                reply["ok"] = True
            except Exception as exc:
                LOG.exception("Brain request failed")
                if (payload is not None and payload.get("command") == "plan"
                        and claimed_episode is not None and brain.episode == claimed_episode):
                    brain.discard_active_episode()
                    claimed_episode = None
                reply = {"req_id": req_id, "ok": False, "error": str(exc)}
            socket.send(pack(reply))
    finally:
        if claimed_episode is not None and brain.episode == claimed_episode:
            LOG.warning("Discarding uncommitted episode %s after client disconnect", claimed_episode)
            brain.discard_active_episode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "rlt_online.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO)
    brain = Brain(config)
    client_lock = threading.Lock()
    def handler(socket):
        with client_lock:
            handle_client(brain, socket)
    with serve(handler, config.get("host", "127.0.0.1"), int(config.get("port", 6016)), max_size=MAX_MESSAGE_BYTES) as server:
        def shutdown(_signal, _frame):
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        try:
            server.serve_forever()
        finally:
            brain.close()


if __name__ == "__main__":
    main()
