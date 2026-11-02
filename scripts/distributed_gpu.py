"""GPU policy and optional online learner, reached only through SSH loopback forwarding.

Run in the *LeRobot 0.6.2* environment, not the OpenPI/JAX SFT environment.
"""

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from scripts.distributed_protocol import TASK, check_rows, check_state, decode_frame
from scripts.rlt_core import Transition, TransitionSource, quantile_normalize
from scripts.distributed_learner import OnlineLearner
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
from lerobot.policies.pi05_rlt.critical import CriticalPhaseDetector
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTokenEncoder, RLTChunkActor
from lerobot.policies.pi05_rlt.vla_compat import extract_embeddings
from lerobot.policies.utils import prepare_observation_for_inference


def load_quantiles(path, state_key, action_key):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for source, target in ((state_key, "state"), (action_key, "action")):
        item = data[source]
        q01, q99 = np.asarray(item["q01"], np.float32), np.asarray(item["q99"], np.float32)
        if q01.shape != (6,) or q99.shape != (6,) or not np.isfinite(q01).all() or not np.all(q99 > q01):
            raise ValueError(f"Invalid six-dimensional quantiles for {source} in {path}")
        result[target] = (q01, q99)
    return result


def absolute_from_delta(model_actions, state, delta_lo, delta_hi):
    """Unnormalize SFT delta joints and add the chunk observation state once."""
    state = check_state(state)
    if model_actions.ndim != 3 or model_actions.shape[2] != 6:
        raise ValueError("Expected batched six-joint normalized SFT actions")
    absolute = (model_actions + 1) * (delta_hi - delta_lo) / 2 + delta_lo
    absolute[:, :, :5] += torch.as_tensor(state[:5], device=model_actions.device)[None, None, :]
    return absolute


class GPUService:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU service requires CUDA")
        self.model = PI05Policy.from_pretrained(args.pi05_path).to(self.device).eval()
        self.pre, _ = make_pre_post_processors(self.model.config, pretrained_path=args.pi05_path)
        if self.model.config.chunk_size != 50 or self.model.config.max_action_dim != 32:
            raise ValueError("Expected H50 π0.5 with 32 padded action dimensions")
        if not self.model.config.use_relative_actions or "gripper" not in self.model.config.relative_exclude_joints:
            raise ValueError("SFT requires delta arm targets and absolute gripper")
        expected = {"observation.images.front", "observation.images.side", "observation.images.empty_camera_0"}
        if set(self.model.config.image_features) != expected:
            raise ValueError(f"Unexpected checkpoint camera contract: {self.model.config.image_features}")

        # The model's action q01/q99 describe DELTAS; Stage 2 actor/replay uses
        # ABSOLUTE joint targets. Mixing the two stats silently moves the arm.
        self.model_stats = load_quantiles(args.openpi_norm_stats, "state", "actions")
        self.absolute_stats = load_quantiles(args.absolute_stats, "observation.state", "action")
        self.state_lo, self.state_hi = [torch.tensor(v, device=self.device) for v in self.absolute_stats["state"]]
        self.action_lo, self.action_hi = [torch.tensor(v, device=self.device) for v in self.absolute_stats["action"]]
        self.delta_lo, self.delta_hi = [torch.tensor(v, device=self.device) for v in self.model_stats["action"]]
        with torch.no_grad():
            for p in self.model.parameters():
                p.requires_grad_(False)

        stage1 = torch.load(args.rlt_checkpoint, map_location=self.device, weights_only=False)
        if stage1.get("rlt_architecture") != "paper_v1" or not stage1.get("image_only", False):
            raise ValueError("Expected image-only paper_v1 Stage 1 checkpoint")
        self.cfg = PI05RLTConfig(mode="online_rl", state_dim=6, action_dim=6,
                                 n_action_steps_rl=10, action_stride=2)
        self.encoder = RLTokenEncoder(self.cfg).to(self.device).eval()
        self.encoder.load_state_dict(stage1["encoder_state_dict"], strict=True)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.actor = RLTChunkActor(self.cfg).to(self.device).eval()
        if args.actor_checkpoint:
            ckpt = torch.load(args.actor_checkpoint, map_location=self.device, weights_only=False)
            if ckpt.get("actor_contract") != "paper_full_output_v1" or ckpt.get("rl_chunk_length") != 10:
                raise ValueError("Incompatible Stage 2 actor checkpoint")
            self.actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
        elif args.mode == "infer":
            raise ValueError("Inference requires --actor-checkpoint")
        self.provenance = {"pi05_path": str(Path(args.pi05_path).resolve()),
                           "rlt_checkpoint": str(Path(args.rlt_checkpoint).resolve()),
                           "openpi_norm_stats": str(Path(args.openpi_norm_stats).resolve()),
                           "absolute_stats": str(Path(args.absolute_stats).resolve()), "task": TASK}
        self.learner = OnlineLearner(self.actor, self.cfg, self.device, args.output,
                                     restore_replay=args.restore_replay,
                                     batch_size=args.batch_size) if args.mode == "train" else None
        self.lock = threading.RLock()
        self.pending = None
        self.next_id = 1
        self.detector = CriticalPhaseDetector(
            args.critical_classifier, self.device, threshold_on=0.75,
            threshold_off=0.45, smooth_steps=4, min_on_chunks=3,
        ) if args.critical_classifier else None
        self.critical = False
        self.detected_once = False
        self.current_episode = None

    def _boundary(self, payload):
        state = check_state(payload["state"])
        raw = {"observation.state": state,
               "observation.images.front": decode_frame(payload["front_jpeg"]),
               "observation.images.side": decode_frame(payload["side_jpeg"])}
        batch = self.pre(prepare_observation_for_inference(raw, self.device, task=TASK))
        with torch.no_grad():
            model_actions = self.model.predict_action_chunk(batch).float()
            if model_actions.shape != (1, 50, 6):
                raise ValueError(f"Unexpected model chunk {tuple(model_actions.shape)}")
            # Original OpenPI quantiles normalize delta action first; convert
            # back to degrees, add the SAME observation state, leave gripper abs.
            absolute = absolute_from_delta(model_actions, state, self.delta_lo, self.delta_hi)
            reference = quantile_normalize(absolute[:, :10], self.action_lo, self.action_hi)
            images, img_masks = self.model._preprocess_images(batch)
            padded = torch.zeros(1, 50, 32, dtype=model_actions.dtype, device=self.device)
            padded[:, :, :6] = model_actions
            embeddings, _, embedding_mask = extract_embeddings(
                self.model, images, img_masks, batch["observation.language.tokens"],
                batch["observation.language.attention_mask"], padded,
                chunk_size=50, max_action_dim=32, image_only=True,
            )
            z_rl = self.encoder(embeddings.float(), mask=embedding_mask)
            proprio = quantile_normalize(torch.as_tensor(state, device=self.device), self.state_lo, self.state_hi)
            boundary = {"z": z_rl[0].cpu().numpy(), "proprio": proprio.cpu().numpy(),
                        "raw_state": state.copy(), "ref": reference[0].cpu().numpy()}
        return boundary, absolute[0, :10], reference

    def _record(self, previous, next_boundary, *, terminal=False, reward=0):
        if self.pending is None or previous["id"] != self.pending["id"]:
            raise ValueError("Stale or missing action-window id; refusing replay append")
        actual = check_rows(previous["actual"])
        count = len(actual)
        source = previous["source"]
        if source not in ("VLA", "ACTOR", "HUMAN"):
            raise ValueError("Unknown action source")
        origin = self.pending
        if self.learner is not None and previous["critical"]:
            normalized = quantile_normalize(torch.tensor(actual, device=self.device),
                                            self.action_lo, self.action_hi).cpu().numpy()
            actions = np.concatenate([normalized, np.repeat(normalized[-1:], 10-count, axis=0)])
            src = {"VLA": TransitionSource.BASE, "ACTOR": TransitionSource.RL,
                   "HUMAN": TransitionSource.HUMAN}[source]
            rewards = np.zeros(10, dtype=np.float32)
            if terminal:
                rewards[count - 1] = reward
            transition = Transition(
                z_rl=origin["z"], proprio=origin["proprio"], ref_chunk=origin["ref"],
                action_chunk=actions, rewards=rewards, done=terminal,
                next_z_rl=next_boundary["z"], next_proprio=next_boundary["proprio"],
                next_ref_chunk=next_boundary["ref"], source=int(src),
                source_chunk=np.full(10, int(src), dtype=np.uint8),
                valid_mask=np.arange(10) < count, executed_steps=count,
                collection_phase="warmup" if previous.get("warmup") else "online",
                success=int(terminal and reward == 1), intervention_flag=(source == "HUMAN"),
                episode_id=int(previous["episode"]), step_id=int(previous["step"]),
            )
            self.learner.add(transition)
        self.pending = None

    def plan(self, payload):
        with self.lock:
            if self.learner is not None:
                self.learner.raise_if_failed()
            if payload.get("task", TASK) != TASK:
                raise ValueError("This RL service accepts only the red-on-green task")
            episode = int(payload["episode"])
            if self.current_episode != episode:
                if self.pending is not None:
                    raise ValueError("Finish the previous episode before planning a new one")
                self.current_episode = episode
                self.critical = self.detected_once = False
                if self.detector:
                    self.detector.reset()
            boundary, absolute, reference = self._boundary(payload)
            if self.pending is not None:
                self._record(payload["previous"], boundary)
            elif payload.get("previous") is not None:
                raise ValueError("Unexpected previous window")
            if self.detector is None or payload.get("manual_override"):
                self.critical = bool(payload.get("critical"))
            elif int(payload["step"]) >= 60 and not (self.detected_once and not self.critical):
                # The classifier checkpoint standardizes RAW degrees using
                # its own mean/std; Actor/Critic use q01/q99-normalized state.
                desired = self.detector.update(boundary["z"], boundary["raw_state"])
                if desired is not None:
                    self.critical = desired
                    self.detected_once |= desired
            actor = bool(payload.get("actor")) and self.critical
            if self.learner is not None:
                actor &= self.learner.updates >= self.args.actor_start_updates
            if actor:
                with (self.learner.lock if self.learner else self.lock):
                    with torch.no_grad():
                        mean, _ = self.actor(torch.tensor(boundary["z"], device=self.device),
                                             torch.tensor(boundary["proprio"], device=self.device), reference)
                        actions = (mean[0] + 1) * (self.action_hi - self.action_lo) / 2 + self.action_lo
                source = "ACTOR"
            else:
                actions, source = absolute, "VLA"
            rows = check_rows(actions.detach().cpu().numpy()).tolist()
            identifier = self.next_id
            self.next_id += 1
            self.pending = {**boundary, "id": identifier}
            return {"ok": True, "id": identifier, "actions": rows, "source": source,
                    "critical": self.critical,
                    "updates": self.learner.updates if self.learner else None}

    def finish(self, payload):
        with self.lock:
            if self.learner is not None:
                self.learner.raise_if_failed()
            episode = int(payload["episode"])
            if self.pending is not None:
                previous = payload.get("previous")
                if previous is None and not payload.get("abort", False):
                    raise ValueError("Episode ended with an unreported action window")
                if previous is not None and int(previous["episode"]) != episode:
                    raise ValueError("Episode id mismatch")
                if not payload.get("abort", False):
                    reward = payload["reward"]
                    if reward not in (0, 1):
                        raise ValueError("Terminal reward must be 0 or 1")
                    self._record(previous, self.pending, terminal=True, reward=reward)
                else:
                    self.pending = None
            path = self.learner.save(episode, self.provenance) if self.learner else None
            return {"ok": True, "checkpoint": str(path) if path else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "infer"), required=True)
    parser.add_argument("--pi05-path", required=True)
    parser.add_argument("--rlt-checkpoint", required=True)
    parser.add_argument("--openpi-norm-stats", required=True)
    parser.add_argument("--absolute-stats", required=True)
    parser.add_argument("--actor-checkpoint")
    parser.add_argument("--critical-classifier", help="Optional Stage 4 classifier checkpoint")
    parser.add_argument("--restore-replay")
    parser.add_argument("--output", default="outputs/distributed_stage2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--actor-start-updates", type=int, default=1000)
    parser.add_argument("--port", type=int, default=6016)
    args = parser.parse_args()
    service = GPUService(args)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/rpc" or int(self.headers.get("Content-Length", 0)) > 2_000_000:
                self.send_error(400)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                command = body["command"]
                if command == "health":
                    result = {"ok": True, "mode": args.mode}
                elif command == "plan":
                    result = service.plan(body)
                elif command == "finish":
                    result = service.finish(body)
                else:
                    raise ValueError("Unknown command")
            except Exception as exc:
                logging.exception("Request failed")
                result = {"ok": False, "error": str(exc)}
            encoded = json.dumps(result, allow_nan=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    # Do not expose model or motor commands directly on the Internet.
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"GPU {args.mode} service listening on 127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        finally:
            if service.learner:
                service.learner.stop()


if __name__ == "__main__":
    main()
