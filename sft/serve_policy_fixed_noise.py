#!/usr/bin/env python3
"""Serve an OpenPI checkpoint over WebSocket with JPEG inputs and fixed diffusion noise.

Purpose:
- Keep the user's current JPEG transport.
- Make π0/π0.5 sampling deterministic across repeated inference calls by using
  one fixed standard-normal noise tensor for every request.
- This is especially useful for diagnosing / preventing chunk-to-chunk
  stochastic action reversals during remote robot deployment.

Compatible command shape:
    python -m sft.serve_policy_fixed_noise \
      --config pi05_so101_pickplace_v3 \
      --checkpoint-dir /path/to/checkpoint \
      --port 6006 \
      --noise-seed 0
"""

from __future__ import annotations

import argparse
import io
import logging
import socket
import time
from typing import Any

import numpy as np
from PIL import Image

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
from sft.so101_config import register


FRONT_JPEG_KEY = "transport/front_image_jpeg"
SIDE_JPEG_KEY = "transport/side_image_jpeg"


def decode_rgb_jpeg(payload: bytes | bytearray | memoryview) -> np.ndarray:
    with Image.open(io.BytesIO(bytes(payload))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


class FixedNoiseJpegPolicy:
    """JPEG transport adapter + deterministic fixed-noise sampler."""

    def __init__(
        self,
        policy: Any,
        *,
        action_horizon: int,
        action_dim: int,
        noise_seed: int,
    ) -> None:
        self._policy = policy
        self._action_horizon = int(action_horizon)
        self._action_dim = int(action_dim)
        self._noise_seed = int(noise_seed)

        rng = np.random.default_rng(self._noise_seed)
        self._noise = rng.standard_normal(
            (self._action_horizon, self._action_dim)
        ).astype(np.float32)

        base_metadata = dict(getattr(policy, "metadata", {}) or {})
        self._metadata = {
            **base_metadata,
            "transport": "jpeg",
            "jpeg_input_keys": [FRONT_JPEG_KEY, SIDE_JPEG_KEY],
            "sampling": "fixed_noise",
            "noise_seed": self._noise_seed,
            "noise_shape": [self._action_horizon, self._action_dim],
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()

        if FRONT_JPEG_KEY not in obs:
            raise KeyError(f"Missing {FRONT_JPEG_KEY}")
        if SIDE_JPEG_KEY not in obs:
            raise KeyError(f"Missing {SIDE_JPEG_KEY}")

        decode_start = time.perf_counter()
        front = decode_rgb_jpeg(obs[FRONT_JPEG_KEY])
        side = decode_rgb_jpeg(obs[SIDE_JPEG_KEY])
        jpeg_decode_ms = (time.perf_counter() - decode_start) * 1000.0

        policy_obs = {
            "observation/front_image": front,
            "observation/side_image": side,
            "observation/state": np.asarray(
                obs["observation/state"],
                dtype=np.float32,
            ),
            "prompt": obs["prompt"],
        }

        # Important: same exact noise tensor for every request.
        result = self._policy.infer(policy_obs, noise=self._noise)

        total_ms = (time.perf_counter() - start) * 1000.0
        result["server_timing"] = {
            "infer_ms": total_ms,
            "jpeg_decode_ms": jpeg_decode_ms,
        }
        result["sampling_info"] = {
            "mode": "fixed_noise",
            "noise_seed": self._noise_seed,
        }
        return result


def resolve_model_shape(training_config: Any) -> tuple[int, int]:
    """Resolve model action horizon/dim from OpenPI TrainingConfig."""
    candidates = [
        getattr(training_config, "model", None),
        getattr(training_config, "model_config", None),
    ]

    for model_config in candidates:
        if model_config is None:
            continue

        action_horizon = getattr(model_config, "action_horizon", None)
        action_dim = getattr(model_config, "action_dim", None)

        if action_horizon is not None and action_dim is not None:
            return int(action_horizon), int(action_dim)

    raise RuntimeError(
        "Could not resolve action_horizon/action_dim from the training config. "
        "Inspect the config object before starting the server."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--default-prompt", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    register()
    training_config = _config.get_config(args.config)
    action_horizon, action_dim = resolve_model_shape(training_config)

    logging.info(
        "Loading policy config=%s checkpoint=%s",
        args.config,
        args.checkpoint_dir,
    )

    policy = _policy_config.create_trained_policy(
        training_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )

    wrapped = FixedNoiseJpegPolicy(
        policy,
        action_horizon=action_horizon,
        action_dim=action_dim,
        noise_seed=args.noise_seed,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    logging.info(
        "Fixed-noise JPEG server host=%s ip=%s port=%d seed=%d noise_shape=(%d,%d)",
        hostname,
        local_ip,
        args.port,
        args.noise_seed,
        action_horizon,
        action_dim,
    )

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped,
        host="127.0.0.1",  # Only accessible through the SSH local tunnel.
        port=args.port,
        metadata=wrapped.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
