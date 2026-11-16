"""GPU-only one-episode inference, replay, learner, and checkpoint smoke test."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from rlt_online.protocol import jpeg
from rlt_online.serve_rlt_brain import Brain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rlt_online.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="rlt_smoke_") as folder:
        config.update(mode="train", output=folder, warmup_journal=None,
                      resume_checkpoint=None, actor_checkpoint=None)
        brain = Brain(config)
        try:
            state = brain.mapping.to_physical((brain.state_low + brain.state_high) / 2).tolist()
            black = jpeg(np.zeros((480, 640, 3), dtype=np.uint8))
            observation = {"state": state, "front": black, "side": black}
            plan = brain.handle({"req_id": 1, "command": "plan", "episode": 1,
                                 "step": 0, "actor": False, "manual_critical": True, **observation})
            actions = plan["actions"][:10]
            boundary = brain.handle({"req_id": 2, "command": "boundary_slice", "episode": 1,
                                     "plan_id": plan["plan_id"], "step": 10,
                                     "actual": actions, "sources": ["VLA"] * 10,
                                     "critical_steps": [True] * 10, "actor": False,
                                     "manual_critical": True,
                                     "interrupted": True, **observation})
            finished = brain.handle({"req_id": 3, "command": "finish", "episode": 1,
                                     "step": 10, "reward": 0, "discard": False})
            assert len(actions) == 10 and boundary["req_id"] == 2
            assert finished["next_episode"] == 2
        finally:
            brain.close()
        assert (Path(folder) / "latest_checkpoint.pt").is_file()
        assert (Path(folder) / "replay_journal.pkl").is_file()
        print("GPU smoke passed: pi0.5 -> encoder -> ten-step replay -> checkpoint")


if __name__ == "__main__":
    main()
