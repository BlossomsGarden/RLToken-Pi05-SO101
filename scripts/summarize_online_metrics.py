"""Summarize autonomous versus human-assisted outcomes from local episode files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, default=Path("rlt_episodes"))
    parser.add_argument("--metrics", type=Path, default=None,
                        help="Optional copied GPU metrics.jsonl")
    args = parser.parse_args()
    records = []
    for path in sorted(args.episodes_dir.glob("ep_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            records.append({"reward": int(archive["reward"]), "source": archive["source"].tolist()})
    autonomous = [record for record in records
                  if "HUMAN" not in record["source"]]
    assisted = [record for record in records
                if "HUMAN" in record["source"]]
    outcome = {"episodes": len(records),
               "autonomous_episodes": len(autonomous),
               "autonomous_successes": sum(record["reward"] == 1 for record in autonomous),
               "assisted_episodes": len(assisted),
               "assisted_successes": sum(record["reward"] == 1 for record in assisted),
               "actor_actions": sum(source == "ACTOR" for record in records for source in record["source"]),
               "vla_actions": sum(source == "VLA" for record in records for source in record["source"]),
               "human_actions": sum(source == "HUMAN" for record in records for source in record["source"])}
    if args.metrics:
        events = [json.loads(line) for line in args.metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
        updates = [event for event in events if event.get("type") == "update"]
        outcome["learner_updates"] = updates[-1]["updates"] if updates else 0
        outcome["last_critic_loss"] = updates[-1]["critic_loss"] if updates else None
    print(json.dumps(outcome, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
