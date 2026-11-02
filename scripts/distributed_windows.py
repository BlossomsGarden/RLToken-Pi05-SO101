"""Windows SO-101 follower + leader HIL controller for the remote RL GPU service.

Action windows run locally at 30 Hz. Nothing moves without --execute; the
server only listens on loopback and must be reached through an SSH -L tunnel.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.distributed_protocol import JOINTS, TASK, check_rows, check_state, encode_frame, request


def action_dict(values):
    row = check_state(values)
    return {f"{name}.pos": float(row[index]) for index, name in enumerate(JOINTS)}


def state_from(mapping):
    return check_state([mapping[f"{name}.pos"] for name in JOINTS])


def present_state(robot):
    """Read just the six joints on each control tick; images are per window."""
    positions = robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries)
    return check_state([positions[name] for name in JOINTS])


def observation(robot, episode, step):
    raw = robot.get_observation()
    return {"state": state_from(raw).tolist(),
            "front_jpeg": encode_frame(raw["front"]),
            "side_jpeg": encode_frame(raw["side"]),
            "task": TASK, "episode": episode, "step": step}


class Controls:
    def __init__(self):
        self.human = False
        self.actor = True
        self.critical = False
        self.stop = False
        self.abort = False
        self.finish = False
        self.emergency = False
        self.manual_override_at = float("-inf")

    def poll(self):
        import msvcrt

        while msvcrt.kbhit():
            key = msvcrt.getwch().lower()
            if key == "h":
                self.human = not self.human
                print(f"HIL {'human' if self.human else 'autonomous'}")
            elif key == "a":
                self.actor = not self.actor
                print(f"Actor {'enabled' if self.actor else 'disabled'}")
            elif key == "c":
                self.critical = not self.critical
                self.manual_override_at = time.monotonic()
                print(f"Critical phase {'on' if self.critical else 'off'}")
            elif key in ("q", "g", "e", "f"):
                self.stop |= key == "q"
                self.abort |= key == "g"
                self.emergency |= key == "e"
                self.finish |= key == "f"


def mirror(leader, follower, enabled, max_align):
    if not enabled:
        leader.disable_torque()
        return
    target = state_from(follower.get_observation())
    actual = state_from(leader.get_action())
    if np.max(np.abs(target - actual)) > max_align:
        raise RuntimeError("Leader is not aligned to follower; align both arms before enabling mirror")
    leader.send_feedback(action_dict(target))
    leader.enable_torque()


def run(args, config):
    mode = request(args.port, {"command": "health"})["mode"]
    print(f"Remote GPU: {mode}; task: {TASK}")
    if not args.execute:
        print("Connection dry run complete; no cameras or motors opened")
        return
    if os.name != "nt":
        raise RuntimeError("This robot driver entry point requires Windows")
    if "LEADER" in config["leader_port"] or not config["leader_port"]:
        raise ValueError("Set the real leader COM port in the local config")

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from lerobot.teleoperators.so_leader.so_leader import SO101Leader

    cameras = {key: OpenCVCameraConfig(**item) for key, item in config["cameras"].items()}
    if set(cameras) != {"front", "side"}:
        raise ValueError("Exactly front and side cameras are required")
    robot = SO101Follower(SO101FollowerConfig(port=config["follower_port"],
                            id=config["follower_id"], cameras=cameras,
                            max_relative_target=config["max_relative_target"]))
    leader = SO101Leader(SO101LeaderConfig(port=config["leader_port"], id=config["leader_id"]))
    robot_connected = leader_connected = False
    controls = Controls()
    try:
        robot.connect()
        robot_connected = True
        leader.connect()
        leader_connected = True
        mirror(leader, robot, True, config["max_leader_alignment_deg"])
        print("Keys: h=HIL, c=critical, a=actor, f=finish, g=skip, e=stop, q=quit")
        for episode in range(1, args.episodes + 1):
            if controls.stop or controls.emergency:
                break
            controls.critical = False
            controls.human = False
            step = 0
            previous = None
            while step < args.steps:
                controls.poll()
                if controls.emergency:
                    robot.bus.disable_torque()
                if controls.stop or controls.abort or controls.emergency or controls.finish:
                    break
                payload = {"command": "plan", **observation(robot, episode, step),
                           "previous": previous,
                           "actor": controls.actor and not controls.human
                           and (mode == "infer" or episode > args.warmup_episodes),
                           "critical": controls.critical,
                           "manual_override": time.monotonic() - controls.manual_override_at < 3.0}
                result = request(args.port, payload, timeout=args.timeout)
                controls.critical = bool(result["critical"])
                proposed = check_rows(result["actions"])
                source = "HUMAN" if controls.human else result["source"]
                window_critical = controls.critical
                window_start = step
                actual = []
                for row in proposed:
                    tick = time.monotonic()
                    was_human = controls.human
                    was_critical = controls.critical
                    controls.poll()
                    if controls.human != was_human:
                        mirror(leader, robot, not controls.human, config["max_leader_alignment_deg"])
                        if actual:
                            break  # Exact partial window, replan at the new state.
                        source = "HUMAN" if controls.human else result["source"]
                    if controls.critical != was_critical and actual:
                        break
                    if controls.critical != was_critical:
                        window_critical = controls.critical
                    if controls.emergency:
                        robot.bus.disable_torque()
                    if controls.stop or controls.abort or controls.emergency or controls.finish:
                        break
                    target = state_from(leader.get_action()) if controls.human else row
                    current = present_state(robot)
                    if not controls.human and np.any(np.abs(target - current) > config["max_tracking_error_deg"]):
                        raise RuntimeError("Tracking watchdog: predicted target is too far from measured pose")
                    if not controls.human:
                        if np.any(np.abs(target - current) > config["max_command_step_deg"]):
                            raise RuntimeError("Joint command step limit exceeded; stopping robot")
                    accepted = state_from(robot.send_action(action_dict(target)))
                    actual.append(accepted.tolist())
                    if not controls.human:
                        leader.send_feedback(action_dict(accepted))
                    step += 1
                    remaining = 1.0 / config["fps"] - (time.monotonic() - tick)
                    if remaining > 0:
                        time.sleep(remaining)
                    if step >= args.steps:
                        break
                previous = ({"id": result["id"], "actual": actual, "source": source,
                             "critical": window_critical, "warmup": mode == "train" and episode <= args.warmup_episodes,
                             "episode": episode, "step": window_start} if actual else None)
                if not actual:
                    # Never silently leave a pending server window behind.
                    controls.abort = True
                    break
            aborted = controls.abort or controls.stop or controls.emergency
            reward = None if aborted else int(input("Episode success? [y/N] ").strip().lower() == "y")
            request(args.port, {"command": "finish", "episode": episode,
                                "previous": previous, "abort": bool(aborted), "reward": reward})
            print(f"Episode {episode}: steps={step}, result={'aborted' if aborted else reward}")
            controls.abort = controls.finish = False
            if controls.emergency:
                break
    finally:
        if leader_connected:
            leader.disable_torque()
            leader.disconnect()
        if robot_connected:
            robot.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/distributed_windows.json"))
    parser.add_argument("--port", type=int, default=6016)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-episodes", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["fps"] != 30 or args.episodes < 1 or args.steps < 1:
        parser.error("Expected 30fps and positive episode/step counts")
    run(args, config)


if __name__ == "__main__":
    main()
