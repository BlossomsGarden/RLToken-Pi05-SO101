"""SO-101 Windows control client for frozen pi0.5, Stage 2 Actor and human takeover."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from pathlib import Path

import numpy as np

from rlt_online.broker import BrainBroker
from rlt_online.joint_mapping import JOINTS
from rlt_online.protocol import jpeg, joints
from sft.camera_env import camera_setting

ROOT = Path(__file__).resolve().parents[1]


def positions(values):
    return joints([values[f"{name}.pos"] for name in JOINTS])


def command(values):
    return {f"{name}.pos": float(values[i]) for i, name in enumerate(JOINTS)}


def actor_requested(keys, args):
    """Smooth 50-step execution records replay; Actor runs in window-replan mode."""
    return keys.actor and not args.no_replan_every_window


def apply_actor_patch(plan, patch, index):
    start = int(patch["start"])
    rows = joints(patch["actions"], rows=True, max_rows=10)
    end = start + len(rows)
    first = max(start, index)
    if start < 0 or end > 50 or first >= end or (index > start and index // 10 != start // 10):
        return False
    plan["actions"][first:end] = rows[first - start:]
    plan["actor_start"] = start
    return True


def observation(robot):
    raw = robot.get_observation()
    return {"state": positions(raw).tolist(), "front": jpeg(raw["front"]), "side": jpeg(raw["side"])}


def read_env(path):
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="gbk", errors="replace").splitlines()
    result = {}
    for raw in lines:
        if "=" not in raw or raw.lstrip().startswith("#"):
            continue
        key, value = raw.split("=", 1)
        parsed = shlex.split(value.strip())
        result[key.strip()] = " ".join(parsed) if parsed else ""
    return result


def configure(cfg, env):
    for key, field in (("FOLLOWER_PORT", "follower_port"), ("FOLLOWER_ID", "follower_id"),
                       ("LEADER_PORT", "leader_port"), ("LEADER_ID", "leader_id")):
        if key in env:
            cfg[field] = env[key]
    for canonical, alias, field in (("FRONT_CAMERA", "CAM_FRONT", "front"),
                                    ("SIDE_CAMERA", "CAM_SIDE", "side")):
        _, value = camera_setting(env, canonical, alias)
        if value is not None:
            cfg["cameras"][field]["index_or_path"] = int(value) if value.isdecimal() else value
    if "RLT_HOST" in env or "RLT_PORT" in env:
        cfg["brain_uri"] = f"ws://{env.get('RLT_HOST', '127.0.0.1')}:{int(env.get('RLT_PORT', 6016))}"
    return cfg


def next_local_episode(folder):
    numbers = [int(p.stem[3:]) for p in folder.glob("ep_*.npz") if p.stem[3:].isdigit()]
    return max(numbers, default=0) + 1


def save_episode(folder, episode, record):
    """Save compact numeric telemetry; JPEG images stay on the transport only."""
    folder.mkdir(parents=True, exist_ok=True)
    aborted = bool(record.get("aborted", False))
    filename = (f"aborted_ep_{episode:04d}_{time.time_ns()}.npz" if aborted
                else f"ep_{episode:04d}.npz")
    destination = folder / filename
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(".npz.tmp")
    steps = record["steps"]
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, episode=np.int32(episode), reward=np.int8(record["reward"]),
            aborted=np.bool_(aborted),
            state=np.asarray([x["state"] for x in steps], dtype=np.float32).reshape(-1, 6),
            action=np.asarray([x["action"] for x in steps], dtype=np.float32).reshape(-1, 6),
            leader=np.asarray([x["leader"] for x in steps], dtype=np.float32).reshape(-1, 6),
            source=np.asarray([x["source"] for x in steps], dtype="U5"),
            critical=np.asarray([x["critical"] for x in steps], dtype=np.bool_),
            critical_prob=np.asarray([x["critical_prob"] for x in steps], dtype=np.float32),
            chunk_id=np.asarray([x["chunk_id"] for x in steps], dtype=np.int64),
            step_ms=np.asarray([x["step_ms"] for x in steps], dtype=np.float32),
            boundary_step=np.asarray([x["step"] for x in record["boundaries"]], dtype=np.int32),
            boundary_req=np.asarray([x["req_id"] for x in record["boundaries"]], dtype=np.int64),
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return destination


def abort_episode(broker, folder, episode, step, record, timeout, error):
    """Best-effort replay discard and local audit after a mid-episode failure."""
    try:
        broker.submit({"command": "finish", "episode": episode, "step": step,
                       "reward": 0, "discard": True}).result(timeout=timeout)
    except Exception as cleanup_error:
        print(f"Brain discard unavailable ({type(cleanup_error).__name__}); "
              "the server will clear the episode when this connection closes")
    record["reward"] = -1
    record["aborted"] = True
    try:
        audit = save_episode(folder, episode, record)
    except Exception as audit_error:
        print(f"Aborted episode audit failed ({type(audit_error).__name__})")
    else:
        print(f"Episode {episode} aborted ({type(error).__name__}); audit saved: {audit}")


class Keys:
    def __init__(self, *, auto_critical, actor=False, actor_mode=True):
        self.human = False
        self.actor = bool(actor)
        self.actor_mode = bool(actor_mode)
        self.manual_critical = None if auto_critical else False
        self.auto_critical = auto_critical
        self.manual_until = None
        self.outcome = None
        self.quit = False
        self.emergency = False
        self.changed = False
        self.actor_changed = False
        self.phase_changed = False

    def poll(self, current_critical=None):
        import msvcrt
        self.changed = self.actor_changed = self.phase_changed = False
        while msvcrt.kbhit():
            key = msvcrt.getwch().lower()
            if key == "h" and not self.human:
                self.human, self.changed = True, True
            elif key == "s" and self.human:
                self.human, self.changed = False, True
            elif key == "c":
                self.manual_critical = not (bool(current_critical) if self.auto_critical else bool(self.manual_critical))
                if self.auto_critical:
                    self.manual_until = time.monotonic() + 3
                self.phase_changed = True
                print(f"Critical phase: {self.manual_critical}")
            elif key == "a":
                if not self.actor_mode:
                    print("Actor requires window replanning; restart without --no-replan-every-window")
                    continue
                self.actor = not self.actor
                self.actor_changed = True
                print(f"Actor execution: {'enabled' if self.actor else 'disabled'}")
            elif key == "f":
                self.outcome = "finish"
            elif key == "g":
                self.outcome = "discard"
            elif key == "q":
                self.quit, self.outcome = True, "discard"
            elif key == "e":
                self.emergency, self.outcome = True, "discard"

    def manual_override(self):
        if not self.auto_critical or (self.manual_until is not None and
                                      time.monotonic() < self.manual_until):
            return self.manual_critical
        return None


class GripperHold:
    def __init__(self, config):
        self.config = config
        self.target = None
        self.count = 0

    def apply(self, robot, requested, actual):
        cfg = self.config
        result = requested.copy()
        result[5] = max(result[5], cfg["min_deg"])
        if self.target is not None:
            if requested[5] >= self.target + cfg["release_margin_deg"]:
                self.target, self.count = None, 0
            else:
                result[5] = self.target
                return result
        current = robot.bus.read("Present_Current", "gripper", normalize=False)
        self.count = self.count + 1 if current >= cfg["current_threshold_raw"] else 0
        if self.count >= cfg["consecutive"] and actual[5] - result[5] >= cfg["closing_error_deg"]:
            self.target = max(cfg["min_deg"], actual[5] - cfg["hold_offset_deg"])
            self.count = 0
            result[5] = self.target
            print("Gripper current HOLD active")
        return result


def safe_action(robot, target, hold, cfg, last, *, human=False, present=None):
    target = joints(target)
    if last[5] < cfg["gripper_hold"]["min_deg"]:
        raise RuntimeError("Gripper below verified safe floor before control")
    low = np.asarray(cfg["physical_min_deg"], dtype=np.float32)
    high = np.asarray(cfg["physical_max_deg"], dtype=np.float32)
    if np.any(target[:5] < low[:5]) or np.any(target[:5] > high[:5]) or target[5] > high[5]:
        raise RuntimeError("Target outside verified physical joint limits")
    if present is None:
        present = positions({f"{name}.pos": value for name, value in
                             robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries).items()})
    if not human and np.any(np.abs(last - present) > np.asarray(cfg["max_tracking_error_deg"])):
        raise RuntimeError("Follower tracking watchdog tripped; control halted")
    limits = np.asarray(cfg["max_command_step_deg"])
    limited = np.clip(target - last, -limits, limits) + last
    if not human:
        limited = hold.apply(robot, limited, present)
    else:
        limited[5] = max(limited[5], cfg["gripper_hold"]["min_deg"])
    if np.any(np.abs(limited - last) > limits + 1e-5):
        raise RuntimeError("Command exceeded the verified rate limit")
    return positions(robot.send_action(command(limited)))


class LeaderMirror:
    """Low-torque, smoothed feedback; a takeover offset avoids manual alignment."""
    def __init__(self, leader, *, ratio=.5, smoothing=.35, max_step_deg=5):
        self.leader, self.smoothing, self.max_step_deg = leader, smoothing, max_step_deg
        self.enabled = False
        if not 0 < ratio <= 1:
            raise ValueError("Leader torque fraction must lie within (0, 1]")
        self.low, self.high = self._reachable_limits(leader)
        # Address 48 is volatile SRAM. Never write Max_Torque_Limit at EEPROM address 16.
        for name in JOINTS:
            leader.bus.write("Torque_Limit", name, int(1000 * ratio), normalize=False)

    @staticmethod
    def _reachable_limits(leader):
        bus = leader.bus
        if not bus.calibration or set(bus.calibration) != set(JOINTS):
            raise ValueError("Leader calibration must cover all six joints before mirroring")
        low, high = [], []
        for name in JOINTS:
            calibration = bus.calibration[name]
            if calibration.range_min >= calibration.range_max:
                raise ValueError(f"Invalid leader calibration range for {name}")
            if name == "gripper":
                low.append(0.0)
                high.append(100.0)
            else:
                resolution = bus.model_resolution_table[bus.motors[name].model] - 1
                half_range = (calibration.range_max - calibration.range_min) * 180 / resolution
                low.append(-half_range)
                high.append(half_range)
        return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)

    def resume(self):
        current = positions(self.leader.get_action())
        self.leader.send_feedback(command(np.clip(current, self.low, self.high)))
        self.leader.enable_torque()
        self.enabled = True

    def feedback(self, follower):
        if self.enabled:
            current = positions(self.leader.get_action())
            target = current + np.clip((follower - current) * self.smoothing,
                                       -self.max_step_deg, self.max_step_deg)
            self.leader.send_feedback(command(np.clip(target, self.low, self.high)))

    def relax(self):
        if self.enabled:
            self.leader.disable_torque()
        self.enabled = False

    def close(self):
        self.relax()


def wait_settled(robot, cfg):
    tolerance = np.asarray(cfg["settle_tolerance_deg"], dtype=np.float32)
    deadline = time.monotonic() + cfg["settle_timeout_s"]
    previous, stable = None, 0
    while time.monotonic() < deadline:
        present = robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries)
        current = joints([present[name] for name in JOINTS])
        stable = stable + 1 if previous is not None and np.all(np.abs(current - previous) <= tolerance) else 0
        if stable >= 4:
            return True
        previous = current
        time.sleep(.05)
    print("Warning: follower did not settle before timeout; continuing with measured pose")
    return False


def reset_follower_torque(robot):
    """Let an operator position the arm, then hold its new pose before moving."""
    robot.bus.disable_torque()
    input("Reset arm and blocks to demo start; Enter to begin: ")
    parked = robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries)
    robot.bus.sync_write("Goal_Position", parked)
    robot.bus.enable_torque()


def terminal_label():
    while True:
        answer = input("本集结果 [y=成功 / n=失败 / Enter=放弃]: ").strip().lower()
        if answer in ("", "y", "n"):
            return {"": "discard", "y": 1, "n": 0}[answer]
        print("Please press y, n or Enter")


def run(args, cfg):
    folder = ROOT / cfg["episodes_dir"]
    broker = BrainBroker(cfg["brain_uri"], timeout=cfg["network_timeout_s"],
                         capacity=cfg["max_pending_boundaries"])
    robot = leader = mirror = None
    try:
        health = broker.submit({"command": "health"}).result(timeout=cfg["network_timeout_s"])
        print(f"Brain: {health['mode']}, next episode {health['next_episode']}")
        if os.name != "nt" or not cfg["calibration_verified"] or not cfg["mapping_verified"] or not cfg["safety_verified"]:
            raise RuntimeError("Windows and verified calibration, mapping, safety settings are required")
        low, high = np.asarray(cfg["physical_min_deg"]), np.asarray(cfg["physical_max_deg"])
        if low.shape != (6,) or high.shape != (6,) or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high):
            raise ValueError("Enter six verified physical minimum and maximum joint limits")
        for name in ("max_command_step_deg", "max_tracking_error_deg", "settle_tolerance_deg"):
            bound = np.asarray(cfg[name], dtype=np.float32)
            if bound.shape != (6,) or not np.isfinite(bound).all() or np.any(bound <= 0):
                raise ValueError(f"Verify six positive joint limits for {name}")
        if health["active_episode"] is not None:
            raise RuntimeError("Brain has unfinished episode; inspect partial data before resuming")
        episode = max(next_local_episode(folder), int(health["next_episode"]))
        if episode != health["next_episode"]:
            print(f"Local files set episode {episode}; remote next is {health['next_episode']}. Inspect lost checkpoint data.")
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.so_follower.so_follower import SO101Follower
        from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
        from lerobot.teleoperators.so_leader.so_leader import SO101Leader
        cameras = {name: OpenCVCameraConfig(**config) for name, config in cfg["cameras"].items()}
        if set(cameras) != {"front", "side"}:
            raise ValueError("Configure and verify both front and side cameras")
        robot = SO101Follower(SO101FollowerConfig(port=cfg["follower_port"], id=cfg["follower_id"],
                               cameras=cameras, max_relative_target=cfg["max_relative_target"]))
        robot.connect(calibrate=not args.no_calibration)
        if not args.execute:
            result = broker.submit({"command": "plan", "episode": episode, "step": 0,
                                    "actor": False, "manual_critical": False if args.no_auto_critical else None,
                                    **observation(robot)}).result(timeout=cfg["network_timeout_s"])
            try:
                current = positions(robot.get_observation())
                first = joints(result["actions"], rows=True)[0]
                if np.max(np.abs(first[:5] - current[:5])) > 40:
                    raise RuntimeError("First mapped action differs from measured state by over 40 degrees")
                print(f"Dry run: state={current.tolist()}, first client action={first.tolist()}; no motion")
            finally:
                broker.submit({"command": "finish", "episode": episode, "step": 0,
                               "reward": 0, "discard": True}).result(timeout=cfg["network_timeout_s"])
            return
        leader = SO101Leader(SO101LeaderConfig(port=cfg["leader_port"], id=cfg["leader_id"]))
        leader.connect(calibrate=not args.no_calibration)
        mirror = LeaderMirror(leader)
        print("a=actor h=takeover s=resume c=critical f=finish g=skip q=quit e=emergency")
        finished = 0
        actor_enabled = False
        while args.episodes == 0 or finished < args.episodes:
            mirror.relax()
            print(f"Episode {episode}: place the arm and blocks at the demo start pose")
            reset_follower_torque(robot)
            keys = Keys(auto_critical=not args.no_auto_critical, actor=actor_enabled,
                        actor_mode=not args.no_replan_every_window)
            mirror.resume()
            step, index, plan = 0, 0, None
            offset = None
            critical, critical_prob = False, float("nan")
            recent, sources, labels, pending = [], [], [], []
            record = {"steps": [], "boundaries": []}
            hold = GripperHold(cfg["gripper_hold"])
            last = positions({f"{name}.pos": value for name, value in
                              robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries).items()})
            last_reported_update = None
            human_start = None

            def submit_slice(*, interrupt=False, defer_close=False, close_critical=None):
                nonlocal recent, sources, labels
                if not recent:
                    return False
                closing = labels[-1] if close_critical is None else bool(close_critical)
                if not closing and not defer_close:
                    recent, sources, labels = [], [], []
                    return False
                if plan is None:
                    raise RuntimeError("Cannot record executed actions without a parent plan")
                snapshot = {} if defer_close else observation(robot)
                payload = {"command": "human_slice" if all(s == "HUMAN" for s in sources) else "boundary_slice",
                           "episode": episode, "plan_id": plan["plan_id"], "step": step,
                           "actual": recent, "sources": sources,
                           "critical_steps": [*labels[:-1], closing],
                           "interrupted": bool(interrupt), "defer_close": defer_close,
                           "actor": actor_requested(keys, args),
                           "manual_critical": keys.manual_override(), **snapshot}
                future = broker.submit(payload)
                pending.append(future)
                record["boundaries"].append({"step": step, "req_id": future.request_id})
                recent, sources, labels = [], [], []
                if len(pending) > cfg["max_pending_boundaries"]:
                    raise RuntimeError("Network backlog exceeded safe boundary limit")
                return True

            def collect_replies(*, wait=False):
                nonlocal critical, critical_prob, last_reported_update, plan
                while pending and (wait or pending[0].done()):
                    reply = pending.pop(0).result(timeout=cfg["network_timeout_s"] if wait else 0)
                    if "critical" in reply:
                        critical = bool(reply["critical"])
                    if reply.get("critical_prob") is not None:
                        critical_prob = float(reply["critical_prob"])
                    updates = reply.get("updates")
                    if updates is not None and (last_reported_update is None or updates // 50 != last_reported_update // 50):
                        print(f"Episode {episode}: learner updates={updates}, critical={critical}")
                    last_reported_update = updates
                    patch = reply.get("patch")
                    if patch and plan is not None and reply.get("plan_id") == plan["plan_id"]:
                        apply_actor_patch(plan, patch, index)

            brain_finished = False
            try:
                while step < args.steps:
                    keys.poll(critical)
                    if keys.phase_changed:
                        critical = bool(keys.manual_critical)
                    if keys.emergency:
                        robot.bus.disable_torque()
                    if keys.outcome is not None:
                        break
                    collect_replies()
                    if (keys.phase_changed or keys.actor_changed) and not keys.human and plan is not None and index < 50:
                        if not submit_slice(interrupt=True, close_critical=critical if keys.phase_changed else None):
                            pending.append(broker.submit({"command": "interrupt_plan", "episode": episode,
                                                           "plan_id": plan["plan_id"], "step": step}))
                        collect_replies(wait=True)
                        plan = None
                    elif keys.phase_changed and keys.human and not keys.changed and plan is not None:
                        submit_slice(interrupt=True, close_critical=critical)
                        pending.append(broker.submit({"command": "snapshot", "episode": episode,
                                                       "plan_id": plan["plan_id"], "step": step,
                                                       **observation(robot)}))
                        human_start = step
                    if keys.changed and keys.human:
                        if plan is None:
                            keys.human = False
                            print("Wait for a plan before taking over")
                        else:
                            mirror.relax()
                            submit_slice(interrupt=True, close_critical=critical if keys.phase_changed else None)
                            pending.append(broker.submit({"command": "snapshot", "episode": episode,
                                                           "plan_id": plan["plan_id"], "step": step,
                                                           **observation(robot)}))
                            human_start = step
                            offset = last - positions(leader.get_action())
                            print("Human takeover: model proposal refreshed without executing it")
                    elif keys.changed and not keys.human:
                        submit_slice(interrupt=True)
                        collect_replies(wait=True)
                        mirror.resume()
                        plan = None
                        print("Autonomous replan from current observation")
                    if not keys.human and (plan is None or index >= 50):
                        submit_slice()
                        collect_replies(wait=True)
                        result = broker.submit({"command": "plan", "episode": episode, "step": step,
                                                "actor": actor_requested(keys, args),
                                                "manual_critical": keys.manual_override(),
                                                **observation(robot)}).result(timeout=cfg["network_timeout_s"])
                        plan = {**result, "actions": joints(result["actions"], rows=True).copy(),
                                "actor_start": 0 if result.get("actor_patch") else -1}
                        critical = bool(result["critical"])
                        if result.get("critical_prob") is not None:
                            critical_prob = float(result["critical_prob"])
                        index = 0
                    tick = time.monotonic()
                    present = positions({f"{name}.pos": value for name, value in
                                         robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries).items()})
                    leader_position = positions(leader.get_action())
                    if keys.human:
                        desired = leader_position + offset
                        source = "HUMAN"
                    else:
                        desired = plan["actions"][index]
                        source = ("ACTOR" if plan["actor_start"] >= 0 and
                                  index // 10 == plan["actor_start"] // 10 else "VLA")
                    last = safe_action(robot, desired, hold, cfg, last, human=keys.human, present=present)
                    if not keys.human:
                        mirror.feedback(last)
                    record["steps"].append({"state": present.tolist(), "action": last.tolist(),
                                            "leader": leader_position.tolist(), "source": source,
                                            "critical": bool(critical), "critical_prob": critical_prob,
                                            "chunk_id": plan["plan_id"], "step_ms": (time.monotonic()-tick)*1000})
                    recent.append(last.tolist())
                    sources.append(source)
                    labels.append(bool(critical))
                    step += 1
                    if not keys.human:
                        index += 1
                    if len(recent) == 10 or (not keys.human and index == 50):
                        if not keys.human and index == 50:
                            wait_settled(robot, cfg)
                        replan = not keys.human and not args.no_replan_every_window and index < 50
                        emitted = submit_slice(interrupt=replan, defer_close=not keys.human and index == 50)
                        if replan and not emitted:
                            pending.append(broker.submit({"command": "interrupt_plan", "episode": episode,
                                                           "plan_id": plan["plan_id"], "step": step}))
                        if replan:
                            collect_replies(wait=True)
                            plan = None
                        if keys.human and human_start is not None and step-human_start >= 10:
                            pending.append(broker.submit({"command": "snapshot", "episode": episode,
                                                           "plan_id": plan["plan_id"], "step": step,
                                                           **observation(robot)}))
                            human_start = step
                    delay = 1/cfg["fps"] - (time.monotonic()-tick)
                    if delay > 0:
                        time.sleep(delay)
                if keys.outcome == "finish" or keys.outcome is None:
                    keys.outcome = terminal_label()
                if step == 0 and keys.outcome != "discard":
                    raise RuntimeError("Cannot label an episode with no executed actions")
                collect_replies(wait=True)
                if keys.outcome != "discard":
                    wait_settled(robot, cfg)
                    final = broker.submit({"command": "terminal_snapshot", "episode": episode,
                                           "plan_id": plan["plan_id"], "step": step,
                                           "actual": recent, "sources": sources,
                                           "critical_steps": labels, "interrupted": True,
                                           "manual_critical": keys.manual_override(), **observation(robot)})
                    record["boundaries"].append({"step": step, "req_id": final.request_id})
                    final.result(timeout=cfg["network_timeout_s"])
                result = broker.submit({"command": "finish", "episode": episode, "step": step,
                                        "reward": keys.outcome if keys.outcome != "discard" else 0,
                                        "discard": keys.outcome == "discard"}).result(timeout=cfg["network_timeout_s"])
                brain_finished = True
                mirror.relax()
                if keys.outcome != "discard":
                    record["reward"] = keys.outcome
                    save_episode(folder, episode, record)
                    finished += 1
                print(f"Episode {episode}: {step} steps, label={keys.outcome}; next={result['next_episode']}")
                actor_enabled = keys.actor
                episode = max(next_local_episode(folder), int(result["next_episode"]))
                if keys.quit or keys.emergency:
                    break
            except BaseException as error:
                if not brain_finished:
                    try:
                        mirror.relax()
                    except Exception:
                        pass
                    abort_episode(broker, folder, episode, step, record,
                                  cfg["network_timeout_s"], error)
                else:
                    print("Brain already acknowledged the result; inspect the local archive and replay")
                raise
    finally:
        broker.close()
        if mirror is not None:
            mirror.close()
        if leader is not None and leader.is_connected:
            leader.disconnect()
        if robot is not None and robot.is_connected:
            robot.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "robot.json")
    default_env = ROOT / ".env"
    if not default_env.is_file():
        default_env = ROOT.parent / ".env"
    parser.add_argument("--env-file", type=Path, default=default_env)
    parser.add_argument("--episodes", type=int, default=0, help="0 continues until q")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--no-replan-every-window", action="store_true")
    parser.add_argument("--no-auto-critical", action="store_true")
    args = parser.parse_args()
    cfg = configure(json.loads(args.config.read_text(encoding="utf-8")), read_env(args.env_file))
    if cfg["fps"] != 30 or args.episodes < 0 or args.steps < 1:
        parser.error("Expected 30 fps, nonnegative episode count and positive step limit")
    run(args, cfg)


if __name__ == "__main__":
    main()
