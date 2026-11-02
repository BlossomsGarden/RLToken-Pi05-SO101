#!/usr/bin/env python3
"""SO-ARM101 OpenPI remote client with JPEG transport, async prefetch,
execute-horizon control, trajectory resampling, no-rebase async execution, command-rate limiting, and tracking watchdog diagnostics.

Designed for a local Windows LeRobot SO-101 follower and two USB cameras.

Works with sft/serve_policy_fixed_noise.py on the OpenPI server.

Safety:
- Default is DRY-RUN.
- Real motion requires --execute.
- Safety is split into command-rate limiting and a tracking-error watchdog.

Key addition in v2:
- The model returns the configured 50-step action chunk.
- --execute-horizon controls how many leading actions are actually executed
  before switching to a newly inferred chunk.
- Optional trajectory resampling stretches the predicted path in time without extrapolation.
- Per-joint command-rate limiting and max tracking error are reported.
"""

from __future__ import annotations

import csv

import argparse
import io
import os
import queue
import shlex
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

EXPECTED_ACTION_DIM = len(MOTOR_NAMES)
MODEL_IMAGE_SIZE = 224
RAW_TWO_IMAGE_BYTES = 2 * MODEL_IMAGE_SIZE * MODEL_IMAGE_SIZE * 3
CLIP_ATOL = 1e-4


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        try:
            parsed = shlex.split(raw_value)
            values[key] = " ".join(parsed) if parsed else ""
        except ValueError as exc:
            raise ValueError(f"Invalid value in {path}: {raw_line}") from exc

    return values


def env_value(env: dict[str, str], name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, env.get(name, default))


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--env-file", type=Path, default=project_root / ".env")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--task", default=None)

    parser.add_argument("--jpeg-quality", type=int, default=90)

    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Total number of robot actions to execute. Only used with --execute.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send actions to the robot. Default is DRY-RUN.",
    )
    parser.add_argument(
        "--execute-horizon",
        type=int,
        default=6,
        help=(
            "How many leading actions from each returned model chunk to execute "
            "before switching to a newly inferred chunk. Default: 6."
        ),
    )
    parser.add_argument(
        "--resample-steps",
        type=int,
        default=None,
        help=(
            "Optional number of execution waypoints obtained by linearly "
            "resampling the leading --execute-horizon model actions. "
            "Example: horizon=10, resample-steps=15 stretches the same "
            "10-action trajectory over 15 commands while preserving endpoints."
        ),
    )
    parser.add_argument(
        "--prefetch-after",
        type=int,
        default=1,
        help=(
            "Submit the next observation for inference after this many actions "
            "of the current executed horizon have been sent. Default: 1."
        ),
    )
    parser.add_argument(
        "--max-boundary-wait",
        type=float,
        default=1.0,
        help="Maximum seconds to wait at a chunk boundary before aborting.",
    )

    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--command-step-limits",
        type=float,
        nargs=6,
        metavar=("PAN", "LIFT", "ELBOW", "WRIST", "ROLL", "GRIPPER"),
        default=(5.0, 7.0, 7.5, 4.5, 1.25, 6.5),
        help=(
            "Per-joint max change between consecutive commanded absolute targets. "
            "Defaults are based on approximately the p99.5 demonstration "
            "|action[t]-action[t-1]| values."
        ),
    )
    parser.add_argument(
        "--tracking-error-limits",
        type=float,
        nargs=6,
        metavar=("PAN", "LIFT", "ELBOW", "WRIST", "ROLL", "GRIPPER"),
        default=(27.0, 37.0, 45.0, 29.0, 10.0, 30.0),
        help=(
            "Per-joint watchdog limits for |safe_target-Present_Position|. "
            "Exceeding one aborts before sending the action. Defaults are "
            "roughly the p99.5 demonstration |action-state| envelope, rounded up."
        ),
    )
    parser.add_argument("--no-calibration", action="store_true")

    return parser


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise ValueError(f"JPEG input must be uint8, got {image.dtype}")
    if image.shape != (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected image shape: {image.shape}")

    buffer = io.BytesIO()
    Image.fromarray(image).save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=False,
        subsampling=2,
    )
    return buffer.getvalue()


def make_observation(
    robot: Any,
    image_tools: Any,
    task: str,
    jpeg_quality: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    start = time.perf_counter()

    raw = robot.get_observation()

    state = np.asarray(
        [raw[f"{motor}.pos"] for motor in MOTOR_NAMES],
        dtype=np.float32,
    )

    front = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(raw["front"], MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
    )
    side = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(raw["side"], MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
    )

    front_jpeg = encode_jpeg(front, jpeg_quality)
    side_jpeg = encode_jpeg(side, jpeg_quality)

    capture_encode_ms = (time.perf_counter() - start) * 1000.0

    obs = {
        "transport/front_image_jpeg": front_jpeg,
        "transport/side_image_jpeg": side_jpeg,
        "observation/state": state,
        "prompt": task,
    }

    stats = {
        "capture_encode_ms": capture_encode_ms,
        "front_bytes": float(len(front_jpeg)),
        "side_bytes": float(len(side_jpeg)),
        "total_bytes": float(len(front_jpeg) + len(side_jpeg)),
    }
    return obs, stats


def validate_actions(actions: Any, current_state: Any) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)

    if actions.ndim == 1:
        actions = actions[None, :]

    if actions.ndim != 2:
        raise ValueError(f"Unexpected action shape: {actions.shape}")

    if actions.shape[1] != EXPECTED_ACTION_DIM:
        raise ValueError(
            f"Expected action shape (N,{EXPECTED_ACTION_DIM}), got {actions.shape}"
        )

    if len(actions) == 0:
        raise ValueError("Received empty action chunk.")

    if not np.isfinite(actions).all():
        raise ValueError("Action chunk contains NaN/Inf.")

    state = np.asarray(current_state, dtype=np.float32)
    if state.shape != (EXPECTED_ACTION_DIM,):
        raise ValueError(f"Unexpected state shape: {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("State contains NaN/Inf.")

    return actions


def action_dict(action_row: Any) -> dict[str, float]:
    values = [float(v) for v in action_row]
    if len(values) != EXPECTED_ACTION_DIM:
        raise ValueError(
            f"Expected {EXPECTED_ACTION_DIM} action values, got {len(values)}."
        )
    return {
        f"{motor}.pos": value
        for motor, value in zip(MOTOR_NAMES, values, strict=True)
    }



def resample_action_sequence(
    actions: np.ndarray,
    output_steps: int | None,
) -> np.ndarray:
    """Linearly resample an absolute-action trajectory along its time axis.

    Endpoints are preserved exactly. This does not extrapolate beyond the model
    chunk; it only slows/smooths traversal through the same predicted path.
    """
    actions = np.asarray(actions, dtype=np.float32)

    if output_steps is None or output_steps == len(actions):
        return actions.copy()

    if len(actions) < 2:
        raise ValueError("Need at least 2 model actions for trajectory resampling.")

    source_t = np.linspace(0.0, 1.0, num=len(actions), dtype=np.float64)
    target_t = np.linspace(0.0, 1.0, num=output_steps, dtype=np.float64)

    resampled = np.empty((output_steps, actions.shape[1]), dtype=np.float32)
    for joint_index in range(actions.shape[1]):
        resampled[:, joint_index] = np.interp(
            target_t,
            source_t,
            actions[:, joint_index].astype(np.float64),
        ).astype(np.float32)

    return resampled

def read_present_position(robot: Any) -> np.ndarray:
    present_dict = robot.bus.sync_read("Present_Position")
    return np.asarray(
        [present_dict[motor] for motor in MOTOR_NAMES],
        dtype=np.float32,
    )


def rate_limit_command(
    requested: np.ndarray,
    previous_command: np.ndarray | None,
    step_limits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Limit command-to-command jumps, not target-to-present tracking lag."""
    requested = np.asarray(requested, dtype=np.float32)

    if previous_command is None:
        return requested.copy(), np.zeros_like(requested)

    delta = requested - previous_command
    safe_delta = np.clip(delta, -step_limits, step_limits)
    safe_target = previous_command + safe_delta
    correction = np.abs(requested - safe_target)
    return safe_target.astype(np.float32), correction.astype(np.float32)


def check_tracking_watchdog(
    safe_target: np.ndarray,
    present: np.ndarray,
    tracking_limits: np.ndarray,
) -> np.ndarray:
    error = np.abs(safe_target - present)
    exceeded = error > tracking_limits
    if np.any(exceeded):
        details = ", ".join(
            f"{MOTOR_NAMES[i]}: error={error[i]:.3f} > limit={tracking_limits[i]:.3f}"
            for i in np.flatnonzero(exceeded)
        )
        raise RuntimeError(
            "TRACKING WATCHDOG TRIGGERED; action was NOT sent. " + details
        )
    return error.astype(np.float32)


@dataclass
class InferenceResult:
    request_id: int
    observation: dict[str, Any]
    observation_stats: dict[str, float]
    result: dict[str, Any] | None
    round_trip_ms: float
    error: BaseException | None


class InferenceWorker:
    """Dedicated network thread; never accesses robot serial/cameras."""

    def __init__(self, host: str, port: int, api_key: str | None, client_cls: Any):
        self._host = host
        self._port = port
        self._api_key = api_key
        self._client_cls = client_cls

        self._requests: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._results: queue.Queue[InferenceResult] = queue.Queue(maxsize=1)

        self._ready = threading.Event()
        self._closed = threading.Event()
        self._connect_error: BaseException | None = None
        self.metadata: dict[str, Any] = {}

        self._thread = threading.Thread(
            target=self._thread_main,
            name="openpi-inference-worker",
            daemon=True,
        )
        self._thread.start()

        if not self._ready.wait(timeout=30.0):
            raise TimeoutError("Timed out connecting inference worker to OpenPI server.")
        if self._connect_error is not None:
            raise RuntimeError("Inference worker failed to connect.") from self._connect_error

    def _thread_main(self) -> None:
        try:
            client = self._client_cls(
                host=self._host,
                port=self._port,
                api_key=self._api_key,
            )
            self.metadata = client.get_server_metadata()
        except BaseException as exc:
            self._connect_error = exc
            self._ready.set()
            return

        self._ready.set()

        while not self._closed.is_set():
            item = self._requests.get()
            if item is None:
                break

            request_id, observation, observation_stats = item

            try:
                start = time.perf_counter()
                result = client.infer(observation)
                round_trip_ms = (time.perf_counter() - start) * 1000.0
                message = InferenceResult(
                    request_id=request_id,
                    observation=observation,
                    observation_stats=observation_stats,
                    result=result,
                    round_trip_ms=round_trip_ms,
                    error=None,
                )
            except BaseException as exc:
                message = InferenceResult(
                    request_id=request_id,
                    observation=observation,
                    observation_stats=observation_stats,
                    result=None,
                    round_trip_ms=float("nan"),
                    error=exc,
                )

            self._results.put(message)

    def submit(
        self,
        request_id: int,
        observation: dict[str, Any],
        observation_stats: dict[str, float],
    ) -> None:
        if self._requests.full():
            raise RuntimeError("Inference request queue is already full.")
        self._requests.put((request_id, observation, observation_stats))

    def get(self, timeout: float | None = None) -> InferenceResult:
        message = self._results.get(timeout=timeout)
        if message.error is not None:
            raise RuntimeError(
                f"OpenPI inference request {message.request_id} failed."
            ) from message.error
        return message

    def try_get(self) -> InferenceResult | None:
        try:
            message = self._results.get_nowait()
        except queue.Empty:
            return None

        if message.error is not None:
            raise RuntimeError(
                f"OpenPI inference request {message.request_id} failed."
            ) from message.error
        return message

    def close(self) -> None:
        self._closed.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass


def describe_inference(message: InferenceResult, actions: np.ndarray) -> None:
    result = message.result or {}
    total_bytes = message.observation_stats["total_bytes"]

    print(
        f"[infer {message.request_id}] "
        f"round_trip={message.round_trip_ms:.1f} ms, "
        f"server={result.get('server_timing')}, "
        f"policy={result.get('policy_timing')}, "
        f"capture+jpeg={message.observation_stats['capture_encode_ms']:.1f} ms, "
        f"jpeg={total_bytes/1024:.1f} KiB, "
        f"chunk={actions.shape}"
    )


def extract_actions(message: InferenceResult) -> np.ndarray:
    result = message.result
    if result is None or "actions" not in result:
        keys = [] if result is None else list(result.keys())
        raise KeyError(f"OpenPI response lacks 'actions'. Keys: {keys}")

    return validate_actions(
        result["actions"],
        message.observation["observation/state"],
    )



def rebase_delta_trained_actions(
    actions: np.ndarray,
    inference_state: np.ndarray,
    execution_state: np.ndarray,
    delta_dims: int = 5,
) -> np.ndarray:
    """Re-anchor server-returned absolute actions to the state at execution time.

    Training used DeltaActions on the first `delta_dims` joints and
    AbsoluteActions on output. Therefore for those joints:

        returned_absolute = predicted_delta + inference_state

    When asynchronous inference is used, `inference_state` can be stale by the
    time the chunk is executed. Recover the predicted deltas and add them to
    the *current* execution state instead:

        rebased_absolute = (returned_absolute - inference_state) + execution_state

    The gripper (dimension 5) is intentionally left unchanged because it was
    trained/executed as an absolute action.
    """
    actions = np.asarray(actions, dtype=np.float32)
    inference_state = np.asarray(inference_state, dtype=np.float32)
    execution_state = np.asarray(execution_state, dtype=np.float32)

    if actions.ndim != 2:
        raise ValueError(f"Expected 2-D action chunk, got {actions.shape}")
    if inference_state.shape != (EXPECTED_ACTION_DIM,):
        raise ValueError(f"Unexpected inference_state shape: {inference_state.shape}")
    if execution_state.shape != (EXPECTED_ACTION_DIM,):
        raise ValueError(f"Unexpected execution_state shape: {execution_state.shape}")
    if not (0 <= delta_dims <= actions.shape[1]):
        raise ValueError(f"Invalid delta_dims={delta_dims}")

    rebased = actions.copy()
    if delta_dims:
        predicted_delta = actions[:, :delta_dims] - inference_state[None, :delta_dims]
        rebased[:, :delta_dims] = (
            execution_state[None, :delta_dims] + predicted_delta
        )
    return rebased.astype(np.float32)


def print_boundary_rebase_diagnostics(
    raw_actions: np.ndarray,
    rebased_actions: np.ndarray,
    inference_state: np.ndarray,
    execution_state: np.ndarray,
    previous_command: np.ndarray | None,
) -> None:
    stale_shift = execution_state - inference_state
    print("Boundary state shift (execution - inference):")
    print(np.round(stale_shift, 3))

    if previous_command is not None:
        raw_jump = raw_actions[0] - previous_command
        rebased_jump = rebased_actions[0] - previous_command
        print("Boundary first-action jump BEFORE rebase:")
        print(np.round(raw_jump, 3))
        print("Boundary first-action jump AFTER rebase:")
        print(np.round(rebased_jump, 3))


def wait_for_follower_settle(
    robot,
    timeout_s: float = 2.0,
    sample_s: float = 0.05,
    motion_threshold_deg: float = 0.5,
    consecutive_samples: int = 4,
):
    """Wait until the real follower has become approximately stationary.

    This does NOT change or resend the target.  The last Goal_Position remains
    active while we wait.  We only watch Present_Position and require several
    consecutive low-motion samples before capturing the next observation.
    """
    start = time.perf_counter()
    previous = read_present_position(robot)

    stable_count = 0
    last_motion = float("inf")

    while True:
        time.sleep(sample_s)

        current = read_present_position(robot)
        delta = np.abs(current - previous)
        last_motion = float(np.max(delta))

        if last_motion <= motion_threshold_deg:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= consecutive_samples:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return current, elapsed_ms, last_motion

        if time.perf_counter() - start >= timeout_s:
            raise RuntimeError(
                "FOLLOWER SETTLE TIMEOUT: "
                f"max motion over last {sample_s*1000:.0f} ms "
                f"was {last_motion:.3f} deg > "
                f"{motion_threshold_deg:.3f} deg."
            )

        previous = current


def main() -> None:
    args = build_parser().parse_args()

    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be in [1,100]")
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if args.execute_horizon < 1:
        raise SystemExit("--execute-horizon must be >= 1")
    if args.resample_steps is not None and args.resample_steps < 2:
        raise SystemExit("--resample-steps must be >= 2 when provided")
    if args.prefetch_after < 1:
        raise SystemExit("--prefetch-after must be >= 1")
    if args.prefetch_after > args.execute_horizon:
        raise SystemExit("--prefetch-after cannot exceed --execute-horizon")
    if args.max_boundary_wait <= 0:
        raise SystemExit("--max-boundary-wait must be positive")
    command_step_limits = np.asarray(args.command_step_limits, dtype=np.float32)
    tracking_error_limits = np.asarray(args.tracking_error_limits, dtype=np.float32)
    if np.any(command_step_limits <= 0):
        raise SystemExit("All --command-step-limits must be positive")
    if np.any(tracking_error_limits <= 0):
        raise SystemExit("All --tracking-error-limits must be positive")

    env = load_env_file(args.env_file)

    follower_port = env_value(env, "FOLLOWER_PORT")
    follower_id = env_value(env, "FOLLOWER_ID")
    front_camera = env_value(env, "FRONT_CAMERA")
    side_camera = env_value(env, "SIDE_CAMERA")
    calibration_root = env_value(env, "HF_LEROBOT_CALIBRATION")

    task = args.task or env_value(env, "TASK_DESCRIPTION")
    fps = args.fps or int(env_value(env, "FPS", "30"))
    width = args.width or int(env_value(env, "CAMERA_WIDTH", "640"))
    height = args.height or int(env_value(env, "CAMERA_HEIGHT", "480"))
    host = args.host or env_value(env, "OPENPI_HOST", "127.0.0.1")
    port = args.port or int(env_value(env, "OPENPI_PORT", "6006"))
    api_key = args.api_key or env_value(env, "OPENPI_API_KEY")

    required = {
        "FOLLOWER_PORT": follower_port,
        "FOLLOWER_ID": follower_id,
        "FRONT_CAMERA": front_camera,
        "SIDE_CAMERA": side_camera,
        "TASK_DESCRIPTION or --task": task,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit("Missing configuration: " + ", ".join(missing))

    # Windows serial ports (COM5) and OpenCV integer indices (0, 1) are not
    # filesystem paths. Leave device validation to LeRobot's connect().
    if os.name != "nt" and not Path(follower_port).exists():
        raise SystemExit(f"Missing follower: {follower_port}")
    for label, path in (("front camera", front_camera), ("side camera", side_camera)):
        if not path.isdecimal() and not Path(path).exists():
            raise SystemExit(f"Missing {label}: {path}")

    if calibration_root:
        os.environ.setdefault("HF_LEROBOT_CALIBRATION", calibration_root)

    try:
        from openpi_client import image_tools, websocket_client_policy
    except ImportError as exc:
        raise SystemExit("openpi-client is not installed.") from exc

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower

    camera_config = {
        "front": OpenCVCameraConfig(
            index_or_path=int(front_camera) if front_camera.isdecimal() else front_camera,
            fps=fps,
            width=width,
            height=height,
        ),
        "side": OpenCVCameraConfig(
            index_or_path=int(side_camera) if side_camera.isdecimal() else side_camera,
            fps=fps,
            width=width,
            height=height,
        ),
    }

    # Disable LeRobot's built-in target-vs-present clamp. v5 uses:
    #   (1) a command-to-command rate limiter, and
    #   (2) a separate target-vs-present tracking watchdog that aborts instead
    #       of continuously rewriting normal demonstration-like commands.
    robot_config = SO101FollowerConfig(
        port=follower_port,
        id=follower_id,
        cameras=camera_config,
        max_relative_target=None,
    )
    robot = SO101Follower(robot_config)

    print("Connecting asynchronous JPEG OpenPI worker...")
    worker = InferenceWorker(
        host=host,
        port=port,
        api_key=api_key,
        client_cls=websocket_client_policy.WebsocketClientPolicy,
    )

    print("Server metadata:", worker.metadata)
    print(f"Task: {task}")
    print(f"JPEG quality: {args.jpeg_quality}")
    print(f"FPS: {fps}")
    print(f"Execute horizon: {args.execute_horizon}")
    print(f"Resample steps: {args.resample_steps}")
    print(f"Prefetch after action: {args.prefetch_after}")
    print("Async delta-action rebase: DISABLED")
    print("Command-step limits:")
    for motor, limit in zip(MOTOR_NAMES, command_step_limits, strict=True):
        print(f"  {motor:<16} {float(limit):.3f}")
    print("Tracking watchdog limits:")
    for motor, limit in zip(MOTOR_NAMES, tracking_error_limits, strict=True):
        print(f"  {motor:<16} {float(limit):.3f}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")

    connected = False

    try:
        robot.connect(calibrate=not args.no_calibration)
        connected = True

        observation, observation_stats = make_observation(
            robot,
            image_tools,
            task,
            args.jpeg_quality,
        )

        request_id = 0
        worker.submit(request_id, observation, observation_stats)
        first = worker.get(timeout=30.0)
        current_actions = extract_actions(first)
        describe_inference(first, current_actions)

        if args.execute_horizon > len(current_actions):
            raise SystemExit(
                f"--execute-horizon={args.execute_horizon} exceeds returned "
                f"chunk length {len(current_actions)}"
            )

        current_state = np.asarray(
            first.observation["observation/state"],
            dtype=np.float32,
        )
        print("Current state:")
        print(current_state)
        print("First predicted action:")
        print(current_actions[0])
        print("First action - state:")
        print(current_actions[0] - current_state)

        if not args.execute:
            print("\nDRY RUN COMPLETE: no action was sent to the robot.")
            return

        total_steps = 0
        chunk_number = 0

        rate_limited_action_count = 0
        rate_limited_joint_event_count = 0

        per_joint_limit_count = {motor: 0 for motor in MOTOR_NAMES}
        per_joint_limit_sum = {motor: 0.0 for motor in MOTOR_NAMES}
        per_joint_limit_max = {motor: 0.0 for motor in MOTOR_NAMES}

        boundary_waits_ms: list[float] = []
        control_intervals_ms: list[float] = []
        inference_round_trips_ms: list[float] = [first.round_trip_ms]
        deadline_miss_count = 0
        previous_command: np.ndarray | None = None
        max_tracking_error = np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32)

        next_request_id = 1

        telemetry_path = Path("gripper_telemetry.csv")
        telemetry_file = telemetry_path.open(
            "w",
            newline="",
            buffering=1,
        )
        telemetry_writer = csv.writer(telemetry_file)
        telemetry_writer.writerow([
            "time_s",
            "step",
            "chunk",
            "chunk_idx",
            "requested_gripper",
            "safe_gripper",
            "present_gripper",
            "present_current_raw",
            "present_load_raw",
            "present_voltage_raw",
            "present_temperature_raw",
            "status_raw",
        ])

        telemetry_start = time.perf_counter()

        # ====================================================
        # Gripper contact / overload protection
        #
        # Smaller normalized gripper value = tighter closing.
        #
        # Empirical data from the real robot:
        # normal current is usually low; sustained 180+ raw
        # appeared when the gripper was mechanically blocked.
        # ====================================================
        SAFE_GRIPPER_MIN = 18.5

        GRIP_CURRENT_TRIGGER = 180
        GRIP_CURRENT_CONSECUTIVE = 2

        # When contact is detected, keep only a small position
        # error so the gripper maintains force without stalling.
        GRIP_HOLD_OFFSET = 1.5

        # Model must clearly command opening before HOLD releases.
        GRIP_RELEASE_MARGIN = 4.0

        # Require meaningful closing error before declaring contact.
        GRIP_MIN_CLOSING_ERROR = 3.0

        grip_hold_active = False
        grip_hold_target = None
        grip_high_current_count = 0

        print(
            f"Gripper telemetry -> {telemetry_path.resolve()}"
        )

        period_s = 1.0 / fps
        last_send_time: float | None = None

        while total_steps < args.steps:
            chunk_number += 1

            model_actions_to_use = min(
                args.execute_horizon,
                len(current_actions),
            )
            model_segment = current_actions[:model_actions_to_use]

            exec_sequence = resample_action_sequence(
                model_segment,
                args.resample_steps,
            )

            actions_to_use = min(
                len(exec_sequence),
                args.steps - total_steps,
            )

            print(
                f"\n===== CHUNK {chunk_number}: "
                f"model {model_actions_to_use}/{len(current_actions)} -> "
                f"execute {actions_to_use}/{len(exec_sequence)} "
                f"resampled action(s) ====="
            )

            next_request_pending = False
            next_message: InferenceResult | None = None

            chunk_start = time.perf_counter()

            for local_index in range(actions_to_use):
                target_time = chunk_start + local_index * period_s
                now = time.perf_counter()

                if target_time > now:
                    time.sleep(target_time - now)
                else:
                    miss_ms = (now - target_time) * 1000.0
                    if local_index > 0 and miss_ms > 2.0:
                        deadline_miss_count += 1

                requested = np.asarray(
                    exec_sequence[local_index],
                    dtype=np.float32,
                ).copy()

                # ------------------------------------------------
                # Gripper command safety + contact HOLD
                # ------------------------------------------------
                raw_gripper = float(requested[5])

                # Keep the original conservative static safety floor.
                if raw_gripper < SAFE_GRIPPER_MIN:
                    requested[5] = SAFE_GRIPPER_MIN
                    print(
                        f"  gripper clamp: "
                        f"{raw_gripper:.3f} -> {SAFE_GRIPPER_MIN:.3f}"
                    )

                # Once contact HOLD is active, ignore further closing
                # requests. Only a clear opening command releases HOLD.
                if grip_hold_active:
                    assert grip_hold_target is not None

                    if raw_gripper >= (
                        grip_hold_target + GRIP_RELEASE_MARGIN
                    ):
                        print(
                            f"  [GRIP HOLD RELEASE] "
                            f"model={raw_gripper:.2f} "
                            f"hold={grip_hold_target:.2f}"
                        )

                        grip_hold_active = False
                        grip_hold_target = None
                        grip_high_current_count = 0

                    else:
                        requested[5] = float(grip_hold_target)

                safe_target, correction = rate_limit_command(
                    requested,
                    previous_command,
                    command_step_limits,
                )
                limited_mask = correction > CLIP_ATOL
                rate_limited = bool(np.any(limited_mask))

                if rate_limited:
                    rate_limited_action_count += 1

                rate_limited_joint_event_count += int(np.count_nonzero(limited_mask))

                limited_desc: list[str] = []
                for joint_index, motor in enumerate(MOTOR_NAMES):
                    if limited_mask[joint_index]:
                        magnitude = float(correction[joint_index])
                        per_joint_limit_count[motor] += 1
                        per_joint_limit_sum[motor] += magnitude
                        per_joint_limit_max[motor] = max(
                            per_joint_limit_max[motor],
                            magnitude,
                        )
                        limited_desc.append(f"{motor}:{magnitude:.3f}")

                # Read follower state immediately before sending and use it only
                # as a watchdog. Normal training-like lag is not clipped away.
                present_before = read_present_position(robot)
                tracking_error = check_tracking_watchdog(
                    safe_target,
                    present_before,
                    tracking_error_limits,
                )
                max_tracking_error = np.maximum(max_tracking_error, tracking_error)

                send_start = time.perf_counter()
                sent_action = robot.send_action(action_dict(safe_target))
                send_done = time.perf_counter()
                previous_command = safe_target.copy()

                if last_send_time is not None:
                    control_intervals_ms.append(
                        (send_done - last_send_time) * 1000.0
                    )
                last_send_time = send_done

                total_steps += 1

                line = (
                    f"step={total_steps}/{args.steps} "
                    f"chunk_idx={local_index} "
                    f"send={(send_done-send_start)*1000.0:.1f}ms "
                    f"rate_limited={rate_limited} "
                    f"track_max={float(tracking_error.max()):.2f}"
                )
                if limited_desc:
                    line += " correction=[" + ", ".join(limited_desc) + "]"
                print(line)

                # ------------------------------------------------
                # Fast gripper current monitor.
                #
                # Read only Present_Current every control cycle.
                # If high current persists for two cycles, verify
                # that the servo is actually failing to reach a
                # tighter target before enabling HOLD.
                # ------------------------------------------------
                try:
                    grip_current_now = robot.bus.read(
                        "Present_Current",
                        "gripper",
                        normalize=False,
                    )
                    grip_current_now = int(grip_current_now)

                    if not grip_hold_active:
                        if grip_current_now >= GRIP_CURRENT_TRIGGER:
                            grip_high_current_count += 1
                        else:
                            grip_high_current_count = 0

                        if (
                            grip_high_current_count
                            >= GRIP_CURRENT_CONSECUTIVE
                        ):
                            present_now = float(
                                read_present_position(robot)[5]
                            )

                            commanded_now = float(safe_target[5])

                            # Positive means the servo is physically
                            # more open than the requested target:
                            # it is trying to close but cannot reach it.
                            closing_error = (
                                present_now - commanded_now
                            )

                            if (
                                closing_error
                                >= GRIP_MIN_CLOSING_ERROR
                            ):
                                grip_hold_target = max(
                                    SAFE_GRIPPER_MIN,
                                    present_now - GRIP_HOLD_OFFSET,
                                )

                                grip_hold_active = True
                                grip_high_current_count = 0

                                print(
                                    "\n"
                                    "  !!! [GRIP HOLD ENGAGED] !!! "
                                    f"current={grip_current_now} "
                                    f"present={present_now:.2f} "
                                    f"sent_target={commanded_now:.2f} "
                                    f"closing_error={closing_error:.2f} "
                                    f"-> hold={grip_hold_target:.2f}"
                                    "\n"
                                )
                            else:
                                # High current without meaningful
                                # closing error is not treated as a
                                # grasp/contact event.
                                grip_high_current_count = 0

                except Exception as e:
                    print(
                        f"  [GRIP CURRENT MONITOR ERROR] {e}"
                    )

                # Read gripper servo telemetry every 5 actions (~6 Hz at 30 Hz).
                # This is diagnostics only: no servo configuration is written.
                if total_steps % 5 == 0:
                    try:
                        current_raw = robot.bus.read(
                            "Present_Current",
                            "gripper",
                            normalize=False,
                        )
                        load_raw = robot.bus.read(
                            "Present_Load",
                            "gripper",
                            normalize=False,
                        )
                        voltage_raw = robot.bus.read(
                            "Present_Voltage",
                            "gripper",
                            normalize=False,
                        )
                        temperature_raw = robot.bus.read(
                            "Present_Temperature",
                            "gripper",
                            normalize=False,
                        )
                        status_raw = robot.bus.read(
                            "Status",
                            "gripper",
                            normalize=False,
                        )

                        present_gripper = float(
                            read_present_position(robot)[5]
                        )

                        telemetry_writer.writerow([
                            f"{time.perf_counter() - telemetry_start:.4f}",
                            total_steps,
                            chunk_number,
                            local_index,
                            f"{float(requested[5]):.4f}",
                            f"{float(safe_target[5]):.4f}",
                            f"{present_gripper:.4f}",
                            current_raw,
                            load_raw,
                            voltage_raw,
                            temperature_raw,
                            status_raw,
                        ])

                        print(
                            "  [GRIP] "
                            f"target={float(safe_target[5]):.2f} "
                            f"present={present_gripper:.2f} "
                            f"current={current_raw} "
                            f"load={load_raw} "
                            f"volt={voltage_raw} "
                            f"temp={temperature_raw} "
                            f"status={status_raw}"
                        )

                    except Exception as e:
                        print(f"  [GRIP TELEMETRY ERROR] {e}")

                should_prefetch = (
                    not next_request_pending
                    and total_steps < args.steps
                    and local_index + 1 >= args.prefetch_after
                    # If prefetch_after == execute_horizon, use synchronous
                    # settle-before-replan.  Do not capture immediately after
                    # sending the final action because the follower is still
                    # physically moving toward that target.
                    and args.prefetch_after < args.execute_horizon
                )

                if should_prefetch:
                    obs, obs_stats = make_observation(
                        robot,
                        image_tools,
                        task,
                        args.jpeg_quality,
                    )

                    worker.submit(
                        next_request_id,
                        obs,
                        obs_stats,
                    )
                    next_request_pending = True

                    print(
                        f"  submitted infer {next_request_id}: "
                        f"capture+jpeg={obs_stats['capture_encode_ms']:.1f}ms, "
                        f"payload={obs_stats['total_bytes']/1024:.1f}KiB"
                    )
                    next_request_id += 1

                if next_request_pending and next_message is None:
                    ready = worker.try_get()
                    if ready is not None:
                        next_message = ready

            if total_steps >= args.steps:
                break

            if not next_request_pending:
                settle_state, settle_ms, settle_motion = wait_for_follower_settle(
                    robot,
                )
                print(
                    f"  settle before infer: {settle_ms:.1f} ms, "
                    f"last_motion={settle_motion:.3f} deg"
                )
                print(
                    "  settled state:",
                    np.array2string(
                        settle_state,
                        precision=3,
                        suppress_small=True,
                    ),
                )

                obs, obs_stats = make_observation(
                    robot,
                    image_tools,
                    task,
                    args.jpeg_quality,
                )
                worker.submit(
                    next_request_id,
                    obs,
                    obs_stats,
                )
                next_request_pending = True
                next_request_id += 1

            boundary_start = time.perf_counter()

            if next_message is None:
                next_message = worker.try_get()

            if next_message is None:
                try:
                    next_message = worker.get(timeout=args.max_boundary_wait)
                except queue.Empty as exc:
                    raise TimeoutError(
                        "Next OpenPI chunk was not ready before "
                        f"--max-boundary-wait={args.max_boundary_wait}s. "
                        "Robot is holding its last commanded position."
                    ) from exc

            boundary_wait_ms = (time.perf_counter() - boundary_start) * 1000.0
            boundary_waits_ms.append(boundary_wait_ms)

            inference_round_trips_ms.append(next_message.round_trip_ms)

            raw_next_actions = extract_actions(next_message)
            inference_state = np.asarray(
                next_message.observation["observation/state"],
                dtype=np.float32,
            )
            execution_state = read_present_position(robot)

            # IMPORTANT:
            # DeltaActions during training encode each future target relative to
            # the observation state, not as an incremental action-to-action delta.
            # The server's AbsoluteActions already reconstructs the correct absolute
            # targets using that observation state. Do NOT rebase the whole chunk to
            # the later execution state, because that double-counts motion already
            # made while the request was in flight.
            current_actions = raw_next_actions.copy()

            describe_inference(next_message, current_actions)

            stale_shift = execution_state - inference_state
            print("Boundary state shift (execution - inference):")
            print(np.round(stale_shift, 3))
            if previous_command is not None:
                raw_jump = current_actions[0] - previous_command
                print("Boundary first-action jump (NO REBASE):")
                print(np.round(raw_jump, 3))

            print(f"Boundary wait: {boundary_wait_ms:.1f} ms")

        print("\n===== ASYNC ROLLOUT SUMMARY =====")
        print(f"Executed actions: {total_steps}")
        print(
            f"Rate-limited actions: {rate_limited_action_count}/{total_steps} "
            f"({100.0*rate_limited_action_count/max(total_steps,1):.1f}%)"
        )
        print(f"Rate-limited joint events: {rate_limited_joint_event_count}")
        print(f"Deadline misses (>2 ms): {deadline_miss_count}")

        if control_intervals_ms:
            arr = np.asarray(control_intervals_ms, dtype=np.float64)
            print(
                "Control interval ms: "
                f"mean={arr.mean():.1f}, "
                f"median={np.median(arr):.1f}, "
                f"max={arr.max():.1f}"
            )
            print(f"Effective control rate: {1000.0/arr.mean():.2f} Hz")

        if boundary_waits_ms:
            arr = np.asarray(boundary_waits_ms, dtype=np.float64)
            print(
                "Boundary wait ms: "
                f"mean={arr.mean():.1f}, "
                f"median={np.median(arr):.1f}, "
                f"max={arr.max():.1f}"
            )

        if inference_round_trips_ms:
            arr = np.asarray(inference_round_trips_ms, dtype=np.float64)
            print(
                "Inference round-trip ms: "
                f"mean={arr.mean():.1f}, "
                f"median={np.median(arr):.1f}, "
                f"max={arr.max():.1f}"
            )

        print("\nPer-joint command-rate limiting:")
        print(
            f"{'joint':<16} {'count':>6} {'mean_corr':>12} {'max_corr':>12} {'max_track':>12}"
        )
        for motor in MOTOR_NAMES:
            count = per_joint_limit_count[motor]
            mean_corr = (
                per_joint_limit_sum[motor] / count if count > 0 else 0.0
            )
            max_corr = per_joint_limit_max[motor]
            joint_index = MOTOR_NAMES.index(motor)
            print(
                f"{motor:<16} {count:>6d} "
                f"{mean_corr:>12.4f} {max_corr:>12.4f} "
                f"{float(max_tracking_error[joint_index]):>12.4f}"
            )

    except KeyboardInterrupt:
        print("\nInterrupted; disconnecting follower.")

    finally:
        worker.close()
        if connected:
            robot.disconnect()
            print("Follower disconnected.")


if __name__ == "__main__":
    main()
