"""Small, strict JSON protocol for a Windows robot and a GPU policy server.

The network carries ONE observation per action window, never one RPC per motor
tick. The server binds loopback; SSH local forwarding is the only entry point.
"""

import base64
import json
import urllib.request

import numpy as np


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TASK = "Pick up the red block and place it on top of the green block."


def check_rows(rows, maximum=10):
    array = np.asarray(rows, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 6 or not 1 <= array.shape[0] <= maximum:
        raise ValueError(f"Expected 1..{maximum} x 6 actions; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Non-finite joint target")
    return array


def check_state(state):
    array = np.asarray(state, dtype=np.float32)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError("Expected six finite joint positions")
    return array


def encode_frame(frame):
    import io

    from PIL import Image

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("Expected uint8 HWC color frame")
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_frame(encoded):
    import io

    from PIL import Image

    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def request(port, payload, *, timeout=60.0):
    data = json.dumps(payload, allow_nan=False).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/rpc", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "Remote policy rejected request"))
    return result
